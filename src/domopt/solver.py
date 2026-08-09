"""Validated entry point for the recommended DOM solver modes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from .baselines import solve_polished_greedy
from .hybrid import ExactLNSConfig, HybridConfig, solve_exact_lns, solve_hybrid
from .objective import evaluate_solution
from .schemas import ProblemData, Solution
from .validation import validate_solution

SolverMode = Literal["fast", "quality", "hybrid"]


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for the final, never-return-invalid solver facade.

    ``fast`` is the measured production default: greedy routing followed by exact
    fixed-assignment quantity recourse. ``quality`` adds adaptive joint-assignment
    exact LNS. ``hybrid`` retains sampler-assisted LNS as an experimental mode; IBM
    execution still requires ``HybridConfig(allow_remote=True, sampler="ibm-qpu")``.
    """

    mode: SolverMode = "fast"
    fast_time_limit_seconds: float = 30.0
    fast_mip_relative_gap: float = 0.0
    fast_seed: int = 0
    milp_backend: str = "scipy-highs"
    thread_count: int | None = None
    exact_lns: ExactLNSConfig = field(default_factory=ExactLNSConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)

    def validate(self) -> None:
        if self.mode not in {"fast", "quality", "hybrid"}:
            raise ValueError("mode must be 'fast', 'quality', or 'hybrid'")
        if self.fast_time_limit_seconds <= 0:
            raise ValueError("fast_time_limit_seconds must be positive")
        if not 0 <= self.fast_mip_relative_gap < 1:
            raise ValueError("fast_mip_relative_gap must be in [0, 1)")
        if not self.milp_backend.strip():
            raise ValueError("milp_backend must be nonempty")
        if self.thread_count is not None and self.thread_count <= 0:
            raise ValueError("thread_count must be positive when provided")
        self.exact_lns.validate()
        self.hybrid.validate()


def solve_dom(
    problem: ProblemData,
    *,
    config: SolverConfig | None = None,
) -> Solution:
    """Solve a DOM instance and independently validate the returned incumbent.

    The facade makes the evidence-backed method hierarchy explicit without hiding
    the underlying solver metadata. It refuses to return an infeasible solution.
    """

    settings = config or SolverConfig()
    settings.validate()

    if settings.mode == "fast":
        solution = solve_polished_greedy(
            problem,
            backend=settings.milp_backend,
            time_limit_seconds=settings.fast_time_limit_seconds,
            mip_relative_gap=settings.fast_mip_relative_gap,
            seed=settings.fast_seed,
            thread_count=settings.thread_count,
        )
        role = "production-default"
    elif settings.mode == "quality":
        solution = solve_exact_lns(
            problem,
            config=replace(
                settings.exact_lns,
                milp_backend=settings.milp_backend,
                thread_count=settings.thread_count,
            ),
        )
        role = "quality-escalation"
    else:
        solution = solve_hybrid(
            problem,
            config=replace(
                settings.hybrid,
                milp_backend=settings.milp_backend,
                thread_count=settings.thread_count,
            ),
        )
        role = "experimental-hybrid-comparator"

    validation = validate_solution(problem, solution)
    if not validation.is_feasible:
        raise RuntimeError(
            "solver returned an invalid incumbent: " + "; ".join(validation.violations)
        )

    objective_value = evaluate_solution(problem, solution).objective_value
    solution.raw_objective = objective_value
    solution.metadata = {
        **solution.metadata,
        "solver_mode": settings.mode,
        "solver_role": role,
        "independently_validated": True,
        "validation_diagnostics": validation.diagnostics,
        "final_objective": objective_value,
    }
    return solution
