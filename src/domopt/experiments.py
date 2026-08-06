"""Reproducible, aggregate-only experiment suite for the challenge notebook."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import solve_default_baseline, solve_greedy_baseline
from .classical import ClassicalSolverError, solve_classical
from .hybrid import HybridConfig, solve_hybrid
from .metrics import compute_metrics
from .poc import (
    limit_candidates,
    prune_pareto_candidates,
    select_penalty_subset,
    select_shortage_subset,
)
from .schemas import ProblemData, Solution
from .synthetic import make_synthetic_problem


@dataclass(frozen=True)
class ExperimentProfile:
    """A complete grid; ``smoke`` is for CI and ``full`` for submission evidence."""

    sizes: tuple[int, ...]
    penalty_scales: tuple[float, ...]
    candidate_counts: tuple[int, ...]
    inventory_shocks: tuple[float, ...]
    seeds: tuple[int, ...]
    noise_levels: tuple[float, ...]
    qubo_one_hot_multipliers: tuple[float, ...]
    qubo_pair_multipliers: tuple[float, ...]
    base_orders: int
    exact_max_orders: int
    hybrid_max_orders: int
    hybrid: HybridConfig


def experiment_profile(name: str = "full") -> ExperimentProfile:
    normalized = name.strip().lower()
    if normalized == "smoke":
        return ExperimentProfile(
            sizes=(4, 8),
            penalty_scales=(0.75, 1.25),
            candidate_counts=(1, 3),
            inventory_shocks=(0.0, 0.2),
            seeds=(3, 11),
            noise_levels=(0.0, 0.02),
            qubo_one_hot_multipliers=(1.05, 1.5),
            qubo_pair_multipliers=(0.0, 1.0),
            base_orders=4,
            exact_max_orders=8,
            hybrid_max_orders=8,
            hybrid=HybridConfig(
                iterations=1,
                neighborhood_orders=4,
                max_qubo_variables=24,
                max_candidates_per_order=4,
                num_reads=4,
                sweeps=20,
                top_k_recourse=1,
                recourse_time_limit_seconds=5,
                initial_method="greedy",
                seed=3,
            ),
        )
    if normalized != "full":
        raise ValueError("profile must be 'smoke' or 'full'")
    return ExperimentProfile(
        sizes=(8, 20, 50, 100, 250, 372),
        penalty_scales=(0.25, 0.5, 1.0, 2.0, 4.0),
        candidate_counts=(1, 2, 4, 6),
        inventory_shocks=(0.0, 0.1, 0.25, 0.4),
        seeds=(3, 11, 29, 47),
        noise_levels=(0.0, 0.01, 0.03, 0.05),
        qubo_one_hot_multipliers=(1.05, 1.25, 1.5, 2.0),
        qubo_pair_multipliers=(0.0, 0.5, 1.0, 2.0),
        base_orders=20,
        exact_max_orders=20,
        hybrid_max_orders=50,
        hybrid=HybridConfig(
            iterations=3,
            neighborhood_orders=8,
            max_qubo_variables=40,
            max_candidates_per_order=5,
            num_reads=32,
            sweeps=100,
            top_k_recourse=2,
            recourse_time_limit_seconds=8,
            initial_method="greedy",
            seed=11,
        ),
    )


def scale_penalties(problem: ProblemData, scale: float) -> ProblemData:
    if scale < 0:
        raise ValueError("penalty scale must be nonnegative")
    lines = problem.order_lines.copy()
    lines["penalty_per_unfilled_case"] *= float(scale)
    orders = problem.orders.copy()
    for column in [
        "penalty_fixed",
        "penalty_per_cut_sku",
        "penalty_minimum",
        "penalty_maximum",
    ]:
        if column in orders.columns:
            orders[column] *= float(scale)
    return replace(
        problem,
        orders=orders,
        order_lines=lines,
        metadata={**problem.metadata, "penalty_scale": float(scale)},
    )


def shock_inventory(problem: ProblemData, fraction: float) -> ProblemData:
    if not 0 <= fraction < 1:
        raise ValueError("inventory shock must be in [0, 1)")
    inventory = problem.inventory.copy()
    inventory["cumulative_available_cases"] = np.floor(
        inventory["cumulative_available_cases"].astype(float) * (1.0 - fraction)
    ).astype(int)
    return replace(
        problem,
        inventory=inventory,
        metadata={**problem.metadata, "inventory_shock_fraction": float(fraction)},
    )


def _record(
    problem: ProblemData,
    solution: Solution,
    *,
    experiment: str,
    level: str,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = compute_metrics(problem, solution)
    validation = metrics.pop("violations", {})
    violation_categories = [
        key.removesuffix("_violations")
        for key, values in validation.items()
        if key.endswith("_violations") and values
    ]
    settings = configuration or {}
    return {
        "experiment": experiment,
        "level": level,
        "order_count": int(problem.orders["order_id"].nunique()),
        "assignment_group_count": int(
            problem.orders.get("assignment_group", problem.orders["order_id"]).nunique()
        ),
        "order_line_count": len(problem.order_lines),
        "candidate_count": len(problem.candidates),
        "candidate_rows_per_assignment_group": len(problem.candidates)
        / max(
            1,
            int(
                problem.orders.get(
                    "assignment_group", problem.orders["order_id"]
                ).nunique()
            ),
        ),
        "validation_violation_count": len(validation.get("violations", [])),
        "validation_categories": ",".join(sorted(violation_categories)),
        "configuration": json.dumps(settings, sort_keys=True),
        **{
            key: value
            for key, value in settings.items()
            if isinstance(value, (bool, int, float, str)) and key != "method"
        },
        **metrics,
    }


def _attempt(
    rows: list[dict[str, Any]],
    problem: ProblemData,
    solver: Callable[[], Solution],
    *,
    experiment: str,
    level: str,
    configuration: dict[str, Any] | None = None,
) -> None:
    try:
        solution = solver()
        rows.append(
            _record(
                problem,
                solution,
                experiment=experiment,
                level=level,
                configuration=configuration,
            )
        )
    except (ClassicalSolverError, ValueError) as error:
        rows.append(
            {
                "experiment": experiment,
                "level": level,
                "order_count": len(problem.orders),
                "order_line_count": len(problem.order_lines),
                "candidate_count": len(problem.candidates),
                "method": "failed",
                "feasible": False,
                "error_type": type(error).__name__,
                "configuration": json.dumps(configuration or {}, sort_keys=True),
            }
        )


EXPERIMENT_NAMES = (
    "solver_comparison",
    "size_scaling",
    "penalty_weight_sensitivity",
    "candidate_count_sensitivity",
    "inventory_shock",
    "quantum_seed_noise",
    "qubo_penalty_sensitivity",
    "pareto_pruning_ablation",
    "batch_strategy_ablation",
    "sampler_ablation",
    "synthetic_coordination_control",
)


def run_challenge_experiments(
    problem: ProblemData,
    *,
    profile: ExperimentProfile | str = "full",
    experiments: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Run selected privacy-safe studies against one common model and validator.

    ``experiments`` lets the notebook checkpoint each study independently. The
    supplied problem is the unpruned real POC focus universe; row-level business
    identifiers are never emitted.
    """

    settings = experiment_profile(profile) if isinstance(profile, str) else profile
    selected = set(EXPERIMENT_NAMES if experiments is None else experiments)
    unknown = selected - set(EXPERIMENT_NAMES)
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")

    rows: list[dict[str, Any]] = []
    pruned = prune_pareto_candidates(problem)
    base = select_shortage_subset(pruned, settings.base_orders)
    penalty_base = (
        select_penalty_subset(pruned, settings.base_orders)
        if "penalty_weight_sensitivity" in selected
        else None
    )

    if "solver_comparison" in selected:
        for method, solver in [
            ("default", lambda: solve_default_baseline(base)),
            ("greedy", lambda: solve_greedy_baseline(base)),
            (
                "classical",
                lambda: solve_classical(
                    base, time_limit_seconds=60, mip_relative_gap=0.01
                ),
            ),
            ("hybrid", lambda: solve_hybrid(base, config=settings.hybrid)),
        ]:
            _attempt(
                rows,
                base,
                solver,
                experiment="solver_comparison",
                level=method,
                configuration={"method": method},
            )

    if "size_scaling" in selected:
        seen_group_counts: set[int] = set()
        for size in settings.sizes:
            instance = select_shortage_subset(pruned, size)
            actual_groups = int(instance.orders["assignment_group"].nunique())
            if actual_groups in seen_group_counts:
                continue
            seen_group_counts.add(actual_groups)
            common = {
                "requested_assignment_groups": size,
                "actual_assignment_groups": actual_groups,
            }
            _attempt(
                rows,
                instance,
                lambda instance=instance: solve_greedy_baseline(instance),
                experiment="size_scaling",
                level=f"groups={actual_groups}:greedy",
                configuration={**common, "method": "greedy"},
            )
            if size <= settings.hybrid_max_orders:
                _attempt(
                    rows,
                    instance,
                    lambda instance=instance: solve_hybrid(
                        instance, config=settings.hybrid
                    ),
                    experiment="size_scaling",
                    level=f"groups={actual_groups}:hybrid",
                    configuration={**common, "method": "hybrid"},
                )
            if size <= settings.exact_max_orders:
                _attempt(
                    rows,
                    instance,
                    lambda instance=instance: solve_classical(
                        instance, time_limit_seconds=60, mip_relative_gap=0.01
                    ),
                    experiment="size_scaling",
                    level=f"groups={actual_groups}:classical",
                    configuration={**common, "method": "classical"},
                )

    if "penalty_weight_sensitivity" in selected:
        assert penalty_base is not None
        for scale in settings.penalty_scales:
            instance = scale_penalties(penalty_base, scale)
            for method, solver in [
                ("greedy", lambda instance=instance: solve_greedy_baseline(instance)),
                (
                    "hybrid",
                    lambda instance=instance: solve_hybrid(
                        instance, config=settings.hybrid
                    ),
                ),
            ]:
                _attempt(
                    rows,
                    instance,
                    solver,
                    experiment="penalty_weight_sensitivity",
                    level=f"scale={scale:g}:{method}",
                    configuration={
                        "penalty_scale": scale,
                        "method": method,
                        "selection_basis": "active_penalty_exposure",
                    },
                )

    if "candidate_count_sensitivity" in selected:
        for count in settings.candidate_counts:
            instance = limit_candidates(base, count)
            for method, solver in [
                ("greedy", lambda instance=instance: solve_greedy_baseline(instance)),
                (
                    "hybrid",
                    lambda instance=instance: solve_hybrid(
                        instance, config=settings.hybrid
                    ),
                ),
            ]:
                _attempt(
                    rows,
                    instance,
                    solver,
                    experiment="candidate_count_sensitivity",
                    level=f"count={count}:{method}",
                    configuration={"candidate_limit": count, "method": method},
                )

    if "inventory_shock" in selected:
        for shock in settings.inventory_shocks:
            instance = shock_inventory(base, shock)
            for method, solver in [
                ("default", lambda instance=instance: solve_default_baseline(instance)),
                ("greedy", lambda instance=instance: solve_greedy_baseline(instance)),
                (
                    "hybrid",
                    lambda instance=instance: solve_hybrid(
                        instance, config=settings.hybrid
                    ),
                ),
            ]:
                _attempt(
                    rows,
                    instance,
                    solver,
                    experiment="inventory_shock",
                    level=f"shock={shock:g}:{method}",
                    configuration={"inventory_shock": shock, "method": method},
                )

    if "quantum_seed_noise" in selected:
        for seed in settings.seeds:
            for noise in settings.noise_levels:
                hybrid = replace(
                    settings.hybrid,
                    seed=seed,
                    qubo_noise_relative_sigma=noise,
                )
                _attempt(
                    rows,
                    base,
                    lambda hybrid=hybrid: solve_hybrid(base, config=hybrid),
                    experiment="quantum_seed_noise",
                    level=f"seed={seed}:noise={noise:g}",
                    configuration={
                        "seed": seed,
                        "coefficient_noise_relative_sigma": noise,
                        "noise_scope": "local QUBO coefficient perturbation; not QPU noise",
                    },
                )

    if "qubo_penalty_sensitivity" in selected:
        for one_hot in settings.qubo_one_hot_multipliers:
            for pair in settings.qubo_pair_multipliers:
                hybrid = replace(
                    settings.hybrid,
                    one_hot_penalty_multiplier=one_hot,
                    pair_penalty_multiplier=pair,
                )
                _attempt(
                    rows,
                    base,
                    lambda hybrid=hybrid: solve_hybrid(base, config=hybrid),
                    experiment="qubo_penalty_sensitivity",
                    level=f"one_hot={one_hot:g}:pair={pair:g}",
                    configuration={
                        "one_hot_penalty_multiplier": one_hot,
                        "pair_penalty_multiplier": pair,
                    },
                )

    if "pareto_pruning_ablation" in selected:
        unpruned_base = select_shortage_subset(problem, settings.base_orders)
        for label, instance in [
            ("without_pruning", unpruned_base),
            ("with_pruning", prune_pareto_candidates(unpruned_base)),
        ]:
            _attempt(
                rows,
                instance,
                lambda instance=instance: solve_hybrid(
                    instance, config=settings.hybrid
                ),
                experiment="pareto_pruning_ablation",
                level=label,
                configuration={"pareto_pruning": label == "with_pruning"},
            )

    if "batch_strategy_ablation" in selected:
        for strategy in ["random", "conflict"]:
            hybrid = replace(settings.hybrid, batch_strategy=strategy)
            _attempt(
                rows,
                base,
                lambda hybrid=hybrid: solve_hybrid(base, config=hybrid),
                experiment="batch_strategy_ablation",
                level=strategy,
                configuration={"batch_strategy": strategy},
            )

    if "sampler_ablation" in selected:
        for sampler in ["random", "simulated_annealing"]:
            hybrid = replace(settings.hybrid, sampler=sampler)
            _attempt(
                rows,
                base,
                lambda hybrid=hybrid: solve_hybrid(base, config=hybrid),
                experiment="sampler_ablation",
                level=sampler,
                configuration={"sampler": sampler},
            )

    if "synthetic_coordination_control" in selected:
        stress = make_synthetic_problem(order_count=8, seed=2)
        stress_hybrid = HybridConfig(
            iterations=4,
            neighborhood_orders=8,
            max_qubo_variables=40,
            max_candidates_per_order=5,
            num_reads=16,
            sweeps=80,
            top_k_recourse=4,
            recourse_time_limit_seconds=8,
            initial_method="greedy",
            seed=2,
        )
        for method, solver in [
            ("greedy", lambda: solve_greedy_baseline(stress)),
            (
                "classical",
                lambda: solve_classical(
                    stress, time_limit_seconds=30, mip_relative_gap=0
                ),
            ),
            ("hybrid", lambda: solve_hybrid(stress, config=stress_hybrid)),
        ]:
            _attempt(
                rows,
                stress,
                solver,
                experiment="synthetic_coordination_control",
                level=method,
                configuration={
                    "method": method,
                    "data_scope": "independently generated synthetic control",
                    "generator_seed": 2,
                },
            )

    result = pd.DataFrame(rows)
    ordered = [
        "experiment",
        "level",
        "method",
        "feasible",
        "order_count",
        "assignment_group_count",
        "order_line_count",
        "candidate_count",
        "objective_value",
        "case_fill_rate",
        "penalty_cost",
        "shipping_cost",
        "reassigned_orders",
        "runtime_seconds",
        "optimality_gap",
        "initial_objective",
        "hybrid_improvement",
        "maximum_qubo_variables",
        "accepted_moves",
        "configuration",
    ]
    return result.reindex(columns=ordered + [c for c in result if c not in ordered])


def run_remote_qpu_validation(
    *,
    sampler: str = "dwave-qpu",
    allow_remote: bool = False,
    num_reads: int = 100,
    seed: int = 19,
) -> pd.DataFrame:
    """Run one privacy-safe hardware check on a generated synthetic QUBO.

    Real Nestle coefficients are deliberately excluded. This validates remote
    submission, embedding, sampling, repair, exact recourse, and metadata capture;
    it is not a quantum-advantage experiment.
    """

    if not allow_remote:
        raise ValueError("Remote QPU validation requires allow_remote=True")
    if sampler not in {"dwave-qpu", "dwave-hybrid"}:
        raise ValueError("sampler must be 'dwave-qpu' or 'dwave-hybrid'")
    problem = make_synthetic_problem(
        order_count=4,
        dc_count=3,
        candidates_per_order=3,
        seed=seed,
    )
    config = HybridConfig(
        iterations=1,
        neighborhood_orders=4,
        max_qubo_variables=20,
        max_candidates_per_order=3,
        sampler=sampler,
        num_reads=num_reads,
        sweeps=50,
        top_k_recourse=3,
        recourse_time_limit_seconds=10,
        initial_method="greedy",
        seed=seed,
        allow_remote=True,
    )
    rows: list[dict[str, Any]] = []
    for method, solver in [
        ("greedy", lambda: solve_greedy_baseline(problem)),
        (
            "classical",
            lambda: solve_classical(
                problem, time_limit_seconds=30, mip_relative_gap=0
            ),
        ),
        ("remote_hybrid", lambda: solve_hybrid(problem, config=config)),
    ]:
        _attempt(
            rows,
            problem,
            solver,
            experiment="remote_qpu_validation",
            level=method,
            configuration={
                "method": method,
                "sampler": sampler if method == "remote_hybrid" else "none",
                "data_scope": "generated synthetic coefficients only",
                "seed": seed,
            },
        )
    return pd.DataFrame(rows)


def write_experiment_results(results: pd.DataFrame, path: str | Path) -> Path:
    """Write aggregate experiment evidence; reject identifier-like columns."""

    forbidden = {"order_id", "sku_id", "dc_id", "candidate_id", "customer", "zip"}
    bad = [column for column in results.columns if column.lower() in forbidden]
    if bad:
        raise ValueError(f"Aggregate experiment output contains forbidden columns: {bad}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    return output
