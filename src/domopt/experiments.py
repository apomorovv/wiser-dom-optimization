"""Reproducible, aggregate-only experiment suite for the challenge notebook."""

from __future__ import annotations

import json
from collections.abc import Callable
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
    base_orders: int
    exact_max_orders: int
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
            base_orders=4,
            exact_max_orders=8,
            hybrid=HybridConfig(
                iterations=1,
                neighborhood_orders=4,
                max_qubo_variables=24,
                max_candidates_per_order=4,
                num_reads=4,
                sweeps=20,
                top_k_recourse=1,
                recourse_time_limit_seconds=5,
                initial_method="default",
                seed=3,
            ),
        )
    if normalized != "full":
        raise ValueError("profile must be 'smoke' or 'full'")
    return ExperimentProfile(
        sizes=(8, 20, 50),
        penalty_scales=(0.5, 1.0, 2.0),
        candidate_counts=(1, 2, 4, 6),
        inventory_shocks=(0.0, 0.1, 0.25),
        seeds=(3, 11, 29, 47),
        noise_levels=(0.0, 0.01, 0.05),
        base_orders=8,
        exact_max_orders=20,
        hybrid=HybridConfig(
            iterations=4,
            neighborhood_orders=8,
            max_qubo_variables=40,
            max_candidates_per_order=5,
            num_reads=24,
            sweeps=100,
            top_k_recourse=4,
            recourse_time_limit_seconds=10,
            initial_method="default",
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
    metrics.pop("violations", None)
    return {
        "experiment": experiment,
        "level": level,
        "order_count": int(problem.orders["order_id"].nunique()),
        "assignment_group_count": int(
            problem.orders.get("assignment_group", problem.orders["order_id"]).nunique()
        ),
        "order_line_count": len(problem.order_lines),
        "candidate_count": len(problem.candidates),
        "configuration": json.dumps(configuration or {}, sort_keys=True),
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


def run_challenge_experiments(
    problem: ProblemData,
    *,
    profile: ExperimentProfile | str = "full",
) -> pd.DataFrame:
    """Run every requested experiment and two challenge comparison controls.

    The supplied ``problem`` should be the unpruned real POC focus universe.
    Results contain aggregate metrics only. No order, SKU, DC, customer, ZIP, or
    commercial lane identifiers are emitted.
    """

    settings = experiment_profile(profile) if isinstance(profile, str) else profile
    rows: list[dict[str, Any]] = []
    pruned = prune_pareto_candidates(problem)
    base = select_shortage_subset(pruned, settings.base_orders)

    # Core comparison required by the challenge: common objective and validator.
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

    # Size scaling on real challenge subsets. Exact MILP is bounded deliberately.
    for size in settings.sizes:
        instance = select_shortage_subset(pruned, size)
        for method, solver in [
            ("greedy", lambda instance=instance: solve_greedy_baseline(instance)),
            (
                "hybrid",
                lambda instance=instance: solve_hybrid(instance, config=settings.hybrid),
            ),
        ]:
            _attempt(
                rows,
                instance,
                solver,
                experiment="size_scaling",
                level=f"requested_orders={size}:{method}",
                configuration={"requested_orders": size, "method": method},
            )
        if size <= settings.exact_max_orders:
            _attempt(
                rows,
                instance,
                lambda instance=instance: solve_classical(
                    instance, time_limit_seconds=60, mip_relative_gap=0.01
                ),
                experiment="size_scaling",
                level=f"requested_orders={size}:classical",
                configuration={"requested_orders": size, "method": "classical"},
            )

    # Penalty sensitivity tests whether routing changes are economically stable.
    for scale in settings.penalty_scales:
        instance = scale_penalties(base, scale)
        for method, solver in [
            ("greedy", lambda instance=instance: solve_greedy_baseline(instance)),
            (
                "hybrid",
                lambda instance=instance: solve_hybrid(instance, config=settings.hybrid),
            ),
        ]:
            _attempt(
                rows,
                instance,
                solver,
                experiment="penalty_weight_sensitivity",
                level=f"scale={scale:g}:{method}",
                configuration={"penalty_scale": scale, "method": method},
            )

    # Candidate-count sensitivity measures search breadth and QUBO width.
    for count in settings.candidate_counts:
        instance = limit_candidates(base, count)
        for method, solver in [
            ("greedy", lambda instance=instance: solve_greedy_baseline(instance)),
            (
                "hybrid",
                lambda instance=instance: solve_hybrid(instance, config=settings.hybrid),
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

    # Inventory shocks test recommendation resilience and graceful degradation.
    for shock in settings.inventory_shocks:
        instance = shock_inventory(base, shock)
        for method, solver in [
            ("default", lambda instance=instance: solve_default_baseline(instance)),
            ("greedy", lambda instance=instance: solve_greedy_baseline(instance)),
            (
                "hybrid",
                lambda instance=instance: solve_hybrid(instance, config=settings.hybrid),
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

    # Local simulator seed and coefficient-noise robustness. This is not QPU noise.
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

    # Pareto pruning ablation on identical order lines and resource tables.
    unpruned_base = select_shortage_subset(problem, settings.base_orders)
    for label, instance in [
        ("without_pruning", unpruned_base),
        ("with_pruning", prune_pareto_candidates(unpruned_base)),
    ]:
        _attempt(
            rows,
            instance,
            lambda instance=instance: solve_hybrid(instance, config=settings.hybrid),
            experiment="pareto_pruning_ablation",
            level=label,
            configuration={"pareto_pruning": label == "with_pruning"},
        )

    # Conflict-aware versus random batches under otherwise identical settings.
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

    # Additional ablation: sampler versus incumbent safety. Random is a weak control.
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

    # Independently generated coordination trap: demonstrates the architectural
    # advantage of revisiting coupled assignments when greedy happens to be myopic.
    # It is intentionally labeled synthetic and cannot support a real-data claim.
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
            lambda: solve_classical(stress, time_limit_seconds=30, mip_relative_gap=0),
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
