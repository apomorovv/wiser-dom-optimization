"""Reproducible, aggregate-only experiment suite for the challenge notebook."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import (
    _InventoryState,
    solve_default_baseline,
    solve_greedy_baseline,
    solve_polished_greedy,
)
from .classical import ClassicalSolverError, solve_classical
from .hybrid import (
    ExactLNSConfig,
    HybridConfig,
    build_neighborhood_qubo,
    solve_exact_lns,
    solve_hybrid,
)
from .metrics import compute_metrics
from .pipeline import current_source_state, problem_fingerprint
from .poc import (
    PocConfig,
    limit_candidates,
    load_poc_problem,
    prune_pareto_candidates,
    select_penalty_subset,
    select_shortage_subset,
)
from .provenance import EXPERIMENT_SCHEMA_VERSION, runtime_environment_json
from .quantum import IBM_MITIGATION_STRATEGIES, QuantumSolverError
from .schemas import ProblemData, Solution
from .synthetic import make_synthetic_problem


@dataclass(frozen=True)
class ExperimentProfile:
    """A complete grid; ``smoke`` is for CI and ``full`` for submission evidence."""

    sizes: tuple[int, ...]
    synthetic_sizes: tuple[int, ...]
    penalty_scales: tuple[float, ...]
    candidate_counts: tuple[int, ...]
    inventory_shocks: tuple[float, ...]
    seeds: tuple[int, ...]
    noise_levels: tuple[float, ...]
    readout_noise_levels: tuple[float, ...]
    qubo_one_hot_multipliers: tuple[float, ...]
    qubo_pair_multipliers: tuple[float, ...]
    base_orders: int
    exact_max_orders: int
    hybrid_max_orders: int
    lns_max_orders: int
    scaling_repetitions: int
    hybrid: HybridConfig
    exact_lns: ExactLNSConfig


def experiment_profile(name: str = "full") -> ExperimentProfile:
    normalized = name.strip().lower()
    if normalized == "smoke":
        return ExperimentProfile(
            sizes=(4, 8),
            synthetic_sizes=(10, 25),
            penalty_scales=(0.75, 1.25),
            candidate_counts=(1, 3),
            inventory_shocks=(0.0, 0.2),
            seeds=(3, 11),
            noise_levels=(0.0, 0.02),
            readout_noise_levels=(0.0, 0.02),
            qubo_one_hot_multipliers=(1.05, 1.5),
            qubo_pair_multipliers=(0.0, 1.0),
            base_orders=8,
            exact_max_orders=8,
            hybrid_max_orders=8,
            lns_max_orders=8,
            scaling_repetitions=1,
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
            exact_lns=ExactLNSConfig(
                iterations=1,
                initial_neighborhood_groups=4,
                minimum_neighborhood_groups=2,
                maximum_neighborhood_groups=6,
                maximum_neighborhood_orders=12,
                maximum_local_fulfillment_variables=3_000,
                local_time_limit_seconds=5,
                mip_relative_gap=0.01,
                seed=3,
            ),
        )
    if normalized != "full":
        raise ValueError("profile must be 'smoke' or 'full'")
    return ExperimentProfile(
        sizes=(8, 20, 50, 100, 250, 372),
        synthetic_sizes=(20, 50, 100, 250, 500),
        penalty_scales=(0.25, 0.5, 1.0, 2.0, 4.0),
        candidate_counts=(1, 2, 4, 6),
        inventory_shocks=(0.0, 0.1, 0.25, 0.4),
        seeds=(3, 11, 29, 47),
        noise_levels=(0.0, 0.01, 0.03, 0.05),
        readout_noise_levels=(0.0, 0.005, 0.01, 0.02),
        qubo_one_hot_multipliers=(1.05, 1.25, 1.5, 2.0),
        qubo_pair_multipliers=(0.0, 0.5, 1.0, 2.0),
        base_orders=20,
        exact_max_orders=20,
        hybrid_max_orders=50,
        lns_max_orders=372,
        scaling_repetitions=3,
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
        exact_lns=ExactLNSConfig(
            iterations=4,
            initial_neighborhood_groups=6,
            minimum_neighborhood_groups=4,
            maximum_neighborhood_groups=12,
            maximum_neighborhood_orders=32,
            maximum_local_fulfillment_variables=6_000,
            local_time_limit_seconds=6,
            mip_relative_gap=0.01,
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
    shocked = replace(
        problem,
        inventory=inventory,
        metadata={**problem.metadata, "inventory_shock_fraction": float(fraction)},
    )
    if {
        "estimated_fill_cases",
        "estimated_fulfilled_value",
    } <= set(problem.candidates.columns):
        state = _InventoryState(inventory)
        lines_by_order = {
            str(order_id): group
            for order_id, group in problem.order_lines.groupby("order_id", sort=False)
        }
        candidates = problem.candidates.copy()
        estimated_cases: list[int] = []
        estimated_values: list[float] = []
        for candidate in candidates.itertuples(index=False):
            total_cases = 0
            total_value = 0.0
            for line in lines_by_order[str(candidate.order_id)].itertuples(index=False):
                available = state.available(
                    str(candidate.dc_id),
                    str(line.sku_id),
                    pd.Timestamp(candidate.pgi_date),
                )
                quantity = min(int(line.demand_cases), int(available))
                total_cases += quantity
                total_value += quantity * float(line.unit_value)
            estimated_cases.append(total_cases)
            estimated_values.append(total_value)
        candidates["estimated_fill_cases"] = estimated_cases
        candidates["estimated_fulfilled_value"] = estimated_values
        orders = problem.orders.copy()
        default_fill = (
            candidates.loc[candidates["is_default"].astype(bool)]
            .set_index("order_id")["estimated_fill_cases"]
            .to_dict()
        )
        orders["default_fillable_cases"] = (
            orders["order_id"].astype(str).map(default_fill).fillna(0).astype(int)
        )
        shocked = replace(
            shocked,
            orders=orders,
            candidates=candidates,
            metadata={
                **shocked.metadata,
                "scenario_candidate_estimates_recomputed": True,
            },
        )
    return shocked


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
        "experiment_schema_version": str(EXPERIMENT_SCHEMA_VERSION),
        "schema_version": problem.metadata.get("schema_version", "unknown"),
        "assumption_version": problem.metadata.get("assumption_version", "unknown"),
        "bundle_sha256": problem.metadata.get("bundle_sha256"),
        "problem_sha256": problem_fingerprint(problem),
        "objective_version": "fulfilled-value-minus-thresholded-penalty-minus-shipping-v2",
        "runtime_environment": runtime_environment_json(),
        **current_source_state(),
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
    except (ClassicalSolverError, QuantumSolverError, ValueError) as error:
        settings = configuration or {}
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
                "error_message": str(error)[:500],
                "configuration": json.dumps(settings, sort_keys=True),
                "experiment_schema_version": str(EXPERIMENT_SCHEMA_VERSION),
                "schema_version": problem.metadata.get("schema_version", "unknown"),
                "assumption_version": problem.metadata.get(
                    "assumption_version", "unknown"
                ),
                "bundle_sha256": problem.metadata.get("bundle_sha256"),
                "problem_sha256": problem_fingerprint(problem),
                "sampler_backend": settings.get("sampler"),
                "runtime_environment": runtime_environment_json(),
                **current_source_state(),
                **{
                    key: value
                    for key, value in settings.items()
                    if isinstance(value, (bool, int, float, str)) and key != "method"
                },
            }
        )


EXPERIMENT_NAMES = (
    "solver_comparison",
    "size_scaling",
    "synthetic_scaling",
    "candidate_dc_scope_sensitivity",
    "penalty_weight_sensitivity",
    "candidate_count_sensitivity",
    "inventory_shock",
    "qubo_coefficient_noise",
    "qaoa_readout_noise",
    "qubo_penalty_sensitivity",
    "pareto_pruning_ablation",
    "batch_strategy_ablation",
    "sampler_ablation",
    "synthetic_coordination_control",
)


def _coordination_control(
    settings: ExperimentProfile,
) -> tuple[ProblemData, HybridConfig]:
    """Return a small control with a verified greedy-assignment gap.

    Seed 6 has a 197.2 objective-unit gap after exact fixed-assignment polish,
    so sampler gains cannot be manufactured by the classical quantity recourse.
    Four groups also keep the feasible gate-model statevector at only 4^4 states.
    """

    problem = make_ibm_hardware_study_problem()
    config = replace(
        settings.hybrid,
        iterations=max(3, settings.hybrid.iterations),
        neighborhood_orders=4,
        max_qubo_variables=24,
        max_candidates_per_order=5,
        # The p=1 Dicke/XY control needs enough finite shots to expose its
        # low-probability improving assignment. Hold the same read budget for
        # every stochastic sampler in this coupled-control family.
        num_reads=max(128, settings.hybrid.num_reads),
        sweeps=max(100, settings.hybrid.sweeps),
        top_k_recourse=max(6, settings.hybrid.top_k_recourse),
        recourse_time_limit_seconds=max(
            5.0, settings.hybrid.recourse_time_limit_seconds
        ),
        seed=6,
        qubo_noise_relative_sigma=0.0,
        qaoa_restarts=max(6, settings.hybrid.qaoa_restarts),
    )
    return problem, config


def make_ibm_hardware_study_problem() -> ProblemData:
    """Return the public synthetic instance shared by IBM and local controls."""

    return make_synthetic_problem(order_count=4, seed=6)


def ibm_hardware_study_logical_qubits() -> int:
    """Derive the IBM circuit width from the actual synthetic study QUBO."""

    problem, settings = _coordination_control(experiment_profile("full"))
    incumbent = solve_greedy_baseline(problem)
    model, _, _, _ = build_neighborhood_qubo(
        problem,
        incumbent,
        one_hot_penalty_multiplier=settings.one_hot_penalty_multiplier,
        pair_penalty_multiplier=settings.pair_penalty_multiplier,
        max_candidates_per_order=settings.max_candidates_per_order,
    )
    return len(model.variable_names)


def _assignment_policy(solution: Solution) -> dict[str, str | None]:
    return {
        str(row.order_id): (
            None if bool(row.is_unassigned) else str(row.candidate_id)
        )
        for row in solution.assignments.itertuples(index=False)
    }


def _solve_fixed_routing_recourse(
    problem: ProblemData,
    policy: dict[str, str | None],
) -> Solution:
    solution = solve_classical(
        problem,
        time_limit_seconds=60,
        fixed_assignments=policy,
    )
    solution.method = "fixed_routing_recourse"
    solution.assignments["method"] = solution.method
    return solution


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
    # Common-objective evidence uses the unpruned universe. The score-based
    # Pareto reduction is deliberately isolated to its own heuristic ablation.
    base = select_shortage_subset(problem, settings.base_orders)
    penalty_base = (
        select_penalty_subset(problem, settings.base_orders)
        if "penalty_weight_sensitivity" in selected
        else None
    )

    if "solver_comparison" in selected:
        for method, solver in [
            ("default", lambda: solve_default_baseline(base)),
            ("greedy", lambda: solve_greedy_baseline(base)),
            (
                "polished_greedy",
                lambda: solve_polished_greedy(
                    base,
                    time_limit_seconds=30,
                    mip_relative_gap=0.01,
                    seed=settings.exact_lns.seed,
                ),
            ),
            (
                "exact_lns",
                lambda: solve_exact_lns(base, config=settings.exact_lns),
            ),
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
            instance = select_shortage_subset(problem, size)
            actual_groups = int(instance.orders["assignment_group"].nunique())
            if actual_groups in seen_group_counts:
                continue
            seen_group_counts.add(actual_groups)
            for repetition in range(settings.scaling_repetitions):
                common = {
                    "requested_assignment_groups": size,
                    "actual_assignment_groups": actual_groups,
                    "repetition": repetition + 1,
                }
                _attempt(
                    rows,
                    instance,
                    lambda instance=instance: solve_greedy_baseline(instance),
                    experiment="size_scaling",
                    level=f"groups={actual_groups}:greedy:rep={repetition + 1}",
                    configuration={**common, "method": "greedy"},
                )
                _attempt(
                    rows,
                    instance,
                    lambda instance=instance, repetition=repetition: solve_polished_greedy(
                        instance,
                        time_limit_seconds=30,
                        mip_relative_gap=0.01,
                        seed=settings.exact_lns.seed + repetition,
                    ),
                    experiment="size_scaling",
                    level=(
                        f"groups={actual_groups}:polished_greedy:rep={repetition + 1}"
                    ),
                    configuration={**common, "method": "polished_greedy"},
                )
                if size <= settings.lns_max_orders:
                    lns = replace(
                        settings.exact_lns,
                        seed=settings.exact_lns.seed + repetition,
                        polish_initial_incumbent=True,
                    )
                    _attempt(
                        rows,
                        instance,
                        lambda instance=instance, lns=lns: solve_exact_lns(
                            instance, config=lns
                        ),
                        experiment="size_scaling",
                        level=f"groups={actual_groups}:exact_lns:rep={repetition + 1}",
                        configuration={**common, "method": "exact_lns"},
                    )
                if size <= settings.hybrid_max_orders:
                    hybrid = replace(
                        settings.hybrid,
                        seed=settings.hybrid.seed + repetition,
                    )
                    _attempt(
                        rows,
                        instance,
                        lambda instance=instance, hybrid=hybrid: solve_hybrid(
                            instance, config=hybrid
                        ),
                        experiment="size_scaling",
                        level=f"groups={actual_groups}:hybrid:rep={repetition + 1}",
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
                        level=f"groups={actual_groups}:classical:rep={repetition + 1}",
                        configuration={**common, "method": "classical"},
                    )

    if "synthetic_scaling" in selected:
        for size in settings.synthetic_sizes:
            for repetition in range(settings.scaling_repetitions):
                generator_seed = 10_000 + 97 * size + repetition
                instance = make_synthetic_problem(
                    order_count=size,
                    seed=generator_seed,
                )
                common = {
                    "requested_assignment_groups": size,
                    "actual_assignment_groups": size,
                    "repetition": repetition + 1,
                    "generator_seed": generator_seed,
                    "data_scope": "independently generated synthetic scaling control",
                }
                for method, solver in [
                    ("greedy", lambda instance=instance: solve_greedy_baseline(instance)),
                    (
                        "exact_lns",
                        lambda instance=instance, repetition=repetition: solve_exact_lns(
                            instance,
                            config=replace(
                                settings.exact_lns,
                                seed=settings.exact_lns.seed + repetition,
                                polish_initial_incumbent=True,
                            ),
                        ),
                    ),
                ]:
                    _attempt(
                        rows,
                        instance,
                        solver,
                        experiment="synthetic_scaling",
                        level=f"orders={size}:{method}:rep={repetition + 1}",
                        configuration={**common, "method": method},
                    )
                if size <= settings.hybrid_max_orders:
                    hybrid = replace(
                        settings.hybrid,
                        seed=settings.hybrid.seed + repetition,
                    )
                    _attempt(
                        rows,
                        instance,
                        lambda instance=instance, hybrid=hybrid: solve_hybrid(
                            instance, config=hybrid
                        ),
                        experiment="synthetic_scaling",
                        level=f"orders={size}:hybrid:rep={repetition + 1}",
                        configuration={**common, "method": "hybrid"},
                    )
                if size <= settings.exact_max_orders:
                    _attempt(
                        rows,
                        instance,
                        lambda instance=instance: solve_classical(
                            instance, time_limit_seconds=60, mip_relative_gap=0.01
                        ),
                        experiment="synthetic_scaling",
                        level=f"orders={size}:classical:rep={repetition + 1}",
                        configuration={**common, "method": "classical"},
                    )

    if "candidate_dc_scope_sensitivity" in selected:
        if problem.source_dir is None:
            raise ValueError(
                "candidate_dc_scope_sensitivity requires a POC problem with source_dir"
            )
        for scope in ["focus_default_dcs", "network_intersection"]:
            scoped_problem = load_poc_problem(
                problem.source_dir,
                config=PocConfig(
                    pareto_prune=False,
                    candidate_dc_scope=scope,
                ),
                strict_bundle_audit=False,
            )
            instance = select_shortage_subset(scoped_problem, settings.base_orders)
            for method, solver in [
                (
                    "polished_greedy",
                    lambda instance=instance: solve_polished_greedy(
                        instance,
                        time_limit_seconds=30,
                        mip_relative_gap=0.01,
                        seed=settings.exact_lns.seed,
                    ),
                ),
                (
                    "exact_lns",
                    lambda instance=instance: solve_exact_lns(
                        instance, config=settings.exact_lns
                    ),
                ),
            ]:
                _attempt(
                    rows,
                    instance,
                    solver,
                    experiment="candidate_dc_scope_sensitivity",
                    level=f"scope={scope}:{method}",
                    configuration={
                        "candidate_dc_scope": scope,
                        "method": method,
                    },
                )

    if "penalty_weight_sensitivity" in selected:
        assert penalty_base is not None
        for scale in settings.penalty_scales:
            instance = scale_penalties(penalty_base, scale)
            for method, solver in [
                ("greedy", lambda instance=instance: solve_greedy_baseline(instance)),
                (
                    "polished_greedy",
                    lambda instance=instance: solve_polished_greedy(
                        instance,
                        time_limit_seconds=30,
                        mip_relative_gap=0.01,
                        seed=settings.exact_lns.seed,
                    ),
                ),
                (
                    "exact_lns",
                    lambda instance=instance: solve_exact_lns(
                        instance, config=settings.exact_lns
                    ),
                ),
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
                    "polished_greedy",
                    lambda instance=instance: solve_polished_greedy(
                        instance,
                        time_limit_seconds=30,
                        mip_relative_gap=0.01,
                        seed=settings.exact_lns.seed,
                    ),
                ),
                (
                    "exact_lns",
                    lambda instance=instance: solve_exact_lns(
                        instance, config=settings.exact_lns
                    ),
                ),
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
        nominal_policy = _assignment_policy(solve_greedy_baseline(base))
        for shock in settings.inventory_shocks:
            instance = shock_inventory(base, shock)
            for method, solver in [
                ("default", lambda instance=instance: solve_default_baseline(instance)),
                ("greedy", lambda instance=instance: solve_greedy_baseline(instance)),
                (
                    "fixed_routing_recourse",
                    lambda instance=instance: _solve_fixed_routing_recourse(
                        instance, nominal_policy
                    ),
                ),
                (
                    "exact_lns",
                    lambda instance=instance: solve_exact_lns(
                        instance, config=settings.exact_lns
                    ),
                ),
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

    if "qubo_coefficient_noise" in selected:
        control, control_hybrid = _coordination_control(settings)
        for seed in settings.seeds:
            for noise in settings.noise_levels:
                hybrid = replace(
                    control_hybrid,
                    seed=seed,
                    qubo_noise_relative_sigma=noise,
                )
                _attempt(
                    rows,
                    control,
                    lambda hybrid=hybrid: solve_hybrid(control, config=hybrid),
                    experiment="qubo_coefficient_noise",
                    level=f"seed={seed}:noise={noise:g}",
                    configuration={
                        "seed": seed,
                        "coefficient_noise_relative_sigma": noise,
                        "noise_scope": "local QUBO coefficient perturbation; not QPU noise",
                        "data_scope": "independently generated coupled synthetic control",
                    },
                )

    if "qaoa_readout_noise" in selected:
        control, control_hybrid = _coordination_control(settings)
        for seed in settings.seeds:
            for probability in settings.readout_noise_levels:
                hybrid = replace(
                    control_hybrid,
                    sampler="qaoa_statevector",
                    iterations=1,
                    seed=seed,
                    qaoa_readout_bitflip_probability=probability,
                )
                _attempt(
                    rows,
                    control,
                    lambda hybrid=hybrid: solve_hybrid(control, config=hybrid),
                    experiment="qaoa_readout_noise",
                    level=f"seed={seed}:readout={probability:g}",
                    configuration={
                        "seed": seed,
                        "qaoa_readout_bitflip_probability": probability,
                        "noise_scope": (
                            "independent symmetric measurement bit flips after "
                            "ideal local QAOA; not gate/decoherence/hardware noise"
                        ),
                        "data_scope": (
                            "independently generated coupled synthetic control"
                        ),
                    },
                )

    if "qubo_penalty_sensitivity" in selected:
        control, control_hybrid = _coordination_control(settings)
        for one_hot in settings.qubo_one_hot_multipliers:
            for pair in settings.qubo_pair_multipliers:
                hybrid = replace(
                    control_hybrid,
                    one_hot_penalty_multiplier=one_hot,
                    pair_penalty_multiplier=pair,
                )
                _attempt(
                    rows,
                    control,
                    lambda hybrid=hybrid: solve_hybrid(control, config=hybrid),
                    experiment="qubo_penalty_sensitivity",
                    level=f"one_hot={one_hot:g}:pair={pair:g}",
                    configuration={
                        "one_hot_penalty_multiplier": one_hot,
                        "pair_penalty_multiplier": pair,
                        "data_scope": "independently generated coupled synthetic control",
                    },
                )

    if "pareto_pruning_ablation" in selected:
        unpruned_base = select_shortage_subset(problem, settings.base_orders)
        for seed in settings.seeds:
            for label, instance in [
                ("without_pruning", unpruned_base),
                ("with_pruning", prune_pareto_candidates(unpruned_base)),
            ]:
                hybrid = replace(settings.hybrid, seed=seed)
                _attempt(
                    rows,
                    instance,
                    lambda instance=instance, hybrid=hybrid: solve_hybrid(
                        instance, config=hybrid
                    ),
                    experiment="pareto_pruning_ablation",
                    level=f"{label}:seed={seed}",
                    configuration={
                        "pareto_pruning": label == "with_pruning",
                        "seed": seed,
                    },
                )

    if "batch_strategy_ablation" in selected:
        for seed in settings.seeds:
            for strategy in ["random", "conflict"]:
                hybrid = replace(
                    settings.hybrid,
                    batch_strategy=strategy,
                    seed=seed,
                )
                _attempt(
                    rows,
                    base,
                    lambda hybrid=hybrid: solve_hybrid(base, config=hybrid),
                    experiment="batch_strategy_ablation",
                    level=f"{strategy}:seed={seed}",
                    configuration={"batch_strategy": strategy, "seed": seed},
                )

    if "sampler_ablation" in selected:
        control, control_hybrid = _coordination_control(settings)
        for sampler in [
            "exact_feasible",
            "random",
            "simulated_annealing",
            "qaoa_statevector",
        ]:
            hybrid = replace(control_hybrid, sampler=sampler)
            _attempt(
                rows,
                control,
                lambda hybrid=hybrid: solve_hybrid(control, config=hybrid),
                experiment="sampler_ablation",
                level=sampler,
                configuration={
                    "sampler": sampler,
                    "data_scope": "independently generated coupled synthetic control",
                },
            )

    if "synthetic_coordination_control" in selected:
        stress, stress_hybrid = _coordination_control(settings)
        for method, solver in [
            ("greedy", lambda: solve_greedy_baseline(stress)),
            (
                "polished_greedy",
                lambda: solve_polished_greedy(
                    stress,
                    time_limit_seconds=30,
                    mip_relative_gap=0,
                    seed=6,
                ),
            ),
            (
                "exact_lns",
                lambda: solve_exact_lns(
                    stress,
                    config=ExactLNSConfig(
                        iterations=1,
                        initial_neighborhood_groups=4,
                        minimum_neighborhood_groups=2,
                        maximum_neighborhood_groups=4,
                        maximum_neighborhood_orders=4,
                        local_time_limit_seconds=5,
                        mip_relative_gap=0,
                        seed=6,
                    ),
                ),
            ),
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
                    "generator_seed": 6,
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
        "requested_value",
        "objective_capture_rate",
        "objective_per_assignment_group",
        "case_fill_rate",
        "penalty_cost",
        "shipping_cost",
        "reassigned_orders",
        "reassigned_assignment_groups",
        "runtime_seconds",
        "optimality_gap",
        "initial_objective",
        "raw_initial_objective",
        "polished_initial_objective",
        "initial_polish_improvement",
        "hybrid_improvement",
        "search_improvement",
        "lns_improvement",
        "total_hybrid_improvement",
        "total_search_improvement",
        "maximum_qubo_variables",
        "accepted_moves",
        "assignment_moves",
        "local_solves",
        "maximum_active_groups",
        "maximum_local_variables",
        "maximum_local_constraints",
        "maximum_local_mip_nodes",
        "experiment_schema_version",
        "schema_version",
        "assumption_version",
        "bundle_sha256",
        "problem_sha256",
        "git_commit",
        "git_dirty",
        "source_state_sha256",
        "runtime_environment",
        "configuration",
    ]
    return result.reindex(columns=ordered + [c for c in result if c not in ordered])


def run_ibm_hardware_study(
    *,
    allow_remote: bool = False,
    backend_name: str | None = None,
    shots: int = 512,
    profile: str = "quick",
    progress_callback: Callable[[pd.DataFrame], None] | None = None,
    existing_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run a matched IBM QPU stress test on the verified coupled control.

    The study sends only independently generated synthetic circuits.  ``quick``
    uses one hardware repetition; ``presentation`` repeats the same compiled
    configuration three times.  The remote matrix is a full factorial over
    p=1/p=2 and all mitigation strategies.  Angle and transpiler seeds are held
    fixed across hardware repetitions so QPU variability is not confounded with
    optimizer or compilation variability.  Successful rows from a verified
    partial checkpoint can be supplied through ``existing_results`` and are
    skipped, while failed variants are retried.
    """

    if not allow_remote:
        raise ValueError("IBM hardware study requires allow_remote=True")
    if shots <= 0:
        raise ValueError("shots must be positive")
    normalized = str(profile).strip().lower()
    if normalized == "quick":
        local_angle_seeds = (6,)
        hardware_repetitions = (1,)
    elif normalized == "presentation":
        local_angle_seeds = (3, 11, 29)
        hardware_repetitions = (1, 2, 3)
    else:
        raise ValueError("IBM hardware profile must be 'quick' or 'presentation'")

    problem, base_hybrid = _coordination_control(experiment_profile("full"))
    hardware_variants = tuple(
        (layers, mitigation)
        for layers in (1, 2)
        for mitigation in IBM_MITIGATION_STRATEGIES
    )
    angle_seed = 6
    transpiler_seed = 20_260_807
    rows: list[dict[str, Any]] = (
        []
        if existing_results is None
        else existing_results.replace({np.nan: None}).to_dict("records")
    )

    def succeeded(level: str) -> bool:
        for row in rows:
            if str(row.get("level")) != level:
                continue
            feasible = str(row.get("feasible", "")).strip().lower() in {
                "true",
                "1",
            }
            if feasible and str(row.get("method")) != "failed":
                return True
        return False

    def persist_progress() -> None:
        if progress_callback is not None:
            progress_callback(pd.DataFrame(rows).copy())

    def attempt_level(
        level: str,
        solver: Callable[[], Solution],
        configuration: dict[str, Any],
    ) -> None:
        if succeeded(level):
            return
        rows[:] = [row for row in rows if str(row.get("level")) != level]
        _attempt(
            rows,
            problem,
            solver,
            experiment="ibm_hardware_stress",
            level=level,
            configuration=configuration,
        )
        persist_progress()

    for method, solver in [
        ("greedy_reference", lambda: solve_greedy_baseline(problem)),
        (
            "exact_reference",
            lambda: solve_classical(
                problem, time_limit_seconds=30, mip_relative_gap=0
            ),
        ),
    ]:
        attempt_level(
            method,
            solver,
            {
                "method": method,
                "data_scope": "independently generated coupled synthetic control",
                "hardware_profile": normalized,
            },
        )

    for seed in local_angle_seeds:
        for layers in (1, 2):
            local = replace(
                base_hybrid,
                iterations=1,
                sampler="qaoa_statevector",
                num_reads=shots,
                qaoa_layers=layers,
                seed=seed,
                allow_remote=False,
            )
            attempt_level(
                f"local_qaoa:p={layers}:angle_seed={seed}",
                lambda local=local: solve_hybrid(problem, config=local),
                {
                    "method": "local_qaoa",
                    "sampler": "qaoa_statevector",
                    "qaoa_layers": layers,
                    "angle_seed": seed,
                    "shots": shots,
                    "data_scope": "independently generated coupled synthetic control",
                    "hardware_profile": normalized,
                },
            )

    for hardware_repetition in hardware_repetitions:
        for layers, mitigation in hardware_variants:
            remote = replace(
                base_hybrid,
                iterations=1,
                sampler="ibm-qpu",
                num_reads=shots,
                qaoa_layers=layers,
                seed=angle_seed,
                allow_remote=True,
                remote_time_limit_seconds=180.0,
                ibm_backend_name=backend_name,
                ibm_mitigation_strategy=mitigation,
                ibm_transpiler_optimization_level=3,
                ibm_transpiler_trials=8,
                ibm_transpiler_seed=transpiler_seed,
            )
            level = (
                f"ibm:p={layers}:{mitigation}:"
                f"hardware_rep={hardware_repetition}"
            )
            attempt_level(
                level,
                lambda remote=remote: solve_hybrid(problem, config=remote),
                {
                    "method": "ibm_qaoa",
                    "sampler": "ibm-qpu",
                    "qaoa_layers": layers,
                    "angle_seed": angle_seed,
                    "transpiler_seed": transpiler_seed,
                    "hardware_repetition": hardware_repetition,
                    "shots": shots,
                    "ibm_mitigation_strategy": mitigation,
                    "hardware_mitigation_strategy": mitigation,
                    "requested_backend": backend_name or "least_busy",
                    "data_scope": "independently generated coupled synthetic control",
                    "hardware_profile": normalized,
                },
            )
    return pd.DataFrame(rows)


def rank_ibm_hardware_strategies(results: pd.DataFrame) -> pd.DataFrame:
    """Rank successful IBM variants using quality first and cost as a tie-breaker.

    Exact feasible-QUBO raw hit rate is the primary hardware outcome. Raw one-hot
    feasibility is next because repaired assignments are not evidence of native
    constraint preservation. Validated assignment gain then confirms end-to-end
    usefulness; quantum usage and wall time break remaining ties. Presentation
    profiles report medians and interquartile ranges across seeds.
    """

    required = {
        "feasible",
        "sampler_backend",
        "qaoa_layers",
        "hardware_mitigation_strategy",
    }
    if missing := required - set(results.columns):
        raise ValueError(f"IBM hardware results are missing {sorted(missing)}")
    results = results.copy()
    for column in [
        "hardware_qubo_optimal_hit_rate",
        "raw_one_hot_rate",
        "search_improvement",
        "hardware_two_qubit_gates",
        "hardware_quantum_seconds",
        "hardware_queue_seconds",
        "hardware_execution_seconds",
        "runtime_seconds",
    ]:
        if column not in results.columns:
            results[column] = np.nan
    feasibility = results["feasible"]
    if feasibility.dtype == bool:
        feasible = feasibility.fillna(False)
    else:
        feasible = feasibility.astype(str).str.lower().isin({"true", "1"})
    attempts = results.loc[results["sampler_backend"].eq("ibm-qpu")].copy()
    if attempts.empty:
        raise ValueError("IBM hardware results contain no QPU attempt rows")
    data = attempts.loc[feasible.loc[attempts.index]].copy()

    group_columns = ["qaoa_layers", "hardware_mitigation_strategy"]
    measures = [
        "hardware_qubo_optimal_hit_rate",
        "raw_one_hot_rate",
        "search_improvement",
        "hardware_two_qubit_gates",
        "hardware_quantum_seconds",
        "hardware_queue_seconds",
        "hardware_execution_seconds",
        "runtime_seconds",
    ]
    attempted = (
        attempts.groupby(group_columns, dropna=False, sort=True)
        .size()
        .rename("attempted_runs")
        .reset_index()
    )
    if data.empty:
        summary = attempted.copy()
        for measure in measures:
            summary[measure] = np.nan
        summary["successful_runs"] = 0
        for measure in [
            "hardware_qubo_optimal_hit_rate",
            "raw_one_hot_rate",
            "runtime_seconds",
        ]:
            summary[f"{measure}_q25"] = np.nan
            summary[f"{measure}_q75"] = np.nan
    else:
        grouped = data.groupby(group_columns, dropna=False, sort=True)
        successful = grouped[measures].median().reset_index()
        successful["successful_runs"] = grouped.size().to_numpy()
        summary = attempted.merge(successful, on=group_columns, how="left")
        summary["successful_runs"] = summary["successful_runs"].fillna(0).astype(int)
        for measure in [
            "hardware_qubo_optimal_hit_rate",
            "raw_one_hot_rate",
            "runtime_seconds",
        ]:
            quantiles = grouped[measure].quantile([0.25, 0.75]).unstack()
            quantiles = quantiles.rename(
                columns={
                    0.25: f"{measure}_q25",
                    0.75: f"{measure}_q75",
                }
            ).reset_index()
            summary = summary.merge(quantiles, on=group_columns, how="left")
    summary["success_rate"] = summary["successful_runs"] / summary["attempted_runs"]
    summary["variant"] = summary.apply(
        lambda row: (
            f"p={int(row['qaoa_layers'])} | "
            f"{row['hardware_mitigation_strategy']}"
        ),
        axis=1,
    )
    summary = summary.sort_values(
        [
            "hardware_qubo_optimal_hit_rate",
            "raw_one_hot_rate",
            "search_improvement",
            "hardware_quantum_seconds",
            "runtime_seconds",
            "hardware_two_qubit_gates",
            "qaoa_layers",
            "hardware_mitigation_strategy",
        ],
        ascending=[False, False, False, True, True, True, True, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    any_success = bool((summary["successful_runs"] > 0).any())
    summary.insert(
        1,
        "selected_best_observed",
        summary["rank"].eq(1) & any_success,
    )
    return summary


def write_experiment_results(results: pd.DataFrame, path: str | Path) -> Path:
    """Write aggregate experiment evidence; reject identifier-like columns."""

    forbidden_fragments = {
        "address",
        "candidate_id",
        "customer",
        "dc_id",
        "delivery_number",
        "load_id",
        "material",
        "order_id",
        "plant",
        "sku_id",
        "zip",
    }
    allowed_identifiers = {"dataset_id", "experiment_id"}
    bad = []
    for column in results.columns:
        normalized = column.lower()
        if normalized in allowed_identifiers:
            continue
        if normalized.endswith("_id") or any(
            fragment in normalized for fragment in forbidden_fragments
        ):
            bad.append(column)
    if bad:
        raise ValueError(f"Aggregate experiment output contains forbidden columns: {bad}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    results.to_csv(temporary, index=False)
    temporary.replace(output)
    return output
