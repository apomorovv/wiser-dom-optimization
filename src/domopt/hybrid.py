"""Quantum-assisted large-neighborhood search with exact classical recourse.

The global solution always remains classically feasible. A bounded assignment
QUBO proposes alternatives for a resource-conflict neighborhood; the detailed
MILP then optimizes fulfillment quantities for each retained proposal before an
improving move can be accepted.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .baselines import (
    _CapacityState,
    _InventoryState,
    _preview_candidate,
    solve_default_baseline,
    solve_greedy_baseline,
)
from .classical import ClassicalSolverError, solve_classical
from .data import normalize_problem_data
from .objective import evaluate_solution
from .quantum import sample_qubo
from .qubo import build_candidate_qubo, perturb_qubo, qubo_energy
from .repair import repair_one_hot
from .resources import (
    candidate_fixed_consumption,
    line_variable_consumption,
    solution_capacity_usage,
    solution_inventory_usage,
    uses_split_pick_accounting,
)
from .schemas import ProblemData, QUBOModel, Solution
from .validation import validate_solution


@dataclass(frozen=True)
class HybridConfig:
    """Configuration for the bounded quantum-classical search."""

    iterations: int = 8
    neighborhood_orders: int = 8
    max_qubo_variables: int = 40
    max_candidates_per_order: int = 5
    sampler: str = "simulated_annealing"
    num_reads: int = 32
    sweeps: int = 200
    top_k_recourse: int = 6
    recourse_time_limit_seconds: float = 10.0
    one_hot_penalty_multiplier: float = 1.25
    pair_penalty_multiplier: float = 1.0
    initial_method: str = "greedy"
    seed: int = 0
    allow_remote: bool = False
    remote_time_limit_seconds: float | None = None
    qubo_noise_relative_sigma: float = 0.0

    def validate(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.neighborhood_orders <= 0 or self.max_qubo_variables <= 0:
            raise ValueError("neighborhood and QUBO limits must be positive")
        if self.max_candidates_per_order <= 0:
            raise ValueError("max_candidates_per_order must be positive")
        if self.max_candidates_per_order + 1 > self.max_qubo_variables:
            raise ValueError(
                "max_qubo_variables must fit one order's retained candidates "
                "plus its unassigned outcome"
            )
        if self.num_reads <= 0 or self.sweeps <= 0 or self.top_k_recourse <= 0:
            raise ValueError("sampling and recourse counts must be positive")
        if self.recourse_time_limit_seconds <= 0:
            raise ValueError("recourse_time_limit_seconds must be positive")
        if self.one_hot_penalty_multiplier <= 1.0:
            raise ValueError("one_hot_penalty_multiplier must be greater than one")
        if self.pair_penalty_multiplier < 0:
            raise ValueError("pair_penalty_multiplier must be nonnegative")
        if self.qubo_noise_relative_sigma < 0:
            raise ValueError("qubo_noise_relative_sigma must be nonnegative")
        if self.initial_method not in {"default", "greedy"}:
            raise ValueError("initial_method must be 'default' or 'greedy'")


def _solution_for_orders(solution: Solution, order_ids: set[str], method: str) -> Solution:
    assignments = solution.assignments.loc[
        solution.assignments["order_id"].astype(str).isin(order_ids)
    ].copy()
    fulfillment = solution.fulfillment.loc[
        solution.fulfillment["order_id"].astype(str).isin(order_ids)
    ].copy()
    assignments["method"] = method
    return Solution(method=method, assignments=assignments, fulfillment=fulfillment)


def residualize_problem(
    problem: ProblemData,
    incumbent: Solution,
    active_orders: set[str],
) -> ProblemData:
    """Remove frozen-order consumption and return a bounded local problem."""

    all_orders = set(problem.orders["order_id"].astype(str))
    if not active_orders or not active_orders <= all_orders:
        raise ValueError("active_orders must be a nonempty subset of problem orders")
    frozen_orders = all_orders - active_orders
    frozen = _solution_for_orders(incumbent, frozen_orders, "frozen")

    inventory_usage = solution_inventory_usage(problem, frozen)
    inventory = problem.inventory.copy()
    residual_inventory: list[int] = []
    for row in inventory.itertuples(index=False):
        key = (str(row.dc_id), str(row.sku_id), pd.Timestamp(row.date))
        residual = float(row.cumulative_available_cases) - inventory_usage.get(key, 0.0)
        if residual < -1e-8:
            raise ValueError(f"Frozen incumbent exceeds inventory at {key}")
        residual_inventory.append(max(0, round(residual)))
    inventory["cumulative_available_cases"] = residual_inventory

    capacity_usage = solution_capacity_usage(problem, frozen)
    capacities = problem.capacities.copy()
    residual_capacity: list[float] = []
    for row in capacities.itertuples(index=False):
        key = (str(row.dc_id), pd.Timestamp(row.date), str(row.resource))
        residual = float(row.capacity) - capacity_usage.get(key, 0.0)
        if residual < -1e-8:
            raise ValueError(f"Frozen incumbent exceeds capacity at {key}")
        residual_capacity.append(max(0.0, residual))
    if not capacities.empty:
        capacities["capacity"] = residual_capacity

    local = ProblemData(
        orders=problem.orders.loc[
            problem.orders["order_id"].astype(str).isin(active_orders)
        ].copy(),
        order_lines=problem.order_lines.loc[
            problem.order_lines["order_id"].astype(str).isin(active_orders)
        ].copy(),
        inventory=inventory,
        candidates=problem.candidates.loc[
            problem.candidates["order_id"].astype(str).isin(active_orders)
        ].copy(),
        capacities=capacities,
        calendar=problem.calendar.copy(),
        metadata={
            **problem.metadata,
            "parent_dataset_id": problem.metadata.get("dataset_id", "unknown"),
            "dataset_id": f"{problem.metadata.get('dataset_id', 'unknown')}::local",
            "active_order_count": len(active_orders),
        },
    )
    return normalize_problem_data(local)


def _unassigned_value(problem: ProblemData, order_id: str) -> float:
    lines = problem.order_lines.loc[
        problem.order_lines["order_id"].astype(str) == str(order_id)
    ]
    return -float(
        (lines["penalty_per_unfilled_case"] * lines["demand_cases"]).sum()
    )


def _plan_usage(
    problem: ProblemData,
    plan: Any,
) -> tuple[
    dict[tuple[Any, ...], float],
    dict[tuple[Any, ...], float],
    dict[tuple[Any, ...], float],
]:
    """Return usage, capacities, and marginal loss coefficients for one plan."""

    usage: defaultdict[tuple[Any, ...], float] = defaultdict(float)
    limits: dict[tuple[Any, ...], float] = {}
    loss: dict[tuple[Any, ...], float] = {}
    split_picks = uses_split_pick_accounting(problem)
    lines = problem.order_lines.loc[
        problem.order_lines["order_id"].astype(str) == str(plan.order_id)
    ].set_index("sku_id")

    for sku_id, quantity in plan.quantities.items():
        if quantity <= 0:
            continue
        line = lines.loc[str(sku_id)]
        marginal = float(line["unit_value"]) + float(line["penalty_per_unfilled_case"])
        checkpoints = problem.inventory.loc[
            (problem.inventory["dc_id"].astype(str) == str(plan.dc_id))
            & (problem.inventory["sku_id"].astype(str) == str(sku_id))
            & (problem.inventory["date"] >= pd.Timestamp(plan.pgi_date))
        ]
        for checkpoint in checkpoints.itertuples(index=False):
            key = (
                "inventory",
                str(plan.dc_id),
                str(sku_id),
                pd.Timestamp(checkpoint.date),
            )
            usage[key] += float(quantity)
            limits[key] = float(checkpoint.cumulative_available_cases)
            loss[key] = marginal

    candidate = problem.candidates.set_index("candidate_id").loc[str(plan.candidate_id)]
    for capacity in problem.capacities.itertuples(index=False):
        if (
            str(capacity.dc_id) != str(plan.dc_id)
            or pd.Timestamp(capacity.date) != pd.Timestamp(plan.pgi_date)
        ):
            continue
        resource = str(capacity.resource)
        key = ("capacity", str(plan.dc_id), pd.Timestamp(plan.pgi_date), resource)
        amount = candidate_fixed_consumption(candidate, resource)
        marginal_candidates: list[float] = []
        for sku_id, quantity in plan.quantities.items():
            if quantity <= 0:
                continue
            line = lines.loc[str(sku_id)]
            variable = line_variable_consumption(
                line,
                quantity,
                resource,
                split_picks=split_picks,
            )
            amount += variable
            if variable > 0:
                case_gain = float(line["unit_value"]) + float(
                    line["penalty_per_unfilled_case"]
                )
                marginal_candidates.append(case_gain * float(quantity) / variable)
        if amount > 0:
            usage[key] = amount
            limits[key] = float(capacity.capacity)
            if marginal_candidates:
                loss[key] = min(marginal_candidates)
            else:
                loss[key] = max(1.0, float(plan.incremental_score) / amount)
    return dict(usage), limits, loss


def _resource_pressure(
    plans: pd.DataFrame,
    resource_limits: dict[tuple[Any, ...], float],
) -> dict[tuple[Any, ...], float]:
    """Estimate higher-order contention without adding QUBO slack variables.

    For each resource, only the largest usage available to an order is counted,
    because the one-hot constraint prevents that order from selecting two plans.
    The pressure is the fraction of possible demand that cannot fit. It lets a
    quadratic model represent a three-or-more-plan overload even when every pair
    fits independently. Detailed feasibility is still delegated to MILP recourse.
    """

    maximum_by_order: defaultdict[
        tuple[Any, ...], dict[str, float]
    ] = defaultdict(dict)
    for row in plans.itertuples(index=False):
        for key, amount in row.usage.items():
            order_id = str(row.order_id)
            maximum_by_order[key][order_id] = max(
                float(amount),
                maximum_by_order[key].get(order_id, 0.0),
            )

    pressure: dict[tuple[Any, ...], float] = {}
    for key, by_order in maximum_by_order.items():
        possible = sum(by_order.values())
        limit = float(resource_limits[key])
        if possible > limit and possible > 0:
            pressure[key] = min(1.0, max(0.0, (possible - limit) / possible))
    return pressure


def build_neighborhood_qubo(
    problem: ProblemData,
    incumbent: Solution,
    *,
    one_hot_penalty_multiplier: float,
    pair_penalty_multiplier: float,
    max_candidates_per_order: int,
) -> tuple[QUBOModel, pd.DataFrame, dict[str, int], pd.DataFrame]:
    """Build the local multiple-choice QUBO and incumbent warm start."""

    records: list[dict[str, Any]] = []
    resource_limits: dict[tuple[Any, ...], float] = {}
    inventory = _InventoryState(problem.inventory)
    capacities = _CapacityState(problem.capacities)
    eligible = problem.candidates.loc[
        problem.candidates["eligible"].astype(bool)
    ].sort_values(["order_id", "pgi_date", "candidate_id"], kind="mergesort")

    for order_id in sorted(problem.orders["order_id"].astype(str)):
        records.append(
            {
                "plan_id": f"unassigned::{order_id}",
                "order_id": order_id,
                "candidate_id": None,
                "value": _unassigned_value(problem, order_id),
                "usage": {},
                "loss": {},
            }
        )
        for _, candidate in eligible.loc[eligible["order_id"] == order_id].iterrows():
            plan = _preview_candidate(problem, candidate, inventory, capacities)
            if plan is None:
                continue
            usage, limits, loss = _plan_usage(problem, plan)
            resource_limits.update(limits)
            records.append(
                {
                    "plan_id": str(plan.candidate_id),
                    "order_id": order_id,
                    "candidate_id": str(plan.candidate_id),
                    "value": float(plan.score),
                    "usage": usage,
                    "loss": loss,
                }
            )

    all_plans = pd.DataFrame(records)
    incumbent_lookup = incumbent.assignments.set_index("order_id")
    retained_groups: list[pd.DataFrame] = []
    for order_id, group in all_plans.groupby("order_id", sort=False):
        unassigned = group.loc[group["candidate_id"].isna()]
        candidates_for_order = group.loc[group["candidate_id"].notna()].sort_values(
            ["value", "plan_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        incumbent_row = incumbent_lookup.loc[str(order_id)]
        incumbent_id = (
            None
            if bool(incumbent_row["is_unassigned"])
            else str(incumbent_row["candidate_id"])
        )
        keep_ids: list[str] = []
        if incumbent_id is not None and incumbent_id in set(
            candidates_for_order["plan_id"].astype(str)
        ):
            keep_ids.append(incumbent_id)
        for plan_id in candidates_for_order["plan_id"].astype(str):
            if plan_id not in keep_ids:
                keep_ids.append(plan_id)
            if len(keep_ids) >= max_candidates_per_order:
                break
        retained_groups.extend(
            [
                unassigned,
                candidates_for_order.loc[
                    candidates_for_order["plan_id"].astype(str).isin(keep_ids)
                ],
            ]
        )
    plans = pd.concat(retained_groups, ignore_index=True)
    group_minimum = plans.groupby("order_id")["value"].transform("min")
    plans["qubo_value"] = plans["value"] - group_minimum

    conflict_rows: list[dict[str, object]] = []
    incident_penalty: defaultdict[str, float] = defaultdict(float)
    resource_pressure = _resource_pressure(plans, resource_limits)
    for left_index in range(len(plans)):
        left = plans.iloc[left_index]
        for right_index in range(left_index + 1, len(plans)):
            right = plans.iloc[right_index]
            if str(left["order_id"]) == str(right["order_id"]):
                continue
            shared = set(left["usage"]) & set(right["usage"])
            penalty = 0.0
            for key in shared:
                left_usage = float(left["usage"][key])
                right_usage = float(right["usage"][key])
                pair_excess = max(
                    0.0,
                    left_usage + right_usage - float(resource_limits[key]),
                )
                congestion = resource_pressure.get(key, 0.0) * min(
                    left_usage,
                    right_usage,
                )
                surrogate_excess = max(pair_excess, congestion)
                if surrogate_excess <= 0:
                    continue
                unit_loss = min(
                    float(left["loss"].get(key, 1.0)),
                    float(right["loss"].get(key, 1.0)),
                )
                penalty += surrogate_excess * max(1e-9, unit_loss)
            penalty *= pair_penalty_multiplier
            if penalty > 0:
                plan_a = str(left["plan_id"])
                plan_b = str(right["plan_id"])
                conflict_rows.append(
                    {"plan_id_a": plan_a, "plan_id_b": plan_b, "penalty": penalty}
                )
                incident_penalty[plan_a] += penalty
                incident_penalty[plan_b] += penalty

    conflicts = pd.DataFrame(
        conflict_rows,
        columns=["plan_id_a", "plan_id_b", "penalty"],
    )
    maximum_value = float(plans["qubo_value"].max()) if not plans.empty else 0.0
    maximum_incident = max(incident_penalty.values(), default=0.0)
    one_hot_penalty = one_hot_penalty_multiplier * (
        maximum_value + maximum_incident + 1.0
    )
    qubo_plans = plans[["plan_id", "order_id", "qubo_value"]].rename(
        columns={"qubo_value": "value"}
    )
    model = build_candidate_qubo(
        qubo_plans,
        one_hot_penalty=one_hot_penalty,
        conflicts=conflicts,
    )

    warm_start = {name: 0 for name in model.variable_names}
    for order_id, group in plans.groupby("order_id", sort=False):
        row = incumbent_lookup.loc[str(order_id)]
        target = (
            f"unassigned::{order_id}"
            if bool(row["is_unassigned"])
            else str(row["candidate_id"])
        )
        available = set(group["plan_id"].astype(str))
        if target not in available:
            target = str(
                group.sort_values(
                    ["value", "plan_id"],
                    ascending=[False, True],
                    kind="mergesort",
                ).iloc[0]["plan_id"]
            )
        warm_start[target] = 1
    return model, plans, warm_start, conflicts


def _resource_signatures(problem: ProblemData) -> dict[str, set[tuple[Any, ...]]]:
    signatures: defaultdict[str, set[tuple[Any, ...]]] = defaultdict(set)
    lines_by_order = {
        str(order_id): set(group["sku_id"].astype(str))
        for order_id, group in problem.order_lines.groupby("order_id", sort=False)
    }
    resources = set(problem.capacities.get("resource", pd.Series(dtype=str)).astype(str))
    for candidate in problem.candidates.loc[
        problem.candidates["eligible"].astype(bool)
    ].itertuples(index=False):
        order_id = str(candidate.order_id)
        dc_id = str(candidate.dc_id)
        date = pd.Timestamp(candidate.pgi_date)
        for sku_id in lines_by_order.get(order_id, set()):
            signatures[order_id].add(("inventory", dc_id, sku_id))
        for resource in resources:
            signatures[order_id].add(("capacity", dc_id, date, resource))
    return dict(signatures)


def _select_neighborhood(
    problem: ProblemData,
    incumbent: Solution,
    config: HybridConfig,
    iteration: int,
) -> set[str]:
    signatures = _resource_signatures(problem)
    inverted: defaultdict[tuple[Any, ...], set[str]] = defaultdict(set)
    for order_id, keys in signatures.items():
        for key in keys:
            inverted[key].add(order_id)
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for order_ids in inverted.values():
        for order_id in order_ids:
            adjacency[order_id].update(order_ids - {order_id})

    unfilled = (
        incumbent.fulfillment.assign(
            weighted_unfilled=lambda frame: frame["unfulfilled_cases"].astype(float)
        )
        .groupby("order_id")["weighted_unfilled"]
        .sum()
        .to_dict()
    )
    candidate_count = (
        problem.candidates.loc[problem.candidates["eligible"].astype(bool)]
        .groupby("order_id")["candidate_id"]
        .nunique()
        .to_dict()
    )
    ranked = sorted(
        problem.orders["order_id"].astype(str),
        key=lambda order_id: (
            -float(unfilled.get(order_id, 0.0)),
            -len(adjacency.get(order_id, set())),
            -int(candidate_count.get(order_id, 0)),
            order_id,
        ),
    )
    seed_order = ranked[iteration % len(ranked)]
    queue: deque[str] = deque([seed_order])
    selected: list[str] = []
    variable_count = 0
    visited: set[str] = set()

    while queue and len(selected) < config.neighborhood_orders:
        order_id = queue.popleft()
        if order_id in visited:
            continue
        visited.add(order_id)
        order_variables = min(
            int(candidate_count.get(order_id, 0)),
            config.max_candidates_per_order,
        ) + 1
        if selected and variable_count + order_variables > config.max_qubo_variables:
            continue
        selected.append(order_id)
        variable_count += order_variables
        neighbors = sorted(
            adjacency.get(order_id, set()),
            key=lambda neighbor: (
                -float(unfilled.get(neighbor, 0.0)),
                -len(adjacency.get(neighbor, set())),
                neighbor,
            ),
        )
        queue.extend(neighbors)

    for order_id in ranked:
        if len(selected) >= config.neighborhood_orders:
            break
        if order_id in selected:
            continue
        order_variables = min(
            int(candidate_count.get(order_id, 0)),
            config.max_candidates_per_order,
        ) + 1
        if selected and variable_count + order_variables > config.max_qubo_variables:
            continue
        selected.append(order_id)
        variable_count += order_variables
    return set(selected)


def _merge_local_solution(
    incumbent: Solution,
    local: Solution,
    active_orders: set[str],
    *,
    runtime_seconds: float,
) -> Solution:
    outside_assignments = incumbent.assignments.loc[
        ~incumbent.assignments["order_id"].astype(str).isin(active_orders)
    ]
    outside_fulfillment = incumbent.fulfillment.loc[
        ~incumbent.fulfillment["order_id"].astype(str).isin(active_orders)
    ]
    assignments = pd.concat(
        [outside_assignments, local.assignments],
        ignore_index=True,
    ).sort_values("order_id", kind="mergesort")
    fulfillment = pd.concat(
        [outside_fulfillment, local.fulfillment],
        ignore_index=True,
    ).sort_values(["order_id", "sku_id"], kind="mergesort")
    assignments["method"] = "hybrid"
    return Solution(
        method="hybrid",
        assignments=assignments.reset_index(drop=True),
        fulfillment=fulfillment.reset_index(drop=True),
        runtime_seconds=runtime_seconds,
    )


def _sample_to_fixed_assignments(
    repaired: dict[str, int],
    plans: pd.DataFrame,
) -> dict[str, str | None]:
    selected = {plan_id for plan_id, value in repaired.items() if int(value) == 1}
    fixed: dict[str, str | None] = {}
    for row in plans.itertuples(index=False):
        if str(row.plan_id) in selected:
            fixed[str(row.order_id)] = (
                None if pd.isna(row.candidate_id) else str(row.candidate_id)
            )
    return fixed


def solve_hybrid(
    problem: ProblemData,
    *,
    config: HybridConfig | None = None,
) -> Solution:
    """Run bounded quantum-assisted LNS and return the best feasible incumbent."""

    settings = config or HybridConfig()
    settings.validate()
    start = perf_counter()
    incumbent = (
        solve_default_baseline(problem)
        if settings.initial_method == "default"
        else solve_greedy_baseline(problem)
    )
    initial_value = evaluate_solution(problem, incumbent).objective_value
    incumbent_value = initial_value
    if not validate_solution(problem, incumbent).is_feasible:
        raise ValueError("The initial classical solution is infeasible")

    history: list[dict[str, Any]] = []
    accepted_moves = 0
    sampler_calls = 0
    recourse_solves = 0
    total_raw_samples = 0
    raw_one_hot_samples = 0
    maximum_qubo_variables = 0

    for iteration in range(settings.iterations):
        active_orders = _select_neighborhood(
            problem,
            incumbent,
            settings,
            iteration,
        )
        local_problem = residualize_problem(problem, incumbent, active_orders)
        local_incumbent = _solution_for_orders(incumbent, active_orders, "local_incumbent")
        model, plans, warm_start, conflicts = build_neighborhood_qubo(
            local_problem,
            local_incumbent,
            one_hot_penalty_multiplier=settings.one_hot_penalty_multiplier,
            pair_penalty_multiplier=settings.pair_penalty_multiplier,
            max_candidates_per_order=settings.max_candidates_per_order,
        )
        if len(model.variable_names) > settings.max_qubo_variables:
            raise ValueError(
                f"Neighborhood has {len(model.variable_names)} QUBO variables; "
                f"limit is {settings.max_qubo_variables}"
            )
        maximum_qubo_variables = max(maximum_qubo_variables, len(model.variable_names))
        sampled_model = perturb_qubo(
            model,
            relative_sigma=settings.qubo_noise_relative_sigma,
            seed=settings.seed + iteration,
        )
        samples = sample_qubo(
            sampled_model,
            method=settings.sampler,
            num_samples=settings.num_reads,
            sweeps=settings.sweeps,
            seed=settings.seed + iteration,
            initial_sample=warm_start,
            allow_remote=settings.allow_remote,
            time_limit_seconds=settings.remote_time_limit_seconds,
        )
        sampler_calls += 1
        total_raw_samples += len(samples)

        candidates: list[tuple[float, tuple[str, ...], dict[str, int]]] = []
        seen_assignments: set[tuple[str, ...]] = set()
        for _, sample in samples.iterrows():
            sample_map = {name: int(sample[name]) for name in model.variable_names}
            valid_one_hot = all(
                sum(sample_map[str(plan_id)] for plan_id in group["plan_id"]) == 1
                for _, group in plans.groupby("order_id", sort=False)
            )
            raw_one_hot_samples += int(valid_one_hot)
            repaired = repair_one_hot(sample_map, plans)
            selected = tuple(
                sorted(plan_id for plan_id, value in repaired.items() if value == 1)
            )
            if selected in seen_assignments:
                continue
            seen_assignments.add(selected)
            vector = np.asarray(
                [repaired[name] for name in model.variable_names],
                dtype=int,
            )
            candidates.append((qubo_energy(model, vector), selected, repaired))

        candidates.sort(key=lambda item: (item[0], item[1]))
        best_iteration = incumbent
        best_iteration_value = incumbent_value
        attempted = 0
        for _, _, repaired in candidates[: settings.top_k_recourse]:
            fixed = _sample_to_fixed_assignments(repaired, plans)
            if set(fixed) != active_orders:
                continue
            attempted += 1
            try:
                local_solution = solve_classical(
                    local_problem,
                    time_limit_seconds=settings.recourse_time_limit_seconds,
                    seed=settings.seed + iteration,
                    fixed_assignments=fixed,
                )
            except ClassicalSolverError:
                continue
            recourse_solves += 1
            merged = _merge_local_solution(
                incumbent,
                local_solution,
                active_orders,
                runtime_seconds=perf_counter() - start,
            )
            validation = validate_solution(problem, merged)
            if not validation.is_feasible:
                continue
            value = evaluate_solution(problem, merged).objective_value
            if value > best_iteration_value + 1e-9:
                best_iteration = merged
                best_iteration_value = value

        improved = best_iteration_value > incumbent_value + 1e-9
        if improved:
            incumbent = best_iteration
            incumbent_value = best_iteration_value
            accepted_moves += 1
        history.append(
            {
                "iteration": iteration,
                "active_orders": len(active_orders),
                "qubo_variables": len(model.variable_names),
                "pair_terms": len(conflicts),
                "unique_repaired_samples": len(candidates),
                "recourse_attempts": attempted,
                "accepted": improved,
                "objective_value": incumbent_value,
            }
        )

    runtime = perf_counter() - start
    assignments = incumbent.assignments.copy()
    assignments["method"] = "hybrid"
    return Solution(
        method="hybrid",
        assignments=assignments,
        fulfillment=incumbent.fulfillment.copy(),
        runtime_seconds=runtime,
        raw_objective=incumbent_value,
        metadata={
            "algorithm": "quantum-assisted-large-neighborhood-search",
            "sampler": settings.sampler,
            "initial_method": settings.initial_method,
            "initial_objective": initial_value,
            "final_objective": incumbent_value,
            "improvement": incumbent_value - initial_value,
            "iterations": settings.iterations,
            "accepted_moves": accepted_moves,
            "sampler_calls": sampler_calls,
            "qpu_calls": (
                sampler_calls if settings.sampler in {"dwave-qpu", "dwave-hybrid"} else 0
            ),
            "recourse_solves": recourse_solves,
            "maximum_qubo_variables": maximum_qubo_variables,
            "maximum_candidates_per_order": settings.max_candidates_per_order,
            "raw_one_hot_rate": (
                raw_one_hot_samples / total_raw_samples if total_raw_samples else 0.0
            ),
            "remote_enabled": settings.allow_remote,
            "qubo_noise_relative_sigma": settings.qubo_noise_relative_sigma,
            "history": history,
            "claim": "No quantum advantage is inferred from this run.",
        },
    )
