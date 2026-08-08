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
    _assignment_units,
    _CapacityState,
    _feasible_decisions,
    _InventoryState,
    solve_default_baseline,
    solve_greedy_baseline,
)
from .classical import ClassicalSolverError, solve_classical
from .data import normalize_problem_data
from .objective import evaluate_solution
from .penalties import order_penalty
from .quantum import IBM_MITIGATION_STRATEGIES, sample_qubo
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
    batch_strategy: str = "conflict"
    polish_initial_incumbent: bool = True
    qaoa_layers: int = 1
    qaoa_restarts: int = 4
    qaoa_mixer_topology: str = "path"
    qaoa_parameters: tuple[float, ...] | None = None
    qaoa_readout_bitflip_probability: float = 0.0
    max_feasible_states: int = 65_536
    ibm_backend_name: str | None = None
    ibm_mitigation_strategy: str = "baseline"
    ibm_transpiler_optimization_level: int = 3
    ibm_transpiler_trials: int = 4
    ibm_transpiler_seed: int | None = None

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
        if self.batch_strategy not in {"conflict", "random"}:
            raise ValueError("batch_strategy must be 'conflict' or 'random'")
        if self.qaoa_layers <= 0 or self.qaoa_restarts <= 0:
            raise ValueError("QAOA layers and restarts must be positive")
        if self.qaoa_mixer_topology not in {"path", "ring"}:
            raise ValueError("qaoa_mixer_topology must be 'path' or 'ring'")
        if self.qaoa_parameters is not None and len(self.qaoa_parameters) != 2 * self.qaoa_layers:
            raise ValueError("qaoa_parameters must contain two values per QAOA layer")
        if not 0 <= self.qaoa_readout_bitflip_probability <= 1:
            raise ValueError(
                "qaoa_readout_bitflip_probability must be between zero and one"
            )
        if self.max_feasible_states <= 0:
            raise ValueError("max_feasible_states must be positive")
        if self.ibm_mitigation_strategy not in IBM_MITIGATION_STRATEGIES:
            raise ValueError(
                "ibm_mitigation_strategy must be one of "
                f"{IBM_MITIGATION_STRATEGIES}"
            )
        if self.ibm_transpiler_optimization_level not in {0, 1, 2, 3}:
            raise ValueError("ibm_transpiler_optimization_level must be 0, 1, 2, or 3")
        if self.ibm_transpiler_trials <= 0:
            raise ValueError("ibm_transpiler_trials must be positive")
        if self.ibm_transpiler_seed is not None and self.ibm_transpiler_seed < 0:
            raise ValueError("ibm_transpiler_seed must be nonnegative")


@dataclass(frozen=True)
class ExactLNSConfig:
    """Configuration for the adaptive exact-MILP neighborhood search.

    The method is the production classical comparator for the sampler-assisted
    path.  Each neighborhood leaves assignment decisions free and solves the
    detailed quantity/penalty recourse jointly in one bounded MILP.
    """

    iterations: int = 6
    initial_neighborhood_groups: int = 8
    minimum_neighborhood_groups: int = 4
    maximum_neighborhood_groups: int = 16
    maximum_neighborhood_orders: int = 40
    maximum_local_fulfillment_variables: int = 6_000
    local_time_limit_seconds: float = 8.0
    mip_relative_gap: float = 0.01
    diversification_interval: int = 3
    initial_method: str = "greedy"
    polish_initial_incumbent: bool = True
    adaptive: bool = True
    seed: int = 0

    def validate(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.minimum_neighborhood_groups <= 0:
            raise ValueError("minimum_neighborhood_groups must be positive")
        if not (
            self.minimum_neighborhood_groups
            <= self.initial_neighborhood_groups
            <= self.maximum_neighborhood_groups
        ):
            raise ValueError(
                "neighborhood group limits must satisfy minimum <= initial <= maximum"
            )
        if self.maximum_neighborhood_orders <= 0:
            raise ValueError("maximum_neighborhood_orders must be positive")
        if self.maximum_local_fulfillment_variables <= 0:
            raise ValueError("maximum_local_fulfillment_variables must be positive")
        if self.local_time_limit_seconds <= 0:
            raise ValueError("local_time_limit_seconds must be positive")
        if not 0 <= self.mip_relative_gap < 1:
            raise ValueError("mip_relative_gap must be in [0, 1)")
        if self.diversification_interval <= 0:
            raise ValueError("diversification_interval must be positive")
        if self.initial_method not in {"default", "greedy"}:
            raise ValueError("initial_method must be 'default' or 'greedy'")


@dataclass(frozen=True)
class _NeighborhoodIndex:
    """Problem-static assignment-group conflicts reused across iterations."""

    members_by_unit: dict[str, frozenset[str]]
    adjacency: dict[str, frozenset[str]]
    candidate_count: dict[str, int]
    fulfillment_variable_estimate: dict[str, int]


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

    local_orders = problem.orders.loc[
        problem.orders["order_id"].astype(str).isin(active_orders)
    ].copy()
    local_lines = problem.order_lines.loc[
        problem.order_lines["order_id"].astype(str).isin(active_orders)
    ].copy()
    local_candidates = problem.candidates.loc[
        problem.candidates["order_id"].astype(str).isin(active_orders)
    ].copy()

    # Keep only resource rows reachable by an eligible local assignment.  Earlier
    # versions passed every global ATP checkpoint and capacity row into each local
    # MILP, producing thousands of coefficient-empty constraints.  Frozen use has
    # already been subtracted above, so this projection is lossless for the active
    # orders and benefits both sampler recourse and exact LNS.
    eligible_local = local_candidates.loc[
        local_candidates["eligible"].astype(bool)
    ]
    reachable_inventory = (
        eligible_local[["order_id", "dc_id"]]
        .merge(local_lines[["order_id", "sku_id"]], on="order_id", how="inner")
        [["dc_id", "sku_id"]]
        .drop_duplicates()
    )
    inventory = inventory.merge(
        reachable_inventory,
        on=["dc_id", "sku_id"],
        how="inner",
        validate="many_to_one",
    )
    reachable_dates = (
        eligible_local[["dc_id", "pgi_date"]]
        .rename(columns={"pgi_date": "date"})
        .drop_duplicates()
    )
    capacities = capacities.merge(
        reachable_dates,
        on=["dc_id", "date"],
        how="inner",
        validate="many_to_one",
    )
    calendar = problem.calendar.merge(
        reachable_dates,
        on=["dc_id", "date"],
        how="inner",
        validate="many_to_one",
    )

    local = ProblemData(
        orders=local_orders,
        order_lines=local_lines,
        inventory=inventory,
        candidates=local_candidates,
        capacities=capacities,
        calendar=calendar,
        metadata={
            **problem.metadata,
            "parent_dataset_id": problem.metadata.get("dataset_id", "unknown"),
            "dataset_id": f"{problem.metadata.get('dataset_id', 'unknown')}::local",
            "active_order_count": len(active_orders),
            "local_inventory_rows": len(inventory),
            "local_capacity_rows": len(capacities),
        },
    )
    return normalize_problem_data(local)


def _unassigned_value(problem: ProblemData, order_id: str) -> float:
    lines = problem.order_lines.loc[
        problem.order_lines["order_id"].astype(str) == str(order_id)
    ].copy()
    lines["unfulfilled_cases"] = lines["demand_cases"]
    return -order_penalty(problem, order_id, lines)


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
    units = _assignment_units(problem)
    incumbent_lookup = incumbent.assignments.set_index("order_id")
    candidate_options = problem.candidates.set_index("candidate_id").get(
        "group_option_id"
    )
    incumbent_targets: dict[str, str] = {}

    for unit_id, members in sorted(units.items()):
        unassigned_id = f"unassigned::{unit_id}"
        records.append(
            {
                "plan_id": unassigned_id,
                "order_id": unit_id,
                "candidate_id": None,
                "value": sum(
                    _unassigned_value(problem, order_id) for order_id in members
                ),
                "usage": {},
                "loss": {},
                "fixed_assignments": {order_id: None for order_id in members},
                "is_unassigned": True,
            }
        )

        member_incumbent = incumbent_lookup.loc[list(members)]
        if member_incumbent["is_unassigned"].astype(bool).all():
            incumbent_targets[unit_id] = unassigned_id
        else:
            chosen_options: set[str] = set()
            for row in member_incumbent.itertuples():
                if bool(row.is_unassigned):
                    continue
                candidate_id = str(row.candidate_id)
                option_id = (
                    str(candidate_options.loc[candidate_id])
                    if candidate_options is not None
                    else candidate_id
                )
                chosen_options.add(option_id)
            if len(chosen_options) != 1:
                raise ValueError(f"Incumbent splits assignment unit {unit_id!r}")
            incumbent_targets[unit_id] = (
                f"option::{unit_id}::{next(iter(chosen_options))}"
            )

        decisions = _feasible_decisions(
            problem,
            unit_id,
            members,
            eligible,
            inventory,
            capacities,
        )
        for decision in decisions:
            usage: defaultdict[tuple[Any, ...], float] = defaultdict(float)
            loss: dict[tuple[Any, ...], float] = {}
            fixed_assignments: dict[str, str | None] = {}
            for member_plan in decision.member_plans:
                member_usage, limits, member_loss = _plan_usage(
                    problem, member_plan
                )
                resource_limits.update(limits)
                for key, amount in member_usage.items():
                    usage[key] += float(amount)
                for key, amount in member_loss.items():
                    loss[key] = min(float(amount), loss.get(key, float("inf")))
                fixed_assignments[member_plan.order_id] = member_plan.candidate_id
            records.append(
                {
                    "plan_id": f"option::{unit_id}::{decision.option_id}",
                    "order_id": unit_id,
                    "candidate_id": decision.option_id,
                    "value": float(decision.score),
                    "usage": dict(usage),
                    "loss": loss,
                    "fixed_assignments": fixed_assignments,
                    "is_unassigned": False,
                }
            )

    all_plans = pd.DataFrame(records)
    retained_groups: list[pd.DataFrame] = []
    for unit_id, group in all_plans.groupby("order_id", sort=False):
        unassigned = group.loc[group["is_unassigned"].astype(bool)]
        candidates_for_order = group.loc[~group["is_unassigned"].astype(bool)].sort_values(
            ["value", "plan_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        incumbent_id = incumbent_targets[str(unit_id)]
        keep_ids: list[str] = []
        if incumbent_id in set(candidates_for_order["plan_id"].astype(str)):
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
    for unit_id, group in plans.groupby("order_id", sort=False):
        target = incumbent_targets[str(unit_id)]
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


def _build_neighborhood_index(problem: ProblemData) -> _NeighborhoodIndex:
    """Build the static conflict graph once for a complete search run."""

    signatures = _resource_signatures(problem)
    if bool(problem.metadata.get("enforce_assignment_group", False)):
        if "assignment_group" not in problem.orders.columns:
            raise ValueError("Group cohesion requires orders.assignment_group")
        if "group_option_id" not in problem.candidates.columns:
            raise ValueError("Group cohesion requires candidates.group_option_id")
        unit_for_order = (
            problem.orders.set_index("order_id")["assignment_group"]
            .astype(str)
            .to_dict()
        )
    else:
        unit_for_order = {
            str(order_id): str(order_id)
            for order_id in problem.orders["order_id"].astype(str)
        }

    members_by_unit: defaultdict[str, set[str]] = defaultdict(set)
    unit_signatures: defaultdict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for order_id in problem.orders["order_id"].astype(str):
        unit = unit_for_order[order_id]
        members_by_unit[unit].add(order_id)
        unit_signatures[unit].update(signatures.get(order_id, set()))

    inverted: defaultdict[tuple[Any, ...], set[str]] = defaultdict(set)
    for unit, keys in unit_signatures.items():
        for key in keys:
            inverted[key].add(unit)
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for units in inverted.values():
        for unit in units:
            adjacency[unit].update(units - {unit})

    eligible = problem.candidates.loc[
        problem.candidates["eligible"].astype(bool)
    ].copy()
    if bool(problem.metadata.get("enforce_assignment_group", False)):
        eligible["decision_unit"] = eligible["order_id"].astype(str).map(
            unit_for_order
        )
        candidate_count = (
            eligible.groupby("decision_unit")["group_option_id"].nunique().to_dict()
        )
    else:
        candidate_count = (
            eligible.groupby(eligible["order_id"].astype(str))["candidate_id"]
            .nunique()
            .to_dict()
        )

    line_count = problem.order_lines.groupby(
        problem.order_lines["order_id"].astype(str)
    ).size().to_dict()
    candidates_per_order = eligible.groupby(
        eligible["order_id"].astype(str)
    )["candidate_id"].nunique().to_dict()
    fulfillment_variable_estimate = {
        unit: sum(
            int(line_count.get(order_id, 0))
            * int(candidates_per_order.get(order_id, 0))
            for order_id in members
        )
        for unit, members in members_by_unit.items()
    }

    return _NeighborhoodIndex(
        members_by_unit={
            unit: frozenset(members) for unit, members in members_by_unit.items()
        },
        adjacency={
            unit: frozenset(adjacency.get(unit, set())) for unit in members_by_unit
        },
        candidate_count={str(unit): int(count) for unit, count in candidate_count.items()},
        fulfillment_variable_estimate={
            str(unit): int(count)
            for unit, count in fulfillment_variable_estimate.items()
        },
    )


def _unit_priority(
    problem: ProblemData,
    incumbent: Solution,
    index: _NeighborhoodIndex,
) -> tuple[list[str], dict[str, float]]:
    """Rank atomic groups by approximate recoverable business loss and conflict."""

    fulfillment = incumbent.fulfillment.merge(
        problem.order_lines[
            ["order_id", "sku_id", "unit_value", "penalty_per_unfilled_case"]
        ],
        on=["order_id", "sku_id"],
        how="left",
        validate="one_to_one",
    )
    fulfillment["weighted_unfilled"] = fulfillment["unfulfilled_cases"].astype(
        float
    ) * (
        fulfillment["unit_value"].astype(float)
        + fulfillment["penalty_per_unfilled_case"].astype(float)
    )
    by_order = fulfillment.groupby("order_id")["weighted_unfilled"].sum().to_dict()
    loss_by_unit = {
        unit: sum(float(by_order.get(order_id, 0.0)) for order_id in members)
        for unit, members in index.members_by_unit.items()
    }
    ranked = sorted(
        index.members_by_unit,
        key=lambda unit: (
            -loss_by_unit.get(unit, 0.0),
            -len(index.adjacency.get(unit, frozenset())),
            -len(index.members_by_unit[unit]),
            unit,
        ),
    )
    return ranked, loss_by_unit


def _select_neighborhood(
    problem: ProblemData,
    incumbent: Solution,
    config: HybridConfig,
    iteration: int,
    *,
    index: _NeighborhoodIndex | None = None,
) -> set[str]:
    neighborhood_index = index or _build_neighborhood_index(problem)
    members_by_unit = neighborhood_index.members_by_unit
    adjacency = neighborhood_index.adjacency
    ranked, unit_unfilled = _unit_priority(problem, incumbent, neighborhood_index)
    unit_variables = {
        unit: min(
            int(neighborhood_index.candidate_count.get(unit, 0)),
            config.max_candidates_per_order,
        )
        + 1
        for unit in members_by_unit
    }
    if config.batch_strategy == "random":
        generator = np.random.default_rng(config.seed + iteration)
        ranked = [ranked[index] for index in generator.permutation(len(ranked))]
        queue: deque[str] = deque(ranked)
    else:
        seed_order = ranked[iteration % len(ranked)]
        queue = deque([seed_order])
    selected_units: list[str] = []
    variable_count = 0
    visited: set[str] = set()

    while queue and sum(len(members_by_unit[u]) for u in selected_units) < config.neighborhood_orders:
        order_id = queue.popleft()
        if order_id in visited:
            continue
        visited.add(order_id)
        order_variables = unit_variables[order_id]
        if variable_count + order_variables > config.max_qubo_variables:
            continue
        if (
            selected_units
            and sum(len(members_by_unit[u]) for u in selected_units)
            + len(members_by_unit[order_id])
            > config.neighborhood_orders
        ):
            continue
        selected_units.append(order_id)
        variable_count += order_variables
        if config.batch_strategy == "conflict":
            neighbors = sorted(
                adjacency.get(order_id, set()),
                key=lambda neighbor: (
                    -unit_unfilled.get(neighbor, 0.0),
                    -len(adjacency.get(neighbor, set())),
                    neighbor,
                ),
            )
            queue.extend(neighbors)

    for order_id in ranked:
        selected_count = sum(len(members_by_unit[u]) for u in selected_units)
        if selected_count >= config.neighborhood_orders:
            break
        if order_id in selected_units:
            continue
        order_variables = unit_variables[order_id]
        if variable_count + order_variables > config.max_qubo_variables:
            continue
        if selected_count + len(members_by_unit[order_id]) > config.neighborhood_orders:
            continue
        selected_units.append(order_id)
        variable_count += order_variables
    if not selected_units:
        raise ValueError(
            "No complete assignment group fits the configured neighborhood/QUBO limits"
        )
    return set().union(*(members_by_unit[unit] for unit in selected_units))


def _select_exact_lns_neighborhood(
    problem: ProblemData,
    incumbent: Solution,
    config: ExactLNSConfig,
    iteration: int,
    target_groups: int,
    index: _NeighborhoodIndex,
) -> tuple[set[str], tuple[str, ...], str]:
    """Select a bounded whole-group neighborhood for one exact local solve."""

    ranked, unit_loss = _unit_priority(problem, incumbent, index)
    diversify = (iteration + 1) % config.diversification_interval == 0
    strategy = "random" if diversify else "conflict"
    if diversify:
        generator = np.random.default_rng(config.seed + iteration)
        ranked = [ranked[i] for i in generator.permutation(len(ranked))]

    seed_unit = ranked[iteration % len(ranked)]
    queue: deque[str] = deque([seed_unit] if not diversify else ranked)
    selected: list[str] = []
    visited: set[str] = set()
    active_order_count = 0
    fulfillment_variables = 0

    while queue and len(selected) < target_groups:
        unit = queue.popleft()
        if unit in visited:
            continue
        visited.add(unit)
        member_count = len(index.members_by_unit[unit])
        unit_variables = index.fulfillment_variable_estimate.get(unit, 0)
        if (
            active_order_count + member_count > config.maximum_neighborhood_orders
            or fulfillment_variables + unit_variables
            > config.maximum_local_fulfillment_variables
        ):
            continue
        selected.append(unit)
        active_order_count += member_count
        fulfillment_variables += unit_variables
        if not diversify:
            neighbors = sorted(
                index.adjacency.get(unit, frozenset()),
                key=lambda neighbor: (
                    -unit_loss.get(neighbor, 0.0),
                    -len(index.adjacency.get(neighbor, frozenset())),
                    neighbor,
                ),
            )
            queue.extend(neighbors)

    for unit in ranked:
        if len(selected) >= target_groups:
            break
        if unit in selected:
            continue
        member_count = len(index.members_by_unit[unit])
        unit_variables = index.fulfillment_variable_estimate.get(unit, 0)
        if (
            active_order_count + member_count > config.maximum_neighborhood_orders
            or fulfillment_variables + unit_variables
            > config.maximum_local_fulfillment_variables
        ):
            continue
        selected.append(unit)
        active_order_count += member_count
        fulfillment_variables += unit_variables

    if not selected:
        raise ValueError("No complete assignment group fits the exact-LNS limits")
    active_orders = set().union(*(index.members_by_unit[unit] for unit in selected))
    return active_orders, tuple(selected), strategy


def _merge_local_solution(
    incumbent: Solution,
    local: Solution,
    active_orders: set[str],
    *,
    runtime_seconds: float,
    method: str = "hybrid",
) -> Solution:
    outside_assignments = incumbent.assignments.loc[
        ~incumbent.assignments["order_id"].astype(str).isin(active_orders)
    ]
    outside_fulfillment = incumbent.fulfillment.loc[
        ~incumbent.fulfillment["order_id"].astype(str).isin(active_orders)
    ]
    assignments = (
        local.assignments.copy()
        if outside_assignments.empty
        else pd.DataFrame.from_records(
            outside_assignments.to_dict("records")
            + local.assignments.to_dict("records"),
            columns=incumbent.assignments.columns,
        )
    ).sort_values("order_id", kind="mergesort")
    fulfillment = (
        local.fulfillment.copy()
        if outside_fulfillment.empty
        else pd.DataFrame.from_records(
            outside_fulfillment.to_dict("records")
            + local.fulfillment.to_dict("records"),
            columns=incumbent.fulfillment.columns,
        )
    ).sort_values(["order_id", "sku_id"], kind="mergesort")
    assignments["method"] = method
    return Solution(
        method=method,
        assignments=assignments.reset_index(drop=True),
        fulfillment=fulfillment.reset_index(drop=True),
        runtime_seconds=runtime_seconds,
    )


def _sample_to_fixed_assignments(
    repaired: dict[str, int],
    plans: pd.DataFrame,
    problem: ProblemData,
) -> dict[str, str | None]:
    del problem  # The selected atomic plans already contain every member mapping.
    selected = {plan_id for plan_id, value in repaired.items() if int(value) == 1}
    fixed: dict[str, str | None] = {}
    for row in plans.itertuples(index=False):
        if str(row.plan_id) in selected:
            fixed.update(
                {
                    str(order_id): (
                        None if candidate_id is None else str(candidate_id)
                    )
                    for order_id, candidate_id in row.fixed_assignments.items()
                }
            )
    return fixed


def _solution_fixed_assignments(solution: Solution) -> dict[str, str | None]:
    """Extract a complete assignment policy for exact quantity recourse."""

    return {
        str(row.order_id): (
            None if bool(row.is_unassigned) else str(row.candidate_id)
        )
        for row in solution.assignments.itertuples(index=False)
    }


def solve_exact_lns(
    problem: ProblemData,
    *,
    config: ExactLNSConfig | None = None,
) -> Solution:
    """Run adaptive conflict-aware exact-MILP large-neighborhood search.

    Unlike the sampler-assisted path, each neighborhood solves assignment and
    fulfillment decisions jointly.  Frozen global consumption is residualized,
    and no move is accepted without an independent full-problem validation.
    """

    settings = config or ExactLNSConfig()
    settings.validate()
    start = perf_counter()
    raw_incumbent = (
        solve_default_baseline(problem)
        if settings.initial_method == "default"
        else solve_greedy_baseline(problem)
    )
    baseline_initialization_seconds = perf_counter() - start
    raw_initial_value = evaluate_solution(problem, raw_incumbent).objective_value
    if not validate_solution(problem, raw_incumbent).is_feasible:
        raise ValueError("The initial classical solution is infeasible")

    incumbent = raw_incumbent
    incumbent_value = raw_initial_value
    polish_start = perf_counter()
    polish_succeeded = False
    if settings.polish_initial_incumbent:
        try:
            polished = solve_classical(
                problem,
                time_limit_seconds=settings.local_time_limit_seconds,
                mip_relative_gap=settings.mip_relative_gap,
                seed=settings.seed,
                fixed_assignments=_solution_fixed_assignments(raw_incumbent),
            )
            polished_value = evaluate_solution(problem, polished).objective_value
            if (
                validate_solution(problem, polished).is_feasible
                and polished_value >= incumbent_value - 1e-9
            ):
                incumbent = polished
                incumbent_value = polished_value
                polish_succeeded = True
        except ClassicalSolverError:
            pass
    initial_polish_seconds = perf_counter() - polish_start
    polished_initial_value = incumbent_value

    index_start = perf_counter()
    neighborhood_index = _build_neighborhood_index(problem)
    neighborhood_index_seconds = perf_counter() - index_start
    total_units = len(neighborhood_index.members_by_unit)
    target_groups = min(settings.initial_neighborhood_groups, total_units)
    minimum_groups = min(settings.minimum_neighborhood_groups, total_units)
    maximum_groups = min(settings.maximum_neighborhood_groups, total_units)

    accepted_moves = 0
    assignment_moves = 0
    local_solves = 0
    local_solve_seconds = 0.0
    residualization_seconds = 0.0
    global_validation_seconds = 0.0
    maximum_local_variables = 0
    maximum_local_constraints = 0
    maximum_local_mip_nodes = 0
    maximum_active_orders = 0
    maximum_active_groups = 0
    history: list[dict[str, Any]] = []

    for iteration in range(settings.iterations):
        active_orders, active_units, strategy = _select_exact_lns_neighborhood(
            problem,
            incumbent,
            settings,
            iteration,
            target_groups,
            neighborhood_index,
        )
        maximum_active_orders = max(maximum_active_orders, len(active_orders))
        maximum_active_groups = max(maximum_active_groups, len(active_units))

        residual_start = perf_counter()
        local_problem = residualize_problem(problem, incumbent, active_orders)
        local_incumbent = _solution_for_orders(
            incumbent, active_orders, "exact_lns_incumbent"
        )
        local_incumbent_value = evaluate_solution(
            local_problem, local_incumbent
        ).objective_value
        residualization_seconds += perf_counter() - residual_start

        solve_start = perf_counter()
        local_solution: Solution | None = None
        solve_error: str | None = None
        try:
            local_solution = solve_classical(
                local_problem,
                time_limit_seconds=settings.local_time_limit_seconds,
                mip_relative_gap=settings.mip_relative_gap,
                seed=settings.seed + iteration,
            )
            local_solves += 1
        except ClassicalSolverError as error:
            solve_error = str(error)
        solve_seconds = perf_counter() - solve_start
        local_solve_seconds += solve_seconds

        improved = False
        assignment_changed = False
        local_delta = 0.0
        global_delta = 0.0
        gap: float | None = None
        local_variables = 0
        local_constraints = 0
        local_best_bound: float | None = None
        local_mip_nodes = 0
        if local_solution is not None:
            gap_value = local_solution.metadata.get("optimality_gap")
            gap = None if gap_value is None else float(gap_value)
            local_variables = int(local_solution.metadata.get("n_variables", 0))
            local_constraints = int(local_solution.metadata.get("n_constraints", 0))
            bound_value = local_solution.metadata.get("best_bound")
            local_best_bound = (
                None if bound_value is None else float(bound_value)
            )
            node_value = local_solution.metadata.get("mip_node_count")
            local_mip_nodes = 0 if node_value is None else int(node_value)
            maximum_local_variables = max(maximum_local_variables, local_variables)
            maximum_local_constraints = max(
                maximum_local_constraints, local_constraints
            )
            maximum_local_mip_nodes = max(
                maximum_local_mip_nodes, local_mip_nodes
            )
            local_value = evaluate_solution(
                local_problem, local_solution
            ).objective_value
            local_delta = local_value - local_incumbent_value
            tolerance = 1e-9 * max(1.0, abs(local_incumbent_value))

            # Global concatenation and validation are relatively expensive on the
            # full POC.  The local objective is exactly additive after residualizing
            # frozen consumption, so a non-improving local solution cannot improve
            # the global incumbent and is rejected before that work.
            if local_delta > tolerance:
                incumbent_policy = _solution_fixed_assignments(local_incumbent)
                proposal_policy = _solution_fixed_assignments(local_solution)
                assignment_changed = proposal_policy != incumbent_policy
                validation_start = perf_counter()
                merged = _merge_local_solution(
                    incumbent,
                    local_solution,
                    active_orders,
                    runtime_seconds=perf_counter() - start,
                    method="exact_lns",
                )
                validation = validate_solution(problem, merged)
                merged_value = evaluate_solution(problem, merged).objective_value
                global_validation_seconds += perf_counter() - validation_start
                global_delta = merged_value - incumbent_value
                if validation.is_feasible and global_delta > tolerance:
                    incumbent = merged
                    incumbent_value = merged_value
                    accepted_moves += 1
                    assignment_moves += int(assignment_changed)
                    improved = True

        expensive = (
            local_solution is None
            or solve_seconds >= 0.9 * settings.local_time_limit_seconds
            or (
                gap is not None
                and gap > max(0.05, 2.0 * settings.mip_relative_gap)
            )
        )
        if settings.adaptive:
            if expensive and target_groups > minimum_groups:
                target_groups -= 1
            elif (
                solve_seconds <= 0.5 * settings.local_time_limit_seconds
                and len(active_units) >= target_groups
                and target_groups < maximum_groups
            ):
                target_groups += 1

        history.append(
            {
                "iteration": iteration,
                "strategy": strategy,
                "active_groups": len(active_units),
                "active_orders": len(active_orders),
                "local_variables": local_variables,
                "local_constraints": local_constraints,
                "local_runtime_seconds": solve_seconds,
                "local_optimality_gap": gap,
                "local_best_bound": local_best_bound,
                "local_mip_node_count": local_mip_nodes,
                "local_objective_delta": local_delta,
                "global_objective_delta": global_delta,
                "assignment_changed": assignment_changed,
                "accepted": improved,
                "next_target_groups": target_groups,
                "error": solve_error,
            }
        )

    runtime = perf_counter() - start
    assignments = incumbent.assignments.copy()
    assignments["method"] = "exact_lns"
    initialization_seconds = (
        baseline_initialization_seconds
        + initial_polish_seconds
        + neighborhood_index_seconds
    )
    return Solution(
        method="exact_lns",
        assignments=assignments,
        fulfillment=incumbent.fulfillment.copy(),
        runtime_seconds=runtime,
        raw_objective=incumbent_value,
        metadata={
            "algorithm": "adaptive-exact-milp-large-neighborhood-search",
            "execution_class": "classical-matheuristic",
            "initial_method": settings.initial_method,
            "raw_initial_objective": raw_initial_value,
            "polished_initial_objective": polished_initial_value,
            "initial_objective": polished_initial_value,
            "search_improvement": incumbent_value - polished_initial_value,
            "total_improvement": incumbent_value - raw_initial_value,
            "initial_polish_improvement": polished_initial_value - raw_initial_value,
            "initial_polish_succeeded": polish_succeeded,
            "polish_initial_incumbent": settings.polish_initial_incumbent,
            "iterations": settings.iterations,
            "accepted_moves": accepted_moves,
            "assignment_moves": assignment_moves,
            "local_solves": local_solves,
            "maximum_active_groups": maximum_active_groups,
            "maximum_active_orders": maximum_active_orders,
            "maximum_local_variables": maximum_local_variables,
            "maximum_local_constraints": maximum_local_constraints,
            "maximum_local_mip_nodes": maximum_local_mip_nodes,
            "initialization_seconds": initialization_seconds,
            "baseline_initialization_seconds": baseline_initialization_seconds,
            "initial_polish_seconds": initial_polish_seconds,
            "neighborhood_index_seconds": neighborhood_index_seconds,
            "residualization_seconds": residualization_seconds,
            "local_solve_seconds": local_solve_seconds,
            "global_validation_seconds": global_validation_seconds,
            "other_seconds": max(
                0.0,
                runtime
                - initialization_seconds
                - residualization_seconds
                - local_solve_seconds
                - global_validation_seconds,
            ),
            "history": history,
            "claim": "No quantum advantage is inferred from this classical run.",
        },
    )


def solve_hybrid(
    problem: ProblemData,
    *,
    config: HybridConfig | None = None,
) -> Solution:
    """Run bounded quantum-assisted LNS and return the best feasible incumbent."""

    settings = config or HybridConfig()
    settings.validate()
    start = perf_counter()
    raw_incumbent = (
        solve_default_baseline(problem)
        if settings.initial_method == "default"
        else solve_greedy_baseline(problem)
    )
    baseline_initialization_seconds = perf_counter() - start
    raw_initial_value = evaluate_solution(problem, raw_incumbent).objective_value
    if not validate_solution(problem, raw_incumbent).is_feasible:
        raise ValueError("The initial classical solution is infeasible")

    # Greedy assignment and greedy quantity allocation are separate components.
    # Polish the quantities once with assignments held fixed so subsequent gains
    # measure sampler-driven assignment changes, not hidden classical recourse.
    incumbent = raw_incumbent
    incumbent_value = raw_initial_value
    polish_start = perf_counter()
    polish_succeeded = False
    if settings.polish_initial_incumbent:
        try:
            polished = solve_classical(
                problem,
                time_limit_seconds=settings.recourse_time_limit_seconds,
                seed=settings.seed,
                fixed_assignments=_solution_fixed_assignments(raw_incumbent),
            )
            polished_value = evaluate_solution(problem, polished).objective_value
            if (
                validate_solution(problem, polished).is_feasible
                and polished_value >= incumbent_value - 1e-9
            ):
                incumbent = polished
                incumbent_value = polished_value
                polish_succeeded = True
        except ClassicalSolverError:
            # The feasible raw incumbent remains a valid fallback.  The metadata
            # makes a failed or disabled polish visible to experiment consumers.
            pass
    initial_polish_seconds = perf_counter() - polish_start
    polished_initial_value = incumbent_value
    initialization_seconds = baseline_initialization_seconds + initial_polish_seconds

    history: list[dict[str, Any]] = []
    accepted_moves = 0
    sampler_calls = 0
    recourse_solves = 0
    total_raw_samples = 0
    raw_one_hot_samples = 0
    maximum_qubo_variables = 0
    qubo_build_seconds = 0.0
    sampling_seconds = 0.0
    sample_decode_repair_seconds = 0.0
    recourse_seconds = 0.0
    sampler_runs: list[dict[str, object]] = []
    hardware_runs: list[dict[str, object]] = []

    neighborhood_index = _build_neighborhood_index(problem)
    for iteration in range(settings.iterations):
        build_start = perf_counter()
        active_orders = _select_neighborhood(
            problem,
            incumbent,
            settings,
            iteration,
            index=neighborhood_index,
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
        qubo_build_seconds += perf_counter() - build_start
        sampling_start = perf_counter()
        samples = sample_qubo(
            sampled_model,
            method=settings.sampler,
            num_samples=settings.num_reads,
            sweeps=settings.sweeps,
            seed=settings.seed + iteration,
            initial_sample=warm_start,
            allow_remote=settings.allow_remote,
            time_limit_seconds=settings.remote_time_limit_seconds,
            qaoa_layers=settings.qaoa_layers,
            qaoa_restarts=settings.qaoa_restarts,
            qaoa_mixer_topology=settings.qaoa_mixer_topology,
            qaoa_parameters=settings.qaoa_parameters,
            qaoa_readout_bitflip_probability=(
                settings.qaoa_readout_bitflip_probability
            ),
            max_feasible_states=settings.max_feasible_states,
            ibm_backend_name=settings.ibm_backend_name,
            ibm_mitigation_strategy=settings.ibm_mitigation_strategy,
            ibm_transpiler_optimization_level=(
                settings.ibm_transpiler_optimization_level
            ),
            ibm_transpiler_trials=settings.ibm_transpiler_trials,
            ibm_transpiler_seed=settings.ibm_transpiler_seed,
        )
        sampling_seconds += perf_counter() - sampling_start
        sampler_info = samples.attrs.get("sampler_info")
        if isinstance(sampler_info, dict):
            sampler_runs.append(dict(sampler_info))
            if sampler_info.get("remote"):
                hardware_runs.append(dict(sampler_info))
        sampler_calls += 1
        total_raw_samples += len(samples)

        sample_processing_start = perf_counter()
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
        sample_decode_repair_seconds += perf_counter() - sample_processing_start
        best_iteration = incumbent
        best_iteration_value = incumbent_value
        attempted = 0
        recourse_start = perf_counter()
        for _, _, repaired in candidates[: settings.top_k_recourse]:
            fixed = _sample_to_fixed_assignments(repaired, plans, local_problem)
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
        recourse_seconds += perf_counter() - recourse_start

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
    if settings.sampler == "ibm-qpu":
        execution_class = "quantum-assisted-hardware"
    elif settings.sampler == "qaoa_statevector":
        execution_class = "quantum-algorithm-simulation"
    elif settings.sampler == "simulated_annealing":
        execution_class = "quantum-inspired-classical"
    else:
        execution_class = "classical-control"
    def hardware_values(key: str) -> list[float]:
        return [
            float(run[key])
            for run in hardware_runs
            if isinstance(run.get(key), (int, float, np.integer, np.floating))
        ]

    def hardware_text_values(key: str) -> list[str]:
        return [
            str(run[key])
            for run in hardware_runs
            if run.get(key) not in {None, ""}
        ]

    def sampler_values(key: str) -> list[float]:
        return [
            float(run[key])
            for run in sampler_runs
            if isinstance(run.get(key), (int, float, np.integer, np.floating))
        ]

    backend_names = sorted(
        {
            str(run["backend_name"])
            for run in hardware_runs
            if run.get("backend_name") is not None
        }
    )
    mitigation_names = sorted(
        {
            str(run["mitigation_strategy"])
            for run in hardware_runs
            if run.get("mitigation_strategy") is not None
        }
    )
    hardware_present = bool(hardware_runs)
    def hardware_sum(key: str) -> float | None:
        values = hardware_values(key)
        return float(sum(values)) if values else None

    quantum_seconds = hardware_sum("hardware_quantum_seconds")
    return Solution(
        method="hybrid",
        assignments=assignments,
        fulfillment=incumbent.fulfillment.copy(),
        runtime_seconds=runtime,
        raw_objective=incumbent_value,
        metadata={
            "algorithm": "sampler-assisted-large-neighborhood-search",
            "execution_class": execution_class,
            "sampler": settings.sampler,
            "initial_method": settings.initial_method,
            "raw_initial_objective": raw_initial_value,
            "polished_initial_objective": polished_initial_value,
            # Backward-compatible name now means the incumbent at sampler entry.
            "initial_objective": polished_initial_value,
            "final_objective": incumbent_value,
            # Sampler attribution excludes the exact fixed-assignment polish.
            "improvement": incumbent_value - polished_initial_value,
            "total_improvement": incumbent_value - raw_initial_value,
            "initial_polish_improvement": polished_initial_value - raw_initial_value,
            "initial_polish_succeeded": polish_succeeded,
            "polish_initial_incumbent": settings.polish_initial_incumbent,
            "iterations": settings.iterations,
            "accepted_moves": accepted_moves,
            "sampler_calls": sampler_calls,
            "qpu_calls": (
                sampler_calls if settings.sampler == "ibm-qpu" else 0
            ),
            "quantum_simulator_calls": (
                sampler_calls if settings.sampler == "qaoa_statevector" else 0
            ),
            "qpu_access_time_microseconds": (
                quantum_seconds * 1_000_000.0
                if quantum_seconds is not None
                else None
            ),
            "hardware_backend": ",".join(backend_names) if backend_names else None,
            "hardware_job_ids": (
                ",".join(hardware_text_values("job_id"))
                if hardware_present
                else None
            ),
            "hardware_created_at": (
                min(hardware_text_values("hardware_created_at"), default=None)
            ),
            "hardware_finished_at": (
                max(hardware_text_values("hardware_finished_at"), default=None)
            ),
            "hardware_physical_qubit_mappings": (
                ";".join(sorted(set(hardware_text_values("physical_qubit_mapping"))))
                if hardware_present
                else None
            ),
            "hardware_calibration_last_update_at": (
                max(hardware_text_values("calibration_last_update_at"), default=None)
            ),
            "hardware_qiskit_versions": (
                ",".join(sorted(set(hardware_text_values("qiskit_version"))))
                if hardware_present
                else None
            ),
            "hardware_qiskit_ibm_runtime_versions": (
                ",".join(
                    sorted(set(hardware_text_values("qiskit_ibm_runtime_version")))
                )
                if hardware_present
                else None
            ),
            "hardware_backend_pending_jobs": (
                min(hardware_values("backend_pending_jobs_at_selection"), default=None)
            ),
            "hardware_mitigation_strategy": (
                ",".join(mitigation_names) if mitigation_names else None
            ),
            "hardware_wall_seconds": (
                hardware_sum("hardware_wall_seconds")
            ),
            "hardware_queue_seconds": (
                hardware_sum("hardware_queue_seconds")
            ),
            "hardware_execution_seconds": hardware_sum(
                "hardware_execution_seconds"
            ),
            "hardware_turnaround_seconds": (
                hardware_sum("hardware_turnaround_seconds")
            ),
            "hardware_quantum_seconds": quantum_seconds,
            "hardware_angle_optimization_seconds": hardware_sum(
                "angle_optimization_seconds"
            ),
            "hardware_circuit_construction_seconds": hardware_sum(
                "circuit_construction_seconds"
            ),
            "hardware_backend_selection_seconds": hardware_sum(
                "backend_selection_seconds"
            ),
            "hardware_transpilation_seconds": hardware_sum(
                "transpilation_seconds"
            ),
            "hardware_primitive_submit_seconds": hardware_sum(
                "primitive_submit_seconds"
            ),
            "hardware_primitive_wait_seconds": hardware_sum(
                "primitive_wait_seconds"
            ),
            "hardware_decode_seconds": hardware_sum("decode_seconds"),
            "hardware_angle_seeds": (
                ",".join(sorted(set(hardware_text_values("angle_seed"))))
                if hardware_present
                else None
            ),
            "hardware_transpiler_base_seeds": (
                ",".join(
                    sorted(set(hardware_text_values("transpiler_base_seed")))
                )
                if hardware_present
                else None
            ),
            "hardware_selected_transpiler_seeds": (
                ",".join(
                    sorted(set(hardware_text_values("selected_transpiler_seed")))
                )
                if hardware_present
                else None
            ),
            "hardware_returned_samples": (
                sum(hardware_values("returned_samples")) if hardware_present else None
            ),
            "hardware_feasible_shots": (
                sum(hardware_values("hardware_feasible_shots"))
                if hardware_present
                else None
            ),
            "hardware_backend_num_qubits": max(
                hardware_values("backend_num_qubits"), default=None
            ),
            "hardware_logical_qubits": max(
                hardware_values("logical_qubits"), default=None
            ),
            "hardware_transpiled_depth": max(
                hardware_values("transpiled_depth"), default=None
            ),
            "hardware_two_qubit_gates": max(
                hardware_values("transpiled_two_qubit_gates"), default=None
            ),
            "hardware_two_qubit_depth": max(
                hardware_values("transpiled_two_qubit_depth"), default=None
            ),
            "hardware_qubo_optimal_hit_rate": (
                float(np.mean(hardware_values("hardware_qubo_optimal_hit_rate")))
                if hardware_values("hardware_qubo_optimal_hit_rate")
                else None
            ),
            "hardware_qubo_near_optimal_1pct_rate": (
                float(
                    np.mean(
                        hardware_values("hardware_qubo_near_optimal_1pct_rate")
                    )
                )
                if hardware_values("hardware_qubo_near_optimal_1pct_rate")
                else None
            ),
            "hardware_mean_feasible_normalized_gap": (
                float(
                    np.mean(
                        hardware_values("hardware_mean_feasible_normalized_gap")
                    )
                )
                if hardware_values("hardware_mean_feasible_normalized_gap")
                else None
            ),
            "hardware_uniform_feasible_optimal_rate": (
                float(np.mean(hardware_values("uniform_feasible_optimal_rate")))
                if hardware_values("uniform_feasible_optimal_rate")
                else None
            ),
            "hardware_uniform_feasible_near_optimal_1pct_rate": (
                float(
                    np.mean(
                        hardware_values("uniform_feasible_near_optimal_1pct_rate")
                    )
                )
                if hardware_values("uniform_feasible_near_optimal_1pct_rate")
                else None
            ),
            "hardware_uniform_feasible_mean_normalized_gap": (
                float(
                    np.mean(
                        hardware_values("uniform_feasible_mean_normalized_gap")
                    )
                )
                if hardware_values("uniform_feasible_mean_normalized_gap")
                else None
            ),
            "hardware_qaoa_parameter_cache_hits": sum(
                1 for run in hardware_runs if bool(run.get("qaoa_parameter_cache_hit"))
            ),
            "qaoa_mixer_topology": settings.qaoa_mixer_topology,
            "hardware_qubo_optimal_hit_rate_given_feasible": (
                float(
                    np.mean(
                        hardware_values(
                            "hardware_qubo_optimal_hit_rate_given_feasible"
                        )
                    )
                )
                if hardware_values(
                    "hardware_qubo_optimal_hit_rate_given_feasible"
                )
                else None
            ),
            "hardware_best_feasible_normalized_gap": min(
                hardware_values("hardware_best_feasible_normalized_gap"),
                default=None,
            ),
            "recourse_solves": recourse_solves,
            "maximum_qubo_variables": maximum_qubo_variables,
            "maximum_candidates_per_order": settings.max_candidates_per_order,
            "initialization_seconds": initialization_seconds,
            "baseline_initialization_seconds": baseline_initialization_seconds,
            "initial_polish_seconds": initial_polish_seconds,
            "qubo_build_seconds": qubo_build_seconds,
            "sampling_seconds": sampling_seconds,
            "sample_decode_repair_seconds": sample_decode_repair_seconds,
            "recourse_seconds": recourse_seconds,
            "other_seconds": max(
                0.0,
                runtime
                - initialization_seconds
                - qubo_build_seconds
                - sampling_seconds
                - sample_decode_repair_seconds
                - recourse_seconds,
            ),
            "raw_one_hot_rate": (
                raw_one_hot_samples / total_raw_samples if total_raw_samples else 0.0
            ),
            "sample_qubo_optimal_hit_rate": (
                float(np.mean(sampler_values("sample_qubo_optimal_hit_rate")))
                if sampler_values("sample_qubo_optimal_hit_rate")
                else None
            ),
            "sample_qubo_near_optimal_1pct_rate": (
                float(np.mean(sampler_values("sample_qubo_near_optimal_1pct_rate")))
                if sampler_values("sample_qubo_near_optimal_1pct_rate")
                else None
            ),
            "sample_mean_feasible_normalized_gap": (
                float(np.mean(sampler_values("sample_mean_feasible_normalized_gap")))
                if sampler_values("sample_mean_feasible_normalized_gap")
                else None
            ),
            "uniform_feasible_optimal_rate": (
                float(np.mean(sampler_values("uniform_feasible_optimal_rate")))
                if sampler_values("uniform_feasible_optimal_rate")
                else None
            ),
            "remote_enabled": settings.allow_remote,
            "qubo_noise_relative_sigma": settings.qubo_noise_relative_sigma,
            "qaoa_readout_bitflip_probability": (
                settings.qaoa_readout_bitflip_probability
            ),
            "batch_strategy": settings.batch_strategy,
            "history": history,
            "hardware_runs": hardware_runs,
            "claim": "No quantum advantage is inferred from this run.",
        },
    )
