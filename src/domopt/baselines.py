"""Deterministic default and sequential greedy baselines."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from time import perf_counter

import numpy as np
import pandas as pd

from .penalties import PenaltyContext, build_penalty_context
from .resources import (
    split_pick_quantities,
    uses_split_pick_accounting,
)
from .schemas import ProblemData, Solution


@dataclass(frozen=True)
class _CandidatePlan:
    candidate_id: str
    order_id: str
    dc_id: str
    pgi_date: pd.Timestamp
    is_default: bool
    shipping_cost: float
    dock_units: float
    quantities: dict[str, int]
    score: float
    incremental_score: float


@dataclass(frozen=True)
class _DecisionPlan:
    """One atomic order or assignment-group decision."""

    unit_id: str
    option_id: str
    member_plans: tuple[_CandidatePlan, ...]
    score: float
    incremental_score: float


@dataclass(frozen=True, slots=True)
class _LineData:
    """Compact order-line record used in the baseline inner loop."""

    sku_id: str
    demand_cases: int
    unit_value: float
    penalty_per_unfilled_case: float
    cases_per_pallet: int | None
    unit_weight: float
    unit_volume: float

    @property
    def marginal_value(self) -> float:
        return self.unit_value + self.penalty_per_unfilled_case


@dataclass(frozen=True, slots=True)
class _CandidateData:
    """Compact candidate record shared by every preview of one decision."""

    candidate_id: str
    order_id: str
    dc_id: str
    pgi_date: pd.Timestamp
    is_default: bool
    shipping_cost: float
    dock_units: float


@dataclass(frozen=True)
class _BaselineCache:
    """Problem-static records reused by every residual candidate preview.

    Pandas remains at the input/output boundary.  The planning loop reads compact
    immutable records and direct dictionaries so its cost follows the retained
    candidates and order lines instead of repeated DataFrame construction.
    """

    lines_by_order: dict[str, tuple[_LineData, ...]]
    ranked_lines_by_order: dict[str, tuple[_LineData, ...]]
    members_by_unit: dict[str, tuple[str, ...]]
    options_by_unit: dict[
        str,
        tuple[tuple[str, tuple[_CandidateData, ...]], ...],
    ]
    penalty_context: PenaltyContext
    unassigned_score_by_order: dict[str, float]
    default_dc_by_order: dict[str, str]
    minimum_divert_by_order: dict[str, int | None]
    split_picks: bool


def _optional_float(value: object, default: float = 0.0) -> float:
    return default if value is None or pd.isna(value) else float(value)


def _line_penalty(
    lines: tuple[_LineData, ...],
    quantities: dict[str, int],
    order_id: str,
    context: PenaltyContext,
) -> float:
    """Evaluate the authoritative penalty without allocating a DataFrame."""

    unfulfilled = [line.demand_cases - int(quantities.get(line.sku_id, 0)) for line in lines]
    linear = sum(
        line.penalty_per_unfilled_case * unmet
        for line, unmet in zip(lines, unfulfilled, strict=True)
    )
    if context.mode == "linear_unmet":
        return float(linear)

    fulfilled = sum(line.demand_cases for line in lines) - sum(unfulfilled)
    if fulfilled >= context.activation_fill_by_order[order_id]:
        return 0.0
    parameters = context.parameters_by_order[order_id]
    raw = (
        float(linear)
        + parameters["fixed"]
        + parameters["per_cut_sku"] * sum(value > 0 for value in unfulfilled)
    )
    penalty = max(raw, parameters["minimum"])
    if parameters["maximum"] > 0:
        penalty = min(penalty, parameters["maximum"])
    return float(penalty)


def _make_baseline_cache(problem: ProblemData) -> _BaselineCache:
    lines_by_order: defaultdict[str, list[_LineData]] = defaultdict(list)
    for row in problem.order_lines.itertuples(index=False):
        per_pallet = getattr(row, "cases_per_pallet", None)
        lines_by_order[str(row.order_id)].append(
            _LineData(
                sku_id=str(row.sku_id),
                demand_cases=int(row.demand_cases),
                unit_value=float(row.unit_value),
                penalty_per_unfilled_case=float(row.penalty_per_unfilled_case),
                cases_per_pallet=(
                    None if per_pallet is None or pd.isna(per_pallet) else int(per_pallet)
                ),
                unit_weight=_optional_float(getattr(row, "unit_weight", 0.0)),
                unit_volume=_optional_float(getattr(row, "unit_volume", 0.0)),
            )
        )
    compact_lines = {
        order_id: tuple(sorted(records, key=lambda line: line.sku_id))
        for order_id, records in lines_by_order.items()
    }
    ranked_lines = {
        order_id: tuple(sorted(records, key=lambda line: (-line.marginal_value, line.sku_id)))
        for order_id, records in compact_lines.items()
    }
    context = build_penalty_context(problem)
    unassigned = {
        order_id: -_line_penalty(lines, {}, order_id, context)
        for order_id, lines in compact_lines.items()
    }
    default_dc = {
        str(row.order_id): str(row.default_dc) for row in problem.orders.itertuples(index=False)
    }

    minimum_divert: dict[str, int | None] = {order_id: None for order_id in compact_lines}
    if bool(problem.metadata.get("enforce_min_divert_improvement", False)):
        minimum_cases = int(problem.metadata.get("min_divert_improvement_cases", 0))
        for row in problem.orders.itertuples(index=False):
            order_id = str(row.order_id)
            default_fillable = getattr(row, "default_fillable_cases", None)
            if default_fillable is None or pd.isna(default_fillable):
                raise ValueError("The minimum-divert rule requires orders.default_fillable_cases")
            fraction_value = getattr(row, "min_divert_improvement_fraction", None)
            fraction = (
                float(problem.metadata.get("min_divert_improvement_fraction", 0.05))
                if fraction_value is None or pd.isna(fraction_value)
                else float(fraction_value)
            )
            demand = sum(line.demand_cases for line in compact_lines[order_id])
            improvement = max(ceil(fraction * demand - 1e-12), minimum_cases)
            minimum_divert[order_id] = min(
                demand,
                int(default_fillable) + improvement,
            )

    members_by_unit = _assignment_units(problem)
    unit_for_order = {
        order_id: unit_id for unit_id, members in members_by_unit.items() for order_id in members
    }
    grouped = bool(problem.metadata.get("enforce_assignment_group", False))
    option_records: defaultdict[
        str,
        defaultdict[str, list[_CandidateData]],
    ] = defaultdict(lambda: defaultdict(list))
    eligible = problem.candidates.loc[problem.candidates["eligible"].astype(bool)]
    for row in eligible.itertuples(index=False):
        order_id = str(row.order_id)
        unit_id = unit_for_order[order_id]
        option_id = str(row.group_option_id) if grouped else str(row.candidate_id)
        dock_units = getattr(row, "dock_units", 1.0)
        option_records[unit_id][option_id].append(
            _CandidateData(
                candidate_id=str(row.candidate_id),
                order_id=order_id,
                dc_id=str(row.dc_id),
                pgi_date=pd.Timestamp(row.pgi_date),
                is_default=bool(row.is_default),
                shipping_cost=float(row.shipping_cost),
                dock_units=(
                    1.0 if dock_units is None or pd.isna(dock_units) else float(dock_units)
                ),
            )
        )
    options_by_unit: dict[
        str,
        tuple[tuple[str, tuple[_CandidateData, ...]], ...],
    ] = {}
    for unit_id, members in members_by_unit.items():
        expected = set(members)
        options: list[tuple[str, tuple[_CandidateData, ...]]] = []
        for option_id, records in option_records.get(unit_id, {}).items():
            if {record.order_id for record in records} != expected:
                continue
            options.append((option_id, tuple(sorted(records, key=lambda record: record.order_id))))
        options_by_unit[unit_id] = tuple(sorted(options, key=lambda item: item[0]))

    return _BaselineCache(
        lines_by_order=compact_lines,
        ranked_lines_by_order=ranked_lines,
        members_by_unit=members_by_unit,
        options_by_unit=options_by_unit,
        penalty_context=context,
        unassigned_score_by_order=unassigned,
        default_dc_by_order=default_dc,
        minimum_divert_by_order=minimum_divert,
        split_picks=uses_split_pick_accounting(problem),
    )


class _InventoryState:
    """Residual cumulative inventory with correct earlier-date consumption."""

    def __init__(self, inventory: pd.DataFrame):
        grouped: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
        ordered = inventory.sort_values(["dc_id", "sku_id", "date"], kind="mergesort")
        for row in ordered.itertuples(index=False):
            key = (str(row.dc_id), str(row.sku_id))
            dates, amounts = grouped.setdefault(key, ([], []))
            dates.append(pd.Timestamp(row.date).value)
            amounts.append(int(row.cumulative_available_cases))
        self._rows: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {
            key: (
                np.asarray(dates, dtype=np.int64),
                np.asarray(amounts, dtype=np.int64),
            )
            for key, (dates, amounts) in grouped.items()
        }
        # Arrays in _rows are immutable base data. Every consume creates a new
        # remaining array, so preview clones need only copy this small dictionary.
        self._remaining: dict[tuple[str, str], np.ndarray] = {}

    def available(self, dc_id: str, sku_id: str, date: pd.Timestamp) -> int:
        key = (dc_id, sku_id)
        rows = self._rows.get(key)
        if rows is None:
            return 0
        dates, base = rows
        covering = dates >= pd.Timestamp(date).value
        if not covering.any():
            return 0
        amounts = self._remaining.get(key, base)
        return max(0, int(amounts[covering].min()))

    def consume(self, dc_id: str, sku_id: str, date: pd.Timestamp, quantity: int) -> None:
        if quantity < 0:
            raise ValueError("Cannot consume a negative inventory quantity")
        if quantity == 0:
            return
        if quantity > self.available(dc_id, sku_id, date):
            raise ValueError(f"Insufficient inventory for dc={dc_id}, sku={sku_id}, date={date}")
        key = (dc_id, sku_id)
        dates, base = self._rows[key]
        updated = self._remaining.get(key, base).copy()
        updated[dates >= pd.Timestamp(date).value] -= int(quantity)
        self._remaining[key] = updated

    def clone(self) -> _InventoryState:
        clone = object.__new__(_InventoryState)
        clone._rows = self._rows
        clone._remaining = dict(self._remaining)
        return clone


class _CapacityState:
    """Residual exact-date capacities used by deterministic baselines."""

    def __init__(self, capacities: pd.DataFrame):
        self._remaining: dict[tuple[str, pd.Timestamp, str], float] = {}
        for row in capacities.itertuples(index=False):
            self._remaining[(str(row.dc_id), pd.Timestamp(row.date), str(row.resource))] = float(
                row.capacity
            )

    def remaining(self, dc_id: str, date: pd.Timestamp, resource: str) -> float:
        return self._remaining.get((dc_id, date, resource), float("inf"))

    def consume(self, dc_id: str, date: pd.Timestamp, resource: str, amount: float) -> None:
        key = (dc_id, date, resource)
        if key not in self._remaining:
            return
        if amount > self._remaining[key] + 1e-9:
            raise ValueError(f"Insufficient {resource} capacity for {dc_id} on {date.date()}")
        self._remaining[key] -= amount

    def clone(self) -> _CapacityState:
        clone = object.__new__(_CapacityState)
        clone._remaining = dict(self._remaining)
        return clone


def _preview_candidate(
    problem: ProblemData,
    candidate: _CandidateData,
    inventory: _InventoryState,
    capacities: _CapacityState,
    cache: _BaselineCache,
) -> _CandidatePlan | None:
    order_id = candidate.order_id
    dc_id = candidate.dc_id
    pgi_date = candidate.pgi_date
    lines = cache.ranked_lines_by_order[order_id]

    dock_units = candidate.dock_units
    if capacities.remaining(dc_id, pgi_date, "dock") < dock_units - 1e-9:
        return None

    # Allocate cases in descending marginal business value. This is a baseline,
    # not the exact optimizer; deterministic tie breaks keep results reproducible.
    residual_resources = {
        resource: capacities.remaining(dc_id, pgi_date, resource)
        for resource in [
            "throughput_cases",
            "case_pick",
            "pallet_pick",
            "weight",
            "volume",
        ]
    }
    quantities: dict[str, int] = {}

    for line in lines:
        maximum = min(
            line.demand_cases,
            inventory.available(dc_id, line.sku_id, pgi_date),
        )

        if np.isfinite(residual_resources["throughput_cases"]):
            maximum = min(maximum, int(np.floor(residual_resources["throughput_cases"])))
        unit_weight = line.unit_weight
        unit_volume = line.unit_volume
        if unit_weight > 0 and np.isfinite(residual_resources["weight"]):
            maximum = min(maximum, int(np.floor(residual_resources["weight"] / unit_weight)))
        if unit_volume > 0 and np.isfinite(residual_resources["volume"]):
            maximum = min(maximum, int(np.floor(residual_resources["volume"] / unit_volume)))

        if cache.split_picks:
            if line.cases_per_pallet is None:
                raise ValueError("Pallet/case-pick accounting requires cases_per_pallet")
            per_pallet = line.cases_per_pallet
            maximum_pallets = int(maximum) // per_pallet
            if np.isfinite(residual_resources["pallet_pick"]):
                maximum_pallets = min(
                    maximum_pallets,
                    int(np.floor(residual_resources["pallet_pick"])),
                )
            loose_limit = int(maximum) - maximum_pallets * per_pallet
            if np.isfinite(residual_resources["case_pick"]):
                loose_limit = min(
                    loose_limit,
                    int(np.floor(residual_resources["case_pick"])),
                )
            loose_limit = min(loose_limit, per_pallet - 1)
            quantity = maximum_pallets * per_pallet + max(0, loose_limit)
            residual_resources["pallet_pick"] -= maximum_pallets
            residual_resources["case_pick"] -= max(0, loose_limit)
        else:
            if np.isfinite(residual_resources["case_pick"]):
                maximum = min(
                    maximum,
                    int(np.floor(residual_resources["case_pick"])),
                )
            quantity = max(0, int(maximum))
            residual_resources["case_pick"] -= quantity

        quantities[line.sku_id] = quantity
        residual_resources["throughput_cases"] -= quantity
        if unit_weight > 0:
            residual_resources["weight"] -= quantity * unit_weight
        if unit_volume > 0:
            residual_resources["volume"] -= quantity * unit_volume

    canonical_lines = cache.lines_by_order[order_id]
    score = (
        sum(line.unit_value * int(quantities.get(line.sku_id, 0)) for line in canonical_lines)
        - _line_penalty(
            canonical_lines,
            quantities,
            order_id,
            cache.penalty_context,
        )
        - candidate.shipping_cost
    )
    unassigned = cache.unassigned_score_by_order[order_id]
    if dc_id != cache.default_dc_by_order[order_id]:
        threshold = cache.minimum_divert_by_order[order_id]
        if threshold is not None and sum(quantities.values()) < threshold:
            return None
    return _CandidatePlan(
        candidate_id=candidate.candidate_id,
        order_id=order_id,
        dc_id=dc_id,
        pgi_date=pgi_date,
        is_default=candidate.is_default,
        shipping_cost=candidate.shipping_cost,
        dock_units=dock_units,
        quantities=quantities,
        score=float(score),
        incremental_score=score - unassigned,
    )


def _commit_plan(
    problem: ProblemData,
    plan: _CandidatePlan,
    inventory: _InventoryState,
    capacities: _CapacityState,
    cache: _BaselineCache,
) -> None:
    lines = {line.sku_id: line for line in cache.lines_by_order[plan.order_id]}
    total_cases = 0
    total_case_picks = 0
    total_pallet_picks = 0
    total_weight = 0.0
    total_volume = 0.0
    for sku_id, quantity in plan.quantities.items():
        inventory.consume(plan.dc_id, sku_id, plan.pgi_date, int(quantity))
        total_cases += int(quantity)
        line = lines[sku_id]
        if cache.split_picks:
            if line.cases_per_pallet is None:
                raise ValueError("Pallet/case-pick accounting requires cases_per_pallet")
            per_pallet = line.cases_per_pallet
            pallets, loose = split_pick_quantities(quantity, per_pallet)
            total_pallet_picks += pallets
            total_case_picks += loose
        else:
            total_case_picks += int(quantity)
        total_weight += int(quantity) * line.unit_weight
        total_volume += int(quantity) * line.unit_volume

    capacities.consume(plan.dc_id, plan.pgi_date, "dock", plan.dock_units)
    capacities.consume(plan.dc_id, plan.pgi_date, "throughput_cases", total_cases)
    capacities.consume(plan.dc_id, plan.pgi_date, "case_pick", total_case_picks)
    capacities.consume(plan.dc_id, plan.pgi_date, "pallet_pick", total_pallet_picks)
    capacities.consume(plan.dc_id, plan.pgi_date, "weight", total_weight)
    capacities.consume(plan.dc_id, plan.pgi_date, "volume", total_volume)


def _assignment_units(problem: ProblemData) -> dict[str, tuple[str, ...]]:
    """Return the atomic decisions used by every baseline.

    Nestle loads may contain several order records. When group cohesion is enabled,
    every member must choose the same DC/date option, so treating members as separate
    greedy decisions can create an infeasible split load.
    """

    if bool(problem.metadata.get("enforce_assignment_group", False)):
        if "assignment_group" not in problem.orders.columns:
            raise ValueError("Group cohesion requires orders.assignment_group")
        return {
            str(group_id): tuple(sorted(group["order_id"].astype(str)))
            for group_id, group in problem.orders.groupby("assignment_group", sort=False)
        }
    return {str(order_id): (str(order_id),) for order_id in problem.orders["order_id"].astype(str)}


def _candidate_options_for_unit(
    problem: ProblemData,
    eligible: pd.DataFrame,
    unit_id: str,
    members: tuple[str, ...],
    *,
    default_only: bool = False,
    cache: _BaselineCache,
) -> list[tuple[str, tuple[_CandidateData, ...]]]:
    if cache.members_by_unit.get(unit_id) != members:
        raise ValueError(f"Unknown or mismatched decision unit {unit_id!r}")
    options = list(cache.options_by_unit[unit_id])
    if default_only:
        options = [
            (option_id, rows) for option_id, rows in options if all(row.is_default for row in rows)
        ]
    return options


def _preview_decision(
    problem: ProblemData,
    unit_id: str,
    option_id: str,
    option_rows: tuple[_CandidateData, ...],
    inventory: _InventoryState,
    capacities: _CapacityState,
    cache: _BaselineCache,
) -> _DecisionPlan | None:
    """Preview all members against private residual states, then commit atomically."""

    trial_inventory = inventory.clone()
    trial_capacities = capacities.clone()
    member_plans: list[_CandidatePlan] = []
    unassigned_score = 0.0

    for candidate in option_rows:
        order_id = candidate.order_id
        plan = _preview_candidate(
            problem,
            candidate,
            trial_inventory,
            trial_capacities,
            cache,
        )
        if plan is None:
            return None
        _commit_plan(problem, plan, trial_inventory, trial_capacities, cache)
        member_plans.append(plan)
        unassigned_score += cache.unassigned_score_by_order[order_id]

    score = sum(plan.score for plan in member_plans)
    return _DecisionPlan(
        unit_id=unit_id,
        option_id=option_id,
        member_plans=tuple(member_plans),
        score=score,
        incremental_score=score - unassigned_score,
    )


def _feasible_decisions(
    problem: ProblemData,
    unit_id: str,
    members: tuple[str, ...],
    eligible: pd.DataFrame,
    inventory: _InventoryState,
    capacities: _CapacityState,
    *,
    default_only: bool = False,
    cache: _BaselineCache | None = None,
) -> list[_DecisionPlan]:
    planning_cache = cache or _make_baseline_cache(problem)
    decisions: list[_DecisionPlan] = []
    for option_id, option_rows in _candidate_options_for_unit(
        problem,
        eligible,
        unit_id,
        members,
        default_only=default_only,
        cache=planning_cache,
    ):
        decision = _preview_decision(
            problem,
            unit_id,
            option_id,
            option_rows,
            inventory,
            capacities,
            planning_cache,
        )
        if decision is not None:
            decisions.append(decision)
    return decisions


def _decision_sort_key(decision: _DecisionPlan) -> tuple[object, ...]:
    return (
        -decision.incremental_score,
        -sum(sum(plan.quantities.values()) for plan in decision.member_plans),
        sum(plan.shipping_cost for plan in decision.member_plans),
        not all(plan.is_default for plan in decision.member_plans),
        decision.option_id,
    )


def _commit_decision(
    problem: ProblemData,
    decision: _DecisionPlan,
    inventory: _InventoryState,
    capacities: _CapacityState,
    selected: dict[str, _CandidatePlan | None],
    cache: _BaselineCache,
) -> None:
    for plan in decision.member_plans:
        _commit_plan(problem, plan, inventory, capacities, cache)
        selected[plan.order_id] = plan


def _solution_from_plans(
    problem: ProblemData,
    method: str,
    selected: dict[str, _CandidatePlan | None],
    runtime_seconds: float,
    cache: _BaselineCache,
) -> Solution:
    default_dc = cache.default_dc_by_order
    assignment_rows: list[dict[str, object]] = []
    fulfillment_rows: list[dict[str, object]] = []

    for order_id in sorted(problem.orders["order_id"].astype(str)):
        plan = selected.get(order_id)
        if plan is None:
            assignment_rows.append(
                {
                    "order_id": order_id,
                    "candidate_id": None,
                    "selected_dc": None,
                    "selected_pgi_date": None,
                    "is_unassigned": True,
                    "is_divert": False,
                    "method": method,
                }
            )
        else:
            assignment_rows.append(
                {
                    "order_id": order_id,
                    "candidate_id": plan.candidate_id,
                    "selected_dc": plan.dc_id,
                    "selected_pgi_date": plan.pgi_date,
                    "is_unassigned": False,
                    "is_divert": plan.dc_id != default_dc[order_id],
                    "method": method,
                }
            )

        for line in cache.lines_by_order[order_id]:
            fulfilled = 0 if plan is None else int(plan.quantities.get(line.sku_id, 0))
            fulfillment_rows.append(
                {
                    "order_id": order_id,
                    "sku_id": line.sku_id,
                    "fulfilled_cases": fulfilled,
                    "unfulfilled_cases": line.demand_cases - fulfilled,
                    "selected_dc": None if plan is None else plan.dc_id,
                    "selected_pgi_date": None if plan is None else plan.pgi_date,
                }
            )

    return Solution(
        method=method,
        assignments=pd.DataFrame(assignment_rows),
        fulfillment=pd.DataFrame(fulfillment_rows),
        runtime_seconds=runtime_seconds,
        metadata={"deterministic": True},
    )


def solve_default_baseline(problem: ProblemData) -> Solution:
    """Keep each order at its earliest eligible default candidate.

    Shared inventory is allocated sequentially in lexicographic ``order_id`` order.
    """

    start = perf_counter()
    inventory = _InventoryState(problem.inventory)
    capacities = _CapacityState(problem.capacities)
    selected: dict[str, _CandidatePlan | None] = {}
    cache = _make_baseline_cache(problem)

    eligible = problem.candidates[problem.candidates["eligible"]].copy()
    units = cache.members_by_unit
    for unit_id, members in sorted(units.items()):
        decisions = _feasible_decisions(
            problem,
            unit_id,
            members,
            eligible,
            inventory,
            capacities,
            default_only=True,
            cache=cache,
        )
        if not decisions:
            selected.update({order_id: None for order_id in members})
            continue
        decisions.sort(
            key=lambda decision: (
                max(plan.pgi_date for plan in decision.member_plans),
                decision.option_id,
            )
        )
        _commit_decision(
            problem,
            decisions[0],
            inventory,
            capacities,
            selected,
            cache,
        )

    return _solution_from_plans(
        problem,
        "default",
        selected,
        runtime_seconds=perf_counter() - start,
        cache=cache,
    )


def solve_greedy_baseline(problem: ProblemData) -> Solution:
    """Sequential scarcity-aware greedy reassignment baseline.

    Rank atomic orders or assignment groups once, then choose the best currently
    feasible option for each unit. Residual inventory and capacities are updated
    immediately, and an assignment group is never split across outcomes.
    """

    start = perf_counter()
    inventory = _InventoryState(problem.inventory)
    capacities = _CapacityState(problem.capacities)
    selected: dict[str, _CandidatePlan | None] = {}
    eligible = problem.candidates[problem.candidates["eligible"]].copy()
    cache = _make_baseline_cache(problem)
    units = cache.members_by_unit

    # Establish one stable priority from the unconstrained initial previews. The
    # previous implementation rescanned every remaining order after every commit,
    # giving quadratic DataFrame work. Each unit is now revisited exactly once
    # against current residual resources, while candidate choice remains dynamic.
    priority: list[tuple[tuple[object, ...], str]] = []
    for unit_id, members in units.items():
        initial = _feasible_decisions(
            problem,
            unit_id,
            members,
            eligible,
            inventory,
            capacities,
            cache=cache,
        )
        positive = [decision for decision in initial if decision.incremental_score > 1e-9]
        if positive:
            positive.sort(key=_decision_sort_key)
            priority.append((_decision_sort_key(positive[0]), unit_id))
        else:
            priority.append(((float("inf"), unit_id), unit_id))

    for _, unit_id in sorted(priority, key=lambda item: (item[0], item[1])):
        members = units[unit_id]
        decisions = _feasible_decisions(
            problem,
            unit_id,
            members,
            eligible,
            inventory,
            capacities,
            cache=cache,
        )
        positive = [decision for decision in decisions if decision.incremental_score > 1e-9]
        if not positive:
            selected.update({order_id: None for order_id in members})
            continue
        positive.sort(key=_decision_sort_key)
        _commit_decision(
            problem,
            positive[0],
            inventory,
            capacities,
            selected,
            cache,
        )

    return _solution_from_plans(
        problem,
        "greedy",
        selected,
        runtime_seconds=perf_counter() - start,
        cache=cache,
    )


def solve_polished_greedy(
    problem: ProblemData,
    *,
    backend: str = "scipy-highs",
    time_limit_seconds: float = 30.0,
    mip_relative_gap: float = 0.0,
    seed: int | None = None,
    thread_count: int | None = None,
) -> Solution:
    """Polish greedy quantities exactly while preserving its assignment policy.

    This is the strongest low-risk production baseline: greedy chooses the whole-load
    routing policy, then a structurally reduced MILP reallocates cases and evaluates
    thresholded penalties exactly.  Failure or degradation falls back to the feasible
    greedy incumbent.
    """

    # Local imports keep the elementary default/greedy module independent at import
    # time while avoiding a circular dependency with hybrid initialization.
    from .classical import ClassicalSolverError, solve_classical
    from .objective import evaluate_solution
    from .validation import validate_solution

    start = perf_counter()
    raw = solve_greedy_baseline(problem)
    raw_value = evaluate_solution(problem, raw).objective_value
    policy = {
        str(row.order_id): (None if bool(row.is_unassigned) else str(row.candidate_id))
        for row in raw.assignments.itertuples(index=False)
    }
    polish_start = perf_counter()
    best = raw
    best_value = raw_value
    succeeded = False
    recourse_metadata: dict[str, object] = {}
    try:
        polished = solve_classical(
            problem,
            backend=backend,
            time_limit_seconds=time_limit_seconds,
            mip_relative_gap=mip_relative_gap,
            seed=seed,
            thread_count=thread_count,
            fixed_assignments=policy,
            minimum_objective=raw_value,
        )
        recourse_metadata = dict(polished.metadata)
        polished_value = evaluate_solution(problem, polished).objective_value
        if validate_solution(problem, polished).is_feasible and polished_value >= raw_value - 1e-9:
            best = polished
            best_value = polished_value
            succeeded = True
    except ClassicalSolverError as error:
        recourse_metadata = {"error": str(error)}

    assignments = best.assignments.copy()
    assignments["method"] = "polished_greedy"
    runtime = perf_counter() - start
    return Solution(
        method="polished_greedy",
        assignments=assignments,
        fulfillment=best.fulfillment.copy(),
        runtime_seconds=runtime,
        raw_objective=best_value,
        metadata={
            "algorithm": "greedy-plus-exact-fixed-assignment-recourse",
            "execution_class": "classical-matheuristic",
            "milp_backend": recourse_metadata.get("milp_backend", backend),
            "thread_count": recourse_metadata.get("thread_count", thread_count),
            "raw_initial_objective": raw_value,
            "initial_objective": raw_value,
            "final_objective": best_value,
            "initial_polish_improvement": best_value - raw_value,
            "total_improvement": best_value - raw_value,
            "polish_succeeded": succeeded,
            "polish_seconds": perf_counter() - polish_start,
            "greedy_seconds": raw.runtime_seconds,
            "recourse_metadata": recourse_metadata,
            "optimality_gap": recourse_metadata.get("optimality_gap"),
            "best_bound": recourse_metadata.get("best_bound"),
            "n_variables": recourse_metadata.get("n_variables"),
            "n_constraints": recourse_metadata.get("n_constraints"),
            "fixed_candidate_columns_removed": recourse_metadata.get(
                "fixed_candidate_columns_removed"
            ),
        },
    )


def run_baselines(problem: ProblemData) -> list[Solution]:
    return [solve_default_baseline(problem), solve_greedy_baseline(problem)]
