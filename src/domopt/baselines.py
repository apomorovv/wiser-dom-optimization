"""Deterministic default and sequential greedy baselines."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from .penalties import order_penalty
from .resources import (
    candidate_fixed_consumption,
    split_pick_quantities,
    uses_split_pick_accounting,
)
from .rules import candidate_is_divert, minimum_divert_fulfillment
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


class _InventoryState:
    """Residual cumulative inventory with correct earlier-date consumption."""

    def __init__(self, inventory: pd.DataFrame):
        grouped: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
        ordered = inventory.sort_values(
            ["dc_id", "sku_id", "date"], kind="mergesort"
        )
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
            raise ValueError(
                f"Insufficient inventory for dc={dc_id}, sku={sku_id}, date={date}"
            )
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


def _order_lines(problem: ProblemData, order_id: str) -> pd.DataFrame:
    return problem.order_lines.loc[problem.order_lines["order_id"] == order_id].copy()


def _unassigned_score(problem: ProblemData, order_id: str, lines: pd.DataFrame) -> float:
    quantities = lines.copy()
    quantities["unfulfilled_cases"] = quantities["demand_cases"]
    return -order_penalty(problem, order_id, quantities)


def _preview_candidate(
    problem: ProblemData,
    candidate: pd.Series,
    inventory: _InventoryState,
    capacities: _CapacityState,
) -> _CandidatePlan | None:
    order_id = str(candidate["order_id"])
    dc_id = str(candidate["dc_id"])
    pgi_date = pd.Timestamp(candidate["pgi_date"])
    lines = _order_lines(problem, order_id)

    dock_units = candidate_fixed_consumption(candidate, "dock")
    if capacities.remaining(dc_id, pgi_date, "dock") < dock_units - 1e-9:
        return None

    # Allocate cases in descending marginal business value. This is a baseline,
    # not the exact optimizer; deterministic tie breaks keep results reproducible.
    lines["marginal_value"] = (
        lines["unit_value"].astype(float)
        + lines["penalty_per_unfilled_case"].astype(float)
    )
    lines = lines.sort_values(
        ["marginal_value", "sku_id"], ascending=[False, True], kind="mergesort"
    )

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
    split_picks = uses_split_pick_accounting(problem)
    quantities: dict[str, int] = {}

    for row in lines.itertuples(index=False):
        demand = int(row.demand_cases)
        maximum = min(demand, inventory.available(dc_id, str(row.sku_id), pgi_date))

        if np.isfinite(residual_resources["throughput_cases"]):
            maximum = min(maximum, int(np.floor(residual_resources["throughput_cases"])))
        unit_weight = float(getattr(row, "unit_weight", 0.0) or 0.0)
        unit_volume = float(getattr(row, "unit_volume", 0.0) or 0.0)
        if unit_weight > 0 and np.isfinite(residual_resources["weight"]):
            maximum = min(maximum, int(np.floor(residual_resources["weight"] / unit_weight)))
        if unit_volume > 0 and np.isfinite(residual_resources["volume"]):
            maximum = min(maximum, int(np.floor(residual_resources["volume"] / unit_volume)))

        if split_picks:
            per_pallet_value = getattr(row, "cases_per_pallet", None)
            if per_pallet_value is None or pd.isna(per_pallet_value):
                raise ValueError(
                    "Pallet/case-pick accounting requires cases_per_pallet"
                )
            per_pallet = int(per_pallet_value)
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

        quantities[str(row.sku_id)] = quantity
        residual_resources["throughput_cases"] -= quantity
        if unit_weight > 0:
            residual_resources["weight"] -= quantity * unit_weight
        if unit_volume > 0:
            residual_resources["volume"] -= quantity * unit_volume

    merged = _order_lines(problem, order_id).copy()
    merged["fulfilled"] = merged["sku_id"].map(quantities).fillna(0).astype(int)
    merged["unfulfilled"] = merged["demand_cases"] - merged["fulfilled"]
    penalty_quantities = merged.rename(columns={"unfulfilled": "unfulfilled_cases"})
    score = float(
        (merged["unit_value"] * merged["fulfilled"]).sum()
        - order_penalty(problem, order_id, penalty_quantities)
        - float(candidate["shipping_cost"])
    )
    unassigned = _unassigned_score(problem, order_id, merged)
    if candidate_is_divert(problem, candidate):
        threshold = minimum_divert_fulfillment(problem, order_id)
        if threshold is not None and int(merged["fulfilled"].sum()) < threshold:
            return None
    return _CandidatePlan(
        candidate_id=str(candidate["candidate_id"]),
        order_id=order_id,
        dc_id=dc_id,
        pgi_date=pgi_date,
        is_default=bool(candidate["is_default"]),
        shipping_cost=float(candidate["shipping_cost"]),
        dock_units=dock_units,
        quantities=quantities,
        score=score,
        incremental_score=score - unassigned,
    )


def _commit_plan(
    problem: ProblemData,
    plan: _CandidatePlan,
    inventory: _InventoryState,
    capacities: _CapacityState,
) -> None:
    lines = _order_lines(problem, plan.order_id).set_index("sku_id")
    total_cases = 0
    total_case_picks = 0
    total_pallet_picks = 0
    total_weight = 0.0
    total_volume = 0.0
    split_picks = uses_split_pick_accounting(problem)
    for sku_id, quantity in plan.quantities.items():
        inventory.consume(plan.dc_id, sku_id, plan.pgi_date, int(quantity))
        total_cases += int(quantity)
        if split_picks:
            per_pallet = int(lines.loc[sku_id, "cases_per_pallet"])
            pallets, loose = split_pick_quantities(quantity, per_pallet)
            total_pallet_picks += pallets
            total_case_picks += loose
        else:
            total_case_picks += int(quantity)
        if "unit_weight" in lines.columns and not pd.isna(lines.loc[sku_id, "unit_weight"]):
            total_weight += int(quantity) * float(lines.loc[sku_id, "unit_weight"])
        if "unit_volume" in lines.columns and not pd.isna(lines.loc[sku_id, "unit_volume"]):
            total_volume += int(quantity) * float(lines.loc[sku_id, "unit_volume"])

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
            for group_id, group in problem.orders.groupby(
                "assignment_group", sort=False
            )
        }
    return {
        str(order_id): (str(order_id),)
        for order_id in problem.orders["order_id"].astype(str)
    }


def _candidate_options_for_unit(
    problem: ProblemData,
    eligible: pd.DataFrame,
    members: tuple[str, ...],
    *,
    default_only: bool = False,
) -> list[tuple[str, pd.DataFrame]]:
    rows = eligible.loc[eligible["order_id"].astype(str).isin(members)].copy()
    if default_only:
        rows = rows.loc[rows["is_default"].astype(bool)]
    grouped = bool(problem.metadata.get("enforce_assignment_group", False))
    option_column = "group_option_id" if grouped else "candidate_id"
    if option_column not in rows.columns:
        raise ValueError(f"Candidate table requires {option_column!r}")

    options: list[tuple[str, pd.DataFrame]] = []
    expected = set(members)
    for option_id, option_rows in rows.groupby(option_column, sort=False):
        if set(option_rows["order_id"].astype(str)) != expected:
            continue
        options.append(
            (
                str(option_id),
                option_rows.sort_values("order_id", kind="mergesort"),
            )
        )
    options.sort(key=lambda item: item[0])
    return options


def _preview_decision(
    problem: ProblemData,
    unit_id: str,
    option_id: str,
    option_rows: pd.DataFrame,
    inventory: _InventoryState,
    capacities: _CapacityState,
) -> _DecisionPlan | None:
    """Preview all members against private residual states, then commit atomically."""

    trial_inventory = inventory.clone()
    trial_capacities = capacities.clone()
    member_plans: list[_CandidatePlan] = []
    unassigned_score = 0.0

    for candidate in option_rows.itertuples(index=False):
        candidate_row = pd.Series(candidate._asdict())
        order_id = str(candidate_row["order_id"])
        plan = _preview_candidate(
            problem,
            candidate_row,
            trial_inventory,
            trial_capacities,
        )
        if plan is None:
            return None
        _commit_plan(problem, plan, trial_inventory, trial_capacities)
        member_plans.append(plan)
        lines = _order_lines(problem, order_id)
        unassigned_score += _unassigned_score(problem, order_id, lines)

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
) -> list[_DecisionPlan]:
    decisions: list[_DecisionPlan] = []
    for option_id, option_rows in _candidate_options_for_unit(
        problem,
        eligible,
        members,
        default_only=default_only,
    ):
        decision = _preview_decision(
            problem,
            unit_id,
            option_id,
            option_rows,
            inventory,
            capacities,
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
) -> None:
    for plan in decision.member_plans:
        _commit_plan(problem, plan, inventory, capacities)
        selected[plan.order_id] = plan


def _solution_from_plans(
    problem: ProblemData,
    method: str,
    selected: dict[str, _CandidatePlan | None],
    runtime_seconds: float,
) -> Solution:
    default_dc = problem.orders.set_index("order_id")["default_dc"].to_dict()
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

        for line in _order_lines(problem, order_id).sort_values("sku_id").itertuples(index=False):
            fulfilled = 0 if plan is None else int(plan.quantities.get(str(line.sku_id), 0))
            fulfillment_rows.append(
                {
                    "order_id": order_id,
                    "sku_id": str(line.sku_id),
                    "fulfilled_cases": fulfilled,
                    "unfulfilled_cases": int(line.demand_cases) - fulfilled,
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

    eligible = problem.candidates[problem.candidates["eligible"]].copy()
    units = _assignment_units(problem)
    for unit_id, members in sorted(units.items()):
        decisions = _feasible_decisions(
            problem,
            unit_id,
            members,
            eligible,
            inventory,
            capacities,
            default_only=True,
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
        )

    return _solution_from_plans(
        problem, "default", selected, runtime_seconds=perf_counter() - start
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
    units = _assignment_units(problem)

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
        )

    return _solution_from_plans(
        problem, "greedy", selected, runtime_seconds=perf_counter() - start
    )


def run_baselines(problem: ProblemData) -> list[Solution]:
    return [solve_default_baseline(problem), solve_greedy_baseline(problem)]
