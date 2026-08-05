"""Deterministic default and sequential greedy baselines."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

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


class _InventoryState:
    """Residual cumulative inventory with correct earlier-date consumption."""

    def __init__(self, inventory: pd.DataFrame):
        self._rows: dict[tuple[str, str], list[list[object]]] = {}
        for (dc_id, sku_id), group in inventory.groupby(["dc_id", "sku_id"], sort=False):
            self._rows[(str(dc_id), str(sku_id))] = [
                [pd.Timestamp(row.date), int(row.cumulative_available_cases)]
                for row in group.sort_values("date").itertuples(index=False)
            ]

    def available(self, dc_id: str, sku_id: str, date: pd.Timestamp) -> int:
        checkpoints = self._rows.get((dc_id, sku_id), [])
        covering = [int(amount) for checkpoint, amount in checkpoints if checkpoint >= date]
        return max(0, min(covering)) if covering else 0

    def consume(self, dc_id: str, sku_id: str, date: pd.Timestamp, quantity: int) -> None:
        if quantity < 0:
            raise ValueError("Cannot consume a negative inventory quantity")
        if quantity > self.available(dc_id, sku_id, date):
            raise ValueError(
                f"Insufficient inventory for dc={dc_id}, sku={sku_id}, date={date}"
            )
        for checkpoint in self._rows.get((dc_id, sku_id), []):
            if checkpoint[0] >= date:
                checkpoint[1] = int(checkpoint[1]) - int(quantity)


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


def _order_lines(problem: ProblemData, order_id: str) -> pd.DataFrame:
    return problem.order_lines.loc[problem.order_lines["order_id"] == order_id].copy()


def _unassigned_score(lines: pd.DataFrame) -> float:
    return -float((lines["penalty_per_unfilled_case"] * lines["demand_cases"]).sum())


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
    score = float(
        (merged["unit_value"] * merged["fulfilled"]).sum()
        - (merged["penalty_per_unfilled_case"] * merged["unfulfilled"]).sum()
        - float(candidate["shipping_cost"])
    )
    unassigned = _unassigned_score(merged)
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
    for order_id in sorted(problem.orders["order_id"].astype(str)):
        options = eligible[
            (eligible["order_id"] == order_id) & eligible["is_default"].astype(bool)
        ].sort_values(["pgi_date", "candidate_id"], kind="mergesort")
        if options.empty:
            selected[order_id] = None
            continue
        plan = _preview_candidate(problem, options.iloc[0], inventory, capacities)
        if plan is None:
            selected[order_id] = None
        else:
            _commit_plan(problem, plan, inventory, capacities)
            selected[order_id] = plan

    return _solution_from_plans(
        problem, "default", selected, runtime_seconds=perf_counter() - start
    )


def solve_greedy_baseline(problem: ProblemData) -> Solution:
    """Sequential scarcity-aware greedy reassignment baseline.

    At each round, select the remaining order/candidate pair with the largest
    improvement over leaving that order unassigned. Residual inventory and
    capacities are updated immediately.
    """

    start = perf_counter()
    inventory = _InventoryState(problem.inventory)
    capacities = _CapacityState(problem.capacities)
    remaining = set(problem.orders["order_id"].astype(str))
    selected: dict[str, _CandidatePlan | None] = {}
    eligible = problem.candidates[problem.candidates["eligible"]].copy()

    while remaining:
        plans: list[_CandidatePlan] = []
        for order_id in sorted(remaining):
            options = eligible[eligible["order_id"] == order_id]
            for _, candidate in options.iterrows():
                plan = _preview_candidate(problem, candidate, inventory, capacities)
                if plan is not None:
                    plans.append(plan)

        positive = [plan for plan in plans if plan.incremental_score > 1e-9]
        if not positive:
            # No remaining assignment improves over the explicit no-assignment option.
            for order_id in sorted(remaining):
                selected[order_id] = None
            break

        # Stable deterministic order: incremental objective, filled cases, lower
        # shipping cost, default before alternate, candidate ID.
        positive.sort(
            key=lambda plan: (
                -plan.incremental_score,
                -sum(plan.quantities.values()),
                plan.shipping_cost,
                not plan.is_default,
                plan.candidate_id,
            )
        )
        chosen = positive[0]
        _commit_plan(problem, chosen, inventory, capacities)
        selected[chosen.order_id] = chosen
        remaining.remove(chosen.order_id)

    return _solution_from_plans(
        problem, "greedy", selected, runtime_seconds=perf_counter() - start
    )


def run_baselines(problem: ProblemData) -> list[Solution]:
    return [solve_default_baseline(problem), solve_greedy_baseline(problem)]
