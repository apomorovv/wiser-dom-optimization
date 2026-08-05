"""Exact deterministic MILP and fixed-assignment recourse solver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .resources import candidate_fixed_consumption, uses_split_pick_accounting
from .rules import candidate_is_divert, minimum_divert_fulfillment
from .schemas import ProblemData, Solution


class ClassicalSolverError(RuntimeError):
    """Raised when the MILP cannot produce a usable solution."""


@dataclass
class _RowBuilder:
    rows: list[int]
    cols: list[int]
    data: list[float]
    lower: list[float]
    upper: list[float]
    row_count: int = 0

    @classmethod
    def create(cls) -> _RowBuilder:
        return cls([], [], [], [], [])

    def add(self, coefficients: Mapping[int, float], lb: float, ub: float) -> None:
        for column, value in coefficients.items():
            if abs(value) > 0:
                self.rows.append(self.row_count)
                self.cols.append(int(column))
                self.data.append(float(value))
        self.lower.append(float(lb))
        self.upper.append(float(ub))
        self.row_count += 1


def solve_classical(
    problem: ProblemData,
    *,
    time_limit_seconds: float = 60.0,
    mip_relative_gap: float = 0.0,
    seed: int | None = None,
    fixed_assignments: Mapping[str, str | None] | None = None,
) -> Solution:
    """Solve the detailed DOM MILP with HiGHS through SciPy.

    fixed_assignments turns the model into an exact fulfillment-recourse
    problem. A value is either a candidate ID or None for no assignment.
    The hybrid optimizer uses this bounded recourse mode after sampling a local
    assignment QUBO.

    seed is recorded for experiment consistency. SciPy's portable HiGHS
    interface does not currently expose a random-seed option.
    """

    recorded_seed = seed
    start = perf_counter()
    fixed = {str(key): value for key, value in (fixed_assignments or {}).items()}

    candidates = (
        problem.candidates.loc[problem.candidates["eligible"].astype(bool)]
        .sort_values(["order_id", "pgi_date", "candidate_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    lines = (
        problem.order_lines.sort_values(["order_id", "sku_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    orders = problem.orders.sort_values(
        "order_id", kind="mergesort"
    ).reset_index(drop=True)
    order_ids = set(orders["order_id"].astype(str))
    unknown_fixed = sorted(set(fixed) - order_ids)
    if unknown_fixed:
        raise ClassicalSolverError(
            f"Fixed assignments reference unknown orders: {unknown_fixed}"
        )

    candidates_by_order: dict[str, list[pd.Series]] = {
        str(order_id): [row for _, row in group.iterrows()]
        for order_id, group in candidates.groupby("order_id", sort=False)
    }

    resources = set(
        problem.capacities.get("resource", pd.Series(dtype=str)).astype(str)
    )
    split_picks = uses_split_pick_accounting(problem) and bool(
        resources & {"case_pick", "pallet_pick"}
    )
    if split_picks:
        if (
            "cases_per_pallet" not in lines.columns
            or lines["cases_per_pallet"].isna().any()
        ):
            raise ClassicalSolverError(
                "Pallet/case-pick accounting requires cases_per_pallet for every order line"
            )
        if (lines["cases_per_pallet"].astype(int) <= 0).any():
            raise ClassicalSolverError("cases_per_pallet must be positive")
    if "pallet_pick" in resources and not split_picks:
        raise ClassicalSolverError(
            "pallet_pick capacity requires metadata.pick_capacity_mode='pallet_case'"
        )

    # Variable order: assignment x, no-assignment z, fulfillment f, unmet u,
    # and optional full-pallet p / loose-case k variables.
    next_index = 0
    x_index: dict[str, int] = {}
    for candidate in candidates.itertuples(index=False):
        x_index[str(candidate.candidate_id)] = next_index
        next_index += 1

    z_index: dict[str, int] = {}
    for order_id in orders["order_id"].astype(str):
        z_index[order_id] = next_index
        next_index += 1

    f_index: dict[tuple[int, str], int] = {}
    for line_id, line in lines.iterrows():
        for candidate in candidates_by_order.get(str(line["order_id"]), []):
            candidate_id = str(candidate["candidate_id"])
            f_index[(int(line_id), candidate_id)] = next_index
            next_index += 1

    u_index: dict[int, int] = {}
    for line_id in lines.index:
        u_index[int(line_id)] = next_index
        next_index += 1

    p_index: dict[tuple[int, str], int] = {}
    k_index: dict[tuple[int, str], int] = {}
    if split_picks:
        for key in f_index:
            p_index[key] = next_index
            next_index += 1
            k_index[key] = next_index
            next_index += 1

    n_variables = next_index
    objective = np.zeros(n_variables, dtype=float)
    lower_bounds = np.zeros(n_variables, dtype=float)
    upper_bounds = np.full(n_variables, np.inf, dtype=float)
    integrality = np.ones(n_variables, dtype=np.uint8)

    for candidate in candidates.itertuples(index=False):
        index = x_index[str(candidate.candidate_id)]
        objective[index] = float(candidate.shipping_cost)
        upper_bounds[index] = 1.0
    for index in z_index.values():
        upper_bounds[index] = 1.0

    for line_id, line in lines.iterrows():
        demand = int(line["demand_cases"])
        for candidate in candidates_by_order.get(str(line["order_id"]), []):
            candidate_id = str(candidate["candidate_id"])
            key = (int(line_id), candidate_id)
            objective[f_index[key]] = -float(line["unit_value"])
            upper_bounds[f_index[key]] = demand
            if split_picks:
                per_pallet = int(line["cases_per_pallet"])
                upper_bounds[p_index[key]] = demand // per_pallet
                upper_bounds[k_index[key]] = per_pallet - 1
        objective[u_index[int(line_id)]] = float(line["penalty_per_unfilled_case"])
        upper_bounds[u_index[int(line_id)]] = demand

    constraints = _RowBuilder.create()

    # Exactly one modeled outcome per order.
    for order_id in orders["order_id"].astype(str):
        coefficients = {z_index[order_id]: 1.0}
        for candidate in candidates_by_order.get(order_id, []):
            coefficients[x_index[str(candidate["candidate_id"])]] = 1.0
        constraints.add(coefficients, 1.0, 1.0)

    # Fix assignment decisions for classical recourse when requested.
    candidate_lookup = candidates.set_index("candidate_id", drop=False)
    for order_id, candidate_id in fixed.items():
        if candidate_id is None:
            constraints.add({z_index[order_id]: 1.0}, 1.0, 1.0)
            continue
        candidate_key = str(candidate_id)
        if candidate_key not in x_index:
            raise ClassicalSolverError(
                f"Fixed candidate {candidate_key!r} is not eligible in this problem"
            )
        candidate = candidate_lookup.loc[candidate_key]
        if isinstance(candidate, pd.DataFrame):
            candidate = candidate.iloc[0]
        if str(candidate["order_id"]) != order_id:
            raise ClassicalSolverError(
                f"Fixed candidate {candidate_key!r} does not belong to order {order_id!r}"
            )
        constraints.add({x_index[candidate_key]: 1.0}, 1.0, 1.0)

    # Demand balance, assignment linking, and optional exact pick decomposition.
    for line_id, line in lines.iterrows():
        order_id = str(line["order_id"])
        demand = int(line["demand_cases"])
        balance = {u_index[int(line_id)]: 1.0}
        for candidate in candidates_by_order.get(order_id, []):
            candidate_id = str(candidate["candidate_id"])
            key = (int(line_id), candidate_id)
            index_f = f_index[key]
            balance[index_f] = 1.0
            constraints.add(
                {index_f: 1.0, x_index[candidate_id]: -float(demand)},
                -np.inf,
                0.0,
            )
            if split_picks:
                per_pallet = int(line["cases_per_pallet"])
                constraints.add(
                    {
                        index_f: 1.0,
                        p_index[key]: -float(per_pallet),
                        k_index[key]: -1.0,
                    },
                    0.0,
                    0.0,
                )
                constraints.add(
                    {
                        k_index[key]: 1.0,
                        x_index[candidate_id]: -float(per_pallet - 1),
                    },
                    -np.inf,
                    0.0,
                )
        constraints.add(balance, float(demand), float(demand))

    # Projected-ATP protection. A fulfillment at tau consumes every checkpoint t >= tau.
    line_ids_by_sku: dict[str, list[int]] = {
        str(sku_id): [int(index) for index in group.index]
        for sku_id, group in lines.groupby("sku_id", sort=False)
    }
    for inventory in problem.inventory.itertuples(index=False):
        checkpoint = pd.Timestamp(inventory.date)
        coefficients: dict[int, float] = {}
        for line_id in line_ids_by_sku.get(str(inventory.sku_id), []):
            order_id = str(lines.loc[line_id, "order_id"])
            for candidate in candidates_by_order.get(order_id, []):
                if (
                    str(candidate["dc_id"]) == str(inventory.dc_id)
                    and pd.Timestamp(candidate["pgi_date"]) <= checkpoint
                ):
                    index = f_index[(line_id, str(candidate["candidate_id"]))]
                    coefficients[index] = coefficients.get(index, 0.0) + 1.0
        constraints.add(
            coefficients,
            -np.inf,
            float(inventory.cumulative_available_cases),
        )

    # Optional exact-date resources.
    if not problem.capacities.empty:
        for capacity in problem.capacities.itertuples(index=False):
            dc_id = str(capacity.dc_id)
            date = pd.Timestamp(capacity.date)
            resource = str(capacity.resource)
            coefficients: dict[int, float] = {}

            if resource == "dock":
                for candidate in candidates.itertuples(index=False):
                    if (
                        str(candidate.dc_id) == dc_id
                        and pd.Timestamp(candidate.pgi_date) == date
                    ):
                        candidate_series = candidate_lookup.loc[
                            str(candidate.candidate_id)
                        ]
                        if isinstance(candidate_series, pd.DataFrame):
                            candidate_series = candidate_series.iloc[0]
                        coefficients[x_index[str(candidate.candidate_id)]] = (
                            candidate_fixed_consumption(candidate_series, "dock")
                        )
            elif resource in {
                "throughput_cases",
                "case_pick",
                "pallet_pick",
                "weight",
                "volume",
            }:
                attribute = {
                    "weight": "unit_weight",
                    "volume": "unit_volume",
                }.get(resource)
                if attribute is not None and attribute not in lines.columns:
                    raise ClassicalSolverError(
                        f"Capacity resource {resource!r} requires order_lines.{attribute}"
                    )

                for line_id, line in lines.iterrows():
                    for candidate in candidates_by_order.get(str(line["order_id"]), []):
                        if (
                            str(candidate["dc_id"]) != dc_id
                            or pd.Timestamp(candidate["pgi_date"]) != date
                        ):
                            continue
                        key = (int(line_id), str(candidate["candidate_id"]))
                        if resource == "pallet_pick":
                            index = p_index[key]
                            coefficient = 1.0
                        elif resource == "case_pick" and split_picks:
                            index = k_index[key]
                            coefficient = 1.0
                        else:
                            index = f_index[key]
                            coefficient = (
                                1.0 if attribute is None else float(line[attribute])
                            )
                        coefficients[index] = coefficients.get(index, 0.0) + coefficient
            else:
                raise ClassicalSolverError(
                    f"Unsupported capacity resource {resource!r}. "
                    "Add its exact linear consumption rule before enabling it."
                )

            constraints.add(coefficients, -np.inf, float(capacity.capacity))

    # Nestle's minimum alternate-fill improvement rule.
    for _, candidate in candidates.iterrows():
        if not candidate_is_divert(problem, candidate):
            continue
        threshold = minimum_divert_fulfillment(problem, str(candidate["order_id"]))
        if threshold is None:
            continue
        candidate_id = str(candidate["candidate_id"])
        coefficients = {x_index[candidate_id]: -float(threshold)}
        for line_id, line in lines.iterrows():
            if str(line["order_id"]) == str(candidate["order_id"]):
                coefficients[f_index[(int(line_id), candidate_id)]] = 1.0
        constraints.add(coefficients, 0.0, np.inf)

    matrix = coo_matrix(
        (constraints.data, (constraints.rows, constraints.cols)),
        shape=(constraints.row_count, n_variables),
    ).tocsr()
    options: dict[str, Any] = {
        "time_limit": float(time_limit_seconds),
        "mip_rel_gap": float(mip_relative_gap),
        "presolve": True,
    }
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            matrix,
            np.asarray(constraints.lower, dtype=float),
            np.asarray(constraints.upper, dtype=float),
        ),
        options=options,
    )

    if result.x is None:
        raise ClassicalSolverError(
            f"MILP returned no solution. status={result.status}, message={result.message}"
        )
    if not bool(result.success) and int(result.status) not in {1}:
        raise ClassicalSolverError(
            f"MILP failed. status={result.status}, message={result.message}"
        )

    values = np.asarray(result.x, dtype=float)
    default_lookup = orders.set_index("order_id")["default_dc"].astype(str).to_dict()
    assignment_rows: list[dict[str, object]] = []
    fulfillment_rows: list[dict[str, object]] = []
    selected_by_order: dict[str, str | None] = {}

    for order_id in orders["order_id"].astype(str):
        selected = [
            str(candidate["candidate_id"])
            for candidate in candidates_by_order.get(order_id, [])
            if values[x_index[str(candidate["candidate_id"])]] > 0.5
        ]
        if len(selected) > 1:
            raise ClassicalSolverError(
                f"Numerical extraction found multiple candidates for order {order_id}: {selected}"
            )
        candidate_id = selected[0] if selected else None
        selected_by_order[order_id] = candidate_id
        if candidate_id is None:
            assignment_rows.append(
                {
                    "order_id": order_id,
                    "candidate_id": None,
                    "selected_dc": None,
                    "selected_pgi_date": None,
                    "is_unassigned": True,
                    "is_divert": False,
                    "method": "classical",
                }
            )
        else:
            candidate = candidate_lookup.loc[candidate_id]
            if isinstance(candidate, pd.DataFrame):
                candidate = candidate.iloc[0]
            dc_id = str(candidate["dc_id"])
            assignment_rows.append(
                {
                    "order_id": order_id,
                    "candidate_id": candidate_id,
                    "selected_dc": dc_id,
                    "selected_pgi_date": pd.Timestamp(candidate["pgi_date"]),
                    "is_unassigned": False,
                    "is_divert": dc_id != default_lookup[order_id],
                    "method": "classical",
                }
            )

    for line_id, line in lines.iterrows():
        order_id = str(line["order_id"])
        candidate_id = selected_by_order[order_id]
        fulfilled = 0
        selected_dc: str | None = None
        selected_date: pd.Timestamp | None = None
        if candidate_id is not None:
            fulfilled = round(values[f_index[(int(line_id), candidate_id)]])
            candidate = candidate_lookup.loc[candidate_id]
            if isinstance(candidate, pd.DataFrame):
                candidate = candidate.iloc[0]
            selected_dc = str(candidate["dc_id"])
            selected_date = pd.Timestamp(candidate["pgi_date"])
        fulfillment_rows.append(
            {
                "order_id": order_id,
                "sku_id": str(line["sku_id"]),
                "fulfilled_cases": fulfilled,
                "unfulfilled_cases": int(line["demand_cases"]) - fulfilled,
                "selected_dc": selected_dc,
                "selected_pgi_date": selected_date,
            }
        )

    best_bound = (
        None
        if getattr(result, "mip_dual_bound", None) is None
        else -float(result.mip_dual_bound)
    )
    optimality_gap = (
        None if getattr(result, "mip_gap", None) is None else float(result.mip_gap)
    )
    return Solution(
        method="classical",
        assignments=pd.DataFrame(assignment_rows),
        fulfillment=pd.DataFrame(fulfillment_rows),
        runtime_seconds=perf_counter() - start,
        raw_objective=-float(result.fun),
        metadata={
            "solver": "scipy.optimize.milp/HiGHS",
            "status": int(result.status),
            "message": str(result.message),
            "best_bound": best_bound,
            "optimality_gap": optimality_gap,
            "mip_node_count": getattr(result, "mip_node_count", None),
            "n_variables": n_variables,
            "n_binary_assignment_variables": len(x_index) + len(z_index),
            "n_fulfillment_variables": len(f_index),
            "n_pick_variables": len(p_index) + len(k_index),
            "n_constraints": constraints.row_count,
            "fixed_assignment_count": len(fixed),
            "seed": recorded_seed,
        },
    )
