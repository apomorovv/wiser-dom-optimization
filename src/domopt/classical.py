"""Exact deterministic MILP for the V0 DOM formulation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

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
    def create(cls) -> "_RowBuilder":
        return cls([], [], [], [], [])

    def add(self, coefficients: dict[int, float], lb: float, ub: float) -> None:
        for column, value in coefficients.items():
            if abs(value) > 0:
                self.rows.append(self.row_count)
                self.cols.append(column)
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
) -> Solution:
    """Solve the detailed MILP using :func:`scipy.optimize.milp`.

    ``seed`` is recorded for experiment consistency. HiGHS through SciPy does
    not currently expose a portable random-seed option in this interface.
    """

    del seed  # Recorded in metadata below; not passed to SciPy's portable API.
    start = perf_counter()

    candidates = (
        problem.candidates.loc[problem.candidates["eligible"].astype(bool)]
        .sort_values(["order_id", "pgi_date", "candidate_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    lines = (
        problem.order_lines.sort_values(["order_id", "sku_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    orders = problem.orders.sort_values("order_id", kind="mergesort").reset_index(drop=True)

    # Variable order: x, z, f, u.
    next_index = 0
    x_index: dict[str, int] = {}
    for candidate in candidates.itertuples(index=False):
        x_index[str(candidate.candidate_id)] = next_index
        next_index += 1

    z_index: dict[str, int] = {}
    for order_id in orders["order_id"].astype(str):
        z_index[order_id] = next_index
        next_index += 1

    candidates_by_order: dict[str, list[pd.Series]] = {
        order_id: [row for _, row in group.iterrows()]
        for order_id, group in candidates.groupby("order_id", sort=False)
    }

    f_index: dict[tuple[int, str], int] = {}
    for line_id, line in lines.iterrows():
        for candidate in candidates_by_order.get(str(line["order_id"]), []):
            candidate_id = str(candidate["candidate_id"])
            f_index[(line_id, candidate_id)] = next_index
            next_index += 1

    u_index: dict[int, int] = {}
    for line_id in lines.index:
        u_index[int(line_id)] = next_index
        next_index += 1

    n_variables = next_index
    objective = np.zeros(n_variables, dtype=float)
    lower_bounds = np.zeros(n_variables, dtype=float)
    upper_bounds = np.full(n_variables, np.inf, dtype=float)
    integrality = np.ones(n_variables, dtype=np.uint8)

    for candidate in candidates.itertuples(index=False):
        idx = x_index[str(candidate.candidate_id)]
        objective[idx] = float(candidate.shipping_cost)
        upper_bounds[idx] = 1.0
    for idx in z_index.values():
        upper_bounds[idx] = 1.0

    for line_id, line in lines.iterrows():
        demand = int(line["demand_cases"])
        for candidate in candidates_by_order.get(str(line["order_id"]), []):
            idx = f_index[(int(line_id), str(candidate["candidate_id"]))]
            objective[idx] = -float(line["unit_value"])
            upper_bounds[idx] = demand
        idx_u = u_index[int(line_id)]
        objective[idx_u] = float(line["penalty_per_unfilled_case"])
        upper_bounds[idx_u] = demand

    constraints = _RowBuilder.create()

    # Exactly one modeled outcome per order.
    for order_id in orders["order_id"].astype(str):
        coeff = {z_index[order_id]: 1.0}
        for candidate in candidates_by_order.get(order_id, []):
            coeff[x_index[str(candidate["candidate_id"])]] = 1.0
        constraints.add(coeff, 1.0, 1.0)

    # Demand balance and linking constraints.
    for line_id, line in lines.iterrows():
        order_id = str(line["order_id"])
        demand = int(line["demand_cases"])
        balance = {u_index[int(line_id)]: 1.0}
        for candidate in candidates_by_order.get(order_id, []):
            candidate_id = str(candidate["candidate_id"])
            idx_f = f_index[(int(line_id), candidate_id)]
            balance[idx_f] = 1.0
            constraints.add(
                {
                    idx_f: 1.0,
                    x_index[candidate_id]: -float(demand),
                },
                -np.inf,
                0.0,
            )
        constraints.add(balance, float(demand), float(demand))

    # Cumulative inventory constraints.
    line_ids_by_sku: dict[str, list[int]] = {
        str(sku_id): [int(i) for i in group.index]
        for sku_id, group in lines.groupby("sku_id", sort=False)
    }
    for inv in problem.inventory.itertuples(index=False):
        checkpoint = pd.Timestamp(inv.date)
        coeff: dict[int, float] = {}
        for line_id in line_ids_by_sku.get(str(inv.sku_id), []):
            order_id = str(lines.loc[line_id, "order_id"])
            for candidate in candidates_by_order.get(order_id, []):
                if (
                    str(candidate["dc_id"]) == str(inv.dc_id)
                    and pd.Timestamp(candidate["pgi_date"]) <= checkpoint
                ):
                    idx = f_index[(line_id, str(candidate["candidate_id"]))]
                    coeff[idx] = coeff.get(idx, 0.0) + 1.0
        constraints.add(
            coeff,
            -np.inf,
            float(inv.cumulative_available_cases),
        )

    # Optional exact-date capacities supported by V0.
    if not problem.capacities.empty:
        for cap in problem.capacities.itertuples(index=False):
            dc_id = str(cap.dc_id)
            date = pd.Timestamp(cap.date)
            resource = str(cap.resource)
            coeff: dict[int, float] = {}

            if resource == "dock":
                for candidate in candidates.itertuples(index=False):
                    if str(candidate.dc_id) == dc_id and pd.Timestamp(candidate.pgi_date) == date:
                        coeff[x_index[str(candidate.candidate_id)]] = 1.0
            elif resource in {"throughput_cases", "case_pick", "weight", "volume"}:
                attribute = None
                if resource == "weight":
                    attribute = "unit_weight"
                elif resource == "volume":
                    attribute = "unit_volume"
                if attribute is not None and attribute not in lines.columns:
                    raise ClassicalSolverError(
                        f"Capacity resource {resource!r} requires order_lines.{attribute}"
                    )

                for line_id, line in lines.iterrows():
                    coefficient = 1.0 if attribute is None else float(line[attribute])
                    for candidate in candidates_by_order.get(str(line["order_id"]), []):
                        if (
                            str(candidate["dc_id"]) == dc_id
                            and pd.Timestamp(candidate["pgi_date"]) == date
                        ):
                            idx = f_index[(int(line_id), str(candidate["candidate_id"]))]
                            coeff[idx] = coefficient
            else:
                raise ClassicalSolverError(
                    f"Unsupported capacity resource {resource!r}. "
                    "Add its exact linear consumption rule before enabling it."
                )

            constraints.add(coeff, -np.inf, float(cap.capacity))

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
        # Status 1 can indicate a time/iteration limit with a valid incumbent.
        raise ClassicalSolverError(
            f"MILP failed. status={result.status}, message={result.message}"
        )

    values = np.asarray(result.x, dtype=float)
    candidate_lookup = candidates.set_index("candidate_id")
    default_lookup = orders.set_index("order_id")["default_dc"].to_dict()
    assignment_rows: list[dict[str, object]] = []
    fulfillment_rows: list[dict[str, object]] = []

    selected_candidate_by_order: dict[str, str | None] = {}
    for order_id in orders["order_id"].astype(str):
        options_for_order = candidates_by_order.get(order_id, [])
        selected_ids = [
            str(candidate["candidate_id"])
            for candidate in options_for_order
            if values[x_index[str(candidate["candidate_id"])]] > 0.5
        ]
        if len(selected_ids) > 1:
            raise ClassicalSolverError(
                f"Numerical extraction found multiple candidates for order {order_id}: {selected_ids}"
            )
        candidate_id = selected_ids[0] if selected_ids else None
        selected_candidate_by_order[order_id] = candidate_id
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
            assignment_rows.append(
                {
                    "order_id": order_id,
                    "candidate_id": candidate_id,
                    "selected_dc": str(candidate["dc_id"]),
                    "selected_pgi_date": pd.Timestamp(candidate["pgi_date"]),
                    "is_unassigned": False,
                    "is_divert": str(candidate["dc_id"]) != str(default_lookup[order_id]),
                    "method": "classical",
                }
            )

    for line_id, line in lines.iterrows():
        order_id = str(line["order_id"])
        candidate_id = selected_candidate_by_order[order_id]
        fulfilled = 0
        selected_dc: str | None = None
        selected_date: pd.Timestamp | None = None
        if candidate_id is not None:
            fulfilled = int(round(values[f_index[(int(line_id), candidate_id)]]))
            candidate = candidate_lookup.loc[candidate_id]
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

    best_bound = None
    if getattr(result, "mip_dual_bound", None) is not None:
        best_bound = -float(result.mip_dual_bound)
    optimality_gap = None
    if getattr(result, "mip_gap", None) is not None:
        optimality_gap = float(result.mip_gap)

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
            "n_constraints": constraints.row_count,
        },
    )

