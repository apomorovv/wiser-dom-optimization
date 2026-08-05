"""Business rules shared by solvers and the independent validator."""

from __future__ import annotations

from math import ceil

import pandas as pd

from .schemas import ProblemData


def minimum_divert_fulfillment(problem: ProblemData, order_id: str) -> int | None:
    """Return the required cases for a non-default assignment.

    The Nestle clarification defines the improvement as percentage points of total
    ordered cases: alternate fill minus default fill must be at least ``delta * Q``.
    The rule is enabled with ``metadata.enforce_min_divert_improvement`` and needs
    ``orders.default_fillable_cases``. An order-level fraction overrides the
    dataset-level default.
    """

    if not bool(problem.metadata.get("enforce_min_divert_improvement", False)):
        return None

    rows = problem.orders.loc[problem.orders["order_id"].astype(str) == str(order_id)]
    if len(rows) != 1:
        raise ValueError(f"Expected one order row for {order_id!r}")
    order = rows.iloc[0]
    if "default_fillable_cases" not in rows.columns or pd.isna(
        order.get("default_fillable_cases")
    ):
        raise ValueError(
            "The minimum-divert rule requires orders.default_fillable_cases"
        )

    if "min_divert_improvement_fraction" in rows.columns and not pd.isna(
        order.get("min_divert_improvement_fraction")
    ):
        fraction = float(order["min_divert_improvement_fraction"])
    else:
        fraction = float(problem.metadata.get("min_divert_improvement_fraction", 0.05))

    demand = int(
        problem.order_lines.loc[
            problem.order_lines["order_id"].astype(str) == str(order_id), "demand_cases"
        ].sum()
    )
    default_fill = int(order["default_fillable_cases"])
    return min(demand, default_fill + ceil(fraction * demand - 1e-12))


def candidate_is_divert(problem: ProblemData, candidate: pd.Series) -> bool:
    defaults = problem.orders.set_index("order_id")["default_dc"].astype(str).to_dict()
    return str(candidate["dc_id"]) != defaults[str(candidate["order_id"])]
