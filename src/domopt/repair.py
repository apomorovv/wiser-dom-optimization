"""Deterministic repair helpers for candidate-plan bitstrings."""

from __future__ import annotations

import pandas as pd


class RepairError(ValueError):
    pass


def repair_one_hot(
    sample: dict[str, int] | pd.Series,
    plans: pd.DataFrame,
) -> dict[str, int]:
    """Repair a sample so exactly one plan is selected per order.

    When an order has zero selected plans, choose the highest-value plan. When it
    has multiple selected plans, keep the selected plan with highest value.
    Ties are broken by ``plan_id``.
    """

    required = {"plan_id", "order_id", "value"}
    if missing := required - set(plans.columns):
        raise RepairError(f"plans is missing {sorted(missing)}")

    repaired = {str(plan_id): int(sample.get(str(plan_id), 0)) for plan_id in plans["plan_id"]}
    for _, group in plans.groupby("order_id", sort=False):
        ordered = group.sort_values(["value", "plan_id"], ascending=[False, True], kind="mergesort")
        selected = [
            str(row.plan_id)
            for row in ordered.itertuples(index=False)
            if repaired[str(row.plan_id)] == 1
        ]
        keep = selected[0] if selected else str(ordered.iloc[0]["plan_id"])
        for plan_id in ordered["plan_id"].astype(str):
            repaired[plan_id] = int(plan_id == keep)
    return repaired


def selected_plans(sample: dict[str, int] | pd.Series) -> list[str]:
    return sorted(str(name) for name, value in sample.items() if int(value) == 1)


