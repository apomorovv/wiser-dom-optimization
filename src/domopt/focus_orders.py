"""Transparent focus-order identification helpers."""

from __future__ import annotations

import pandas as pd


class FocusOrderError(ValueError):
    pass


def identify_focus_orders(
    orders: pd.DataFrame,
    order_lines: pd.DataFrame,
    default_fulfillment: pd.DataFrame,
) -> pd.DataFrame:
    """Identify orders whose default plan cannot fully satisfy demand.

    ``default_fulfillment`` must contain one row per order–SKU pair with a
    ``default_fulfilled_cases`` column. More detailed business filters should be
    added explicitly rather than embedded as undocumented conditions.
    """

    required_lines = {"order_id", "sku_id", "demand_cases"}
    required_fill = {"order_id", "sku_id", "default_fulfilled_cases"}
    if missing := required_lines - set(order_lines.columns):
        raise FocusOrderError(f"order_lines is missing {sorted(missing)}")
    if missing := required_fill - set(default_fulfillment.columns):
        raise FocusOrderError(f"default_fulfillment is missing {sorted(missing)}")

    merged = order_lines.merge(
        default_fulfillment,
        on=["order_id", "sku_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["default_fulfilled_cases"].isna().any():
        raise FocusOrderError("Default fulfillment is missing for at least one order line")

    merged["line_shortfall"] = (
        merged["demand_cases"].astype(int)
        - merged["default_fulfilled_cases"].astype(int)
    ).clip(lower=0)
    summary = (
        merged.groupby("order_id", as_index=False)
        .agg(
            requested_cases=("demand_cases", "sum"),
            default_fulfilled_cases=("default_fulfilled_cases", "sum"),
            shortfall_cases=("line_shortfall", "sum"),
        )
    )
    summary["default_fill_rate"] = (
        summary["default_fulfilled_cases"] / summary["requested_cases"].where(
            summary["requested_cases"] > 0, 1
        )
    )
    summary["focus_reason"] = "inventory_shortage"
    focus = summary.loc[summary["shortfall_cases"] > 0].copy()
    return orders.merge(focus, on="order_id", how="inner", validate="one_to_one")



