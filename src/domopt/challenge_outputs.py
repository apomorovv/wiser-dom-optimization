"""Privacy-preserving audit of the provided Nestle POC output tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ChallengeOutputError(ValueError):
    pass


ORDER_REQUIRED = {
    "SalesDocument/GroupingIndicator",
    "LoadNumber",
    "IsDivert",
    "DefaultDC",
    "RecommendedDC",
    "OrderedQty_Cases",
    "Default_Qty_Cases",
    "Divert_Qty_Cases",
}
SKU_REQUIRED = {
    "SalesDocument/GroupingIndicator",
    "MaterialNumber",
    "IsDivert",
    "DefaultDC",
    "RecommendedDC",
    "OrderedQty_Cases",
    "Default_Qty_Cases",
    "Divert_Qty_Cases",
}


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ChallengeOutputError(f"{name} is missing columns: {missing}")


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ChallengeOutputError(f"{column!r} contains nonnumeric values")
        frame[column] = values.astype(float)


def summarize_challenge_outputs(
    order_level_path: str | Path,
    sku_level_path: str | Path,
    *,
    include_commercial_metrics: bool = False,
) -> dict[str, Any]:
    """Validate the two output tables and return aggregate-only diagnostics.

    The function never returns order, customer, load, SKU, or DC identifiers.
    Commercial totals are excluded unless explicitly requested for an authorized,
    private analysis.
    """

    orders = pd.read_csv(
        order_level_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    skus = pd.read_csv(
        sku_level_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    _require(orders, ORDER_REQUIRED, "order-level output")
    _require(skus, SKU_REQUIRED, "SKU-level output")

    order_key = "SalesDocument/GroupingIndicator"
    if orders[order_key].duplicated().any():
        raise ChallengeOutputError("order-level output must contain one row per order")
    if skus.duplicated([order_key, "MaterialNumber"]).any():
        raise ChallengeOutputError("SKU-level output contains duplicate order-SKU rows")
    if set(skus[order_key]) != set(orders[order_key]):
        raise ChallengeOutputError("Order keys do not reconcile across output tables")

    numeric_columns = [
        "OrderedQty_Cases",
        "Default_Qty_Cases",
        "Divert_Qty_Cases",
    ]
    _numeric(orders, numeric_columns)
    _numeric(skus, numeric_columns)

    sku_rollup = skus.groupby(order_key, as_index=False)[numeric_columns].sum()
    reconciliation = orders[[order_key, *numeric_columns]].merge(
        sku_rollup,
        on=order_key,
        suffixes=("_order", "_sku"),
        validate="one_to_one",
    )
    mismatch_count = 0
    for column in numeric_columns:
        mismatch_count += int(
            (
                np.abs(
                    reconciliation[f"{column}_order"]
                    - reconciliation[f"{column}_sku"]
                )
                > 1e-6
            ).sum()
        )

    diverted = orders["IsDivert"].eq("Non-Default")
    fulfilled = np.where(
        diverted,
        orders["Divert_Qty_Cases"],
        orders["Default_Qty_Cases"],
    ).astype(float)
    ordered = orders["OrderedQty_Cases"].astype(float).to_numpy()
    total_ordered = float(ordered.sum())
    total_fulfilled = float(fulfilled.sum())
    result: dict[str, Any] = {
        "order_rows": len(orders),
        "sku_rows": len(skus),
        "unique_orders": int(orders[order_key].nunique()),
        "unique_loads": int(orders["LoadNumber"].nunique()),
        "distribution_center_count": len(set(orders["DefaultDC"]) | set(orders["RecommendedDC"])),
        "diverted_orders": int(diverted.sum()),
        "default_orders": int((~diverted).sum()),
        "requested_cases": total_ordered,
        "fulfilled_cases": total_fulfilled,
        "case_fill_rate": (
            total_fulfilled / total_ordered if total_ordered > 0 else 1.0
        ),
        "order_sku_reconciliation_mismatches": mismatch_count,
        "contains_raw_identifiers": False,
        "commercial_metrics_included": bool(include_commercial_metrics),
    }

    if include_commercial_metrics:
        optional = [
            "DefaultDCShippingCost",
            "DivertedDCShippingCost",
            "PenaltyIfNotDiverted",
            "PenaltyIfDiverted",
            "DefaultRevenue",
            "DivertRevenue",
        ]
        _numeric(orders, [column for column in optional if column in orders.columns])
        if {"DefaultDCShippingCost", "DivertedDCShippingCost"} <= set(orders.columns):
            result["selected_shipping_cost"] = float(
                np.where(
                    diverted,
                    orders["DivertedDCShippingCost"],
                    orders["DefaultDCShippingCost"],
                )
                .astype(float)
                .sum()
            )
        if {"PenaltyIfNotDiverted", "PenaltyIfDiverted"} <= set(orders.columns):
            result["selected_penalty_cost"] = float(
                np.where(
                    diverted,
                    orders["PenaltyIfDiverted"],
                    orders["PenaltyIfNotDiverted"],
                )
                .astype(float)
                .sum()
            )
    return result
