"""Authoritative order-penalty calculations for synthetic and POC data."""

from __future__ import annotations

from math import ceil

import pandas as pd

from .schemas import ProblemData

LINEAR_UNMET = "linear_unmet"
THRESHOLDED_CUT = "thresholded_cut"


def penalty_mode(problem: ProblemData) -> str:
    """Return the configured penalty model and reject unknown values."""

    mode = str(problem.metadata.get("penalty_mode", LINEAR_UNMET)).strip().lower()
    if mode not in {LINEAR_UNMET, THRESHOLDED_CUT}:
        raise ValueError(f"Unsupported penalty_mode {mode!r}")
    return mode


def order_penalty_parameters(problem: ProblemData, order_id: str) -> dict[str, float]:
    """Return normalized order-level parameters for the thresholded POC rule."""

    rows = problem.orders.loc[problem.orders["order_id"].astype(str) == str(order_id)]
    if len(rows) != 1:
        raise ValueError(f"Expected one order row for {order_id!r}")
    row = rows.iloc[0]

    def value(column: str, default: float = 0.0) -> float:
        raw = row.get(column, default)
        return default if pd.isna(raw) else float(raw)

    return {
        "threshold": value("penalty_threshold_fraction"),
        "fixed": value("penalty_fixed"),
        "per_cut_sku": value("penalty_per_cut_sku"),
        "minimum": value("penalty_minimum"),
        "maximum": value("penalty_maximum"),
    }


def penalty_activation_fill_cases(problem: ProblemData, order_id: str) -> int:
    """Smallest integer fill that avoids the order-level penalty."""

    demand = int(
        problem.order_lines.loc[
            problem.order_lines["order_id"].astype(str) == str(order_id),
            "demand_cases",
        ].sum()
    )
    threshold = order_penalty_parameters(problem, order_id)["threshold"]
    return min(demand, max(0, ceil(threshold * demand - 1e-12)))


def order_penalty(problem: ProblemData, order_id: str, quantities: pd.DataFrame) -> float:
    """Evaluate one order's penalty from demand and unfulfilled quantities.

    ``quantities`` must contain ``demand_cases``, ``unfulfilled_cases``, and
    ``penalty_per_unfilled_case``. In the POC rule, a partially unfilled order
    incurs no penalty when its integer case fill meets the configured threshold.
    Below that threshold, the raw penalty is clipped to the optional minimum and
    maximum supplied in the challenge data.
    """

    required = {"demand_cases", "unfulfilled_cases", "penalty_per_unfilled_case"}
    missing = required - set(quantities.columns)
    if missing:
        raise ValueError(f"Penalty quantities are missing columns: {sorted(missing)}")

    unfulfilled = quantities["unfulfilled_cases"].astype(float)
    linear = float(
        (quantities["penalty_per_unfilled_case"].astype(float) * unfulfilled).sum()
    )
    if penalty_mode(problem) == LINEAR_UNMET:
        return linear

    demand = float(quantities["demand_cases"].astype(float).sum())
    fulfilled = demand - float(unfulfilled.sum())
    required_fill = penalty_activation_fill_cases(problem, order_id)
    if fulfilled >= required_fill - 1e-9:
        return 0.0

    parameters = order_penalty_parameters(problem, order_id)
    cut_skus = int((unfulfilled > 1e-9).sum())
    raw = linear + parameters["fixed"] + parameters["per_cut_sku"] * cut_skus
    penalty = max(raw, parameters["minimum"])
    if parameters["maximum"] > 0:
        penalty = min(penalty, parameters["maximum"])
    return float(penalty)


def total_penalty(problem: ProblemData, quantities: pd.DataFrame) -> float:
    """Evaluate the total penalty for a complete canonical line table."""

    total = 0.0
    for order_id, group in quantities.groupby("order_id", sort=False):
        total += order_penalty(problem, str(order_id), group)
    return float(total)
