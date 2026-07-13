"""Candidate generation and hard-feasibility filtering."""

from __future__ import annotations

import pandas as pd


class CandidateGenerationError(ValueError):
    """Raised when candidate inputs are ambiguous or inconsistent."""


def filter_feasible_candidates(
    candidates: pd.DataFrame,
    *,
    orders: pd.DataFrame | None = None,
    calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return candidates that satisfy documented hard filters.

    Required candidate columns are ``candidate_id``, ``order_id``, ``dc_id``,
    ``pgi_date``, ``shipping_cost``, ``is_default``, and ``eligible``.
    Optional ``arrival_date`` is compared with the order's requested delivery
    date when both are available.
    """

    required = {
        "candidate_id",
        "order_id",
        "dc_id",
        "pgi_date",
        "shipping_cost",
        "is_default",
        "eligible",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise CandidateGenerationError(f"Candidate table is missing {sorted(missing)}")

    result = candidates.copy()
    result["pgi_date"] = pd.to_datetime(result["pgi_date"], errors="raise")
    result = result.loc[result["eligible"].astype(bool)].copy()

    if calendar is not None and not calendar.empty:
        required_calendar = {"dc_id", "date", "is_open"}
        missing_calendar = required_calendar - set(calendar.columns)
        if missing_calendar:
            raise CandidateGenerationError(
                f"Calendar table is missing {sorted(missing_calendar)}"
            )
        cal = calendar.copy()
        cal["date"] = pd.to_datetime(cal["date"], errors="raise")
        cal = cal[["dc_id", "date", "is_open"]].rename(columns={"date": "pgi_date"})
        result = result.merge(
            cal,
            on=["dc_id", "pgi_date"],
            how="left",
            validate="many_to_one",
        )
        # A missing calendar record is not silently interpreted as closed.
        result = result.loc[result["is_open"].fillna(True).astype(bool)].drop(
            columns=["is_open"]
        )

    if orders is not None and "arrival_date" in result.columns:
        required_orders = {"order_id", "requested_delivery_date"}
        missing_orders = required_orders - set(orders.columns)
        if missing_orders:
            raise CandidateGenerationError(f"Order table is missing {sorted(missing_orders)}")
        order_dates = orders[["order_id", "requested_delivery_date"]].copy()
        order_dates["requested_delivery_date"] = pd.to_datetime(
            order_dates["requested_delivery_date"], errors="raise"
        )
        result["arrival_date"] = pd.to_datetime(result["arrival_date"], errors="coerce")
        result = result.merge(order_dates, on="order_id", how="left", validate="many_to_one")
        valid_arrival = result["arrival_date"].isna() | (
            result["arrival_date"] <= result["requested_delivery_date"]
        )
        result = result.loc[valid_arrival].drop(columns=["requested_delivery_date"])

    if result.duplicated(["order_id", "dc_id", "pgi_date"]).any():
        raise CandidateGenerationError(
            "More than one candidate remains for the same (order_id, dc_id, pgi_date)"
        )

    return result.sort_values(
        ["order_id", "pgi_date", "shipping_cost", "candidate_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def generate_candidates(
    orders: pd.DataFrame,
    lane_options: pd.DataFrame,
    *,
    calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build canonical candidates from already-calculated lane/date options.

    This function intentionally does not guess lead-time calendars or inventory
    rules. ``lane_options`` must already contain one row per proposed order–DC–PGI
    combination and the required canonical candidate fields.
    """

    candidates = lane_options.copy()
    if "candidate_id" not in candidates.columns:
        candidates["candidate_id"] = candidates.apply(
            lambda row: f"{row['order_id']}__{row['dc_id']}__{pd.Timestamp(row['pgi_date']).date()}",
            axis=1,
        )
    if "eligible" not in candidates.columns:
        candidates["eligible"] = True
    if "is_default" not in candidates.columns:
        defaults = orders.set_index("order_id")["default_dc"].to_dict()
        candidates["is_default"] = candidates.apply(
            lambda row: str(row["dc_id"]) == str(defaults[str(row["order_id"])]), axis=1
        )
    return filter_feasible_candidates(candidates, orders=orders, calendar=calendar)

