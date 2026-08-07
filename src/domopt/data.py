"""Loading, normalization, integrity checks, and synthetic data helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import (
    ASSUMPTION_VERSION,
    CALENDAR_COLUMNS,
    CANDIDATES_COLUMNS,
    CAPACITY_COLUMNS,
    INVENTORY_COLUMNS,
    ORDER_LINES_COLUMNS,
    ORDERS_COLUMNS,
    SCHEMA_VERSION,
    ProblemData,
)


class DataValidationError(ValueError):
    """Raised when canonical input data violate the documented contract."""


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {missing}")


def _parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        raise DataValidationError("Boolean field contains a missing value")
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "y", "yes"}:
        return True
    if normalized in {"false", "0", "n", "no"}:
        return False
    raise DataValidationError(f"Cannot parse Boolean value {value!r}")


def _as_string(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            if frame[column].isna().any():
                raise DataValidationError(f"Column {column!r} contains missing identifiers")
            frame[column] = frame[column].astype(str).str.strip()
            invalid = frame[column].str.lower().isin({"", "null", "none", "nan"})
            if invalid.any():
                raise DataValidationError(
                    f"Column {column!r} contains empty or textual-null identifiers"
                )


def _as_date(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.isna().any() and frame[column].notna().any():
                bad = frame.loc[parsed.isna() & frame[column].notna(), column].head().tolist()
                raise DataValidationError(f"Invalid dates in {column!r}: {bad}")
            frame[column] = parsed


def _as_numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    integer: bool = False,
    nonnegative: bool = True,
) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            bad = frame.loc[values.isna(), column].head().tolist()
            raise DataValidationError(f"Invalid numeric values in {column!r}: {bad}")
        finite = np.isfinite(values.to_numpy(dtype=float))
        if not finite.all():
            bad = frame.loc[~finite, column].head().tolist()
            raise DataValidationError(f"Nonfinite numeric values in {column!r}: {bad}")
        if nonnegative and (values < 0).any():
            raise DataValidationError(f"Column {column!r} contains negative values")
        if integer:
            rounded = np.rint(values.to_numpy(dtype=float))
            if not np.allclose(values.to_numpy(dtype=float), rounded, atol=1e-9):
                raise DataValidationError(f"Column {column!r} must contain integers")
            frame[column] = rounded.astype(np.int64)
        else:
            frame[column] = values.astype(float)


def normalize_problem_data(problem: ProblemData) -> ProblemData:
    """Return a normalized deep copy of a canonical problem instance."""

    orders = problem.orders.copy(deep=True)
    lines = problem.order_lines.copy(deep=True)
    inventory = problem.inventory.copy(deep=True)
    candidates = problem.candidates.copy(deep=True)
    capacities = problem.capacities.copy(deep=True)
    calendar = problem.calendar.copy(deep=True)

    _require_columns(orders, ORDERS_COLUMNS, "orders")
    _require_columns(lines, ORDER_LINES_COLUMNS, "order_lines")
    _require_columns(inventory, INVENTORY_COLUMNS, "inventory")
    _require_columns(candidates, CANDIDATES_COLUMNS, "candidates")
    _require_columns(capacities, CAPACITY_COLUMNS, "capacities")
    _require_columns(calendar, CALENDAR_COLUMNS, "calendar")

    _as_string(orders, ["order_id", "default_dc", "assignment_group"])
    _as_string(lines, ["order_id", "sku_id"])
    _as_string(inventory, ["dc_id", "sku_id"])
    _as_string(
        candidates,
        ["candidate_id", "order_id", "dc_id", "group_option_id"],
    )
    if not capacities.empty:
        _as_string(capacities, ["dc_id", "resource", "unit"])
    if not calendar.empty:
        _as_string(calendar, ["dc_id"])

    _as_date(orders, ["requested_delivery_date", "default_pgi_date"])
    _as_date(inventory, ["date"])
    _as_date(candidates, ["pgi_date", "arrival_date"])
    _as_date(capacities, ["date"])
    _as_date(calendar, ["date"])

    _as_numeric(orders, ["default_fillable_cases", "priority"], integer=True)
    _as_numeric(
        orders,
        [
            "min_divert_improvement_fraction",
            "penalty_threshold_fraction",
            "penalty_fixed",
            "penalty_per_cut_sku",
            "penalty_minimum",
            "penalty_maximum",
        ],
        nonnegative=True,
    )
    _as_numeric(lines, ["demand_cases", "cases_per_pallet"], integer=True)
    _as_numeric(
        lines,
        [
            "unit_value",
            "penalty_per_unfilled_case",
            "unit_weight",
            "unit_volume",
        ],
    )
    _as_numeric(
        inventory,
        [
            "cumulative_available_cases",
            "reserved_default_cases",
            "source_inventory_cases",
        ],
        integer=True,
    )
    if "dock_units" not in candidates.columns:
        candidates["dock_units"] = 1.0
    _as_numeric(candidates, ["shipping_cost", "distance", "dock_units"])
    _as_numeric(candidates, ["lead_time_days"], integer=True)
    _as_numeric(capacities, ["capacity"])

    candidates["is_default"] = candidates["is_default"].map(_parse_bool)
    candidates["eligible"] = candidates["eligible"].map(_parse_bool)
    if "is_top_customer" in orders.columns:
        orders["is_top_customer"] = orders["is_top_customer"].map(_parse_bool)
    if "forecast_required" in lines.columns:
        lines["forecast_required"] = lines["forecast_required"].map(_parse_bool)
    if not calendar.empty:
        calendar["is_open"] = calendar["is_open"].map(_parse_bool)

    normalized = ProblemData(
        orders=orders,
        order_lines=lines,
        inventory=inventory,
        candidates=candidates,
        capacities=capacities,
        calendar=calendar,
        metadata=dict(problem.metadata),
        source_dir=problem.source_dir,
    )
    validate_problem_data(normalized)
    return normalized


def validate_problem_data(problem: ProblemData) -> None:
    """Validate canonical keys, foreign keys, candidate logic, and inventory monotonicity."""

    orders = problem.orders
    lines = problem.order_lines
    inventory = problem.inventory
    candidates = problem.candidates
    capacities = problem.capacities
    calendar = problem.calendar

    if orders["order_id"].duplicated().any():
        duplicate = orders.loc[orders["order_id"].duplicated(), "order_id"].tolist()
        raise DataValidationError(f"Duplicate order_id values: {duplicate[:5]}")
    if lines.duplicated(["order_id", "sku_id"]).any():
        raise DataValidationError("order_lines contains duplicate (order_id, sku_id) rows")
    if inventory.duplicated(["dc_id", "sku_id", "date"]).any():
        raise DataValidationError("inventory contains duplicate (dc_id, sku_id, date) rows")
    if candidates["candidate_id"].duplicated().any():
        raise DataValidationError("candidates contains duplicate candidate_id values")
    if candidates.duplicated(["order_id", "dc_id", "pgi_date"]).any():
        raise DataValidationError(
            "candidates contains duplicate (order_id, dc_id, pgi_date) rows"
        )
    if not capacities.empty and capacities.duplicated(["dc_id", "date", "resource"]).any():
        raise DataValidationError("capacities contains duplicate key rows")
    if not calendar.empty and calendar.duplicated(["dc_id", "date"]).any():
        raise DataValidationError("calendar contains duplicate key rows")

    order_ids = set(orders["order_id"])
    unknown_lines = sorted(set(lines["order_id"]) - order_ids)
    unknown_candidates = sorted(set(candidates["order_id"]) - order_ids)
    if unknown_lines:
        raise DataValidationError(
            f"order_lines references unknown orders: {unknown_lines[:5]}"
        )
    if unknown_candidates:
        raise DataValidationError(
            f"candidates references unknown orders: {unknown_candidates[:5]}"
        )
    missing_line_orders = sorted(order_ids - set(lines["order_id"]))
    if missing_line_orders:
        raise DataValidationError(
            f"Every order must have at least one line; missing for {missing_line_orders[:5]}"
        )
    if (lines["demand_cases"].astype(int) <= 0).any():
        raise DataValidationError("order_lines.demand_cases must be positive")
    if "cases_per_pallet" in lines and (
        lines["cases_per_pallet"].astype(int) <= 0
    ).any():
        raise DataValidationError("order_lines.cases_per_pallet must be positive")

    eligible_candidates = candidates[candidates["eligible"]]
    missing_candidate_orders = sorted(order_ids - set(eligible_candidates["order_id"]))
    if missing_candidate_orders:
        # No-assignment preserves mathematical feasibility, but an absent candidate
        # set is almost always an upstream data error.
        raise DataValidationError(
            "Every order must have at least one eligible assignment candidate; "
            "missing for "
            f"{missing_candidate_orders[:5]}"
        )

    eligible_default_counts = (
        eligible_candidates.loc[eligible_candidates["is_default"]]
        .groupby("order_id")
        .size()
        .reindex(sorted(order_ids), fill_value=0)
    )
    if not eligible_default_counts.eq(1).all():
        bad = eligible_default_counts.loc[~eligible_default_counts.eq(1)].index.tolist()
        raise DataValidationError(
            "Every order must have exactly one eligible default candidate; "
            f"invalid for {bad[:5]}"
        )

    default_map = orders.set_index("order_id")["default_dc"].to_dict()
    incorrect_default = candidates[
        candidates["is_default"]
        != candidates.apply(
            lambda row: row["dc_id"] == default_map[row["order_id"]],
            axis=1,
        )
    ]
    if not incorrect_default.empty:
        ids = incorrect_default["candidate_id"].head().tolist()
        raise DataValidationError(f"Incorrect is_default flag for candidates: {ids}")

    if not calendar.empty:
        open_lookup = calendar.set_index(["dc_id", "date"])["is_open"].to_dict()
        closed_selected = []
        for row in eligible_candidates.itertuples(index=False):
            key = (row.dc_id, row.pgi_date)
            if key in open_lookup and not bool(open_lookup[key]):
                closed_selected.append(row.candidate_id)
        if closed_selected:
            raise DataValidationError(
                "Eligible candidates occur on closed dates: " f"{closed_selected[:5]}"
            )

    if "arrival_date" in eligible_candidates:
        requested = eligible_candidates["order_id"].map(
            orders.set_index("order_id")["requested_delivery_date"]
        )
        late_alternates = eligible_candidates.loc[
            ~eligible_candidates["is_default"]
            & eligible_candidates["arrival_date"].notna()
            & eligible_candidates["arrival_date"].gt(requested),
            "candidate_id",
        ]
        if not late_alternates.empty:
            raise DataValidationError(
                "Eligible alternate candidates arrive after requested delivery: "
                f"{late_alternates.head().tolist()}"
            )

    inventory_policy = str(
        problem.metadata.get("inventory_policy", "projected_atp")
    ).lower()
    if inventory_policy not in {"projected_atp", "cumulative_receipts"}:
        raise DataValidationError(
            "metadata.inventory_policy must be 'projected_atp' or 'cumulative_receipts'"
        )
    if inventory_policy == "cumulative_receipts":
        for (dc_id, sku_id), group in inventory.groupby(
            ["dc_id", "sku_id"], sort=False
        ):
            ordered = group.sort_values("date")["cumulative_available_cases"].to_numpy()
            if np.any(np.diff(ordered) < 0):
                raise DataValidationError(
                    f"Cumulative inventory decreases for dc={dc_id}, sku={sku_id}"
                )

    if "min_divert_improvement_fraction" in orders.columns:
        fractions = orders["min_divert_improvement_fraction"].dropna().astype(float)
        if (fractions > 1).any():
            raise DataValidationError(
                "min_divert_improvement_fraction must be between zero and one"
            )
    if bool(problem.metadata.get("enforce_min_divert_improvement", False)):
        if "default_fillable_cases" not in orders.columns:
            raise DataValidationError(
                "Minimum-divert enforcement requires orders.default_fillable_cases"
            )
        if orders["default_fillable_cases"].isna().any():
            raise DataValidationError(
                "orders.default_fillable_cases contains missing values"
            )

    if str(problem.metadata.get("penalty_mode", "linear_unmet")) == "thresholded_cut":
        required_penalty_columns = {
            "penalty_threshold_fraction",
            "penalty_fixed",
            "penalty_per_cut_sku",
            "penalty_minimum",
            "penalty_maximum",
        }
        missing_penalty = required_penalty_columns - set(orders.columns)
        if missing_penalty:
            raise DataValidationError(
                "thresholded_cut penalty mode requires order columns: "
                f"{sorted(missing_penalty)}"
            )
        thresholds = orders["penalty_threshold_fraction"].astype(float)
        if ((thresholds < 0) | (thresholds > 1)).any():
            raise DataValidationError(
                "penalty_threshold_fraction must be between zero and one"
            )

    if bool(problem.metadata.get("enforce_assignment_group", False)):
        if "assignment_group" not in orders.columns:
            raise DataValidationError(
                "Group cohesion requires orders.assignment_group"
            )
        if "group_option_id" not in candidates.columns:
            raise DataValidationError(
                "Group cohesion requires candidates.group_option_id"
            )
        order_group = orders.set_index("order_id")["assignment_group"].astype(str)
        option_sets = {
            str(order_id): frozenset(group["group_option_id"].astype(str))
            for order_id, group in eligible_candidates.groupby("order_id", sort=False)
        }
        for group_id, members in orders.groupby("assignment_group", sort=False):
            member_ids = members["order_id"].astype(str).tolist()
            reference = option_sets[member_ids[0]]
            if any(option_sets[member] != reference for member in member_ids[1:]):
                raise DataValidationError(
                    "All orders in an assignment group must expose identical eligible "
                    f"group options; mismatch in group {group_id!r}"
                )
            for member in member_ids:
                if str(order_group.loc[member]) != str(group_id):
                    raise DataValidationError(
                        f"Inconsistent assignment-group mapping for order {member!r}"
                    )


def load_problem_data(data_dir: str | Path) -> ProblemData:
    """Load canonical CSV/JSON files from ``data_dir``."""

    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {path}")

    def read_required(name: str) -> pd.DataFrame:
        file_path = path / name
        if not file_path.exists():
            raise FileNotFoundError(f"Required data file is missing: {file_path}")
        return pd.read_csv(file_path, dtype=str, keep_default_na=False)

    orders = read_required("orders.csv")
    lines = read_required("order_lines.csv")
    inventory = read_required("inventory.csv")
    candidates = read_required("candidates.csv")
    capacities = read_required("capacities.csv")

    calendar_path = path / "calendar.csv"
    if calendar_path.exists():
        calendar = pd.read_csv(calendar_path, dtype=str, keep_default_na=False)
    else:
        calendar = pd.DataFrame(columns=sorted(CALENDAR_COLUMNS))

    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    return normalize_problem_data(
        ProblemData(
            orders=orders,
            order_lines=lines,
            inventory=inventory,
            candidates=candidates,
            capacities=capacities,
            calendar=calendar,
            metadata=metadata,
            source_dir=path,
        )
    )


def _serialize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].dt.strftime("%Y-%m-%d")
    return result


def save_problem_data(problem: ProblemData, data_dir: str | Path) -> Path:
    """Write one canonical problem instance to disk."""

    normalized = normalize_problem_data(problem)
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)

    _serialize_dates(normalized.orders).to_csv(path / "orders.csv", index=False)
    _serialize_dates(normalized.order_lines).to_csv(path / "order_lines.csv", index=False)
    _serialize_dates(normalized.inventory).to_csv(path / "inventory.csv", index=False)
    _serialize_dates(normalized.candidates).to_csv(path / "candidates.csv", index=False)
    _serialize_dates(normalized.capacities).to_csv(path / "capacities.csv", index=False)
    if not normalized.calendar.empty:
        _serialize_dates(normalized.calendar).to_csv(path / "calendar.csv", index=False)

    metadata = {
        "dataset_id": "unknown",
        "schema_version": SCHEMA_VERSION,
        "assumption_version": ASSUMPTION_VERSION,
        "currency": "unspecified",
        "quantity_unit": "cases",
        "inventory_policy": "projected_atp",
        **normalized.metadata,
    }
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n"
    )
    return path


def make_tiny_problem_data() -> ProblemData:
    """Construct the exact two-order synthetic instance documented in the repository."""

    orders = pd.DataFrame(
        [
            {
                "order_id": "O1",
                "default_dc": "D1",
                "requested_delivery_date": "2026-07-15",
                "default_pgi_date": "2026-07-14",
                "default_fillable_cases": 5,
                "min_divert_improvement_fraction": 0.05,
            },
            {
                "order_id": "O2",
                "default_dc": "D1",
                "requested_delivery_date": "2026-07-15",
                "default_pgi_date": "2026-07-14",
                "default_fillable_cases": 7,
                "min_divert_improvement_fraction": 0.05,
            },
        ]
    )
    lines = pd.DataFrame(
        [
            {
                "order_id": order_id,
                "sku_id": sku_id,
                "demand_cases": demand,
                "unit_value": 10,
                "penalty_per_unfilled_case": 20,
            }
            for order_id, sku_id, demand in [
                ("O1", "A", 4),
                ("O1", "B", 2),
                ("O2", "A", 3),
                ("O2", "B", 4),
            ]
        ]
    )
    inventory = pd.DataFrame(
        [
            {
                "dc_id": dc_id,
                "sku_id": sku_id,
                "date": "2026-07-14",
                "cumulative_available_cases": available,
            }
            for dc_id, sku_id, available in [
                ("D1", "A", 3),
                ("D1", "B", 4),
                ("D2", "A", 4),
                ("D2", "B", 2),
            ]
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": f"{order_id}_{dc_id}_T1",
                "order_id": order_id,
                "dc_id": dc_id,
                "pgi_date": "2026-07-14",
                "shipping_cost": 0 if dc_id == "D1" else 4,
                "is_default": dc_id == "D1",
                "eligible": True,
            }
            for order_id in ["O1", "O2"]
            for dc_id in ["D1", "D2"]
        ]
    )
    capacities = pd.DataFrame(columns=["dc_id", "date", "resource", "capacity", "unit"])
    calendar = pd.DataFrame(
        [
            {"dc_id": "D1", "date": "2026-07-14", "is_open": True},
            {"dc_id": "D2", "date": "2026-07-14", "is_open": True},
        ]
    )
    metadata = {
        "dataset_id": "tiny",
        "schema_version": SCHEMA_VERSION,
        "assumption_version": ASSUMPTION_VERSION,
        "currency": "synthetic_units",
        "quantity_unit": "cases",
        "inventory_policy": "projected_atp",
        "pick_capacity_mode": "auto",
        "enforce_min_divert_improvement": True,
        "min_divert_improvement_fraction": 0.05,
    }
    return normalize_problem_data(
        ProblemData(
            orders=orders,
            order_lines=lines,
            inventory=inventory,
            candidates=candidates,
            capacities=capacities,
            calendar=calendar,
            metadata=metadata,
        )
    )
