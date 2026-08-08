"""Independent feasibility validation for solver outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .resources import solution_capacity_usage
from .rules import minimum_divert_fulfillment
from .schemas import ProblemData, Solution, ValidationResult

_TOL = 1e-8


def _date(value: object) -> pd.Timestamp | pd.NaT:
    try:
        if value is None or pd.isna(value):
            return pd.NaT
        return pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT


def _strict_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"expected a boolean value, received {value!r}")


def validate_solution(problem: ProblemData, solution: Solution) -> ValidationResult:
    assignment_violations: list[str] = []
    demand_violations: list[str] = []
    inventory_violations: list[str] = []
    eligibility_violations: list[str] = []
    capacity_violations: list[str] = []
    schema_violations: list[str] = []

    assignments = solution.assignments.copy()
    fulfillment = solution.fulfillment.copy()
    order_ids = set(problem.orders["order_id"].astype(str))
    line_keys = set(
        map(
            tuple,
            problem.order_lines[["order_id", "sku_id"]]
            .astype(str)
            .itertuples(index=False, name=None),
        )
    )
    enabled_capacity_resources = sorted(
        problem.capacities.get("resource", pd.Series(dtype=str)).astype(str).unique()
    )
    diagnostics: dict[str, object] = {
        "validation_tolerance": _TOL,
        "checked_orders": len(order_ids),
        "checked_order_lines": len(line_keys),
        "assignment_row_count": len(assignments),
        "fulfillment_row_count": len(fulfillment),
        "maximum_demand_balance_abs_error": 0.0,
        "maximum_fulfilled_integrality_error": 0.0,
        "maximum_unfulfilled_integrality_error": 0.0,
        "maximum_inventory_excess_cases": 0.0,
        "maximum_capacity_excess": 0.0,
        "capacity_constraints_enabled": not problem.capacities.empty,
        "capacity_constraint_rows": len(problem.capacities),
        "enabled_capacity_resources": "|".join(enabled_capacity_resources) or "none",
        "throughput_capacity_enabled": "throughput_cases" in enabled_capacity_resources,
        "case_pick_capacity_enabled": "case_pick" in enabled_capacity_resources,
        "pallet_pick_capacity_enabled": "pallet_pick" in enabled_capacity_resources,
    }

    required_assignment_columns = {
        "order_id",
        "candidate_id",
        "selected_dc",
        "selected_pgi_date",
        "is_unassigned",
        "is_divert",
        "method",
    }
    missing = required_assignment_columns - set(assignments.columns)
    if missing:
        schema_violations.append(f"assignments missing columns {sorted(missing)}")

    required_fulfillment_columns = {
        "order_id",
        "sku_id",
        "fulfilled_cases",
        "unfulfilled_cases",
        "selected_dc",
        "selected_pgi_date",
    }
    missing_f = required_fulfillment_columns - set(fulfillment.columns)
    if missing_f:
        schema_violations.append(f"fulfillment missing columns {sorted(missing_f)}")

    if schema_violations:
        return ValidationResult(
            is_feasible=False,
            schema_violations=schema_violations,
            diagnostics=diagnostics,
        )

    for column in ["is_unassigned", "is_divert"]:
        parsed: list[bool] = []
        for index, value in assignments[column].items():
            try:
                parsed.append(_strict_bool(value))
            except ValueError as error:
                schema_violations.append(
                    f"assignments row {index} column {column}: {error}"
                )
                parsed.append(False)
        assignments[column] = parsed

    if assignments["order_id"].duplicated().any():
        duplicates = assignments.loc[
            assignments["order_id"].duplicated(keep=False), "order_id"
        ].unique().tolist()
        assignment_violations.append(
            f"duplicate assignment rows for orders {duplicates[:5]}"
        )

    assignment_ids = set(assignments["order_id"].astype(str))
    missing_orders = sorted(order_ids - assignment_ids)
    extra_orders = sorted(assignment_ids - order_ids)
    if missing_orders:
        assignment_violations.append(
            f"orders missing assignment rows: {missing_orders[:5]}"
        )
    if extra_orders:
        assignment_violations.append(
            f"assignment rows reference unknown orders: {extra_orders[:5]}"
        )

    candidate_lookup = problem.candidates.set_index("candidate_id", drop=False)
    default_lookup = problem.orders.set_index("order_id")["default_dc"].to_dict()
    assignment_lookup: dict[str, dict[str, object]] = {}

    for row in assignments.itertuples(index=False):
        order_id = str(row.order_id)
        if order_id in assignment_lookup:
            continue
        is_unassigned = bool(row.is_unassigned)
        candidate_id = (
            None
            if pd.isna(row.candidate_id) or str(row.candidate_id).strip() == ""
            else str(row.candidate_id)
        )
        selected_dc = (
            None
            if pd.isna(row.selected_dc) or str(row.selected_dc).strip() == ""
            else str(row.selected_dc)
        )
        selected_date = _date(row.selected_pgi_date)

        if is_unassigned:
            if candidate_id is not None or selected_dc is not None or not pd.isna(selected_date):
                assignment_violations.append(
                    f"unassigned order {order_id} contains selected candidate/DC/date"
                )
            if bool(row.is_divert):
                assignment_violations.append(
                    f"unassigned order {order_id} cannot be a divert"
                )
        else:
            if candidate_id is None:
                assignment_violations.append(
                    f"assigned order {order_id} has no candidate_id"
                )
            elif candidate_id not in candidate_lookup.index:
                eligibility_violations.append(
                    f"order {order_id} selects unknown candidate {candidate_id}"
                )
            else:
                candidate = candidate_lookup.loc[candidate_id]
                if isinstance(candidate, pd.DataFrame):
                    candidate = candidate.iloc[0]
                if str(candidate["order_id"]) != order_id:
                    eligibility_violations.append(
                        f"candidate {candidate_id} belongs to "
                        f"{candidate['order_id']}, not {order_id}"
                    )
                if not bool(candidate["eligible"]):
                    eligibility_violations.append(
                        f"order {order_id} selects ineligible candidate {candidate_id}"
                    )
                if selected_dc != str(candidate["dc_id"]):
                    assignment_violations.append(
                        f"order {order_id} selected_dc does not match candidate {candidate_id}"
                    )
                if (
                    pd.isna(selected_date)
                    or selected_date != pd.Timestamp(candidate["pgi_date"])
                ):
                    assignment_violations.append(
                        f"order {order_id} selected_pgi_date does not match "
                        f"candidate {candidate_id}"
                    )
                expected_divert = str(candidate["dc_id"]) != default_lookup.get(order_id)
                if bool(row.is_divert) != expected_divert:
                    assignment_violations.append(
                        f"order {order_id} has incorrect is_divert flag"
                    )

        assignment_lookup[order_id] = {
            "is_unassigned": is_unassigned,
            "candidate_id": candidate_id,
            "selected_dc": selected_dc,
            "selected_pgi_date": selected_date,
        }

    if bool(problem.metadata.get("enforce_assignment_group", False)):
        if "assignment_group" not in problem.orders.columns:
            schema_violations.append(
                "group cohesion requires orders.assignment_group"
            )
        else:
            for group_id, group in problem.orders.groupby(
                "assignment_group", sort=False
            ):
                members = [
                    assignment_lookup.get(str(order_id))
                    for order_id in group["order_id"]
                ]
                members = [member for member in members if member is not None]
                if len(members) <= 1:
                    continue
                outcomes = {
                    (
                        bool(member["is_unassigned"]),
                        member["selected_dc"],
                        member["selected_pgi_date"],
                    )
                    for member in members
                }
                if len(outcomes) > 1:
                    assignment_violations.append(
                        f"assignment group {group_id!r} is split across outcomes"
                    )

    if fulfillment.duplicated(["order_id", "sku_id"]).any():
        demand_violations.append("duplicate fulfillment rows for an order–SKU pair")

    fulfillment_keys = set(
        map(
            tuple,
            fulfillment[["order_id", "sku_id"]]
            .astype(str)
            .itertuples(index=False, name=None),
        )
    )
    missing_lines = sorted(line_keys - fulfillment_keys)
    extra_lines = sorted(fulfillment_keys - line_keys)
    if missing_lines:
        demand_violations.append(f"missing fulfillment rows: {missing_lines[:5]}")
    if extra_lines:
        demand_violations.append(f"unknown fulfillment rows: {extra_lines[:5]}")

    demands = (
        problem.order_lines.set_index(["order_id", "sku_id"])["demand_cases"]
        .to_dict()
    )
    positive_usage: list[dict[str, object]] = []

    for row in fulfillment.itertuples(index=False):
        key = (str(row.order_id), str(row.sku_id))
        if key not in demands:
            continue
        try:
            fulfilled = float(row.fulfilled_cases)
            unfulfilled = float(row.unfulfilled_cases)
        except (TypeError, ValueError):
            demand_violations.append(f"nonnumeric quantities for line {key}")
            continue

        if not np.isfinite(fulfilled) or not np.isfinite(unfulfilled):
            demand_violations.append(f"nonfinite quantities for line {key}")
            continue

        if fulfilled < -_TOL or unfulfilled < -_TOL:
            demand_violations.append(f"negative quantities for line {key}")
        fulfilled_integrality_error = abs(fulfilled - round(fulfilled))
        unfulfilled_integrality_error = abs(unfulfilled - round(unfulfilled))
        diagnostics["maximum_fulfilled_integrality_error"] = max(
            float(diagnostics["maximum_fulfilled_integrality_error"]),
            fulfilled_integrality_error,
        )
        diagnostics["maximum_unfulfilled_integrality_error"] = max(
            float(diagnostics["maximum_unfulfilled_integrality_error"]),
            unfulfilled_integrality_error,
        )
        if fulfilled_integrality_error > _TOL:
            demand_violations.append(f"fulfilled_cases is not integer for line {key}")
        if unfulfilled_integrality_error > _TOL:
            demand_violations.append(f"unfulfilled_cases is not integer for line {key}")
        demand = float(demands[key])
        balance_error = abs(fulfilled + unfulfilled - demand)
        diagnostics["maximum_demand_balance_abs_error"] = max(
            float(diagnostics["maximum_demand_balance_abs_error"]),
            balance_error,
        )
        if balance_error > _TOL:
            demand_violations.append(
                f"demand balance fails for {key}: {fulfilled}+{unfulfilled}!={demand}"
            )

        assignment = assignment_lookup.get(key[0])
        row_dc = (
            None
            if pd.isna(row.selected_dc) or str(row.selected_dc).strip() == ""
            else str(row.selected_dc)
        )
        row_date = _date(row.selected_pgi_date)
        if assignment is None:
            continue
        if assignment["is_unassigned"]:
            if fulfilled > _TOL:
                demand_violations.append(f"unassigned order {key[0]} has positive fulfillment")
            if row_dc is not None or not pd.isna(row_date):
                demand_violations.append(
                    f"unassigned order line {key} contains selected DC/date"
                )
        else:
            if row_dc != assignment["selected_dc"]:
                demand_violations.append(f"selected DC mismatch for line {key}")
            if pd.isna(row_date) or row_date != assignment["selected_pgi_date"]:
                demand_violations.append(f"selected date mismatch for line {key}")
            if fulfilled > _TOL:
                positive_usage.append(
                    {
                        "order_id": key[0],
                        "sku_id": key[1],
                        "dc_id": row_dc,
                        "date": row_date,
                        "fulfilled_cases": fulfilled,
                    }
                )

    usage = pd.DataFrame(positive_usage)
    inventory = problem.inventory.copy()
    if not usage.empty:
        usage["date"] = pd.to_datetime(usage["date"])
        inventory["date"] = pd.to_datetime(inventory["date"])
        usage_groups = {
            (str(dc_id), str(sku_id)): group
            for (dc_id, sku_id), group in usage.groupby(
                ["dc_id", "sku_id"], sort=False
            )
        }
        inventory_groups = {
            (str(dc_id), str(sku_id)): group.sort_values("date")
            for (dc_id, sku_id), group in inventory.groupby(
                ["dc_id", "sku_id"], sort=False
            )
        }

        for key, group in usage_groups.items():
            checkpoints = inventory_groups.get(key)
            if checkpoints is None:
                for date in group["date"]:
                    inventory_violations.append(
                        "no cumulative inventory checkpoint covers "
                        f"dc={key[0]}, sku={key[1]}, date={pd.Timestamp(date).date()}"
                    )
                continue
            last_checkpoint = pd.Timestamp(checkpoints["date"].max())
            for date in group.loc[group["date"] > last_checkpoint, "date"]:
                inventory_violations.append(
                    "no cumulative inventory checkpoint covers "
                    f"dc={key[0]}, sku={key[1]}, date={pd.Timestamp(date).date()}"
                )

        # A shipment on date t consumes every cumulative checkpoint at t or later.
        # Grouping and searchsorted replace the previous inventory-row-by-row scans.
        for key, checkpoints in inventory_groups.items():
            group = usage_groups.get(key)
            if group is None:
                continue
            daily = (
                group.groupby("date", as_index=False)["fulfilled_cases"]
                .sum()
                .sort_values("date")
            )
            usage_dates = daily["date"].to_numpy(dtype="datetime64[ns]")
            cumulative = daily["fulfilled_cases"].to_numpy(dtype=float).cumsum()
            checkpoint_dates = checkpoints["date"].to_numpy(dtype="datetime64[ns]")
            positions = np.searchsorted(usage_dates, checkpoint_dates, side="right") - 1
            consumed = np.where(positions >= 0, cumulative[np.maximum(positions, 0)], 0.0)
            available = checkpoints["cumulative_available_cases"].to_numpy(dtype=float)
            diagnostics["maximum_inventory_excess_cases"] = max(
                float(diagnostics["maximum_inventory_excess_cases"]),
                float(np.maximum(consumed - available, 0.0).max(initial=0.0)),
            )
            exceeded = np.flatnonzero(consumed > available + _TOL)
            for index in exceeded:
                row = checkpoints.iloc[int(index)]
                inventory_violations.append(
                    f"inventory exceeded at dc={key[0]}, sku={key[1]}, "
                    f"date={pd.Timestamp(row['date']).date()}: used={consumed[index]}, "
                    f"available={row['cumulative_available_cases']}"
                )

    fulfilled_by_order = fulfillment.groupby("order_id")["fulfilled_cases"].sum()
    for row in assignments.loc[
        (~assignments["is_unassigned"].astype(bool))
        & assignments["is_divert"].astype(bool)
    ].itertuples(index=False):
        try:
            threshold = minimum_divert_fulfillment(problem, str(row.order_id))
        except ValueError as error:
            schema_violations.append(str(error))
            continue
        if threshold is not None:
            fulfilled = float(fulfilled_by_order.get(str(row.order_id), 0.0))
            if fulfilled + _TOL < threshold:
                demand_violations.append(
                    f"divert improvement fails for order {row.order_id}: "
                    f"fulfilled={fulfilled}, required={threshold}"
                )

    capacities = problem.capacities
    if not capacities.empty:
        sanitized_solution = Solution(
            method=solution.method,
            assignments=assignments,
            fulfillment=fulfillment,
            runtime_seconds=solution.runtime_seconds,
            raw_objective=solution.raw_objective,
            metadata=solution.metadata,
        )
        try:
            capacity_usage = solution_capacity_usage(problem, sanitized_solution)
        except (TypeError, ValueError, OverflowError) as error:
            schema_violations.append(str(error))
            capacity_usage = {}
        for cap in capacities.itertuples(index=False):
            date = pd.Timestamp(cap.date)
            resource = str(cap.resource)
            capacity = float(cap.capacity)
            consumed = float(capacity_usage.get((str(cap.dc_id), date, resource), 0.0))
            diagnostics["maximum_capacity_excess"] = max(
                float(diagnostics["maximum_capacity_excess"]),
                max(0.0, consumed - capacity),
            )

            if consumed > capacity + _TOL:
                capacity_violations.append(
                    f"capacity exceeded for dc={cap.dc_id}, date={date.date()}, "
                    f"resource={resource}: used={consumed}, capacity={capacity}"
                )

    all_violations = (
        assignment_violations
        + demand_violations
        + inventory_violations
        + eligibility_violations
        + capacity_violations
        + schema_violations
    )
    return ValidationResult(
        is_feasible=not all_violations,
        assignment_violations=assignment_violations,
        demand_violations=demand_violations,
        inventory_violations=inventory_violations,
        eligibility_violations=eligibility_violations,
        capacity_violations=capacity_violations,
        schema_violations=schema_violations,
        diagnostics=diagnostics,
    )
