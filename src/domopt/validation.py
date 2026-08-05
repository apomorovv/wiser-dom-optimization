"""Independent feasibility validation for solver outputs."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .resources import solution_capacity_usage
from .rules import minimum_divert_fulfillment
from .schemas import ProblemData, Solution, ValidationResult


_TOL = 1e-8


def _date(value: object) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return pd.NaT
    return pd.Timestamp(value)


def validate_solution(problem: ProblemData, solution: Solution) -> ValidationResult:
    assignment_violations: list[str] = []
    demand_violations: list[str] = []
    inventory_violations: list[str] = []
    eligibility_violations: list[str] = []
    capacity_violations: list[str] = []
    schema_violations: list[str] = []

    assignments = solution.assignments.copy()
    fulfillment = solution.fulfillment.copy()
    order_ids = set(problem.orders["order_id"])
    line_keys = set(
        map(tuple, problem.order_lines[["order_id", "sku_id"]].itertuples(index=False, name=None))
    )

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
        )

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

        if fulfilled < -_TOL or unfulfilled < -_TOL:
            demand_violations.append(f"negative quantities for line {key}")
        if not np.isclose(fulfilled, round(fulfilled), atol=_TOL):
            demand_violations.append(f"fulfilled_cases is not integer for line {key}")
        if not np.isclose(unfulfilled, round(unfulfilled), atol=_TOL):
            demand_violations.append(f"unfulfilled_cases is not integer for line {key}")
        demand = float(demands[key])
        if not np.isclose(fulfilled + unfulfilled, demand, atol=_TOL):
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
        for row in usage.itertuples(index=False):
            compatible = inventory[
                (inventory["dc_id"] == row.dc_id)
                & (inventory["sku_id"] == row.sku_id)
                & (inventory["date"] >= row.date)
            ]
            if compatible.empty:
                inventory_violations.append(
                    f"no cumulative inventory checkpoint covers dc={row.dc_id}, "
                    f"sku={row.sku_id}, date={row.date.date()}"
                )

        for inv in inventory.itertuples(index=False):
            consumed = usage[
                (usage["dc_id"] == inv.dc_id)
                & (usage["sku_id"] == inv.sku_id)
                & (usage["date"] <= inv.date)
            ]["fulfilled_cases"].sum()
            if consumed > float(inv.cumulative_available_cases) + _TOL:
                inventory_violations.append(
                    f"inventory exceeded at dc={inv.dc_id}, sku={inv.sku_id}, "
                    f"date={pd.Timestamp(inv.date).date()}: used={consumed}, "
                    f"available={inv.cumulative_available_cases}"
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
        try:
            capacity_usage = solution_capacity_usage(problem, solution)
        except ValueError as error:
            schema_violations.append(str(error))
            capacity_usage = {}
        for cap in capacities.itertuples(index=False):
            date = pd.Timestamp(cap.date)
            resource = str(cap.resource)
            capacity = float(cap.capacity)
            consumed = float(capacity_usage.get((str(cap.dc_id), date, resource), 0.0))

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
    )
