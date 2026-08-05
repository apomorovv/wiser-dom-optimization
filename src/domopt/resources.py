"""Common inventory and capacity accounting.

Keeping this logic in one place prevents the baseline, hybrid residual model, and
validator from silently using different definitions of dock, pallet, or case picks.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from math import floor

import pandas as pd

from .schemas import ProblemData, Solution

SUPPORTED_CAPACITY_RESOURCES = {
    "dock",
    "throughput_cases",
    "case_pick",
    "pallet_pick",
    "weight",
    "volume",
}


def cases_per_pallet(line: Mapping[str, object] | pd.Series) -> int | None:
    value = line.get("cases_per_pallet")
    if value is None or pd.isna(value):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def split_pick_quantities(quantity: float, per_pallet: int) -> tuple[int, int]:
    """Return ``(full_pallets, loose_cases)`` for an integer case quantity."""

    cases = round(float(quantity))
    if cases < 0 or per_pallet <= 0:
        raise ValueError(
            "Pick decomposition requires nonnegative cases and a positive pallet size"
        )
    return divmod(cases, int(per_pallet))


def candidate_fixed_consumption(
    candidate: Mapping[str, object] | pd.Series,
    resource: str,
) -> float:
    if resource == "dock":
        value = candidate.get("dock_units", 1.0)
        return 1.0 if value is None or pd.isna(value) else float(value)
    column = f"fixed_{resource}"
    value = candidate.get(column, 0.0)
    return 0.0 if value is None or pd.isna(value) else float(value)


def line_variable_consumption(
    line: Mapping[str, object] | pd.Series,
    quantity: float,
    resource: str,
    *,
    split_picks: bool,
) -> float:
    cases = round(float(quantity))
    if resource == "throughput_cases":
        return float(cases)
    if resource in {"case_pick", "pallet_pick"}:
        per_pallet = cases_per_pallet(line)
        if split_picks:
            if per_pallet is None:
                raise ValueError(
                    f"Resource {resource!r} requires order_lines.cases_per_pallet"
                )
            pallets, loose = split_pick_quantities(cases, per_pallet)
            return float(loose if resource == "case_pick" else pallets)
        if resource == "pallet_pick":
            if per_pallet is None:
                raise ValueError("pallet_pick requires order_lines.cases_per_pallet")
            return float(floor(cases / per_pallet))
        return float(cases)
    if resource == "weight":
        if "unit_weight" not in line or pd.isna(line.get("unit_weight")):
            raise ValueError("weight capacity requires order_lines.unit_weight")
        return float(cases) * float(line["unit_weight"])
    if resource == "volume":
        if "unit_volume" not in line or pd.isna(line.get("unit_volume")):
            raise ValueError("volume capacity requires order_lines.unit_volume")
        return float(cases) * float(line["unit_volume"])
    if resource == "dock":
        return 0.0
    raise ValueError(f"Unsupported capacity resource {resource!r}")


def uses_split_pick_accounting(problem: ProblemData) -> bool:
    resources = set(problem.capacities.get("resource", pd.Series(dtype=str)).astype(str))
    configured = str(problem.metadata.get("pick_capacity_mode", "auto")).lower()
    if configured == "pallet_case":
        return True
    if configured == "cases":
        return False
    return "pallet_pick" in resources


def solution_capacity_usage(
    problem: ProblemData,
    solution: Solution,
) -> dict[tuple[str, pd.Timestamp, str], float]:
    """Compute exact-date resource use for a complete or partial solution."""

    split_picks = uses_split_pick_accounting(problem)
    capacities = problem.capacities
    resources = set(capacities.get("resource", pd.Series(dtype=str)).astype(str))
    unsupported = resources - SUPPORTED_CAPACITY_RESOURCES
    if unsupported:
        raise ValueError(f"Unsupported capacity resources: {sorted(unsupported)}")

    usage: defaultdict[tuple[str, pd.Timestamp, str], float] = defaultdict(float)
    candidates = problem.candidates.set_index("candidate_id", drop=False)
    assigned = solution.assignments.loc[
        ~solution.assignments["is_unassigned"].astype(bool)
    ]
    for row in assigned.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        if candidate_id not in candidates.index:
            raise ValueError(
                f"Cannot compute capacity for unknown candidate {candidate_id!r}"
            )
        candidate = candidates.loc[candidate_id]
        if isinstance(candidate, pd.DataFrame):
            candidate = candidate.iloc[0]
        dc_id = str(candidate["dc_id"])
        date = pd.Timestamp(candidate["pgi_date"])
        for resource in resources:
            fixed = candidate_fixed_consumption(candidate, resource)
            if fixed:
                usage[(dc_id, date, resource)] += fixed

    lines = problem.order_lines.set_index(["order_id", "sku_id"], drop=False)
    for row in solution.fulfillment.itertuples(index=False):
        quantity = round(float(row.fulfilled_cases))
        if quantity <= 0 or pd.isna(row.selected_dc) or pd.isna(row.selected_pgi_date):
            continue
        key = (str(row.order_id), str(row.sku_id))
        if key not in lines.index:
            continue
        line = lines.loc[key]
        if isinstance(line, pd.DataFrame):
            line = line.iloc[0]
        dc_id = str(row.selected_dc)
        date = pd.Timestamp(row.selected_pgi_date)
        for resource in resources - {"dock"}:
            usage[(dc_id, date, resource)] += line_variable_consumption(
                line,
                quantity,
                resource,
                split_picks=split_picks,
            )
    return dict(usage)


def solution_inventory_usage(
    problem: ProblemData,
    solution: Solution,
) -> dict[tuple[str, str, pd.Timestamp], float]:
    """Return consumption through every projected-ATP checkpoint."""

    result: defaultdict[tuple[str, str, pd.Timestamp], float] = defaultdict(float)
    checkpoints = {
        (str(dc), str(sku)): sorted(pd.Timestamp(date) for date in group["date"])
        for (dc, sku), group in problem.inventory.groupby(["dc_id", "sku_id"], sort=False)
    }
    for row in solution.fulfillment.itertuples(index=False):
        quantity = float(row.fulfilled_cases)
        if quantity <= 0 or pd.isna(row.selected_dc) or pd.isna(row.selected_pgi_date):
            continue
        dc_id = str(row.selected_dc)
        sku_id = str(row.sku_id)
        pgi_date = pd.Timestamp(row.selected_pgi_date)
        for checkpoint in checkpoints.get((dc_id, sku_id), []):
            if pgi_date <= checkpoint:
                result[(dc_id, sku_id, checkpoint)] += quantity
    return dict(result)
