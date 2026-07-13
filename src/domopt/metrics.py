"""Common business and computational metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .objective import evaluate_solution
from .schemas import ProblemData, Solution
from .validation import validate_solution


def compute_metrics(problem: ProblemData, solution: Solution) -> dict[str, Any]:
    validation = validate_solution(problem, solution)
    objective = evaluate_solution(problem, solution)

    fulfillment = solution.fulfillment.copy()
    lines = problem.order_lines.copy()
    merged = lines.merge(
        fulfillment[["order_id", "sku_id", "fulfilled_cases"]],
        on=["order_id", "sku_id"],
        how="left",
        validate="one_to_one",
    )

    total_cases = float(merged["demand_cases"].sum())
    fulfilled_cases = float(merged["fulfilled_cases"].sum())
    total_requested_value = float((merged["unit_value"] * merged["demand_cases"]).sum())

    case_fill_rate = fulfilled_cases / total_cases if total_cases > 0 else 1.0
    value_fill_rate = (
        objective.fulfilled_value / total_requested_value
        if total_requested_value > 0
        else 1.0
    )

    assignments = solution.assignments
    reassigned = int(
        ((~assignments["is_unassigned"].astype(bool)) & assignments["is_divert"].astype(bool)).sum()
    )
    unassigned = int(assignments["is_unassigned"].astype(bool).sum())

    metrics: dict[str, Any] = {
        "method": solution.method,
        "dataset_id": problem.metadata.get("dataset_id", "unknown"),
        "feasible": bool(validation.is_feasible),
        **objective.to_dict(),
        "fulfilled_cases": fulfilled_cases,
        "requested_cases": total_cases,
        "case_fill_rate": float(case_fill_rate),
        "value_fill_rate": float(value_fill_rate),
        "reassigned_orders": reassigned,
        "unassigned_orders": unassigned,
        "runtime_seconds": float(solution.runtime_seconds),
        "best_bound": solution.metadata.get("best_bound"),
        "optimality_gap": solution.metadata.get("optimality_gap"),
        "violations": validation.to_dict(),
    }
    return metrics

