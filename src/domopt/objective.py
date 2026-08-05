"""Single authoritative objective calculation for every solver."""

from __future__ import annotations

from .penalties import total_penalty
from .schemas import ObjectiveBreakdown, ProblemData, Solution


class ObjectiveEvaluationError(ValueError):
    """Raised when a solution cannot be evaluated unambiguously."""


def evaluate_solution(problem: ProblemData, solution: Solution) -> ObjectiveBreakdown:
    """Recompute fulfilled value, unmet penalty, and shipping cost.

    This function deliberately ignores a solver's reported objective. It evaluates the
    canonical fulfillment and assignment tables against canonical problem parameters.
    """

    lines = problem.order_lines[
        [
            "order_id",
            "sku_id",
            "demand_cases",
            "unit_value",
            "penalty_per_unfilled_case",
        ]
    ].copy()
    fulfillment = solution.fulfillment.copy()

    required_fulfillment = {
        "order_id",
        "sku_id",
        "fulfilled_cases",
        "unfulfilled_cases",
    }
    missing = required_fulfillment - set(fulfillment.columns)
    if missing:
        raise ObjectiveEvaluationError(
            f"Fulfillment table is missing columns: {sorted(missing)}"
        )
    if fulfillment.duplicated(["order_id", "sku_id"]).any():
        raise ObjectiveEvaluationError(
            "Fulfillment table must contain one row per (order_id, sku_id)"
        )

    merged = lines.merge(
        fulfillment[
            ["order_id", "sku_id", "fulfilled_cases", "unfulfilled_cases"]
        ],
        on=["order_id", "sku_id"],
        how="left",
        validate="one_to_one",
    )
    if merged[["fulfilled_cases", "unfulfilled_cases"]].isna().any().any():
        missing_lines = merged.loc[
            merged["fulfilled_cases"].isna(), ["order_id", "sku_id"]
        ].to_dict("records")
        raise ObjectiveEvaluationError(
            f"Solution is missing fulfillment rows: {missing_lines[:5]}"
        )

    fulfilled_value = float(
        (merged["unit_value"].astype(float) * merged["fulfilled_cases"].astype(float)).sum()
    )
    penalty_cost = total_penalty(problem, merged)

    assignments = solution.assignments.copy()
    required_assignments = {"order_id", "candidate_id", "is_unassigned"}
    missing_assignment = required_assignments - set(assignments.columns)
    if missing_assignment:
        raise ObjectiveEvaluationError(
            f"Assignment table is missing columns: {sorted(missing_assignment)}"
        )
    if assignments["order_id"].duplicated().any():
        raise ObjectiveEvaluationError("Assignment table must contain one row per order")

    selected = assignments.loc[~assignments["is_unassigned"].astype(bool)].copy()
    if selected.empty:
        shipping_cost = 0.0
    else:
        candidate_cost = problem.candidates[
            ["candidate_id", "order_id", "shipping_cost"]
        ]
        selected = selected.merge(
            candidate_cost,
            on=["candidate_id", "order_id"],
            how="left",
            validate="one_to_one",
        )
        if selected["shipping_cost"].isna().any():
            bad = selected.loc[
                selected["shipping_cost"].isna(), ["order_id", "candidate_id"]
            ].to_dict("records")
            raise ObjectiveEvaluationError(f"Unknown selected candidates: {bad[:5]}")
        shipping_cost = float(selected["shipping_cost"].astype(float).sum())

    return ObjectiveBreakdown(
        fulfilled_value=fulfilled_value,
        penalty_cost=penalty_cost,
        shipping_cost=shipping_cost,
    )


