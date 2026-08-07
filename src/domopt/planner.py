"""Planner-facing comparison of a validated solution with the default policy."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .baselines import solve_default_baseline
from .penalties import build_penalty_context, order_penalty
from .resources import solution_capacity_usage, solution_inventory_usage
from .schemas import ProblemData, Solution
from .validation import validate_solution


def _candidate_costs(problem: ProblemData) -> dict[str, float]:
    return (
        problem.candidates.set_index("candidate_id")["shipping_cost"]
        .astype(float)
        .to_dict()
    )


def _order_economics(
    problem: ProblemData,
    solution: Solution,
) -> dict[str, dict[str, float]]:
    fulfillment = solution.fulfillment.copy()
    lines = problem.order_lines.merge(
        fulfillment[["order_id", "sku_id", "fulfilled_cases", "unfulfilled_cases"]],
        on=["order_id", "sku_id"],
        how="left",
        validate="one_to_one",
    )
    result: dict[str, dict[str, float]] = {}
    penalty_context = build_penalty_context(problem)
    for order_id, group in lines.groupby("order_id", sort=False):
        result[str(order_id)] = {
            "requested_cases": float(group["demand_cases"].sum()),
            "fulfilled_cases": float(group["fulfilled_cases"].sum()),
            "fulfilled_value": float(
                (group["unit_value"] * group["fulfilled_cases"]).sum()
            ),
            "penalty_cost": order_penalty(
                problem,
                str(order_id),
                group,
                context=penalty_context,
            ),
        }
    return result


def _screened_alternative(
    problem: ProblemData,
    order_id: str,
    selected_id: object,
) -> pd.Series | None:
    """Return the strongest eligible row-level alternative for planner review.

    This is deliberately labeled a screen, not a feasible recommendation: shared
    resources and assignment-group cohesion require a new joint optimization.
    """

    candidates = problem.candidates.loc[
        (problem.candidates["order_id"].astype(str) == order_id)
        & problem.candidates["eligible"].astype(bool)
    ].copy()
    if not pd.isna(selected_id):
        candidates = candidates.loc[
            candidates["candidate_id"].astype(str) != str(selected_id)
        ]
    if candidates.empty:
        return None
    candidates["_estimated_value"] = (
        pd.to_numeric(candidates["estimated_fulfilled_value"], errors="coerce").fillna(
            0.0
        )
        if "estimated_fulfilled_value" in candidates
        else 0.0
    )
    return candidates.sort_values(
        ["_estimated_value", "shipping_cost", "pgi_date", "candidate_id"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).iloc[0]


def _binding_constraint_note(
    problem: ProblemData,
    order_id: str,
    selected_candidate: pd.Series | None,
    inventory_usage: dict[tuple[str, str, pd.Timestamp], float],
    capacity_usage: dict[tuple[str, pd.Timestamp, str], float],
) -> str:
    if selected_candidate is None:
        return "Unassigned after joint objective and feasibility evaluation."
    dc_id = str(selected_candidate["dc_id"])
    pgi_date = pd.Timestamp(selected_candidate["pgi_date"])
    slacks: list[tuple[float, str]] = []
    skus = set(
        problem.order_lines.loc[
            problem.order_lines["order_id"].astype(str) == order_id, "sku_id"
        ].astype(str)
    )
    for row in problem.inventory.loc[
        problem.inventory["dc_id"].astype(str).eq(dc_id)
        & problem.inventory["sku_id"].astype(str).isin(skus)
        & problem.inventory["date"].ge(pgi_date)
    ].itertuples(index=False):
        key = (dc_id, str(row.sku_id), pd.Timestamp(row.date))
        limit = float(row.cumulative_available_cases)
        slack = limit - float(inventory_usage.get(key, 0.0))
        slacks.append((slack / max(1.0, limit), f"ATP {row.sku_id} on {row.date}"))
    for row in problem.capacities.loc[
        problem.capacities["dc_id"].astype(str).eq(dc_id)
        & problem.capacities["date"].eq(pgi_date)
    ].itertuples(index=False):
        key = (dc_id, pgi_date, str(row.resource))
        limit = float(row.capacity)
        slack = limit - float(capacity_usage.get(key, 0.0))
        slacks.append((slack / max(1.0, limit), f"{row.resource} on {row.date}"))
    if not slacks:
        return "No modeled inventory/capacity row applies; objective trade-off governs."
    relative_slack, label = min(slacks, key=lambda item: (item[0], item[1]))
    if relative_slack <= 1e-7:
        return f"Binding modeled resource: {label}."
    return f"Tightest modeled resource: {label} ({relative_slack:.1%} slack)."


def build_planner_table(
    problem: ProblemData,
    solution: Solution,
    *,
    default_solution: Solution | None = None,
) -> pd.DataFrame:
    """Explain each recommendation relative to the feasible default baseline.

    This table intentionally contains order and DC identifiers. Keep it in the
    approved challenge environment unless those identifiers are synthetic.
    """

    baseline = default_solution or solve_default_baseline(problem)
    for name, candidate in [("solution", solution), ("default baseline", baseline)]:
        validation = validate_solution(problem, candidate)
        if not validation.is_feasible:
            raise ValueError(f"Cannot build planner view from infeasible {name}")

    candidate_cost = _candidate_costs(problem)
    selected = solution.assignments.set_index("order_id")
    default = baseline.assignments.set_index("order_id")
    selected_economics = _order_economics(problem, solution)
    default_economics = _order_economics(problem, baseline)
    order_lookup = problem.orders.set_index("order_id")
    candidate_lookup = problem.candidates.set_index("candidate_id", drop=False)
    inventory_usage = solution_inventory_usage(problem, solution)
    capacity_usage = solution_capacity_usage(problem, solution)

    rows: list[dict[str, object]] = []
    for order_id in sorted(problem.orders["order_id"].astype(str)):
        selected_row = selected.loc[order_id]
        default_row = default.loc[order_id]
        selected_values = selected_economics[order_id]
        default_values = default_economics[order_id]
        selected_id = selected_row["candidate_id"]
        default_id = default_row["candidate_id"]
        selected_shipping = (
            0.0 if pd.isna(selected_id) else candidate_cost[str(selected_id)]
        )
        default_shipping = 0.0 if pd.isna(default_id) else candidate_cost[str(default_id)]
        fill_uplift = (
            selected_values["fulfilled_cases"] - default_values["fulfilled_cases"]
        )
        value_uplift = (
            selected_values["fulfilled_value"] - default_values["fulfilled_value"]
        )
        penalty_avoided = (
            default_values["penalty_cost"] - selected_values["penalty_cost"]
        )
        shipping_change = selected_shipping - default_shipping
        net_change = value_uplift + penalty_avoided - shipping_change

        if bool(selected_row["is_unassigned"]):
            decision = "UNASSIGNED"
            reason = "No assignment selected after feasibility and objective evaluation."
        elif bool(selected_row["is_divert"]):
            decision = "DIVERT"
            reason = (
                f"Validated net objective change {net_change:+.2f}; "
                f"case-fill change {fill_uplift:+.0f}."
            )
        else:
            decision = "KEEP DEFAULT"
            reason = "Default DC remains the best accepted feasible assignment."

        order = order_lookup.loc[order_id]
        assignment_group = str(order.get("assignment_group", order_id))
        selected_candidate = None
        if not pd.isna(selected_id) and str(selected_id) in candidate_lookup.index:
            selected_candidate = candidate_lookup.loc[str(selected_id)]
            if isinstance(selected_candidate, pd.DataFrame):
                selected_candidate = selected_candidate.iloc[0]
        arrival_date = (
            pd.NaT
            if selected_candidate is None
            else pd.Timestamp(selected_candidate.get("arrival_date"))
        )
        requested_date = pd.Timestamp(order["requested_delivery_date"])
        alternative = _screened_alternative(problem, order_id, selected_id)
        binding_note = _binding_constraint_note(
            problem,
            order_id,
            selected_candidate,
            inventory_usage,
            capacity_usage,
        )
        rows.append(
            {
                "order_id": order_id,
                "assignment_group": assignment_group,
                "decision": decision,
                "default_dc": str(order["default_dc"]),
                "recommended_dc": selected_row["selected_dc"],
                "requested_cases": selected_values["requested_cases"],
                "default_fill_cases": default_values["fulfilled_cases"],
                "recommended_fill_cases": selected_values["fulfilled_cases"],
                "fill_uplift_cases": fill_uplift,
                "penalty_avoided": penalty_avoided,
                "shipping_cost_change": shipping_change,
                "net_objective_change": net_change,
                "requested_delivery_date": requested_date,
                "recommended_pgi_date": selected_row["selected_pgi_date"],
                "expected_arrival_date": arrival_date,
                "on_time": bool(pd.notna(arrival_date) and arrival_date <= requested_date),
                "binding_constraint_note": binding_note,
                "screened_alternative_dc": (
                    None if alternative is None else str(alternative["dc_id"])
                ),
                "screened_alternative_pgi_date": (
                    pd.NaT if alternative is None else alternative["pgi_date"]
                ),
                "alternative_review_note": (
                    "No other eligible row-level candidate."
                    if alternative is None
                    else "Eligible screen only; reoptimize the full assignment group "
                    "and shared resources before acceptance."
                ),
                "reason": reason,
            }
        )
    table = pd.DataFrame(rows)
    table["group_fill_uplift_cases"] = table.groupby("assignment_group")[
        "fill_uplift_cases"
    ].transform("sum")
    table["group_net_objective_change"] = table.groupby("assignment_group")[
        "net_objective_change"
    ].transform("sum")
    leaders = table.groupby("assignment_group")["order_id"].transform("min")
    table["group_role"] = table["order_id"].eq(leaders).map(
        {True: "leader", False: "member"}
    )
    return table.sort_values(
        ["decision", "net_objective_change", "order_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )


def write_planner_artifacts(
    problem: ProblemData,
    solution: Solution,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write detailed CSV and a compact one-page Markdown planner view."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = build_planner_table(problem, solution)
    csv_path = output / "planner_recommendations.csv"
    markdown_path = output / "planner_view.md"
    table.to_csv(csv_path, index=False)

    diverted = int(table["decision"].eq("DIVERT").sum())
    diverted_groups = int(
        table.loc[table["decision"].eq("DIVERT"), "assignment_group"].nunique()
    )
    fill_uplift = float(table["fill_uplift_cases"].sum())
    net_change = float(table["net_objective_change"].sum())
    priority = table.loc[table["decision"].eq("DIVERT")].head(20)
    lines = [
        "# Planner recommendation",
        "",
        f"Validated method: `{solution.method}`.",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Orders reviewed | {len(table):,} |",
        f"| Recommended diversions | {diverted:,} |",
        f"| Diverted assignment groups | {diverted_groups:,} |",
        f"| Case-fill change vs default | {fill_uplift:+,.0f} |",
        f"| Net objective change vs default | {net_change:+,.2f} |",
        "",
        "## Priority diversions",
        "",
    ]
    if priority.empty:
        lines.append("No diversion was accepted over the feasible default policy.")
    else:
        lines.extend(
            [
                "| Order | Default | Recommended | Fill change | Net change | Constraint |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        for row in priority.itertuples(index=False):
            lines.append(
                f"| {row.order_id} | {row.default_dc} | {row.recommended_dc} | "
                f"{row.fill_uplift_cases:+.0f} | {row.net_objective_change:+.2f} | "
                f"{row.binding_constraint_note} |"
            )
    lines.extend(
        [
            "",
            (
                "Every recommendation passed assignment, demand, eligibility, inventory, "
                "capacity, and minimum-divert checks. Review the detailed CSV before release; "
                "it contains order and DC identifiers."
            ),
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, markdown_path
