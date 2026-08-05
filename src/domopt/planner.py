"""Planner-facing comparison of a validated solution with the default policy."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .baselines import solve_default_baseline
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
    for order_id, group in lines.groupby("order_id", sort=False):
        result[str(order_id)] = {
            "requested_cases": float(group["demand_cases"].sum()),
            "fulfilled_cases": float(group["fulfilled_cases"].sum()),
            "fulfilled_value": float(
                (group["unit_value"] * group["fulfilled_cases"]).sum()
            ),
            "penalty_cost": float(
                (
                    group["penalty_per_unfilled_case"]
                    * group["unfulfilled_cases"]
                ).sum()
            ),
        }
    return result


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
        rows.append(
            {
                "order_id": order_id,
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
                "requested_delivery_date": order["requested_delivery_date"],
                "recommended_pgi_date": selected_row["selected_pgi_date"],
                "reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values(
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
                "| Order | Default | Recommended | Fill change | Net change |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in priority.itertuples(index=False):
            lines.append(
                f"| {row.order_id} | {row.default_dc} | {row.recommended_dc} | "
                f"{row.fill_uplift_cases:+.0f} | {row.net_objective_change:+.2f} |"
            )
    lines.extend(
        [
            "",
            "Every recommendation passed assignment, demand, eligibility, inventory, "
            "capacity, and minimum-divert checks. Review the detailed CSV before release; "
            "it contains order and DC identifiers.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, markdown_path
