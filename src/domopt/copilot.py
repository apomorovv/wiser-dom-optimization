"""Rules-based explanations for aggregate challenge experiment results."""

from __future__ import annotations

import pandas as pd

SAFE_REQUIRED_COLUMNS = {"experiment", "method", "feasible"}
FORBIDDEN_IDENTIFIER_FRAGMENTS = {
    "address",
    "candidate_id",
    "customer",
    "dc_id",
    "delivery_number",
    "load_id",
    "material",
    "order_id",
    "plant",
    "sku_id",
    "zip",
}


def validate_copilot_data(results: pd.DataFrame) -> None:
    missing = SAFE_REQUIRED_COLUMNS - set(results.columns)
    if missing:
        raise ValueError(f"Experiment results are missing columns: {sorted(missing)}")
    forbidden = {
        column
        for column in results.columns
        if column.lower().endswith("_id")
        or any(
            fragment in column.lower()
            for fragment in FORBIDDEN_IDENTIFIER_FRAGMENTS
        )
    } - {"dataset_id", "experiment_id"}
    if forbidden:
        raise ValueError(
            "The copilot accepts aggregate results only; remove identifier columns: "
            f"{sorted(forbidden)}"
        )


def _number(value: object) -> str:
    if pd.isna(value):
        return "not reported"
    return f"{float(value):,.3f}"


def answer_experiment_question(question: str, results: pd.DataFrame) -> str:
    """Answer common planner questions without an external model or data transfer."""

    validate_copilot_data(results)
    text = question.strip().lower()
    feasibility = results["feasible"]
    if feasibility.dtype == bool:
        feasible_mask = feasibility.fillna(False)
    else:
        feasible_mask = feasibility.astype(str).str.lower().isin({"true", "1"})
    feasible = results.loc[feasible_mask].copy()
    if feasible.empty:
        return "No feasible run is available, so the copilot cannot compare methods."

    if any(term in text for term in ["quantum advantage", "qpu", "is it quantum"]):
        return (
            "No quantum advantage is established. The suite includes a local "
            "constraint-preserving QAOA statevector simulation; simulated annealing remains "
            "quantum-inspired, and exact recourse/validation remain classical. A hardware "
            "claim needs approved IBM QPU runs, matched end-to-end budgets, repeated "
            "trials, and uncertainty bounds."
        )

    if any(term in text for term in ["best", "recommend", "winner"]):
        # Raw objective totals are comparable only on an identical instance and
        # economic scale. The common solver comparison is the sole default scope.
        comparable = feasible.loc[
            feasible["experiment"] == "solver_comparison"
        ].dropna(subset=["objective_value"])
        if comparable.empty:
            return (
                "No common-instance solver comparison is loaded. Choose one experiment "
                "and level before asking for a winner; raw objectives across subset sizes "
                "or penalty scales are not comparable."
            )
        quality_leader = comparable.sort_values(
            ["objective_value", "runtime_seconds"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        best_objective = float(quality_leader["objective_value"])
        tolerance = 1e-8 * max(1.0, abs(best_objective))
        competitive = comparable.loc[
            comparable["objective_value"].astype(float) >= best_objective - tolerance
        ]
        operational = competitive.sort_values(
            "runtime_seconds", kind="mergesort"
        ).iloc[0]
        certificate = comparable.loc[
            comparable["method"].eq("classical")
            & comparable.get(
                "optimality_gap", pd.Series(index=comparable.index, dtype=float)
            ).fillna(float("inf"))
            .le(1e-9)
        ]
        proof = (
            " The full MILP supplies a zero-gap certificate on this instance."
            if not certificate.empty
            else " No zero-gap full-MILP certificate is loaded for this instance."
        )
        return (
            f"On the common solver-comparison instance, the highest feasible objective is "
            f"{_number(quality_leader['objective_value'])}, from "
            f"{quality_leader['method']} ({quality_leader.get('level', '')}). "
            f"Among methods tied within numerical tolerance, {operational['method']} is "
            f"fastest at {_number(operational['runtime_seconds'])} s and is the evidence-based "
            f"operational choice.{proof}"
        )

    if any(term in text for term in ["noise", "seed", "robust"]):
        rows = feasible.loc[feasible["experiment"] == "qubo_coefficient_noise"]
        if rows.empty:
            return "No seed/noise experiment rows are loaded."
        improvement = rows["hybrid_improvement"].dropna()
        return (
            f"Across {len(rows)} local seed/noise runs, hybrid improvement ranged from "
            f"{_number(improvement.min())} to {_number(improvement.max())}. All loaded rows "
            "are feasible. This is simulator/coefficient robustness, not physical QPU noise."
        )

    if any(term in text for term in ["pareto", "prun"]):
        rows = feasible.loc[feasible["experiment"] == "pareto_pruning_ablation"]
        if rows.empty:
            return "No Pareto-pruning ablation rows are loaded."
        if "pareto_pruning" in rows:
            grouped = rows.groupby("pareto_pruning", as_index=False).agg(
                candidate_count=("candidate_count", "median"),
                objective_value=("objective_value", "median"),
                runtime_seconds=("runtime_seconds", "median"),
            )
            summary = "; ".join(
                f"{'with' if row.pareto_pruning else 'without'} pruning: "
                f"{int(row.candidate_count)} candidates, median objective "
                f"{_number(row.objective_value)}, {_number(row.runtime_seconds)} s"
                for row in grouped.itertuples(index=False)
            )
        else:
            summary = "; ".join(
                f"{row.level}: {int(row.candidate_count)} candidates, "
                f"objective {_number(row.objective_value)}, "
                f"{_number(row.runtime_seconds)} s"
                for row in rows.itertuples(index=False)
            )
        return (
            f"Pareto ablation — {summary}. Pruning is beneficial only when it reduces search "
            "width/runtime without lowering the validated objective on that instance. It is "
            "a heuristic, not a globally lossless dominance rule."
        )

    if any(term in text for term in ["batch", "conflict", "random"]):
        rows = feasible.loc[feasible["experiment"] == "batch_strategy_ablation"]
        if rows.empty:
            return "No batching ablation rows are loaded."
        strategy_column = "batch_strategy" if "batch_strategy" in rows else "level"
        summary = rows.groupby(strategy_column, as_index=False).agg(
            objective_value=("objective_value", "median"),
            runtime_seconds=("runtime_seconds", "median"),
        )
        best = summary.sort_values(
            ["objective_value", "runtime_seconds"], ascending=[False, True]
        ).iloc[0]
        return (
            f"The stronger median loaded batching result is {best[strategy_column]}: "
            f"objective {_number(best['objective_value'])} in "
            f"{_number(best['runtime_seconds'])} s across the loaded seeds."
        )

    if any(term in text for term in ["inventory", "shock", "shortage"]):
        rows = feasible.loc[feasible["experiment"] == "inventory_shock"]
        if rows.empty:
            return "No inventory-shock rows are loaded."
        minimum_fill = rows["case_fill_rate"].min()
        maximum_fill = rows["case_fill_rate"].max()
        return (
            f"Across the loaded inventory shocks, case fill ranges from "
            f"{minimum_fill:.2%} to {maximum_fill:.2%}. Compare methods at each identical "
            "shock level, and distinguish fixed-routing recourse from post-shock "
            "reoptimization; do not compare raw objectives across different penalty scales."
        )

    if any(term in text for term in ["scale", "runtime", "large"]):
        rows = feasible.loc[feasible["experiment"] == "size_scaling"]
        if rows.empty:
            return "No size-scaling rows are loaded."
        largest_count = int(rows["order_count"].max())
        largest = rows.loc[rows["order_count"].eq(largest_count)]
        summaries = []
        for method, group in largest.groupby("method", sort=True):
            summaries.append(
                f"{method}: median {_number(group['runtime_seconds'].median())} s"
            )
        return (
            f"The largest loaded scale contains {largest_count} actual orders. "
            + "; ".join(summaries)
            + ". Compare medians across repeated rows and do not infer missing methods failed; "
            "some methods are intentionally capped below the largest scale."
        )

    return (
        "Ask about the best validated method, size scaling, penalty sensitivity, candidate "
        "counts, inventory shocks, seed/noise robustness, Pareto pruning, conflict batching, "
        "or why the current evidence does not establish quantum advantage."
    )
