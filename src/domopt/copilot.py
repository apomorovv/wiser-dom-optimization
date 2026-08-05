"""Rules-based explanations for aggregate challenge experiment results."""

from __future__ import annotations

import pandas as pd

SAFE_REQUIRED_COLUMNS = {"experiment", "method", "feasible"}
FORBIDDEN_IDENTIFIER_COLUMNS = {
    "order_id",
    "sku_id",
    "dc_id",
    "candidate_id",
    "customer_number",
    "customer_name",
    "zip_code",
}


def validate_copilot_data(results: pd.DataFrame) -> None:
    missing = SAFE_REQUIRED_COLUMNS - set(results.columns)
    if missing:
        raise ValueError(f"Experiment results are missing columns: {sorted(missing)}")
    forbidden = {
        column for column in results.columns if column.lower() in FORBIDDEN_IDENTIFIER_COLUMNS
    }
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
    feasible = results.loc[results["feasible"].fillna(False).astype(bool)].copy()
    if feasible.empty:
        return "No feasible run is available, so the copilot cannot compare methods."

    if any(term in text for term in ["quantum advantage", "qpu", "is it quantum"]):
        return (
            "No quantum advantage is established. The current quantum-labelled study uses "
            "local QUBO coefficient perturbations and simulated annealing; exact MILP recourse "
            "and the validator remain classical. A QPU claim needs approved hardware runs, "
            "embedding overhead, matched time budgets, repeated trials, and uncertainty bounds."
        )

    if any(term in text for term in ["best", "recommend", "winner"]):
        comparable = feasible.dropna(subset=["objective_value"])
        if comparable.empty:
            return "Feasible runs exist, but none reports a comparable objective value."
        best = comparable.sort_values(
            ["objective_value", "runtime_seconds"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        return (
            f"The highest recorded feasible objective is {_number(best['objective_value'])}, "
            f"from {best['method']} in {best['experiment']} ({best.get('level', '')}). "
            "Treat exact MILP as the proof benchmark when its gap is zero; use hybrid when a "
            "bounded-time, feasibility-preserving search is the operational goal."
        )

    if any(term in text for term in ["noise", "seed", "robust"]):
        rows = feasible.loc[feasible["experiment"] == "quantum_seed_noise"]
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
        summary = "; ".join(
            f"{row.level}: {int(row.candidate_count)} candidates, "
            f"objective {_number(row.objective_value)}, {_number(row.runtime_seconds)} s"
            for row in rows.itertuples(index=False)
        )
        return (
            f"Pareto ablation — {summary}. Pruning is beneficial only when it reduces search "
            "width/runtime without lowering the validated objective."
        )

    if any(term in text for term in ["batch", "conflict", "random"]):
        rows = feasible.loc[feasible["experiment"] == "batch_strategy_ablation"]
        if rows.empty:
            return "No batching ablation rows are loaded."
        best = rows.sort_values(
            ["objective_value", "runtime_seconds"], ascending=[False, True]
        ).iloc[0]
        return (
            f"The stronger loaded batching result is {best['level']}: objective "
            f"{_number(best['objective_value'])} in {_number(best['runtime_seconds'])} s. "
            "Use multiple seeds before treating a small one-run difference as stable."
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
            "shock level; do not compare raw objectives across different penalty scales."
        )

    if any(term in text for term in ["scale", "runtime", "large"]):
        rows = feasible.loc[feasible["experiment"] == "size_scaling"]
        if rows.empty:
            return "No size-scaling rows are loaded."
        largest = rows.sort_values("order_count").iloc[-1]
        return (
            f"The largest loaded run contains {int(largest['order_count'])} actual orders. "
            f"Its {largest['method']} runtime is {_number(largest['runtime_seconds'])} s and "
            f"the maximum local QUBO width is {_number(largest.get('maximum_qubo_variables'))}."
        )

    return (
        "Ask about the best validated method, size scaling, penalty sensitivity, candidate "
        "counts, inventory shocks, seed/noise robustness, Pareto pruning, conflict batching, "
        "or why the current evidence does not establish quantum advantage."
    )
