"""Streamlit planner copilot for aggregate, privacy-safe DOM experiment evidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from domopt.copilot import answer_experiment_question, validate_copilot_data

st.set_page_config(
    page_title="WISER DOM Planner Copilot",
    page_icon="📦",
    layout="wide",
)
st.title("WISER DOM Planner Copilot")
st.caption(
    "Explore validated aggregate experiments. No external LLM is called and no raw "
    "order, customer, SKU, DC, ZIP, or lane data is accepted."
)


@st.cache_data
def read_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


uploaded = st.sidebar.file_uploader("Aggregate experiment CSV", type=["csv"])
default_path = Path("runs/challenge-study/aggregate_results.csv")
if uploaded is not None:
    results = pd.read_csv(uploaded)
    source_label = uploaded.name
elif default_path.exists():
    results = read_results(str(default_path))
    source_label = str(default_path)
else:
    st.info(
        "Run `python scripts/run_challenge_study.py --bundle-dir /approved/path "
        "--profile full` or upload its aggregate CSV."
    )
    st.stop()

try:
    validate_copilot_data(results)
except ValueError as error:
    st.error(str(error))
    st.stop()

st.sidebar.success(f"Loaded {len(results)} aggregate rows from {source_label}")
experiments = sorted(results["experiment"].dropna().astype(str).unique())
selected_experiments = st.sidebar.multiselect(
    "Experiments", experiments, default=experiments
)
visible = results.loc[results["experiment"].isin(selected_experiments)].copy()

overview, explorer, copilot, methodology = st.tabs(
    ["Overview", "Experiment explorer", "Copilot", "Methodology"]
)

with overview:
    feasible = visible["feasible"].fillna(False).astype(bool)
    columns = st.columns(4)
    columns[0].metric("Runs", len(visible))
    columns[1].metric("Feasible", f"{feasible.mean():.1%}" if len(visible) else "—")
    columns[2].metric(
        "Largest instance",
        int(visible["order_count"].max()) if "order_count" in visible and len(visible) else "—",
    )
    columns[3].metric(
        "Largest local QUBO",
        int(visible["maximum_qubo_variables"].max())
        if "maximum_qubo_variables" in visible
        and visible["maximum_qubo_variables"].notna().any()
        else "—",
    )

    core = visible.loc[visible["experiment"] == "solver_comparison"]
    if not core.empty:
        figure = px.bar(
            core,
            x="method",
            y="objective_value",
            color="feasible",
            hover_data=["case_fill_rate", "runtime_seconds", "optimality_gap"],
            title="Common-objective solver comparison",
        )
        st.plotly_chart(figure, width="stretch")

with explorer:
    experiment = st.selectbox("Experiment", experiments)
    subset = results.loc[results["experiment"] == experiment].copy()
    metric_options = [
        column
        for column in [
            "objective_value",
            "case_fill_rate",
            "hybrid_improvement",
            "runtime_seconds",
            "candidate_count",
            "maximum_qubo_variables",
        ]
        if column in subset.columns
    ]
    metric = st.selectbox("Metric", metric_options)
    figure = px.bar(
        subset,
        x="level",
        y=metric,
        color="method",
        hover_data=["feasible", "configuration"],
        title=f"{experiment}: {metric}",
    )
    st.plotly_chart(figure, width="stretch")
    st.dataframe(
        subset.drop(columns=["configuration"], errors="ignore"),
        width="stretch",
        hide_index=True,
    )

with copilot:
    st.write(
        "This deterministic copilot explains only the loaded aggregate evidence; it does "
        "not send data to an external model."
    )
    question = st.text_input(
        "Question",
        placeholder="Did Pareto pruning help?",
    )
    quick_questions = st.pills(
        "Try one",
        [
            "Which method is best?",
            "Did Pareto pruning help?",
            "Conflict or random batches?",
            "How robust is QUBO noise?",
            "Is this quantum advantage?",
        ],
    )
    prompt = question or quick_questions
    if prompt:
        st.markdown(answer_experiment_question(prompt, visible))

with methodology:
    st.markdown(
        """
        - Candidate assignments are proposed by a bounded QUBO neighborhood.
        - Exact MILP recourse determines SKU quantities and enforces inventory, dock,
          capacity, penalty, diversion, and load-cohesion rules.
        - An independent validator recomputes every hard constraint and objective component.
        - A move is accepted only when feasible and strictly better than the incumbent.
        - Seed/coefficient perturbations are local simulator robustness tests, not QPU noise.
        - Physical quantum advantage is not claimed.
        """
    )
