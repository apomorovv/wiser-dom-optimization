"""Interactive local cockpit for auditing, running, and explaining the DOM solver."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st

from domopt.classical import available_cpu_count, available_milp_backends
from domopt.hybrid import ExactLNSConfig, HybridConfig
from domopt.metrics import compute_metrics
from domopt.poc import (
    POC_INPUT_FILENAMES,
    PocConfig,
    audit_poc_bundle,
    load_poc_problem,
    select_shortage_subset,
)
from domopt.solver import SolverConfig, solve_dom
from domopt.validation import validate_solution

st.set_page_config(
    page_title="DOM Solver Cockpit",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #11221c; --muted: #607168; --mint: #d8f3e5; --lime: #d6f36c; }
    .stApp { background: linear-gradient(145deg, #f7faf6 0%, #eef5ef 60%, #e8f0eb 100%); }
    [data-testid="stSidebar"] { background: #10251d; }
    [data-testid="stSidebar"] * { color: #f3f8f4; }
    [data-testid="stMetric"] {
      background: rgba(255,255,255,.84); border: 1px solid #d9e4dc;
      padding: 1rem 1.1rem; border-radius: 16px; box-shadow: 0 8px 30px rgba(26,53,42,.05);
    }
    .hero {
      padding: 1.4rem 1.6rem; border-radius: 24px; color: white;
      background: radial-gradient(circle at 85% 0%, #436c57 0, #16372a 44%, #0e251c 100%);
      box-shadow: 0 20px 60px rgba(16,47,35,.18); margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: clamp(2rem, 4vw, 3.7rem); letter-spacing: -.045em; }
    .hero p { max-width: 760px; color: #d9e9df; font-size: 1.05rem; margin-bottom: .2rem; }
    .eyebrow { color: #d6f36c; font-size: .78rem; letter-spacing: .16em; font-weight: 700; }
    .pipeline { display: grid; grid-template-columns: repeat(4, 1fr); gap: .7rem; margin: .4rem 0 1.2rem; }
    .stage { background: rgba(255,255,255,.8); border: 1px solid #d7e4da; border-radius: 14px; padding: .8rem; }
    .stage b { color: #17392c; display: block; }
    .stage span { color: #687b70; font-size: .82rem; }
    .ok { color: #1f7a4f; font-weight: 700; }
    .warn { color: #a05a16; font-weight: 700; }
    @media (max-width: 900px) { .pipeline { grid-template-columns: repeat(2, 1fr); } }
    </style>
    """,
    unsafe_allow_html=True,
)


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


@st.cache_resource(show_spinner=False)
def _load_problem(bundle_dir: str) -> object:
    return load_poc_problem(
        Path(bundle_dir),
        config=PocConfig(pareto_prune=False),
    )


def _solution_archive(solution: object, metrics: dict[str, object], validation: object) -> bytes:
    payload = BytesIO()
    with ZipFile(payload, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("assignments.csv", solution.assignments.to_csv(index=False))
        archive.writestr("fulfillment.csv", solution.fulfillment.to_csv(index=False))
        archive.writestr(
            "metrics.json",
            json.dumps(metrics, indent=2, sort_keys=True, default=_json_default) + "\n",
        )
        archive.writestr(
            "validation.json",
            json.dumps(
                validation.to_dict(),
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            + "\n",
        )
        archive.writestr(
            "solver_metadata.json",
            json.dumps(
                solution.metadata,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            + "\n",
        )
    return payload.getvalue()


def _mode_description(mode: str) -> str:
    return {
        "fast": "Greedy whole-load routing, then exact fixed-policy quantity recourse.",
        "quality": "Adaptive exact MILP large-neighborhood search over sparse conflicts.",
        "hybrid": "Sampler-proposed local assignments with exact MILP feasibility recourse.",
    }[mode]


st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">WISER · DISTRIBUTED ORDER MANAGEMENT</div>
      <h1>Solver cockpit</h1>
      <p>Audit the five-file challenge bundle, choose a decision budget, run the real
      optimization engine, and inspect every feasibility and search diagnostic.</p>
    </section>
    <div class="pipeline">
      <div class="stage"><b>1 · Compile once</b><span>Integer-indexed candidates, lines, penalties</span></div>
      <div class="stage"><b>2 · Discover locally</b><span>Lazy inverted resource postings</span></div>
      <div class="stage"><b>3 · Optimize</b><span>Bounded exact or sampler-assisted recourse</span></div>
      <div class="stage"><b>4 · Certify</b><span>Independent global feasibility validation</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

cpu_count = available_cpu_count()
backend_availability = available_milp_backends()
backend_options = [name for name, installed in backend_availability.items() if installed]
default_bundle = os.environ.get("DOMOPT_BUNDLE_DIR", "")

with st.sidebar:
    st.subheader("Run configuration")
    bundle_text = st.text_input(
        "Challenge bundle directory",
        value=default_bundle,
        placeholder="/path/to/five/canonical/csv/files",
        help="The files stay on the machine running this app.",
    )
    mode = st.radio(
        "Decision mode",
        options=["fast", "quality", "hybrid"],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    st.caption(_mode_description(mode))
    backend = st.selectbox(
        "MILP backend",
        options=backend_options,
        index=backend_options.index("scipy-highs"),
        help="Install the optional open-source-solvers extra to add native HiGHS and SCIP.",
    )
    threads = st.number_input(
        "CPU thread cap",
        min_value=1,
        max_value=cpu_count,
        value=cpu_count,
        step=1,
        help=f"Detected process CPU budget: {cpu_count}. No machine-specific value is hard-coded.",
    )
    use_subset = st.checkbox("Run a shortage-ranked subset", value=False)
    subset_groups = st.number_input(
        "Assignment groups",
        min_value=1,
        value=20,
        step=1,
        disabled=not use_subset,
    )
    time_limit = st.number_input("MILP time limit (seconds)", min_value=1.0, value=30.0, step=1.0)
    seed = st.number_input("Deterministic seed", min_value=0, value=11, step=1)

    iterations = 4
    initial_groups = 6
    sampler = "simulated_annealing"
    if mode == "quality":
        iterations = st.slider("LNS iterations", 1, 20, 4)
        initial_groups = st.slider("Initial neighborhood groups", 2, 24, 6)
    elif mode == "hybrid":
        iterations = st.slider("Hybrid iterations", 1, 12, 3)
        sampler = st.selectbox(
            "Local proposal engine",
            ["simulated_annealing", "exact", "random", "qaoa_statevector"],
        )

    run_clicked = st.button("Run and certify", type="primary", width="stretch")

if not bundle_text:
    st.info("Point the cockpit at a local challenge bundle to begin. Expected filenames:")
    st.dataframe(
        pd.DataFrame(
            [
                {"role": role, "required filename": filename}
                for role, filename in POC_INPUT_FILENAMES.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.stop()

bundle_path = Path(bundle_text).expanduser()
try:
    audit = audit_poc_bundle(bundle_path)
except Exception as error:  # noqa: BLE001 - UI boundary must surface all audit failures
    st.error(f"Bundle audit failed: {error}")
    st.stop()

audit_column, scope_column = st.columns([1.35, 1])
with audit_column:
    st.subheader("Input contract")
    st.dataframe(audit, hide_index=True, width="stretch")
with scope_column:
    total_bytes = int(audit["bytes"].sum())
    st.subheader("Bundle status")
    st.markdown(
        '<span class="ok">✓ All five source tables are readable</span>', unsafe_allow_html=True
    )
    st.caption(f"{int(audit['rows'].sum()):,} source rows · {total_bytes / 1_048_576:.1f} MiB")

if run_clicked:
    try:
        with st.spinner("Loading and compiling the challenge instance…"):
            problem = _load_problem(str(bundle_path.resolve()))
            if use_subset:
                problem = select_shortage_subset(problem, int(subset_groups))

        exact_config = ExactLNSConfig(
            iterations=int(iterations),
            initial_neighborhood_groups=int(initial_groups),
            minimum_neighborhood_groups=min(4, int(initial_groups)),
            maximum_neighborhood_groups=max(12, int(initial_groups)),
            local_time_limit_seconds=float(time_limit),
            mip_relative_gap=0.01,
            milp_backend=backend,
            thread_count=int(threads),
            seed=int(seed),
        )
        hybrid_config = HybridConfig(
            iterations=int(iterations),
            sampler=sampler,
            recourse_time_limit_seconds=float(time_limit),
            milp_backend=backend,
            thread_count=int(threads),
            seed=int(seed),
        )
        config = SolverConfig(
            mode=mode,
            fast_time_limit_seconds=float(time_limit),
            fast_mip_relative_gap=0.01,
            fast_seed=int(seed),
            milp_backend=backend,
            thread_count=int(threads),
            exact_lns=exact_config,
            hybrid=hybrid_config,
        )
        with st.spinner(f"Running {mode} mode and independently validating the incumbent…"):
            solution = solve_dom(problem, config=config)
            metrics = compute_metrics(problem, solution)
            validation = validate_solution(problem, solution)
        st.session_state["domopt_result"] = {
            "problem": problem,
            "solution": solution,
            "metrics": metrics,
            "validation": validation,
            "config": asdict(config),
        }
    except Exception as error:  # noqa: BLE001 - UI boundary must surface solver failures
        st.exception(error)

result = st.session_state.get("domopt_result")
if result is None:
    st.caption("The bundle is valid. Configure a mode in the sidebar, then run and certify.")
    st.stop()

problem = result["problem"]
solution = result["solution"]
metrics = result["metrics"]
validation = result["validation"]

st.subheader("Certified outcome")
metric_columns = st.columns(5)
metric_columns[0].metric("Objective", f"{metrics['objective_value']:,.0f}")
metric_columns[1].metric("Case fill", f"{metrics['case_fill_rate']:.1%}")
metric_columns[2].metric("Runtime", f"{metrics['runtime_seconds']:.2f}s")
metric_columns[3].metric("Reassigned groups", f"{metrics['reassigned_assignment_groups']:,}")
metric_columns[4].metric("Feasible", "YES" if validation.is_feasible else "NO")

overview_tab, search_tab, decisions_tab, validation_tab = st.tabs(
    ["Business outcome", "Search telemetry", "Decisions", "Validation"]
)

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Objective bridge")
        bridge = pd.DataFrame(
            {
                "component": ["Fulfilled value", "Shipping cost", "Penalty cost"],
                "value": [
                    float(metrics["fulfilled_value"]),
                    -float(metrics["shipping_cost"]),
                    -float(metrics["penalty_cost"]),
                ],
            }
        ).set_index("component")
        st.bar_chart(bridge)
    with right:
        st.markdown("#### Assignment disposition")
        assigned = solution.assignments.copy()
        disposition = pd.Series("Default", index=assigned.index)
        disposition.loc[assigned["is_divert"].astype(bool)] = "Reassigned"
        disposition.loc[assigned["is_unassigned"].astype(bool)] = "Unassigned"
        st.bar_chart(disposition.value_counts().rename("orders"))
        st.caption(
            f"{len(problem.orders):,} orders · {len(problem.order_lines):,} lines · "
            f"{len(problem.candidates):,} candidates"
        )

with search_tab:
    telemetry_keys = {
        "greedy_seconds": "Greedy initialization",
        "initial_polish_seconds": "Exact policy polish",
        "neighborhood_index_seconds": "Sparse index",
        "residualization_seconds": "Residual models",
        "local_solve_seconds": "Local MILPs",
        "global_validation_seconds": "Global validation",
        "qubo_build_seconds": "QUBO construction",
        "sampling_seconds": "Proposal sampling",
        "recourse_seconds": "Exact recourse",
    }
    telemetry = pd.DataFrame(
        [
            {"stage": label, "seconds": float(solution.metadata[key])}
            for key, label in telemetry_keys.items()
            if solution.metadata.get(key) is not None
        ]
    )
    if not telemetry.empty:
        st.bar_chart(telemetry.set_index("stage"))
    history = solution.metadata.get("history")
    if history:
        history_frame = pd.DataFrame(history)
        chart_columns = [
            column
            for column in ["local_objective_delta", "global_objective_delta"]
            if column in history_frame
        ]
        if chart_columns:
            st.line_chart(history_frame.set_index("iteration")[chart_columns])
        st.dataframe(history_frame, hide_index=True, width="stretch")
    else:
        st.caption("This mode performs one greedy construction and one exact policy polish.")
    with st.expander("Solver metadata"):
        st.json(solution.metadata, expanded=False)

with decisions_tab:
    assignments_tab, fulfillment_tab = st.tabs(["Assignments", "Fulfillment"])
    with assignments_tab:
        st.dataframe(solution.assignments, hide_index=True, width="stretch")
        st.download_button(
            "Download assignments.csv",
            solution.assignments.to_csv(index=False),
            "assignments.csv",
            "text/csv",
        )
    with fulfillment_tab:
        st.dataframe(solution.fulfillment, hide_index=True, width="stretch")
        st.download_button(
            "Download fulfillment.csv",
            solution.fulfillment.to_csv(index=False),
            "fulfillment.csv",
            "text/csv",
        )

with validation_tab:
    if validation.is_feasible:
        st.success(
            "Independent validation passed every schema, assignment, demand, inventory, eligibility, and capacity check."
        )
    else:
        st.error("Independent validation failed.")
        st.write(validation.violations)
    diagnostics = pd.DataFrame(
        [
            {
                "check": key,
                "value": (
                    json.dumps(value, default=_json_default)
                    if isinstance(value, (dict, list, tuple, set))
                    else str(value)
                ),
            }
            for key, value in validation.diagnostics.items()
        ]
    )
    st.dataframe(diagnostics, hide_index=True, width="stretch")

archive = _solution_archive(solution, metrics, validation)
st.download_button(
    "Download complete certified run",
    archive,
    f"dom-solver-{solution.method}-certified.zip",
    "application/zip",
    type="primary",
)
