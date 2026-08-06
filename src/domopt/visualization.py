"""Submission-quality plots for privacy-safe aggregate experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_METHOD_STYLES = {
    "default": {"marker": "D", "linestyle": ":"},
    "greedy": {"marker": "o", "linestyle": "-"},
    "classical": {"marker": "^", "linestyle": "-."},
    "hybrid": {"marker": "s", "linestyle": "--"},
}


def _save(figure: plt.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _feasible(frame: pd.DataFrame) -> pd.DataFrame:
    if "feasible" not in frame:
        return frame.copy()
    values = frame["feasible"]
    if values.dtype == bool:
        mask = values
    else:
        mask = values.astype(str).str.lower().isin({"true", "1"})
    return frame.loc[mask].copy()


def plot_method_objectives(metrics: pd.DataFrame, output_path: str | Path) -> Path:
    required = {"method", "objective_value", "feasible"}
    if missing := required - set(metrics.columns):
        raise ValueError(f"metrics is missing {sorted(missing)}")
    valid = _feasible(metrics)
    if valid.empty:
        raise ValueError("There are no feasible methods to plot")

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(valid["method"], valid["objective_value"], color="#2563eb")
    axis.set(xlabel="Method", ylabel="Validated objective", title="DOM objective by method")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _save(figure, output_path)


def _solver_comparison(frame: pd.DataFrame, output: Path) -> Path:
    data = _feasible(frame).sort_values("objective_value", ascending=False)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = ["#64748b", "#0ea5e9", "#16a34a", "#7c3aed"][: len(data)]
    axes[0, 0].bar(data["method"], data["objective_value"], color=colors)
    axes[0, 0].set(title="Validated objective", ylabel="Source currency")
    axes[0, 1].bar(data["method"], 100 * data["case_fill_rate"], color=colors)
    axes[0, 1].set(title="Case fill rate", ylabel="Percent")
    axes[1, 0].bar(data["method"], data["runtime_seconds"], color=colors)
    axes[1, 0].set(title="Solver runtime", ylabel="Seconds")
    axes[1, 0].set_yscale("log")
    axes[1, 1].bar(
        data["method"],
        data["penalty_cost"],
        label="Unmet penalty",
        color="#ef4444",
    )
    axes[1, 1].bar(
        data["method"],
        data["shipping_cost"],
        bottom=data["penalty_cost"],
        label="Shipping",
        color="#f59e0b",
    )
    axes[1, 1].set(title="Cost composition", ylabel="Source currency")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle("Common-objective solver comparison", fontsize=14)
    figure.tight_layout()
    return _save(figure, output)


def _size_scaling(frame: pd.DataFrame, output: Path) -> Path:
    data = _feasible(frame).copy()
    x_column = "actual_assignment_groups" if "actual_assignment_groups" in data else "assignment_group_count"
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for method, group in data.groupby("method", sort=True):
        group = group.sort_values(x_column)
        style = _METHOD_STYLES.get(method, {"marker": "o", "linestyle": "-"})
        axes[0].plot(
            group[x_column], group["runtime_seconds"], label=method, **style
        )
        axes[1].plot(
            group[x_column], group["objective_value"], label=method, **style
        )
    greedy = data.loc[data["method"] == "greedy"].sort_values(x_column)
    axes[2].plot(
        greedy[x_column],
        greedy["candidate_count"],
        marker="o",
        label="Candidate rows",
    )
    hybrid = data.loc[data["method"] == "hybrid"].sort_values(x_column)
    if not hybrid.empty:
        axes[2].plot(
            hybrid[x_column],
            hybrid["maximum_qubo_variables"],
            marker="s",
            label="Maximum local QUBO variables",
        )
    axes[0].set(title="Runtime scaling", xlabel="Assignment groups", ylabel="Seconds")
    axes[0].set_yscale("log")
    axes[1].set(title="Objective scaling", xlabel="Assignment groups", ylabel="Objective")
    axes[2].set(title="Model-size growth", xlabel="Assignment groups", ylabel="Count")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    return _save(figure, output)


def _line_sensitivity(
    frame: pd.DataFrame,
    output: Path,
    *,
    x: str,
    x_label: str,
    title: str,
) -> Path:
    data = _feasible(frame)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    measures = [
        ("objective_value", "Validated objective"),
        ("case_fill_rate", "Case fill rate"),
        ("runtime_seconds", "Runtime (seconds)"),
    ]
    for method, group in data.groupby("method", sort=True):
        group = group.sort_values(x)
        style = _METHOD_STYLES.get(method, {"marker": "o", "linestyle": "-"})
        for axis, (column, label) in zip(axes, measures):
            values = 100 * group[column] if column == "case_fill_rate" else group[column]
            axis.plot(group[x], values, label=method, **style)
            axis.set(xlabel=x_label, ylabel=label)
    axes[2].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    return _save(figure, output)


def _business_penalty(frame: pd.DataFrame, output: Path) -> Path:
    data = _feasible(frame)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    measures = [
        ("case_fill_rate", "Case fill rate", 100.0),
        ("penalty_cost", "Realized penalty cost", 1.0),
        ("reassigned_orders", "Reassigned orders", 1.0),
    ]
    for method, group in data.groupby("method", sort=True):
        group = group.sort_values("penalty_scale")
        style = _METHOD_STYLES.get(method, {"marker": "o", "linestyle": "-"})
        for axis, (column, label, multiplier) in zip(axes, measures):
            axis.plot(
                group["penalty_scale"],
                multiplier * group[column],
                label=method,
                **style,
            )
            axis.set(xlabel="Business penalty multiplier", ylabel=label)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Penalty-active order sensitivity", fontsize=14)
    figure.tight_layout()
    return _save(figure, output)


def _heatmap(
    axis: plt.Axes,
    table: pd.DataFrame,
    *,
    title: str,
    x_label: str,
    y_label: str,
    value_format: str,
) -> None:
    values = table.to_numpy(dtype=float)
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(table.columns)), [f"{value:g}" for value in table.columns])
    axis.set_yticks(range(len(table.index)), [f"{value:g}" for value in table.index])
    axis.set(xlabel=x_label, ylabel=y_label, title=title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    format(value, value_format),
                    ha="center",
                    va="center",
                    color="white" if value < np.nanmean(values) else "black",
                    fontsize=8,
                )
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _quantum_noise(frame: pd.DataFrame, output: Path) -> Path:
    data = _feasible(frame)
    improvement = data.pivot_table(
        index="seed",
        columns="coefficient_noise_relative_sigma",
        values="hybrid_improvement",
        aggfunc="mean",
    )
    one_hot = data.pivot_table(
        index="seed",
        columns="coefficient_noise_relative_sigma",
        values="raw_one_hot_rate",
        aggfunc="mean",
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _heatmap(
        axes[0],
        improvement,
        title="Hybrid improvement",
        x_label="Relative coefficient noise",
        y_label="Seed",
        value_format=".1f",
    )
    _heatmap(
        axes[1],
        one_hot,
        title="Raw one-hot rate",
        x_label="Relative coefficient noise",
        y_label="Seed",
        value_format=".2f",
    )
    figure.suptitle("Local QUBO robustness (not physical QPU noise)", fontsize=14)
    figure.tight_layout()
    return _save(figure, output)


def _qubo_penalties(frame: pd.DataFrame, output: Path) -> Path:
    data = _feasible(frame)
    objective = data.pivot_table(
        index="one_hot_penalty_multiplier",
        columns="pair_penalty_multiplier",
        values="hybrid_improvement",
        aggfunc="mean",
    )
    one_hot = data.pivot_table(
        index="one_hot_penalty_multiplier",
        columns="pair_penalty_multiplier",
        values="raw_one_hot_rate",
        aggfunc="mean",
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _heatmap(
        axes[0],
        objective,
        title="Improvement over incumbent",
        x_label="Conflict-pair multiplier",
        y_label="One-hot multiplier",
        value_format=".1f",
    )
    _heatmap(
        axes[1],
        one_hot,
        title="Raw one-hot rate",
        x_label="Conflict-pair multiplier",
        y_label="One-hot multiplier",
        value_format=".2f",
    )
    figure.suptitle("QUBO penalty calibration", fontsize=14)
    figure.tight_layout()
    return _save(figure, output)


def _ablations(frame: pd.DataFrame, output: Path) -> Path:
    data = _feasible(frame).copy()
    data["label"] = data["experiment"].str.replace("_", " ") + "\n" + data["level"]
    figure, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.55 * len(data))))
    improvement = data["hybrid_improvement"].fillna(0)
    axes[0].barh(data["label"], improvement, color="#7c3aed")
    axes[0].set(xlabel="Improvement over incumbent", title="Solution effect")
    if np.allclose(improvement, 0):
        axes[0].set_xlim(-1, 1)
        axes[0].text(
            0.5,
            0.5,
            "No accepted improvement in this profile",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    axes[1].barh(data["label"], data["runtime_seconds"], color="#0ea5e9")
    axes[1].set(xlabel="Seconds", title="Runtime effect")
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return _save(figure, output)


def _hybrid_timing(frame: pd.DataFrame, output: Path) -> Path:
    data = _feasible(frame.loc[frame["method"] == "hybrid"]).copy()
    data = data.loc[
        data["experiment"].isin({"solver_comparison", "size_scaling"})
    ].sort_values(["experiment", "assignment_group_count"])
    data["label"] = data.apply(
        lambda row: (
            "core"
            if row["experiment"] == "solver_comparison"
            else f"{int(row['assignment_group_count'])} groups"
        ),
        axis=1,
    )
    stages = [
        ("initialization_seconds", "Initial greedy", "#64748b"),
        ("qubo_build_seconds", "QUBO build", "#0ea5e9"),
        ("sampling_seconds", "Sampling", "#7c3aed"),
        ("recourse_seconds", "Exact recourse", "#16a34a"),
        ("other_seconds", "Other", "#f59e0b"),
    ]
    figure, axis = plt.subplots(figsize=(9, max(3.5, 0.6 * len(data))))
    left = np.zeros(len(data))
    for column, label, color in stages:
        values = pd.to_numeric(data.get(column, 0), errors="coerce").fillna(0).to_numpy()
        axis.barh(data["label"], values, left=left, label=label, color=color)
        left += values
    axis.set(xlabel="Seconds", title="Hybrid runtime by stage")
    axis.legend(ncol=3, fontsize=8)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return _save(figure, output)


def plot_challenge_results(
    results: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write every applicable chart and return its stable path."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    def add(name: str, experiment: str, builder) -> None:
        frame = results.loc[results["experiment"] == experiment].copy()
        if not frame.empty:
            generated[name] = builder(frame, root / f"{name}.png")

    add("solver_comparison", "solver_comparison", _solver_comparison)
    add("size_scaling", "size_scaling", _size_scaling)
    add(
        "business_penalty_sensitivity",
        "penalty_weight_sensitivity",
        _business_penalty,
    )
    add(
        "candidate_count_sensitivity",
        "candidate_count_sensitivity",
        lambda frame, path: _line_sensitivity(
            frame,
            path,
            x="candidate_limit",
            x_label="Candidates retained per assignment group",
            title="Candidate-column sensitivity",
        ),
    )
    add(
        "inventory_shock",
        "inventory_shock",
        lambda frame, path: _line_sensitivity(
            frame,
            path,
            x="inventory_shock",
            x_label="Inventory reduction fraction",
            title="Inventory-shock robustness",
        ),
    )
    add("quantum_noise", "quantum_seed_noise", _quantum_noise)
    add("qubo_penalty_sensitivity", "qubo_penalty_sensitivity", _qubo_penalties)

    ablation_names = {
        "pareto_pruning_ablation",
        "batch_strategy_ablation",
        "sampler_ablation",
    }
    ablations = results.loc[results["experiment"].isin(ablation_names)].copy()
    if not ablations.empty:
        generated["ablations"] = _ablations(ablations, root / "ablations.png")

    hybrid_timing = results.loc[
        results["experiment"].isin({"solver_comparison", "size_scaling"})
    ].copy()
    if not hybrid_timing.empty and "initialization_seconds" in hybrid_timing:
        generated["hybrid_timing"] = _hybrid_timing(
            hybrid_timing, root / "hybrid_timing.png"
        )

    synthetic = results.loc[
        results["experiment"] == "synthetic_coordination_control"
    ].copy()
    if not synthetic.empty:
        generated["synthetic_coordination_control"] = plot_method_objectives(
            synthetic,
            root / "synthetic_coordination_control.png",
        )
    return generated


def plot_hardware_benchmark(
    benchmark: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot synthetic QUBO-scoring throughput by workload and backend."""

    required = {"backend", "variables", "samples", "samples_per_second"}
    if missing := required - set(benchmark.columns):
        raise ValueError(f"benchmark is missing {sorted(missing)}")
    data = benchmark.copy()
    data["workload"] = data.apply(
        lambda row: f"n={int(row['variables'])}, reads={int(row['samples'])}",
        axis=1,
    )
    pivot = data.pivot(index="workload", columns="backend", values="samples_per_second")
    figure, axis = plt.subplots(figsize=(10, max(4, 0.55 * len(pivot))))
    pivot.plot.barh(ax=axis)
    axis.set(
        xlabel="QUBO samples scored per second",
        ylabel="Synthetic workload",
        title="CPU/GPU QUBO scoring crossover",
    )
    axis.set_xscale("log")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(title="Backend")
    figure.tight_layout()
    return _save(figure, output_path)
