"""Create privacy-safe publication figures from audited local aggregate evidence.

The underlying CSV files remain outside the submission branch. Point this script at
the audited notebook output directories and it writes only normalized real-data or
independently generated synthetic figures under ``results/final``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from domopt.visualization import (
    plot_ibm_backend_snapshot,
    plot_ibm_hardware_study,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "final" / "figures"
DEFAULT_IBM_OUTPUT = ROOT / "results" / "final" / "ibm" / "figures"

BLUE = "#1769AA"
CYAN = "#47B8E0"
NAVY = "#0B2E4F"
GRAY = "#B8C0C8"
DARK_GRAY = "#52606D"
ORANGE = "#E67E22"
PURPLE = "#7C3AED"
GREEN = "#168A57"

LABELS = {
    "default": "Default",
    "greedy": "Greedy",
    "polished_greedy": "Polished\ngreedy",
    "exact_lns": "Exact LNS",
    "classical": "Full MILP\n(time-limited)",
    "hybrid": "Hybrid SA",
    "fixed_routing_recourse": "Frozen routing",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.8,
        }
    )


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _read(tables: Path, filename: str) -> pd.DataFrame:
    path = tables / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required audited evidence table is missing: {path}")
    return pd.read_csv(path, low_memory=False)


def solver_summary(tables: Path, output: Path) -> Path:
    data = _read(tables, "solver_comparison.csv")[
        ["method", "objective_capture_rate", "case_fill_rate", "runtime_seconds"]
    ].copy()
    order = [
        "default",
        "greedy",
        "polished_greedy",
        "exact_lns",
        "classical",
        "hybrid",
    ]
    data = data.set_index("method").loc[order].reset_index()
    x = np.arange(len(data))
    width = 0.34
    colors = [GRAY, DARK_GRAY, BLUE, CYAN, NAVY, ORANGE]

    figure, (quality, runtime) = plt.subplots(
        1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.55, 1]}
    )
    quality.bar(
        x - width / 2,
        100 * data["objective_capture_rate"],
        width,
        color=colors,
        alpha=0.96,
        label="Objective capture",
    )
    quality.bar(
        x + width / 2,
        100 * data["case_fill_rate"],
        width,
        color="white",
        edgecolor=colors,
        linewidth=1.8,
        label="Case fill",
    )
    quality.set_title("Greedy captures most of the gain; safeguards retain the frontier", loc="left")
    quality.set_ylabel("Percent")
    quality.set_xticks(x, [LABELS[value] for value in order])
    minimum = 100 * min(
        float(data["objective_capture_rate"].min()),
        float(data["case_fill_rate"].min()),
    )
    maximum = 100 * max(
        float(data["objective_capture_rate"].max()),
        float(data["case_fill_rate"].max()),
    )
    quality.set_ylim(np.floor(minimum - 1.5), np.ceil(maximum + 2.0))
    quality.legend(frameon=False, ncols=2, loc="upper left")
    for index, row in data.iterrows():
        quality.text(
            index - width / 2,
            100 * row["objective_capture_rate"] + 0.18,
            f"{100 * row['objective_capture_rate']:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )

    runtime.barh(
        np.arange(len(data)),
        data["runtime_seconds"],
        color=colors,
        height=0.64,
    )
    runtime.set_title("The hierarchy exposes an explicit quality-latency trade-off", loc="left")
    runtime.set_xlabel("End-to-end runtime (seconds, log scale)")
    runtime.set_yticks(
        np.arange(len(data)),
        [LABELS[value].replace("\n", " ") for value in order],
    )
    runtime.set_xscale("log")
    runtime.invert_yaxis()
    for index, value in enumerate(data["runtime_seconds"]):
        runtime.text(value * 1.08, index, f"{value:.2f}s", va="center", fontsize=8)
    figure.tight_layout()
    return _save(figure, output / "submission_solver_summary.png")


def scaling_summary(tables: Path, output: Path) -> Path:
    data = _read(tables, "size_scaling.csv")
    selected = {
        "greedy": ("Greedy", DARK_GRAY, "o"),
        "polished_greedy": ("Polished greedy", BLUE, "s"),
        "exact_lns": ("Exact LNS", ORANGE, "^"),
        "hybrid": ("Hybrid", PURPLE, "D"),
    }
    figure, axis = plt.subplots(figsize=(8.8, 4.6))
    for method, (label, color, marker) in selected.items():
        subset = data.loc[data["method"].eq(method)]
        grouped = subset.groupby("actual_assignment_groups")["runtime_seconds"]
        summary = grouped.agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        axis.plot(
            summary.index,
            summary["median"],
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6,
            label=label,
        )
        axis.fill_between(
            summary.index,
            summary["q25"],
            summary["q75"],
            color=color,
            alpha=0.12,
        )
    axis.set_title("Bounded search scales through all 372 real assignment groups", loc="left")
    axis.set_xlabel("Assignment groups")
    axis.set_ylabel("Runtime (seconds, log scale)")
    axis.set_yscale("log")
    axis.legend(frameon=False, ncols=2, loc="upper left")
    axis.text(
        0.99,
        0.05,
        "Median across 3 repetitions; shaded band = IQR",
        transform=axis.transAxes,
        ha="right",
        color=DARK_GRAY,
        fontsize=9,
    )
    figure.tight_layout()
    return _save(figure, output / "submission_scaling_summary.png")


def robustness_summary(tables: Path, output: Path) -> Path:
    data = _read(tables, "inventory_shock.csv")
    data = data.loc[data["method"].isin(
        ["fixed_routing_recourse", "greedy", "exact_lns", "hybrid"]
    )].copy()
    colors = {
        "fixed_routing_recourse": GRAY,
        "greedy": DARK_GRAY,
        "exact_lns": ORANGE,
        "hybrid": PURPLE,
    }
    markers = {
        "fixed_routing_recourse": "x",
        "greedy": "o",
        "exact_lns": "^",
        "hybrid": "D",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.35))
    for method, group in data.groupby("method", sort=False):
        group = group.sort_values("inventory_shock")
        for axis, column, ylabel in (
            (axes[0], "objective_capture_rate", "Objective capture (%)"),
            (axes[1], "case_fill_rate", "Case fill (%)"),
        ):
            axis.plot(
                100 * group["inventory_shock"],
                100 * group[column],
                color=colors[method],
                marker=markers[method],
                linewidth=2,
                markersize=5,
                label=LABELS[method].replace("\n", " "),
            )
            axis.set(xlabel="Inventory reduction (%)", ylabel=ylabel)
            axis.axvline(55, color="#94A3B8", linestyle=":", linewidth=1.2)
    axes[0].set_title("Adaptive routing remains feasible after frozen routing fails", loc="left")
    axes[1].set_title("Exact LNS retains the strongest service frontier", loc="left")
    axes[0].annotate(
        "Frozen routing infeasible\nfrom 60% reduction",
        xy=(58, 73.2),
        xytext=(31, 74.0),
        arrowprops={"arrowstyle": "->", "color": DARK_GRAY},
        color=DARK_GRAY,
        fontsize=8.5,
    )
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return _save(figure, output / "submission_robustness_summary.png")


def coordination_summary(tables: Path, output: Path) -> Path:
    data = _read(tables, "synthetic_coordination_control.csv")[
        ["method", "objective_value", "runtime_seconds"]
    ].copy()
    order = ["greedy", "polished_greedy", "exact_lns", "classical", "hybrid"]
    data = data.set_index("method").loc[order].reset_index()
    greedy_objective = float(data.loc[data["method"].eq("greedy"), "objective_value"].iloc[0])
    data["gain"] = data["objective_value"] - greedy_objective
    colors = [GRAY, DARK_GRAY, ORANGE, NAVY, PURPLE]
    figure, (gain, runtime) = plt.subplots(1, 2, figsize=(10.8, 4.1))
    gain.bar(np.arange(len(data)), data["gain"], color=colors)
    gain.set_xticks(np.arange(len(data)), [LABELS[value] for value in order])
    gain.set(ylabel="Synthetic objective gain over greedy", title="Coordinated search escapes the greedy trap")
    gain.axhline(197.2, color=GREEN, linestyle="--", linewidth=1.2)
    runtime.barh(np.arange(len(data)), data["runtime_seconds"], color=colors)
    runtime.set_yticks(np.arange(len(data)), [LABELS[value].replace("\n", " ") for value in order])
    runtime.set_xscale("log")
    runtime.invert_yaxis()
    runtime.set(xlabel="Runtime (seconds, log scale)", title="Exact controls remain the speed reference")
    figure.tight_layout()
    return _save(figure, output / "submission_coordination_summary.png")


def ibm_depth_summary(ibm_tables: Path, output: Path) -> Path:
    data = _read(ibm_tables, "ibm_hardware_stress.csv")
    feasible = data["feasible"].astype(str).str.lower().isin({"true", "1"})
    qpu = data.loc[data["sampler_backend"].eq("ibm-qpu") & feasible]
    depth = qpu.groupby("qaoa_layers").agg(
        one_hot=("raw_one_hot_rate", "median"),
        exact_hit=("hardware_qubo_optimal_hit_rate", "median"),
        two_qubit_gates=("hardware_two_qubit_gates", "median"),
        transpiled_depth=("hardware_transpiled_depth", "median"),
    )
    x = np.arange(len(depth))
    labels = [f"p={int(value)}" for value in depth.index]

    figure, (quality, circuit) = plt.subplots(1, 2, figsize=(10.5, 4.25))
    quality.bar(x - 0.18, 100 * depth["one_hot"], 0.36, color=BLUE, label="Raw one-hot")
    quality.bar(
        x + 0.18,
        100 * depth["exact_hit"],
        0.36,
        color=ORANGE,
        label="Exact optimum hit",
    )
    quality.axhline(
        100 / 256,
        color=DARK_GRAY,
        linestyle="--",
        linewidth=1.4,
        label="Uniform feasible optimum rate",
    )
    quality.set_title("Shallow QAOA retains feasible-subspace structure", loc="left")
    quality.set_ylabel("Percent of all shots")
    quality.set_xticks(x, labels)
    quality.set_ylim(0, 75)
    for index, value in enumerate(100 * depth["exact_hit"]):
        quality.text(
            index + 0.18,
            max(1.0, value + 0.8),
            f"{value:.3f}%",
            ha="center",
            va="bottom",
            color=ORANGE,
            fontsize=8,
        )
    quality.legend(frameon=False, fontsize=8)

    circuit.bar(x - 0.18, depth["two_qubit_gates"], 0.36, color=NAVY, label="Two-qubit gates")
    circuit.bar(x + 0.18, depth["transpiled_depth"], 0.36, color=CYAN, label="Circuit depth")
    circuit.set_title("The p=2 implementation is roughly ten times deeper", loc="left")
    circuit.set_ylabel("Median transpiled count")
    circuit.set_xticks(x, labels)
    circuit.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Dicke/XY-QAOA on ibm_marrakesh: 18 jobs, 8,192 shots each",
        x=0.06,
        ha="left",
        fontsize=10,
        color=DARK_GRAY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(figure, output / "submission_ibm_depth_summary.png")


def ibm_study_figures(ibm_root: Path, output: Path) -> list[Path]:
    tables = ibm_root / "tables"
    results = _read(tables, "ibm_hardware_stress.csv")
    paths = [plot_ibm_hardware_study(results, output / "ibm_hardware_stress.png")]
    snapshot = ibm_root / "ibm_backend_snapshot.csv"
    if snapshot.is_file():
        paths.append(
            plot_ibm_backend_snapshot(
                pd.read_csv(snapshot),
                output / "ibm_backend_queue.png",
            )
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        required=True,
        help="Audited notebook/full directory containing a tables subdirectory.",
    )
    parser.add_argument(
        "--ibm-root",
        type=Path,
        required=True,
        help="Audited notebook/ibm-presentation directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ibm-output-dir", type=Path, default=DEFAULT_IBM_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tables = args.study_root / "tables"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ibm_output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    paths = [
        solver_summary(tables, args.output_dir),
        scaling_summary(tables, args.output_dir),
        robustness_summary(tables, args.output_dir),
        coordination_summary(tables, args.output_dir),
        ibm_depth_summary(args.ibm_root / "tables", args.output_dir),
        *ibm_study_figures(args.ibm_root, args.ibm_output_dir),
    ]
    for path in paths:
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
