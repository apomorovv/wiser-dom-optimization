"""Create privacy-safe publication figures from the curated aggregate evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final"
OUTPUT = RESULTS / "figures"

BLUE = "#1769AA"
CYAN = "#47B8E0"
NAVY = "#0B2E4F"
GRAY = "#B8C0C8"
DARK_GRAY = "#52606D"
ORANGE = "#E67E22"

LABELS = {
    "default": "Default",
    "greedy": "Greedy",
    "polished_greedy": "Polished\ngreedy",
    "exact_lns": "Exact LNS",
    "classical": "Full exact\nMILP",
    "hybrid": "Hybrid SA",
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


def solver_summary() -> Path:
    data = pd.read_csv(RESULTS / "tables" / "solver_comparison.csv")
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
    quality.set_title("Four methods reach the same validated quality frontier", loc="left")
    quality.set_ylabel("Percent")
    quality.set_xticks(x, [LABELS[value] for value in order])
    quality.set_ylim(58, 72)
    quality.legend(frameon=False, ncols=2, loc="upper left")
    for index, row in data.iterrows():
        quality.text(
            index - width / 2,
            100 * row["objective_capture_rate"] + 0.25,
            f"{100 * row['objective_capture_rate']:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    runtime.barh(
        np.arange(len(data)),
        data["runtime_seconds"],
        color=colors,
        height=0.64,
    )
    runtime.set_title("Polished greedy is the practical default", loc="left")
    runtime.set_xlabel("End-to-end runtime (seconds, log scale)")
    runtime.set_yticks(np.arange(len(data)), [LABELS[value].replace("\n", " ") for value in order])
    runtime.set_xscale("log")
    runtime.invert_yaxis()
    for index, value in enumerate(data["runtime_seconds"]):
        runtime.text(value * 1.08, index, f"{value:.2f}s", va="center", fontsize=8)
    figure.tight_layout()
    path = OUTPUT / "submission_solver_summary.png"
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def scaling_summary() -> Path:
    data = pd.read_csv(RESULTS / "tables" / "size_scaling.csv")
    selected = {
        "greedy": ("Greedy", DARK_GRAY, "o"),
        "polished_greedy": ("Polished greedy", BLUE, "s"),
        "exact_lns": ("Exact LNS", ORANGE, "^"),
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
    axis.set_title("Bounded exact LNS scales to the full 372-group study", loc="left")
    axis.set_xlabel("Assignment groups")
    axis.set_ylabel("Runtime (seconds, log scale)")
    axis.set_yscale("log")
    axis.legend(frameon=False, ncols=3, loc="upper left")
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
    path = OUTPUT / "submission_scaling_summary.png"
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def ibm_depth_summary() -> Path:
    data = pd.read_csv(RESULTS / "ibm" / "ibm_hardware_stress.csv")
    qpu = data.loc[data["sampler_backend"].eq("ibm-qpu") & data["feasible"].astype(bool)]
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
    quality.set_title("Hardware sample quality falls with depth", loc="left")
    quality.set_ylabel("Percent of shots")
    quality.set_xticks(x, labels)
    quality.set_ylim(0, 60)
    for index, value in enumerate(100 * depth["exact_hit"]):
        quality.text(
            index + 0.18,
            max(1.0, value + 0.8),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            color=ORANGE,
            fontsize=8,
        )
    quality.legend(frameon=False, fontsize=8)

    circuit.bar(x - 0.18, depth["two_qubit_gates"], 0.36, color=NAVY, label="Two-qubit gates")
    circuit.bar(x + 0.18, depth["transpiled_depth"], 0.36, color=CYAN, label="Circuit depth")
    circuit.set_title("The p=2 circuit is substantially larger", loc="left")
    circuit.set_ylabel("Median transpiled count")
    circuit.set_xticks(x, labels)
    circuit.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "IBM Marrakesh: 18 matched QPU jobs, 512 shots each",
        x=0.06,
        ha="left",
        fontsize=10,
        color=DARK_GRAY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUTPUT / "submission_ibm_depth_summary.png"
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _style()
    for path in (solver_summary(), scaling_summary(), ibm_depth_summary()):
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
