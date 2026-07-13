"""Minimal plotting utilities for validated experiment summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_method_objectives(metrics: pd.DataFrame, output_path: str | Path) -> Path:
    required = {"method", "objective_value", "feasible"}
    if missing := required - set(metrics.columns):
        raise ValueError(f"metrics is missing {sorted(missing)}")
    valid = metrics.loc[metrics["feasible"].astype(bool)].copy()
    if valid.empty:
        raise ValueError("There are no feasible methods to plot")

    figure, axis = plt.subplots()
    axis.bar(valid["method"], valid["objective_value"])
    axis.set_xlabel("Method")
    axis.set_ylabel("Validated objective")
    axis.set_title("DOM objective by method")
    figure.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path

