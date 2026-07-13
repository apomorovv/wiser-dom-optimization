#!/usr/bin/env python3
"""Run the default and sequential greedy DOM baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from domopt.baselines import run_baselines
from domopt.data import load_problem_data
from domopt.pipeline import write_solution_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default="baselines")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    problem = load_problem_data(args.data_dir)
    failed = False
    for solution in run_baselines(problem):
        metrics = write_solution_artifacts(
            problem,
            solution,
            args.output_dir / solution.method,
            experiment_id=args.experiment_id,
            seed=args.seed,
        )
        print(
            f"{solution.method:>8s}: feasible={metrics['feasible']} "
            f"objective={metrics['objective_value']:.6f} "
            f"fill={metrics['case_fill_rate']:.4f}"
        )
        failed = failed or not bool(metrics["feasible"])
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

