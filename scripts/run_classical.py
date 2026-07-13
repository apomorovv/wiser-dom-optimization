#!/usr/bin/env python3
"""Run the exact/bounded classical MILP for a canonical DOM instance."""

from __future__ import annotations

import argparse
from pathlib import Path

from domopt.classical import solve_classical
from domopt.data import load_problem_data
from domopt.pipeline import write_solution_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default="classical")
    parser.add_argument("--time-limit-seconds", type=float, default=60.0)
    parser.add_argument("--mip-relative-gap", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    problem = load_problem_data(args.data_dir)
    solution = solve_classical(
        problem,
        time_limit_seconds=args.time_limit_seconds,
        mip_relative_gap=args.mip_relative_gap,
        seed=args.seed,
    )
    metrics = write_solution_artifacts(
        problem,
        solution,
        args.output_dir,
        experiment_id=args.experiment_id,
        seed=args.seed,
        configuration={
            "time_limit_seconds": args.time_limit_seconds,
            "mip_relative_gap": args.mip_relative_gap,
        },
    )
    print(f"feasible={metrics['feasible']}")
    print(f"objective_value={metrics['objective_value']:.6f}")
    print(f"case_fill_rate={metrics['case_fill_rate']:.6f}")
    print(f"reassigned_orders={metrics['reassigned_orders']}")
    return 0 if bool(metrics["feasible"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())

