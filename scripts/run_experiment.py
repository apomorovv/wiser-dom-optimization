#!/usr/bin/env python3
"""Run comparable DOM methods through the common validation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from domopt.data import load_problem_data
from domopt.hybrid import HybridConfig
from domopt.pipeline import run_methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["default", "greedy", "classical", "hybrid"],
        choices=["default", "greedy", "classical", "hybrid"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit-seconds", type=float, default=60.0)
    parser.add_argument(
        "--hybrid-config",
        type=Path,
        help="YAML file containing either HybridConfig fields or a 'hybrid' mapping.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("experiments/registry.csv"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    problem = load_problem_data(args.data_dir)
    hybrid_config = None
    if args.hybrid_config is not None:
        payload = yaml.safe_load(
            args.hybrid_config.read_text(encoding="utf-8")
        ) or {}
        hybrid_config = HybridConfig(**dict(payload.get("hybrid", payload)))
    summary = run_methods(
        problem,
        args.methods,
        args.output_dir,
        experiment_id=args.experiment_id,
        seed=args.seed,
        time_limit_seconds=args.time_limit_seconds,
        registry_path=args.registry,
        hybrid_config=hybrid_config,
    )
    visible = [
        "method",
        "feasible",
        "objective_value",
        "case_fill_rate",
        "shipping_cost",
        "penalty_cost",
        "reassigned_orders",
        "runtime_seconds",
    ]
    print(summary[visible].to_string(index=False))
    return 0 if bool(summary["feasible"].all()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

