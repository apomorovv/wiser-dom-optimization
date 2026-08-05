#!/usr/bin/env python3
"""Run the quantum-assisted large-neighborhood DOM optimizer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from domopt.data import load_problem_data
from domopt.hybrid import HybridConfig, solve_hybrid
from domopt.pipeline import write_solution_artifacts
from domopt.planner import write_planner_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-id", default="hybrid")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_data: dict[str, object] = {}
    if args.config is not None:
        config_data = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    data_dir = args.data_dir or config_data.get("data_dir")
    output_dir = args.output_dir or config_data.get("output_dir")
    if data_dir is None or output_dir is None:
        raise SystemExit("Provide data_dir and --output-dir, or set both in --config")

    hybrid_data = dict(config_data.get("hybrid", {}))
    settings = HybridConfig(**hybrid_data)
    problem = load_problem_data(Path(data_dir))
    solution = solve_hybrid(problem, config=settings)
    metrics = write_solution_artifacts(
        problem,
        solution,
        Path(output_dir),
        experiment_id=args.experiment_id,
        seed=settings.seed,
        configuration={"hybrid": asdict(settings)},
    )
    write_planner_artifacts(problem, solution, Path(output_dir))
    print(json.dumps(metrics, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
