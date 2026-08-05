#!/usr/bin/env python3
"""Run all real-data challenge experiments and write aggregate-only evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from domopt.experiments import run_challenge_experiments, write_experiment_results
from domopt.poc import PocConfig, audit_poc_bundle, audit_poc_outputs, load_poc_problem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/challenge_experiments.csv"),
    )
    args = parser.parse_args()

    audit = audit_poc_bundle(args.bundle_dir)
    print(audit[["role", "filename", "readable"]].to_string(index=False))
    problem = load_poc_problem(
        args.bundle_dir,
        config=PocConfig(pareto_prune=False),
        strict_bundle_audit=False,
    )
    print(audit_poc_outputs(args.bundle_dir, problem))
    results = run_challenge_experiments(problem, profile=args.profile)
    output = write_experiment_results(results, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
