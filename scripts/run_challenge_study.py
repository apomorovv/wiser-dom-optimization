#!/usr/bin/env python3
"""Run all real-data challenge experiments and write aggregate-only evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from domopt.experiments import (
    EXPERIMENT_NAMES,
    run_challenge_experiments,
    write_experiment_results,
)
from domopt.poc import (
    POC_REFERENCE_FILENAMES,
    PocConfig,
    audit_poc_bundle,
    audit_poc_outputs,
    load_poc_problem,
)
from domopt.visualization import plot_challenge_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/challenge-study/aggregate_results.csv"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("runs/challenge-study/figures"),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENT_NAMES,
        help="Run only the named studies; the default runs the complete profile.",
    )
    args = parser.parse_args()

    audit = audit_poc_bundle(args.bundle_dir)
    print(audit[["role", "filename", "readable"]].to_string(index=False))
    problem = load_poc_problem(
        args.bundle_dir,
        config=PocConfig(pareto_prune=False),
        strict_bundle_audit=False,
    )
    if all((args.bundle_dir / name).is_file() for name in POC_REFERENCE_FILENAMES.values()):
        print(audit_poc_outputs(args.bundle_dir, problem))
    else:
        print("Optional recommendation-output audit skipped: reference CSVs absent")
    results = run_challenge_experiments(
        problem,
        profile=args.profile,
        experiments=args.experiments,
    )
    output = write_experiment_results(results, args.output)
    figures = plot_challenge_results(results, args.figures_dir)
    print(f"Wrote {len(results)} rows to {output}")
    for name, path in figures.items():
        print(f"figure {name:>32}: {path}")
    infeasible = results.loc[~results["feasible"].fillna(False)]
    if not infeasible.empty:
        diagnostic_columns = [
            "experiment",
            "level",
            "validation_categories",
            "validation_violation_count",
            "error_type",
        ]
        details = infeasible[
            [column for column in diagnostic_columns if column in infeasible]
        ].to_dict("records")
        raise RuntimeError(f"Experiment suite returned infeasible rows: {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
