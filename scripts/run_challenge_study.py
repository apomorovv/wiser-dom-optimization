#!/usr/bin/env python3
"""Run all real-data challenge experiments and write aggregate-only evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from domopt.checkpoints import (
    challenge_results_root,
    checkpoint_identity,
    write_checkpoint,
)
from domopt.experiments import (
    EXPERIMENT_NAMES,
    experiment_profile,
    run_challenge_experiments,
    write_experiment_results,
)
from domopt.pipeline import current_source_state
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
        help=(
            "Aggregate CSV path. Default: "
            "results/challenge-study/cli/<profile>/aggregate_results.csv"
        ),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        help="Figure directory. Defaults to a figures/ directory beside --output.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENT_NAMES,
        help="Run only the named studies; the default runs the complete profile.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Allow a full evidence run from uncommitted source. The resulting "
            "manifest will still record the dirty source hash."
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    default_root = challenge_results_root(project_root, producer="cli") / args.profile
    output_path = args.output or default_root / "aggregate_results.csv"
    figures_dir = args.figures_dir or output_path.parent / "figures"
    source_state = current_source_state()
    if (
        args.profile == "full"
        and source_state.get("git_dirty") is True
        and not args.allow_dirty
    ):
        raise RuntimeError(
            "Refusing a full evidence run from a dirty Git worktree. Commit the "
            "source first, or pass --allow-dirty for an explicitly provisional run."
        )

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
    settings = experiment_profile(args.profile)
    results = run_challenge_experiments(
        problem,
        profile=settings,
        experiments=args.experiments,
    )
    output = write_experiment_results(results, output_path)
    identity = checkpoint_identity(
        problem,
        profile=args.profile,
        experiment="cli_challenge_suite",
        configuration={
            **asdict(settings),
            "experiments": list(args.experiments or EXPERIMENT_NAMES),
            "producer": "cli",
        },
    )
    _, manifest = write_checkpoint(results, output, identity)
    figures = plot_challenge_results(results, figures_dir)
    print(f"Wrote {len(results)} rows to {output}")
    print(f"Wrote provenance manifest to {manifest}")
    print(
        "Notebook evidence is isolated under "
        f"{challenge_results_root(project_root, producer='notebook')}"
    )
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
