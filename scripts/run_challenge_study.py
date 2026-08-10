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
from domopt.poc import (
    POC_INPUT_FILENAMES,
    POC_REFERENCE_FILENAMES,
    PocConfig,
    PocDataError,
    audit_poc_bundle,
    audit_poc_outputs,
    load_poc_problem,
)

DEFAULT_BUNDLE_RELATIVE = Path("data/raw/nestle_challenge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        help=(
            "Directory containing the five canonical runtime CSVs. Default: "
            "data/raw/nestle_challenge under the repository root."
        ),
    )
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
    return parser


def _has_runtime_bundle(path: Path) -> bool:
    return path.is_dir() and all(
        (path / filename).is_file() for filename in POC_INPUT_FILENAMES.values()
    )


def resolve_bundle_dir(
    requested: Path | None,
    *,
    project_root: Path,
    working_directory: Path | None = None,
    home_directory: Path | None = None,
) -> Path:
    """Resolve a bundle path without silently substituting a different dataset."""

    cwd = working_directory or Path.cwd()
    home = home_directory or Path.home()
    if requested is None:
        candidate = project_root / DEFAULT_BUNDLE_RELATIVE
    else:
        if requested.parts and requested.parts[0] == "~":
            candidate = home.joinpath(*requested.parts[1:])
        else:
            candidate = requested.expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
    candidate = candidate.resolve()
    if candidate.is_dir():
        return candidate

    suggestions: list[Path] = []
    prepared = (project_root / DEFAULT_BUNDLE_RELATIVE).resolve()
    if _has_runtime_bundle(prepared):
        suggestions.append(prepared)
    if requested is not None and requested.is_absolute() and len(requested.parts) > 1:
        home_candidate = home.joinpath(*requested.parts[1:]).resolve()
        if _has_runtime_bundle(home_candidate) and home_candidate not in suggestions:
            suggestions.append(home_candidate)

    lines = [f"Challenge bundle directory does not exist: {candidate}."]
    if suggestions:
        lines.append("Existing challenge bundle candidate(s):")
        lines.extend(f"  - {path}" for path in suggestions)
        lines.append(
            "Pass one of those directories with --bundle-dir; an explicit invalid path "
            "is never replaced silently."
        )
    else:
        lines.append(
            "First run scripts/prepare_challenge_bundle.py, then pass its --output-dir "
            "to this command."
        )
    if requested is not None and requested.is_absolute():
        lines.append(
            "Reminder: /Wiser/... starts at the filesystem root, while ~/Wiser/... "
            "starts in your home directory."
        )
    raise PocDataError("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    try:
        bundle_dir = resolve_bundle_dir(args.bundle_dir, project_root=project_root)
        audit = audit_poc_bundle(bundle_dir)
        problem = load_poc_problem(
            bundle_dir,
            config=PocConfig(pareto_prune=False),
            strict_bundle_audit=False,
        )
    except PocDataError as error:
        parser.error(str(error))

    default_root = challenge_results_root(project_root, producer="cli") / args.profile
    output_path = (
        args.output.expanduser() if args.output else default_root / "aggregate_results.csv"
    )
    figures_dir = (
        args.figures_dir.expanduser()
        if args.figures_dir
        else output_path.parent / "figures"
    )
    print(audit[["role", "filename", "readable"]].to_string(index=False))
    if all((bundle_dir / name).is_file() for name in POC_REFERENCE_FILENAMES.values()):
        print(audit_poc_outputs(bundle_dir, problem))
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
    from domopt.visualization import plot_challenge_results

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
