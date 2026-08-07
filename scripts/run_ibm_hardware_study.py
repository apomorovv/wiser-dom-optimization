#!/usr/bin/env python3
"""Run the privacy-safe IBM QPU stress study and save presentation graphics."""

from __future__ import annotations

import argparse
from pathlib import Path

from domopt.checkpoints import checkpoint_identity, write_checkpoint
from domopt.experiments import (
    make_ibm_hardware_study_problem,
    rank_ibm_hardware_strategies,
    run_ibm_hardware_study,
    write_experiment_results,
)
from domopt.hardware import discover_ibm_backends
from domopt.pipeline import current_source_state
from domopt.visualization import plot_ibm_backend_snapshot, plot_ibm_hardware_study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", help="IBM backend name; default selects least busy")
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument(
        "--profile",
        choices=["quick", "presentation"],
        default="quick",
        help="quick submits 4 jobs; presentation repeats the matrix over 3 seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/challenge-study/cli/ibm-hardware"),
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Required privacy gate; the study sends generated synthetic circuits only",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow an explicitly provisional hardware run from uncommitted source",
    )
    args = parser.parse_args()
    if not args.allow_remote:
        parser.error("--allow-remote is required after IBM account/privacy approval")
    if current_source_state().get("git_dirty") is True and not args.allow_dirty:
        parser.error(
            "commit the worktree before hardware execution, or pass --allow-dirty "
            "for an explicitly provisional run"
        )

    queue = discover_ibm_backends(min_num_qubits=16)
    selected = args.backend or str(
        queue.loc[queue["selected_least_busy"], "backend"].iloc[0]
    )
    if selected not in set(queue["backend"].astype(str)):
        parser.error(f"Requested backend {selected!r} is not in the eligible queue snapshot")
    queue["selected_for_study"] = queue["backend"].astype(str).eq(selected)
    output = args.output_dir
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output / "ibm_backend_snapshot.csv", index=False)
    plot_ibm_backend_snapshot(queue, figures / "ibm_backend_queue.png")

    results_path = output / "ibm_hardware_stress.csv"
    results = run_ibm_hardware_study(
        allow_remote=True,
        backend_name=selected,
        shots=args.shots,
        profile=args.profile,
        progress_callback=lambda frame: write_experiment_results(frame, results_path),
        allow_dirty=args.allow_dirty,
    )
    csv_path = write_experiment_results(results, results_path)
    identity = checkpoint_identity(
        make_ibm_hardware_study_problem(),
        profile=f"ibm-{args.profile}",
        experiment="ibm_hardware_stress",
        configuration={
            "backend": selected,
            "shots": args.shots,
            "hardware_profile": args.profile,
            "data_scope": "independently generated coupled synthetic control",
        },
    )
    _, manifest_path = write_checkpoint(results, csv_path, identity)
    ranking = rank_ibm_hardware_strategies(results)
    ranking_path = write_experiment_results(
        ranking,
        output / "ibm_strategy_ranking.csv",
    )
    plot_ibm_hardware_study(results, figures / "ibm_hardware_stress.png")
    print(queue.to_string(index=False))
    print(f"Selected backend: {selected}")
    print("Observed hardware-strategy ranking:")
    print(ranking.to_string(index=False))
    print(f"Wrote {len(results)} rows to {csv_path}")
    print(f"Wrote provenance manifest to {manifest_path}")
    print(f"Wrote strategy ranking to {ranking_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
