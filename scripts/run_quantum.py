#!/usr/bin/env python3
"""Build and sample a standalone reduced fixed-plan QUBO diagnostic.

This script expects ``plans.csv`` in ``--data-dir`` with columns ``plan_id``,
``order_id``, and ``value``. An optional ``conflicts.csv`` may contain
``plan_id_a`` and ``plan_id_b``. Converting detailed DOM fulfillment variables
into fixed plans is intentionally a separate, auditable preprocessing stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from domopt.quantum import sample_qubo
from domopt.qubo import build_candidate_qubo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=[
            "exact",
            "random",
            "simulated_annealing",
            "dwave-qpu",
            "dwave-hybrid",
        ],
        default="exact",
    )
    parser.add_argument("--one-hot-penalty", type=float, default=1000.0)
    parser.add_argument("--conflict-penalty", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--sweeps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Acknowledge approval to send QUBO coefficients to a remote service.",
    )
    parser.add_argument("--remote-time-limit-seconds", type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plans_path = args.data_dir / "plans.csv"
    if not plans_path.exists():
        raise FileNotFoundError(
            f"Missing {plans_path}. Generate fixed candidate plans before QUBO sampling."
        )
    plans = pd.read_csv(plans_path)
    conflicts_path = args.data_dir / "conflicts.csv"
    conflicts = pd.read_csv(conflicts_path) if conflicts_path.exists() else None

    model = build_candidate_qubo(
        plans,
        one_hot_penalty=args.one_hot_penalty,
        conflicts=conflicts,
        conflict_penalty=args.conflict_penalty,
    )
    samples = sample_qubo(
        model,
        method=args.method,
        num_samples=args.num_samples,
        sweeps=args.sweeps,
        seed=args.seed,
        allow_remote=args.allow_remote,
        time_limit_seconds=args.remote_time_limit_seconds,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.output_dir / "raw_samples.csv", index=False)
    payload = {
        "variable_names": list(model.variable_names),
        "Q": np.asarray(model.Q).tolist(),
        "constant": model.constant,
        "metadata": model.metadata,
    }
    (args.output_dir / "qubo.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(samples.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

