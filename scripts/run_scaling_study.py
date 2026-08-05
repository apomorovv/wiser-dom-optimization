#!/usr/bin/env python3
"""Benchmark baselines, exact MILP, hybrid scaling, and QUBO coefficient noise."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from domopt.baselines import solve_default_baseline, solve_greedy_baseline
from domopt.classical import solve_classical
from domopt.hybrid import HybridConfig, solve_hybrid
from domopt.metrics import compute_metrics
from domopt.synthetic import make_synthetic_problem


def _csv_numbers(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="20,50,100")
    parser.add_argument("--noise", default="0,0.01,0.05")
    parser.add_argument("--classical-max-orders", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("results/scaling.csv"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for order_count in _csv_numbers(args.sizes, int):
        problem = make_synthetic_problem(order_count=order_count, seed=args.seed)
        methods = [
            ("default", solve_default_baseline(problem)),
            ("greedy", solve_greedy_baseline(problem)),
        ]
        if order_count <= args.classical_max_orders:
            methods.append(
                (
                    "classical",
                    solve_classical(problem, time_limit_seconds=60, mip_relative_gap=0.01),
                )
            )
        for name, solution in methods:
            rows.append(
                {
                    "order_count": order_count,
                    "noise_relative_sigma": 0.0,
                    "seed": args.seed,
                    **compute_metrics(problem, solution),
                    "method": name,
                }
            )

        for noise in _csv_numbers(args.noise, float):
            settings = HybridConfig(
                iterations=6,
                neighborhood_orders=8,
                max_qubo_variables=40,
                sampler="simulated_annealing",
                num_reads=32,
                sweeps=100,
                top_k_recourse=4,
                seed=args.seed,
                qubo_noise_relative_sigma=noise,
            )
            solution = solve_hybrid(
                problem,
                config=settings,
            )
            rows.append(
                {
                    "order_count": order_count,
                    "noise_relative_sigma": noise,
                    "seed": args.seed,
                    "hybrid_iterations": settings.iterations,
                    "hybrid_neighborhood_orders": settings.neighborhood_orders,
                    "hybrid_qubo_cap": settings.max_qubo_variables,
                    "hybrid_candidate_cap": settings.max_candidates_per_order,
                    "hybrid_num_reads": settings.num_reads,
                    "hybrid_sweeps": settings.sweeps,
                    "hybrid_top_k_recourse": settings.top_k_recourse,
                    **compute_metrics(problem, solution),
                }
            )

    output = pd.DataFrame(rows).drop(columns=["violations"], errors="ignore")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
