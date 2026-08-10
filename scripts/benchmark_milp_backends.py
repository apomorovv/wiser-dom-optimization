#!/usr/bin/env python3
"""Compare installed MILP adapters on one identically compiled DOM instance."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from domopt.classical import (
    ClassicalSolverError,
    available_cpu_count,
    available_milp_backends,
    solve_classical,
)
from domopt.objective import evaluate_solution
from domopt.poc import PocConfig, load_poc_problem, select_shortage_subset
from domopt.validation import validate_solution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["scipy-highs", "highspy", "scip", "gurobi"],
        choices=["scipy-highs", "highspy", "scip", "gurobi"],
    )
    parser.add_argument(
        "--assignment-groups",
        type=int,
        default=20,
        help="Shortage-ranked group count; use 0 for the complete instance.",
    )
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument(
        "--threads",
        type=int,
        help="Optional cap for native adapters; default uses the visible CPU budget.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.assignment_groups < 0:
        raise SystemExit("--assignment-groups must be nonnegative")
    if args.time_limit <= 0:
        raise SystemExit("--time-limit must be positive")
    if not 0 <= args.mip_gap < 1:
        raise SystemExit("--mip-gap must be in [0, 1)")
    if args.threads is not None and args.threads <= 0:
        raise SystemExit("--threads must be positive")

    problem = load_poc_problem(
        args.bundle_dir,
        config=PocConfig(pareto_prune=False),
    )
    if args.assignment_groups:
        problem = select_shortage_subset(problem, args.assignment_groups)

    installed = available_milp_backends()
    rows: list[dict[str, object]] = []
    for backend in args.backends:
        common = {
            "milp_backend": backend,
            "installed": installed[backend],
            "assignment_group_count": int(problem.orders["assignment_group"].nunique()),
            "order_count": len(problem.orders),
            "order_line_count": len(problem.order_lines),
            "candidate_count": len(problem.candidates),
            "visible_cpu_count": available_cpu_count(),
            "requested_thread_count": args.threads,
            "time_limit_seconds": args.time_limit,
            "requested_mip_gap": args.mip_gap,
            "seed": args.seed,
        }
        if not installed[backend]:
            rows.append({**common, "status": "not installed", "feasible": False})
            continue
        try:
            solution = solve_classical(
                problem,
                backend=backend,
                time_limit_seconds=args.time_limit,
                mip_relative_gap=args.mip_gap,
                seed=args.seed,
                thread_count=args.threads,
            )
            validation = validate_solution(problem, solution)
            objective = evaluate_solution(problem, solution)
            rows.append(
                {
                    **common,
                    "status": solution.metadata.get("message"),
                    "feasible": validation.is_feasible,
                    "objective_value": objective.objective_value,
                    "runtime_seconds": solution.runtime_seconds,
                    "optimality_gap": solution.metadata.get("optimality_gap"),
                    "best_bound": solution.metadata.get("best_bound"),
                    "mip_node_count": solution.metadata.get("mip_node_count"),
                    "n_variables": solution.metadata.get("n_variables"),
                    "n_constraints": solution.metadata.get("n_constraints"),
                    "effective_thread_count": solution.metadata.get("thread_count"),
                    "solver": solution.metadata.get("solver"),
                    "validation_violation_count": len(validation.violations),
                }
            )
        except ClassicalSolverError as error:
            rows.append(
                {
                    **common,
                    "status": "solver error",
                    "feasible": False,
                    "error": str(error),
                }
            )

    results = pd.DataFrame(rows)
    display_columns = [
        "milp_backend",
        "installed",
        "status",
        "feasible",
        "objective_value",
        "runtime_seconds",
        "optimality_gap",
        "effective_thread_count",
    ]
    print(
        results[[column for column in display_columns if column in results]].to_string(index=False)
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"Wrote aggregate backend comparison to {args.output}")
    return 0 if bool(results.get("feasible", pd.Series(dtype=bool)).any()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
