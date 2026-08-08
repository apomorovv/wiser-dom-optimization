#!/usr/bin/env python3
"""Create a public-safe synthetic DOM instance."""

from __future__ import annotations

import argparse
from pathlib import Path

from domopt.data import save_problem_data
from domopt.synthetic import make_synthetic_problem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--orders", type=int, default=100)
    parser.add_argument("--dcs", type=int, default=4)
    parser.add_argument("--skus", type=int, default=12)
    parser.add_argument("--candidates-per-order", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    problem = make_synthetic_problem(
        order_count=args.orders,
        dc_count=args.dcs,
        sku_count=args.skus,
        candidates_per_order=args.candidates_per_order,
        seed=args.seed,
    )
    save_problem_data(problem, args.output_dir)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

