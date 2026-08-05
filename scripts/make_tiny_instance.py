#!/usr/bin/env python3
"""Generate the canonical two-order synthetic DOM instance."""

from __future__ import annotations

import argparse
from pathlib import Path

from domopt.data import make_tiny_problem_data, save_problem_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic/tiny"),
        help="Directory that will receive canonical CSV/JSON files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = save_problem_data(make_tiny_problem_data(), args.output_dir)
    print(f"Wrote tiny DOM instance to {output}")
    for path in sorted(output.iterdir()):
        if path.is_file():
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


