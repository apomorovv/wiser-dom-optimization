#!/usr/bin/env python3
"""Create a clean, runtime-only Nestle challenge input directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from domopt.poc import prepare_poc_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the five optimizer inputs to stable names and optionally retain "
            "the two supplied recommendation outputs for reconciliation."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Exclude the two optional recommendation-output CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copied = prepare_poc_bundle(
        args.source_dir,
        args.output_dir,
        include_reference_outputs=not args.runtime_only,
    )
    for role, path in sorted(copied.items()):
        print(f"{role:>14}  {path.name}  {path.stat().st_size} bytes")
    print(f"Prepared {len(copied)} files in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

