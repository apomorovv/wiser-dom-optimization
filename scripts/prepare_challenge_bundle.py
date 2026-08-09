#!/usr/bin/env python3
"""Create a clean, runtime-only Nestle challenge input directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from domopt.poc import PocDataError, prepare_poc_bundle


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir.expanduser()
    try:
        copied = prepare_poc_bundle(
            args.source_dir.expanduser(),
            output_dir,
            include_reference_outputs=not args.runtime_only,
        )
    except PocDataError as error:
        parser.error(str(error))
    for role, path in sorted(copied.items()):
        print(f"{role:>14}  {path.name}  {path.stat().st_size} bytes")
    prepared = output_dir.resolve()
    print(f"Prepared {len(copied)} files in {prepared}")
    print("Next:")
    print(
        "  python scripts/run_challenge_study.py "
        f'--bundle-dir "{prepared}" --profile smoke'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
