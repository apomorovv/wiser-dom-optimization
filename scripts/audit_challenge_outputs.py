#!/usr/bin/env python3
"""Validate challenge output CSVs without emitting raw identifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from domopt.challenge_outputs import summarize_challenge_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("order_level_csv", type=Path)
    parser.add_argument("sku_level_csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-commercial-metrics",
        action="store_true",
        help="Use only for an authorized private analysis.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = summarize_challenge_outputs(
        args.order_level_csv,
        args.sku_level_csv,
        include_commercial_metrics=args.include_commercial_metrics,
    )
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

