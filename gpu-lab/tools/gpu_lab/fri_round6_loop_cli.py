"""Command line for the fail-closed FRI round-6 preflight lane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .fri_round6_loop import preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-index", required=True, type=Path)
    parser.add_argument("--module-index-sha256", required=True)
    parser.add_argument("--module-sha256", required=True)
    parser.add_argument("--build-recipe-sha256", required=True)
    for fixture in ("primary", "hostile"):
        parser.add_argument(f"--{fixture}", required=True, type=Path)
        parser.add_argument(f"--{fixture}-sha256", required=True)
        parser.add_argument(f"--{fixture}-index", required=True, type=Path)
        parser.add_argument(f"--{fixture}-index-sha256", required=True)
    parser.add_argument("--harness", required=True, type=Path)
    parser.add_argument("--harness-sha256", required=True)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--target-sm", required=True, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument(
        "--correctness-only-unsealed",
        action="store_true",
        help="accept only the explicitly non-production captured-unsealed fixture class",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate and seal identities without launching the GPU runner",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        record = preflight(args)
    except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
        print(f"fri-round6-loop: {error}", file=sys.stderr)
        return 1
    print(json.dumps(record, indent=2, sort_keys=True))
    print(
        "FRI round-6 preflight PASS: correctness-only-unsealed; "
        "production/correctness/performance admission all FALSE",
        file=sys.stderr,
    )
    return 0
