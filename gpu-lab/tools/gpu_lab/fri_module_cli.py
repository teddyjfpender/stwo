"""Dedicated command line for the exact FRI module build boundary."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .fri_module import build_fri_module, validate_fri_module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--nvcc", required=True)
        command.add_argument("--host-compiler", required=True)
        command.add_argument("--sm", required=True, type=int)
        command.add_argument("--source", required=True, type=Path)
        command.add_argument("--abi", required=True, type=Path)
        command.add_argument("--repo-root", required=True, type=Path)
        command.add_argument("--output-dir", required=True, type=Path)
        command.add_argument("--index", required=True, type=Path)
        command.add_argument("--stamp", required=True, type=Path)
        command.add_argument("--depfile", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        (build_fri_module if args.command == "build" else validate_fri_module)(args)
    except (ValueError, TypeError, KeyError, IndexError, OSError,
            subprocess.CalledProcessError) as error:
        print(f"fri-module: {error}", file=sys.stderr)
        return 1
    return 0
