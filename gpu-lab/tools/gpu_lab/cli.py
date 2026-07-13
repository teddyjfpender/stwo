"""Command-line routing for GPU lab."""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

from .artifacts import prepare
from .baselines import BASELINE_ROOT, add_result_arguments, compare_baseline
from .common import LAB_ROOT, sha256_file
from .identity import build_module, write_toolchain_identity
from .loop import capture_environment, write_loop_record
from .oracle_seal import seal_oracle
from .results import validate_result
from .selftest import self_test
from .semantics import make_tiny_fixture, validate_fixture

DESCRIPTION = """GPU-lab build identity, independent tiny oracle, and replay preparation.

Only the `run` phase needs CUDA.  Fixture creation and validation are stdlib-only
so they remain usable on a developer host and cannot accidentally initialize a
candidate device.
"""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=DESCRIPTION)
    sub = result.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make-tiny")
    make.add_argument("output", type=Path)

    validate = sub.add_parser("validate-fixture")
    validate.add_argument("fixture", type=Path)

    build = sub.add_parser("build-module")
    build.add_argument("--nvcc", required=True)
    build.add_argument("--host-compiler", required=True)
    build.add_argument("--sm", required=True, type=int)
    build.add_argument("--source", required=True, type=Path)
    build.add_argument("--abi", required=True, type=Path)
    build.add_argument("--repo-root", required=True, type=Path)
    build.add_argument("--workspace-root", required=True, type=Path)
    build.add_argument("--generator", action="append", default=[], type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--index", required=True, type=Path)
    build.add_argument("--stamp", required=True, type=Path)
    build.add_argument("--repro-check", action="store_true")

    toolchain = sub.add_parser("toolchain-identity")
    toolchain.add_argument("--nvcc", required=True)
    toolchain.add_argument("--host-compiler", required=True)
    toolchain.add_argument("--output", required=True, type=Path)

    prep = sub.add_parser("prepare")
    prep.add_argument("--fixture", required=True, type=Path)
    prep.add_argument("--oracle-index", required=True, type=Path)
    prep.add_argument("--module-index", required=True, type=Path)
    prep.add_argument("--abi", required=True, type=Path)
    prep.add_argument("--sm", required=True, type=int)
    prep.add_argument("--execution", required=True, type=Path)
    prep.add_argument("--replay", required=True, type=Path)
    prep.add_argument("--plan", required=True, type=Path)

    seal = sub.add_parser("seal-oracle")
    seal.add_argument("--fixture", required=True, type=Path)
    seal.add_argument("--artifact", required=True, type=Path)
    seal.add_argument("--exporter", required=True, type=Path)
    seal.add_argument("--output", required=True, type=Path)

    result_cmd = sub.add_parser("validate-result")
    result_cmd.add_argument("--result", required=True, type=Path)
    result_cmd.add_argument("--fixture", required=True, type=Path)
    result_cmd.add_argument("--oracle-index", required=True, type=Path)
    result_cmd.add_argument("--module-index", required=True, type=Path)
    result_cmd.add_argument("--execution", required=True, type=Path)
    result_cmd.add_argument("--replay", required=True, type=Path)
    result_cmd.add_argument("--plan", required=True, type=Path)
    result_cmd.add_argument("--harness", required=True, type=Path)
    result_cmd.add_argument("--mode", required=True, choices=("correctness", "benchmark"))
    result_cmd.add_argument("--allow-performance-failure", action="store_true")

    environment = sub.add_parser("capture-environment")
    environment.add_argument("--device", required=True)
    environment.add_argument("--output", required=True, type=Path)

    compare = sub.add_parser("compare-baseline")
    add_result_arguments(compare)
    compare.add_argument("--harness", required=True, type=Path)
    compare.add_argument("--environment-before", required=True, type=Path)
    compare.add_argument("--environment-after", required=True, type=Path)
    compare.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--allow-performance-failure", action="store_true")
    compare.set_defaults(mode="benchmark")

    loop = sub.add_parser("write-loop")
    loop.add_argument("--phase", action="append", required=True)
    loop.add_argument("--artifact", action="append", required=True)
    loop.add_argument("--total", required=True)
    loop.add_argument("--environment-before", required=True, type=Path)
    loop.add_argument("--environment-after", required=True, type=Path)
    loop.add_argument("--harness", required=True, type=Path)
    loop.add_argument("--orchestrator", required=True, type=Path)
    loop.add_argument("--slo-checker", required=True, type=Path)
    loop.add_argument("--staging-record", required=True, type=Path)
    loop.add_argument("--runtime-mode", action="append", required=True)
    loop.add_argument("--benchmark-exit-code", required=True, type=int)
    loop.add_argument("--performance-status", required=True,
                      choices=("admissible", "non_admissible"))
    loop.add_argument("--output", required=True, type=Path)

    sub.add_parser("self-test")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "make-tiny":
            make_tiny_fixture(args.output)
        elif args.command == "validate-fixture":
            validate_fixture(args.fixture)
            print(f"fixture valid: {args.fixture} sha256={sha256_file(args.fixture)}")
        elif args.command == "build-module":
            build_module(args)
        elif args.command == "toolchain-identity":
            write_toolchain_identity(args)
        elif args.command == "prepare":
            prepare(args)
        elif args.command == "seal-oracle":
            seal_oracle(args.fixture, args.artifact, args.exporter, args.output)
            print(f"oracle wrapper sealed: {args.output} sha256={sha256_file(args.output)}")
        elif args.command == "validate-result":
            validate_result(args)
            print(f"result valid: {args.result}")
        elif args.command == "capture-environment":
            capture_environment(args.device, args.output)
            print(f"GPU environment recorded: {args.output}")
        elif args.command == "compare-baseline":
            compare_baseline(args)
        elif args.command == "write-loop":
            write_loop_record(args)
            print(f"loop record written: {args.output}")
        elif args.command == "self-test":
            self_test(LAB_ROOT)
        else:
            raise AssertionError(args.command)
    except (ValueError, TypeError, KeyError, IndexError, OSError, struct.error,
            subprocess.CalledProcessError) as error:
        print(f"gpu-lab: {error}", file=sys.stderr)
        return 1
    return 0
