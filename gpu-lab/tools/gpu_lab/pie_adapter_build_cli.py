"""Command line entry point for one non-admitting sealed PIE-adapter build."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .common import sha256_file
from .pie_adapter_build_execution import execute


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "source-closure", "cache-identity", "cargo", "rustc",
                 "native-compiler-linker", "native-archiver"):
        destination = name.replace("-", "_")
        parser.add_argument(f"--{name}", required=True, type=Path, dest=destination)
        parser.add_argument(f"--{name}-sha256", required=True,
                            dest=f"{destination}_sha256")
    parser.add_argument("--builder-image-id", required=True)
    parser.add_argument("--builder-image-digest", required=True)
    parser.add_argument("--environment-json", required=True,
                        help="exact JSON object passed as the builder's complete environment")
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        execute(args)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"pie-adapter-build-execute: FAIL: {error}", file=sys.stderr)
        return 2
    print(f"PIE_ADAPTER_BUILD_EXECUTION=PASS record={args.record} "
          f"sha256={sha256_file(args.record)} build_execution_attested=true "
          "builder_runtime_attested=false "
          "production_admissible=false correctness_admissible=false "
          "performance_admissible=false")
    return 0
