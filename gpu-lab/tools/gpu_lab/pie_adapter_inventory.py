"""Generate the exact pre-build source inventory for the host PIE adapter."""

from __future__ import annotations

import argparse
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from .common import WORKSPACE_ROOT, canonical_bytes, require, sha256_bytes
from .immutable_output import write_immutable_bytes
from .pie_adapter_receipt import (
    ADAPTER_PROFILE,
    ADAPTER_REPOSITORY,
    ADAPTER_ROOT,
    INVENTORY_SCHEMA,
    REPOSITORIES,
    TARGET_NAMESPACE,
    discover_expected_sources,
    validate_inventory,
)


def parse_locator(value: str, label: str) -> dict[str, str]:
    """Parse the CLI's unambiguous ``repository:relative/path`` locator."""
    require(isinstance(value, str), f"{label} locator must be text")
    repository, separator, path = value.partition(":")
    require(separator == ":" and repository in REPOSITORIES and path,
            f"{label} must be REPOSITORY:RELATIVE_PATH")
    return {"repository": repository, "path": path}


def _require_isolated_target(target_directory: dict[str, str],
                             sources: list[dict[str, str]],
                             workspace_root: Path) -> None:
    repository = target_directory["repository"]
    relative = PurePosixPath(target_directory["path"])
    for source in sources:
        if source["repository"] != repository:
            continue
        source_path = PurePosixPath(source["path"])
        require(not source_path.is_relative_to(relative),
                "PIE adapter target directory contains a compiler input")
        require(not relative.is_relative_to(source_path),
                "PIE adapter target directory descends from a compiler input")
    repository_root = workspace_root / repository
    root_status = repository_root.lstat()
    require(stat.S_ISDIR(root_status.st_mode) and not stat.S_ISLNK(root_status.st_mode),
            "PIE adapter target repository root is not a real directory")
    current = repository_root
    for part in relative.parts:
        current /= part
        try:
            status = current.lstat()
        except FileNotFoundError:
            break
        require(stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode),
                f"PIE adapter target path crosses a non-directory or symlink: {current}")
    try:
        current = repository_root.joinpath(*relative.parts)
        current.lstat()
    except FileNotFoundError:
        return
    raise ValueError("PIE adapter target directory must not exist before inventory generation")


def make_inventory(target: str, target_directory: dict[str, str],
                   adapter_executable: dict[str, str],
                   workspace_root: Path = WORKSPACE_ROOT) -> dict[str, Any]:
    """Return one validated, non-admitting inventory for a fresh Linux build."""
    sources = discover_expected_sources(workspace_root)
    manifest = {"repository": ADAPTER_REPOSITORY, "path": f"{ADAPTER_ROOT}/Cargo.toml"}
    value: dict[str, Any] = {
        "schema_version": INVENTORY_SCHEMA,
        "repository_roots": {name: name for name in REPOSITORIES},
        "expected_sources": sources,
        "build": {
            "manifest": manifest,
            "cargo_lock": {"repository": ADAPTER_REPOSITORY,
                           "path": f"{ADAPTER_ROOT}/Cargo.lock"},
            "rust_toolchain": {"repository": ADAPTER_REPOSITORY,
                               "path": f"{ADAPTER_ROOT}/rust-toolchain.toml"},
            "profile": ADAPTER_PROFILE,
            "features": [],
            "default_features": True,
            "target": target,
            "target_directory": target_directory,
            "working_directory": f"<workspace>/{ADAPTER_REPOSITORY}/{ADAPTER_ROOT}",
            "environment": {
                "CARGO_TARGET_DIR": (
                    f"<workspace>/{target_directory['repository']}/{target_directory['path']}"
                )
            },
            "command": [
                "cargo", "build", "--manifest-path", "Cargo.toml", "--release",
                "--locked", "--offline", "--target", target,
            ],
        },
        "generated_outputs": [adapter_executable],
        "adapter_executable": adapter_executable,
    }
    validate_inventory(value, workspace_root)
    _require_isolated_target(target_directory, sources, workspace_root)
    return value


def write_inventory(output: Path, target: str, target_directory: dict[str, str],
                    adapter_executable: dict[str, str],
                    workspace_root: Path = WORKSPACE_ROOT) -> tuple[dict[str, Any], str]:
    inventory = make_inventory(
        target, target_directory, adapter_executable, workspace_root,
    )
    namespace_path = workspace_root / ADAPTER_REPOSITORY / TARGET_NAMESPACE
    output_path = output.resolve(strict=False)
    require(not output_path.is_relative_to(namespace_path.resolve(strict=False)),
            "PIE adapter inventory output must be outside the receipt-targets namespace")
    payload = canonical_bytes(inventory) + b"\n"
    sources = [workspace_root / item["repository"] / item["path"]
               for item in inventory["expected_sources"]]

    def revalidate(_: Path) -> None:
        require(make_inventory(target, target_directory, adapter_executable, workspace_root)
                == inventory, "PIE adapter inventory inputs changed before publication")

    write_immutable_bytes(output, payload, sources, _before_publish=revalidate)
    return inventory, sha256_bytes(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True,
                        help="exact Rust Linux target triple")
    parser.add_argument("--target-directory", required=True,
                        help="fresh REPOSITORY:RELATIVE_PATH for CARGO_TARGET_DIR")
    parser.add_argument("--adapter-executable", required=True,
                        help="exact REPOSITORY:RELATIVE_PATH of the release adapter")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None, *,
         workspace_root: Path = WORKSPACE_ROOT) -> int:
    args = _parser().parse_args(argv)
    try:
        _, digest = write_inventory(
            args.output,
            args.target,
            parse_locator(args.target_directory, "target directory"),
            parse_locator(args.adapter_executable, "adapter executable"),
            workspace_root,
        )
    except (OSError, ValueError) as error:
        print(f"pie-adapter-inventory: FAIL: {error}", file=sys.stderr)
        return 2
    print(f"PIE_ADAPTER_INVENTORY=PASS path={args.output} sha256={digest} "
          "build_execution_attested=false production_admissible=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
