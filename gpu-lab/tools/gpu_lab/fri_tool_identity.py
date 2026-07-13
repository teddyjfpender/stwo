"""Explicit build-tool closure for the exact FRI module compiler."""

from __future__ import annotations

from typing import Any

from .common import (
    LAB_ROOT,
    canonical_bytes,
    require,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
    sha256_file,
)


FRI_BUILD_TOOL_PATHS = [
    "tools/fri-module",
    "tools/gpu_lab/__init__.py",
    "tools/gpu_lab/fri_module_cli.py",
    "tools/gpu_lab/fri_discovery.py",
    "tools/gpu_lab/fri_module.py",
    "tools/gpu_lab/fri_publication.py",
    "tools/gpu_lab/fri_staging.py",
    "tools/gpu_lab/fri_tool_identity.py",
    "tools/gpu_lab/common.py",
    "tools/gpu_lab/identity.py",
    "tools/gpu_lab/immutable_output.py",
]


def fri_build_tool_sources() -> list[dict[str, str]]:
    paths = [LAB_ROOT / relative for relative in FRI_BUILD_TOOL_PATHS]
    require(all(path.is_file() for path in paths), "FRI build-tool closure is incomplete")
    return [{"path": relative, "sha256": sha256_file(path)}
            for relative, path in zip(FRI_BUILD_TOOL_PATHS, paths)]


def fri_build_tool_sha256() -> str:
    return sha256_bytes(canonical_bytes(fri_build_tool_sources()))


def validate_fri_build_tool(sources: Any, digest: Any) -> None:
    require(isinstance(sources, list), "FRI build-tool source closure is missing")
    for source in sources:
        require(isinstance(source, dict), "FRI build-tool identity must be an object")
        require_exact_keys(source, {"path", "sha256"}, "FRI build-tool identity")
        require_sha256(source["sha256"], "FRI build-tool source sha256")
    require([source["path"] for source in sources] == FRI_BUILD_TOOL_PATHS,
            "FRI build-tool source path/order differs")
    require_sha256(digest, "FRI build-tool sha256")
    require(digest == sha256_bytes(canonical_bytes(sources)),
            "FRI build-tool hash does not bind its source closure")
