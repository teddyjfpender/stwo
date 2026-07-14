"""Explicit source closure for the sole-purpose FRI round-6 loop."""

from __future__ import annotations

from typing import Any

from .common import LAB_ROOT, canonical_bytes, require, require_exact_keys, require_sha256
from .common import sha256_bytes, sha256_file


LOOP_TOOL_PATHS = [
    "tools/fri-round6-loop",
    "tools/gpu_lab/__init__.py",
    "tools/gpu_lab/common.py",
    "tools/gpu_lab/identity.py",
    "tools/gpu_lab/immutable_output.py",
    "tools/gpu_lab/fri_discovery.py",
    "tools/gpu_lab/fri_staging.py",
    "tools/gpu_lab/fri_tool_identity.py",
    "tools/gpu_lab/fri_module.py",
    "tools/gpu_lab/fri_publication.py",
    "tools/gpu_lab/fri_round6_fixture.py",
    "tools/gpu_lab/fri_round6_execution.py",
    "tools/gpu_lab/fri_round6_process.py",
    "tools/gpu_lab/sealed_process.py",
    "tools/gpu_lab/fri_round6_loop.py",
    "tools/gpu_lab/fri_round6_loop_cli.py",
    "tools/gpu_lab/fri_round6_loop_identity.py",
]


def loop_tool_sources() -> list[dict[str, str]]:
    paths = [LAB_ROOT / relative for relative in LOOP_TOOL_PATHS]
    require(all(path.is_file() and not path.is_symlink() for path in paths),
            "FRI loop-tool closure is incomplete")
    return [{"path": relative, "sha256": sha256_file(path)}
            for relative, path in zip(LOOP_TOOL_PATHS, paths)]


def loop_tool_sha256() -> str:
    return sha256_bytes(canonical_bytes(loop_tool_sources()))


def validate_loop_tool(sources: Any, digest: Any) -> None:
    require(isinstance(sources, list), "FRI loop-tool source closure is missing")
    require([source.get("path") for source in sources if isinstance(source, dict)]
            == LOOP_TOOL_PATHS, "FRI loop-tool source path/order differs")
    for source in sources:
        require(isinstance(source, dict), "FRI loop-tool identity must be an object")
        require_exact_keys(source, {"path", "sha256"}, "FRI loop-tool identity")
        require_sha256(source["sha256"], "FRI loop-tool source sha256")
    require_sha256(digest, "FRI loop-tool sha256")
    require(digest == sha256_bytes(canonical_bytes(sources)),
            "FRI loop-tool hash does not bind its source closure")
