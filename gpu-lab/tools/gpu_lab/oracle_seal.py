"""Deterministic sealing of a reviewed production host-oracle wrapper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from .common import WORKSPACE_ROOT, canonical_bytes, require, sha256_file
from .immutable_output import write_immutable_bytes
from .indexed_fixture import SCHEMA_INDEXED_FIXTURE, validate_indexed_oracle
from .oracle import indexed_production_crosscheck
from .semantics import validate_fixture


def _workspace_file(path: Path, workspace: Path, label: str) -> Path:
    root = workspace.resolve(strict=True)
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(),
            f"{label} is not a regular file inside the workspace")
    return resolved


def _workspace_output(path: Path, workspace: Path) -> Path:
    resolved = path.resolve(strict=False)
    require(resolved.is_relative_to(workspace.resolve(strict=True)),
            "oracle wrapper output escapes the workspace")
    return resolved


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    status = path.stat()
    return (status.st_dev, status.st_ino, status.st_size,
            status.st_mtime_ns, status.st_ctime_ns)


def _while_executable_stable(path: Path, operation: Callable[[str], Any]) -> Any:
    before_identity, before_hash = _file_identity(path), sha256_file(path)
    result = operation(before_hash)
    after_hash, after_identity = sha256_file(path), _file_identity(path)
    require(before_hash == after_hash and before_identity == after_identity,
            "exporter executable changed while its oracle wrapper was sealed")
    return result


def _require_distinct(paths: list[tuple[str, Path]]) -> None:
    for index, (left_label, left) in enumerate(paths):
        for right_label, right in paths[index + 1:]:
            require(not left.samefile(right),
                    f"{left_label} aliases {right_label}")


def seal_oracle(
    fixture_path: Path, artifact_path: Path, exporter_path: Path, output: Path,
    workspace_root: Path = WORKSPACE_ROOT,
) -> dict[str, Any]:
    """Seal one v2 fixture/artifact using the actual stable exporter executable."""
    workspace = workspace_root.resolve(strict=True)
    fixture_file = _workspace_file(fixture_path, workspace, "fixture")
    artifact_file = _workspace_file(artifact_path, workspace, "oracle artifact")
    exporter_file = _workspace_file(exporter_path, workspace, "exporter executable")
    output_file = _workspace_output(output, workspace)
    require(os.access(exporter_file, os.X_OK), "exporter file is not executable")
    _require_distinct([
        ("fixture", fixture_file), ("oracle artifact", artifact_file),
        ("exporter executable", exporter_file),
    ])

    def build(executable_hash: str) -> tuple[dict[str, Any], list[Path]]:
        fixture_hash, artifact_hash = sha256_file(fixture_file), sha256_file(artifact_file)
        fixture = validate_fixture(fixture_file)
        require(fixture.get("schema_version") == SCHEMA_INDEXED_FIXTURE,
                "seal-oracle accepts only production indexed fixtures")
        require(fixture["exporter_executable_sha256"] == executable_hash,
                "fixture was emitted by another exporter executable")
        validate_indexed_oracle(artifact_file, fixture_file, fixture)
        closure = indexed_production_crosscheck(workspace)
        require(fixture["production_crosscheck"] == closure,
                "fixture production cross-check differs from sealing workspace")
        artifact_relative = artifact_file.relative_to(workspace).as_posix()
        require(artifact_relative.startswith(("evidence/", "scratchpad/")),
                "oracle artifact must live under evidence/ or scratchpad/")
        wrapper = {
            "schema_version": "stwo.gpu-lab.host-oracle-index.v1",
            "fixture_sha256": fixture_hash,
            "oracle_artifact": {"path": artifact_relative, "sha256": artifact_hash},
            "exporter": {**closure, "executable_sha256": executable_hash},
        }
        require(sha256_file(fixture_file) == fixture_hash
                and sha256_file(artifact_file) == artifact_hash
                and indexed_production_crosscheck(workspace) == closure,
                "oracle inputs changed while their wrapper was sealed")
        sources = [fixture_file, artifact_file, exporter_file]
        sources.extend(workspace / source["path"] for source in closure["sources"])
        return wrapper, sources

    wrapper, sources = _while_executable_stable(exporter_file, build)
    require(sha256_file(exporter_file) == wrapper["exporter"]["executable_sha256"],
            "exporter executable changed before wrapper installation")
    write_immutable_bytes(output_file, canonical_bytes(wrapper) + b"\n", sources)
    return wrapper
