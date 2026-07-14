"""Fail-closed orchestration for the typed FRI round-6 replay runner."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import REPO_ROOT, canonical_bytes, load_json, require, require_exact_keys
from .common import require_int, require_sha256, sha256_bytes, sha256_file
from .fri_discovery import validate_closure
from .fri_module import ABI_RELATIVE, _validate_index_document, validate_fri_abi
from .fri_round6_fixture import (
    fixture_identity,
    validate_fixture,
    validate_fixture_identity,
    validate_pair,
)
from .fri_round6_loop_identity import loop_tool_sha256, loop_tool_sources, validate_loop_tool
from .fri_tool_identity import fri_build_tool_sha256, fri_build_tool_sources
from .fri_tool_identity import validate_fri_build_tool
from .immutable_output import guard_output, write_immutable_bytes


RECORD_SCHEMA = "stwo.gpu-lab.fri-round6-loop-record.v1"
RESULT_SCHEMA = "stwo.gpu-lab.fri-round6-result.v1"
EVIDENCE_MODE = "correctness-only-unsealed"
VALIDATION_WORDS = 416
SEMANTIC_ROLES = {
    "FRI ABI", "FRI build recipe", "FRI cubin", "FRI module index",
    "FRI recipe-module binding", "FRI runner", "hostile FRI index",
    "hostile FRI payload", "primary FRI index", "primary FRI payload",
}


@dataclass(frozen=True)
class BoundFile:
    role: str
    path: Path
    sha256: str
    byte_length: int
    identity: tuple[int, int, int, int, int, int, int]
    executable: bool = False

    def record(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "bytes": self.byte_length}


def _identity(status: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        status.st_dev, status.st_ino, status.st_size,
        status.st_mtime_ns, status.st_ctime_ns,
        stat.S_IMODE(status.st_mode), status.st_nlink,
    )


def _reject_symlink_components(path: Path, role: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in [*reversed(absolute.parents), absolute]:
        require(not component.is_symlink(), f"{role} path contains a symlink: {component}")
    return absolute


def observe_file(
    path: Path, role: str, *, executable: bool = False, read_only: bool = False,
) -> BoundFile:
    absolute = _reject_symlink_components(path, role)
    resolved = absolute.resolve(strict=True)
    require(resolved == absolute, f"{role} path changed while it was resolved")
    before = resolved.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_size > 0, f"{role} is not a nonempty file")
    require(not executable or before.st_mode & 0o111, f"{role} is not executable")
    require(not read_only or (stat.S_IMODE(before.st_mode) == 0o400 and before.st_nlink == 1),
            f"{role} is not a single-link owner-read-only file")
    digest = sha256_file(resolved)
    rebound = _reject_symlink_components(path, role).resolve(strict=True)
    require(rebound == resolved, f"{role} path changed while it was hashed")
    after = rebound.stat()
    require(_identity(before) == _identity(after), f"{role} changed while it was hashed")
    return BoundFile(role, resolved, digest, before.st_size, _identity(before), executable)


def bind_file(
    path: Path, expected_sha256: str, role: str, *, executable: bool = False,
) -> BoundFile:
    require_sha256(expected_sha256, f"{role} sha256")
    bound = observe_file(path, role, executable=executable)
    require(bound.sha256 == expected_sha256, f"{role} sha256 mismatch")
    return bound


def recheck(bound: BoundFile) -> None:
    require(bind_file(bound.path, bound.sha256, bound.role,
                      executable=bound.executable) == bound,
            f"{bound.role} identity changed")


def _module_bundle(args: Any) -> tuple[dict[str, Any], list[BoundFile]]:
    index_bound = bind_file(args.module_index, args.module_index_sha256, "FRI module index")
    require(index_bound.path.name == f"{index_bound.sha256}.module-index.json",
            "FRI authority is not the immutable content-addressed module index")
    index = load_json(index_bound.path)
    require(index_bound.path.read_bytes() == canonical_bytes(index),
            "FRI immutable module index is not canonical")
    abi_path = (REPO_ROOT / ABI_RELATIVE).resolve()
    abi = validate_fri_abi(abi_path, REPO_ROOT)
    module_path, recipe = _validate_index_document(
        index, abi, REPO_ROOT, index_bound.path.parent, args.target_sm,
    )
    require(index["build_recipe_hash"] == args.build_recipe_sha256,
            "FRI build recipe sha256 differs from the externally pinned value")
    require(index["module_content_sha256"] == args.module_sha256,
            "FRI module sha256 differs from the externally pinned value")
    closure_paths = validate_closure(recipe["source_closure"], REPO_ROOT, check_bytes=True)
    validate_fri_build_tool(recipe["build_tool_sources"], recipe["build_tool_sha256"])
    require(recipe["build_tool_sources"] == fri_build_tool_sources()
            and recipe["build_tool_sha256"] == fri_build_tool_sha256(),
            "FRI module build-tool closure differs from the live reviewed tool")
    binding_path = index_bound.path.parent / f"{args.build_recipe_sha256}.module-sha256"
    require(binding_path.read_bytes() == (args.module_sha256 + "\n").encode(),
            "FRI recipe-module binding bytes differ")
    binding_sha256 = sha256_file(binding_path)
    bounds = [
        index_bound,
        bind_file(Path(index["build_recipe_path"]), args.build_recipe_sha256, "FRI build recipe"),
        bind_file(module_path, args.module_sha256, "FRI cubin"),
        bind_file(binding_path, binding_sha256, "FRI recipe-module binding"),
        bind_file(abi_path, recipe["abi_sha256"], "FRI ABI"),
    ]
    bounds.extend(bind_file(path, source["sha256"], f"FRI source {source['path']}")
                  for path, source in zip(closure_paths, recipe["source_closure"]))
    bounds.extend(bind_file(REPO_ROOT / "gpu-lab" / source["path"], source["sha256"],
                            f"FRI build tool {source['path']}")
                  for source in recipe["build_tool_sources"])
    metadata = {
        "abi_sha256": recipe["abi_sha256"],
        "source_closure_sha256": recipe["source_closure_sha256"],
        "build_tool_sha256": recipe["build_tool_sha256"],
        "toolchain_sha256": recipe["toolchain_sha256"],
    }
    return metadata, bounds


def _fixture_bundle(args: Any) -> tuple[dict[str, Any], list[BoundFile]]:
    require(args.correctness_only_unsealed,
            "production mode is closed: no reviewed production FRI fixture schema is registered")
    bounds = [
        bind_file(args.primary, args.primary_sha256, "primary FRI payload"),
        bind_file(args.primary_index, args.primary_index_sha256, "primary FRI index"),
        bind_file(args.hostile, args.hostile_sha256, "hostile FRI payload"),
        bind_file(args.hostile_index, args.hostile_index_sha256, "hostile FRI index"),
    ]
    primary = validate_fixture(bounds[1].path, bounds[1].sha256,
                               bounds[0].path, bounds[0].sha256, "primary")
    hostile = validate_fixture(bounds[3].path, bounds[3].sha256,
                               bounds[2].path, bounds[2].sha256, "hostile")
    validate_pair(primary, hostile)
    return fixture_identity(primary), bounds


def _loop_tool_bounds(sources: list[dict[str, str]]) -> list[BoundFile]:
    return [bind_file(REPO_ROOT / "gpu-lab" / source["path"], source["sha256"],
                      f"FRI loop tool {source['path']}") for source in sources]


def _artifact_records(bounds: list[BoundFile]) -> dict[str, dict[str, Any]]:
    selected = {
        bound.role: bound.record() for bound in bounds
        if bound.role in SEMANTIC_ROLES
    }
    require(len(selected) == 10, "FRI preflight artifact roles are incomplete")
    return dict(sorted(selected.items()))


def _require_semantic_distinct(bounds: list[BoundFile]) -> None:
    selected = [bound for bound in bounds if bound.role in SEMANTIC_ROLES]
    require(len(selected) == len(SEMANTIC_ROLES)
            and {bound.role for bound in selected} == SEMANTIC_ROLES,
            "FRI semantic artifact roles are incomplete")
    inodes = [(bound.identity[0], bound.identity[1]) for bound in selected]
    require(len(inodes) == len(set(inodes)), "FRI semantic artifacts contain an inode alias")
    paths = [bound.path for bound in selected]
    require(len(paths) == len(set(paths)), "FRI semantic artifacts contain a path alias")


def validate_runner_result(value: Any, expected: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(value, dict), "FRI runner result must be an object")
    require_exact_keys(value, {
        "schema_version", "passed", "standalone_admissible", "performance_admissible", "segment",
        "module_content_sha256", "module_index_sha256", "build_recipe_sha256",
        "primary_fixture_sha256", "primary_fixture_index_sha256",
        "hostile_fixture_sha256", "hostile_fixture_index_sha256",
        "harness_executable_sha256", "device", "graph_contract", "correctness",
    }, "FRI runner result")
    require(value["schema_version"] == RESULT_SCHEMA and value["passed"] is True
            and value["standalone_admissible"] is False
            and value["performance_admissible"] is False,
            "FRI runner did not produce a passed, independently non-admissible result")
    require(value["segment"] == "GraphSegment::FriLayer(7)/FriRound(6)",
            "FRI runner segment differs")
    bindings = {
        "module_content_sha256": "module_sha256",
        "module_index_sha256": "module_index_sha256",
        "build_recipe_sha256": "build_recipe_sha256",
        "primary_fixture_sha256": "primary_sha256",
        "primary_fixture_index_sha256": "primary_index_sha256",
        "hostile_fixture_sha256": "hostile_sha256",
        "hostile_fixture_index_sha256": "hostile_index_sha256",
        "harness_executable_sha256": "harness_sha256",
    }
    for result_name, expected_name in bindings.items():
        require_sha256(value[result_name], f"FRI result {result_name}")
        require(value[result_name] == expected[expected_name],
                f"FRI result {result_name} differs")
    device = value["device"]
    require(isinstance(device, dict), "FRI result device must be an object")
    require_exact_keys(device, {"name", "uuid", "ordinal", "target_sm", "driver_version"},
                       "FRI result device")
    require_int(device["driver_version"], "FRI driver version", 1)
    ordinal = require_int(device["ordinal"], "FRI device ordinal")
    target_sm = require_int(device["target_sm"], "FRI device target SM", 50)
    require(isinstance(device["name"], str) and device["name"]
            and isinstance(device["uuid"], str)
            and re.fullmatch(r"[0-9a-f]{32}", device["uuid"]) is not None
            and ordinal == expected["device"] and target_sm == expected["target_sm"],
            "FRI result device identity differs")
    graph = value["graph_contract"]
    expected_graph = {
        "kernels": 7, "device_copies": 6, "entry_log": 6,
        "exit_log": 3, "packed_leaf_log": 2,
    }
    require(isinstance(graph, dict), "FRI result graph contract must be an object")
    require_exact_keys(graph, set(expected_graph), "FRI result graph contract")
    for name, expected_value in expected_graph.items():
        require(require_int(graph[name], f"FRI graph contract {name}") == expected_value,
                f"FRI result graph contract {name} differs")
    correctness = value["correctness"]
    require(isinstance(correctness, dict), "FRI result correctness must be an object")
    names = {"primary_eager", "primary_graph", "hostile_eager", "hostile_graph",
             "stale_cursor_status_order", "reset_replay"}
    require_exact_keys(correctness, names, "FRI result correctness")
    for name in names - {"stale_cursor_status_order"}:
        check = correctness[name]
        require(isinstance(check, dict), f"FRI result {name} must be an object")
        require_exact_keys(check, {"passed", "checked_words", "error"}, f"FRI result {name}")
        checked_words = require_int(check["checked_words"], f"FRI result {name} checked words")
        require(check["passed"] is True and checked_words == VALIDATION_WORDS
                and isinstance(check["error"], str) and check["error"] == "",
                f"FRI result {name} did not pass exact validation")
    stale = correctness["stale_cursor_status_order"]
    require(isinstance(stale, dict), "FRI stale-cursor result must be an object")
    require_exact_keys(stale, {"passed", "error"}, "FRI stale-cursor result")
    require(stale["passed"] is True and isinstance(stale["error"], str) and stale["error"],
            "FRI stale-cursor mutation was not rejected")
    return value


def validate_record(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "FRI loop record must be an object")
    require_exact_keys(value, {
        "schema_version", "evidence_mode", "preflight_only", "passed",
        "production_admissible", "correctness_admissible", "performance_admissible",
        "target_sm", "device",
        "artifacts", "module_identity", "fixture_identity",
        "loop_tool_sources", "loop_tool_sha256",
        "runner_result",
    }, "FRI loop record")
    require(value["schema_version"] == RECORD_SCHEMA
            and value["evidence_mode"] == EVIDENCE_MODE
            and value["preflight_only"] is True and value["passed"] is True
            and value["production_admissible"] is False
            and value["correctness_admissible"] is False
            and value["performance_admissible"] is False
            and value["runner_result"] is None,
            "FRI preflight record overclaims its evidence")
    require_int(value["target_sm"], "FRI record target SM", 50)
    require(value["target_sm"] <= 999, "FRI record target SM exceeds the supported range")
    require_int(value["device"], "FRI record device ordinal")
    artifacts = value["artifacts"]
    require(isinstance(artifacts, dict), "FRI record artifacts must be an object")
    require(set(artifacts) == SEMANTIC_ROLES, "FRI record artifact roles differ")
    for role, artifact in artifacts.items():
        require(isinstance(artifact, dict), f"FRI record {role} must be an object")
        require_exact_keys(artifact, {"path", "sha256", "bytes"}, f"FRI record {role}")
        require(isinstance(artifact["path"], str) and Path(artifact["path"]).is_absolute(),
                f"FRI record {role} path is not absolute")
        require_sha256(artifact["sha256"], f"FRI record {role} sha256")
        require_int(artifact["bytes"], f"FRI record {role} bytes", 1)
    require(len({artifact["path"] for artifact in artifacts.values()}) == len(artifacts),
            "FRI record artifacts contain a path alias")
    identity = value["module_identity"]
    require(isinstance(identity, dict), "FRI record module identity must be an object")
    require_exact_keys(identity, {
        "abi_sha256", "source_closure_sha256", "build_tool_sha256", "toolchain_sha256",
    }, "FRI record module identity")
    for name, digest in identity.items():
        require_sha256(digest, f"FRI record module identity {name}")
    validate_fixture_identity(value["fixture_identity"])
    validate_loop_tool(value["loop_tool_sources"], value["loop_tool_sha256"])
    return value


def preflight(args: Any) -> dict[str, Any]:
    require(args.preflight_only, "FRI execution plumbing is not enabled in this preflight slice")
    require_int(args.target_sm, "FRI target SM", 50)
    require(args.target_sm <= 999, "FRI target SM exceeds the supported range")
    require_int(args.device, "FRI device ordinal")
    module_identity, module_bounds = _module_bundle(args)
    fixtures, fixture_bounds = _fixture_bundle(args)
    harness = bind_file(args.harness, args.harness_sha256, "FRI runner", executable=True)
    require(harness.path.name == "stwo-gpu-lab-fri-round6", "FRI runner filename differs")
    tool_sources = loop_tool_sources()
    tool_digest = loop_tool_sha256()
    validate_loop_tool(tool_sources, tool_digest)
    tool_bounds = _loop_tool_bounds(tool_sources)
    bounds = [*module_bounds, *fixture_bounds, harness, *tool_bounds]
    _require_semantic_distinct(bounds)
    _reject_symlink_components(args.record.parent, "FRI record output")
    require(args.record.parent.is_dir(), "FRI record output directory is missing")
    guard_output(args.record, [bound.path for bound in bounds])
    for bound in bounds:
        recheck(bound)
    require(loop_tool_sources() == tool_sources and loop_tool_sha256() == tool_digest,
            "FRI loop-tool closure changed during preflight")
    record = {
        "schema_version": RECORD_SCHEMA,
        "evidence_mode": EVIDENCE_MODE,
        "preflight_only": True,
        "passed": True,
        "production_admissible": False,
        "correctness_admissible": False,
        "performance_admissible": False,
        "target_sm": args.target_sm,
        "device": args.device,
        "artifacts": _artifact_records(bounds),
        "module_identity": module_identity,
        "fixture_identity": fixtures,
        "loop_tool_sources": tool_sources,
        "loop_tool_sha256": tool_digest,
        "runner_result": None,
    }
    validate_record(record)
    write_immutable_bytes(args.record, canonical_bytes(record) + b"\n",
                          [bound.path for bound in bounds])
    installed = load_json(args.record)
    require(installed == record and sha256_file(args.record) ==
            sha256_bytes(canonical_bytes(record) + b"\n"),
            "installed FRI preflight record differs")
    for bound in bounds:
        recheck(bound)
    require(loop_tool_sources() == tool_sources and loop_tool_sha256() == tool_digest,
            "FRI loop-tool closure changed after record installation")
    return installed
