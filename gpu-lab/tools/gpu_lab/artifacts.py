"""Canonical replay, launch-plan, and execution-manifest encodings."""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
from pathlib import Path
from typing import Any

from .common import (
    PLAN_MAGIC,
    PLAN_BINDING_VERSION,
    PLAN_VERSION,
    REPLAY_MAGIC,
    REPLAY_VERSION,
    REPO_ROOT,
    SCHEMA_EXECUTION,
    canonical_bytes,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    WORKSPACE_ROOT,
)
from .identity import validate_abi, validate_module_index
from .oracle import validate_oracle_index
from .semantics import validate_fixture


def flatten_columns(columns: list[list[int]], rows: int, label: str) -> list[int]:
    require(all(len(column) == rows for column in columns), f"{label}: column length mismatch")
    return [word for column in columns for word in column]


def reverse_rows(columns: list[list[int]]) -> list[list[int]]:
    """Derive a second case solely by reindexing independently expected rows."""
    return [list(reversed(column)) for column in columns]


def encode_replay(fixture: dict[str, Any]) -> bytes:
    require(fixture.get("schema_version") != "stwo.gpu-lab.semantic-fixture-index.v2",
            "indexed fixtures must use the bounded replay writer")
    payload = fixture["semantic_payload"]
    rows = payload["row_count"]
    inputs = payload["inputs"]
    expected = payload["expected"]
    table = payload["tables"]["address_to_id"]
    primary_inputs = [inputs["segment_start"], inputs["enabler"], inputs["iota"]]
    primary_expected = [
        *expected["output_columns"], *expected["lookup_words"], *expected["sub_words"],
    ]
    words = [
        *flatten_columns(primary_inputs, rows, "inputs"),
        *table,
        *flatten_columns(primary_expected, rows, "expected"),
        *flatten_columns(reverse_rows(primary_inputs), rows, "mutated inputs"),
        *flatten_columns(reverse_rows(primary_expected), rows, "mutated expected"),
    ]
    require(all(isinstance(word, int) and 0 <= word <= 0xFFFF_FFFF for word in words),
            "replay contains a non-u32 word")
    header = struct.pack("<8s8I", REPLAY_MAGIC, REPLAY_VERSION, rows, len(table), 3, 3, 14, 6, 0)
    return header + struct.pack(f"<{len(words)}I", *words)


def write_replay(fixture: dict[str, Any], path: Path, fixture_path: Path | None = None) -> None:
    if fixture.get("schema_version") == "stwo.gpu-lab.semantic-fixture-index.v2":
        from .indexed_fixture import write_indexed_replay

        require(fixture_path is not None, "indexed replay conversion requires its fixture path")
        write_indexed_replay(fixture, fixture_path, path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_replay(fixture))


def verify_canonical_replay(
    fixture: dict[str, Any], fixture_path: Path, replay_path: Path,
) -> None:
    """Reconstruct canonical bytes; never accept a replay from its claimed hash alone."""
    with tempfile.TemporaryDirectory(prefix="gpu-lab-replay-") as temporary:
        canonical = Path(temporary) / "canonical.replay"
        write_replay(fixture, canonical, fixture_path)
        require(canonical.stat().st_size == replay_path.stat().st_size,
                "replay size differs from canonical fixture encoding")
        with canonical.open("rb") as expected, replay_path.open("rb") as actual:
            while expected_block := expected.read(1024 * 1024):
                require(actual.read(len(expected_block)) == expected_block,
                        "replay bytes are not the canonical encoding of the fixture")
            require(not actual.read(1), "replay contains trailing non-canonical bytes")


def encode_plan(execution: dict[str, Any]) -> bytes:
    layout = execution["physical_layout"]
    kernel = execution["kernel"]
    launch = kernel["launch"]
    symbol = kernel["symbol"].encode("utf-8")
    module_path = execution["program_image"]["path"].encode("utf-8")
    require(symbol and b"\0" not in symbol, "kernel symbol is not plan-safe")
    require(module_path and b"\0" not in module_path, "module path is not plan-safe")
    block = launch["block"]
    header = struct.pack(
        "<8s15I",
        PLAN_MAGIC,
        PLAN_VERSION,
        PLAN_BINDING_VERSION,
        execution["target"]["sm"],
        layout["input_columns"],
        layout["table_pointer_slots"],
        layout["table_stride_words"],
        layout["output_columns"],
        layout["lookup_words_per_row"],
        layout["sub_words_per_row"],
        block[0], block[1], block[2],
        launch["dynamic_shared_bytes"],
        len(symbol),
        len(module_path),
    )
    identities = b"".join(bytes.fromhex(value) for value in (
        execution["program_image"]["module_content_sha256"],
        execution["program_image"]["build_recipe_hash"],
        execution["semantic_fixture_sha256"],
        execution["artifacts"]["replay_image_sha256"],
        execution["kernel"]["abi_sha256"],
        execution["host_oracle"]["index_sha256"],
    ))
    return header + symbol + module_path + identities


def write_plan(execution: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_plan(execution))


def execution_document(
    fixture: dict[str, Any], fixture_hash: str, replay_hash: str,
    module: dict[str, Any], module_path: Path, abi: dict[str, Any],
    abi_path: Path, repo_root: Path, target_sm: int,
    oracle_identity: dict[str, str],
) -> dict[str, Any]:
    source = repo_root / abi["source"]
    program_image_hash = sha256_bytes(canonical_bytes({
        "schema_version": abi["schema_version"],
        "semantic_operation": abi["semantic_operation"],
        "semantic_ir_hash": abi["semantic_ir_hash"],
        "source_sha256": sha256_file(source),
        "abi_sha256": sha256_file(abi_path),
    }))
    semantic_binding = (
        {"semantic_identity": fixture["semantic_identity"],
         "full_proof_semantic_hash": fixture["full_proof_semantic_hash"]}
        if "semantic_identity" in fixture
        else {"proof_semantic_hash": fixture["proof_semantic_hash"]}
    )
    return {
        "schema_version": SCHEMA_EXECUTION,
        "semantic_fixture_sha256": fixture_hash,
        **semantic_binding,
        "target": {"sm": target_sm},
        "program_image": {
            "program_image_hash": program_image_hash,
            "build_recipe_hash": module["build_recipe_hash"],
            "module_content_sha256": module["module_content_sha256"],
            "path": str(module_path.resolve()),
        },
        "kernel": {
            "abi_sha256": sha256_file(abi_path),
            "semantic_operation": abi["semantic_operation"],
            "symbol": abi["symbol"],
            "abi_version": abi["abi_version"],
            "arguments": abi["arguments"],
            "launch": abi["launch"],
            "module_globals": abi["module_globals"],
        },
        "host_oracle": oracle_identity,
        "physical_layout": {
            "word_type": "u32_le",
            "input_columns": 3,
            "table_pointer_slots": 37,
            "table_slot_bindings": ["address_to_id", *([None] * 36)],
            "table_stride_words": 3,
            "output_columns": 3,
            "lookup_words_per_row": 14,
            "sub_words_per_row": 6,
            "device_pointer_values_serialized": False,
        },
        "effects": abi["effects"],
        "artifacts": {"replay_image_sha256": replay_hash},
    }


def prepare(args: argparse.Namespace) -> None:
    fixture = validate_fixture(args.fixture)
    module = load_json(args.module_index)
    abi = load_json(args.abi)
    abi_path = args.abi.resolve()
    validate_abi(abi, abi_path, REPO_ROOT)
    oracle_identity = validate_oracle_index(
        args.oracle_index, args.fixture, fixture, WORKSPACE_ROOT
    )
    module_path = validate_module_index(module, abi, abi_path, REPO_ROOT)
    require(module["target_sm"] == args.sm, "module target and requested target differ")
    fixture_hash = sha256_file(args.fixture)
    write_replay(fixture, args.replay, args.fixture)
    replay_hash = sha256_file(args.replay)
    execution = execution_document(
        fixture, fixture_hash, replay_hash, module, module_path, abi, abi_path, REPO_ROOT, args.sm,
        oracle_identity,
    )
    write_plan(execution, args.plan)
    execution["artifacts"]["launch_plan_sha256"] = sha256_file(args.plan)
    args.execution.parent.mkdir(parents=True, exist_ok=True)
    args.execution.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "fixture_sha256": fixture_hash,
        "module_sha256": module["module_content_sha256"],
        "execution_sha256": sha256_file(args.execution),
        "plan_sha256": sha256_file(args.plan),
        "replay_sha256": replay_hash,
    }, sort_keys=True))
