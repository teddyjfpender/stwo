"""Independent fixture semantics and fail-closed semantic validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import (
    M31_P,
    SCHEMA_FIXTURE,
    load_json,
    require,
    require_exact_keys,
    require_words,
)


def m31_add(lhs: int, rhs: int) -> int:
    return (lhs + rhs) % M31_P


def pedersen_oracle(payload: dict[str, Any]) -> dict[str, list[int]]:
    """Independent scalar semantics for Cairo's PedersenBuiltin witness row."""
    rows = payload["row_count"]
    inputs = payload["inputs"]
    table = payload["tables"]["address_to_id"]
    segment_start = inputs["segment_start"]
    enabler = inputs["enabler"]
    iota = inputs["iota"]
    require_words(segment_start, rows, "segment_start")
    require_words(enabler, rows, "enabler")
    require_words(iota, rows, "iota")
    require(all(value == 1 for value in enabler),
            "slab semantics requires enabler to be the canonical constant one")
    require(isinstance(table, list) and len(table) >= 4,
            "address_to_id table must contain address zero plus 1..3")
    require(
        all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value < M31_P
            for value in table),
        "non-canonical table word",
    )

    output = [[] for _ in range(3)]
    lookup = [[] for _ in range(14)]
    sub = [[] for _ in range(6)]
    for row in range(rows):
        address = m31_add(segment_start[row], (3 * iota[row]) % M31_P)
        addresses = [m31_add(address, offset) for offset in range(3)]
        require(all(0 < index < len(table) for index in addresses),
                f"row {row} reaches address zero or leaves address_to_id")
        values = [table[index] for index in addresses]
        for column, value in enumerate(values):
            output[column].append(value)
        for column, value in enumerate([*addresses, *values]):
            sub[column].append(value)
        lookup_row = [
            1_444_891_767, address, values[0],
            1_444_891_767, addresses[1], values[1],
            1_444_891_767, addresses[2], values[2],
            520_578_465, values[0], values[1], values[2], 1,
        ]
        for column, value in enumerate(lookup_row):
            lookup[column].append(value)
    return {"output_columns": output, "lookup_words": lookup, "sub_words": sub}


def make_tiny_fixture(path: Path) -> dict[str, Any]:
    rows = 32
    table = [((index * 65_537) + 97) % M31_P for index in range(1024)]
    segment_start = [((row * 11 + 5) % 97) or 1 for row in range(rows)]
    iota = [(row * 7 + 3) % 101 for row in range(rows)]
    # Accepted proof-semantic rows stay inside the canonical host memory domain.
    segment_start[-2:] = [len(table) - 48, 1]
    iota[-2:] = [15, 0]
    payload: dict[str, Any] = {
        "field": "M31",
        "row_count": rows,
        "inputs": {
            "segment_start": segment_start,
            "enabler": [1] * rows,
            "iota": iota,
        },
        "tables": {"address_to_id": table},
    }
    payload["expected"] = pedersen_oracle(payload)
    from .indexed_fixture import reference_evaluator_identity, semantic_identity

    slab_identity, descriptor = semantic_identity()
    reference_sources, reference_closure = reference_evaluator_identity()
    proof_identity = {
        "kind": "synthetic_kernel_slice",
        "id": "tiny.witness_pedersen_builtin.v1",
        "full_proof": False,
    }
    fixture = {
        "schema_version": SCHEMA_FIXTURE,
        "fixture_id": "tiny.witness_pedersen_builtin.v1",
        "fixture_class": "tiny",
        "proof_identity": proof_identity,
        "semantic_identity": slab_identity,
        "full_proof_semantic_hash": None,
        "boundary": {
            "operation": "cairo.witness.pedersen_builtin",
            "entry": descriptor["boundary"]["entry"],
            "exit": descriptor["boundary"]["exit"],
        },
        "oracle": {
            "implementation": "gpu-lab/tools/lab.py:pedersen_oracle",
            "version": "candidate-free-scalar-v2",
            "candidate_independent": True,
            "provenance": "reviewed candidate-free PedersenBuiltin slab descriptor",
            "reference_closure_sha256": reference_closure,
            "reference_sources": reference_sources,
        },
        "semantic_payload": payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    return fixture


def _validate_identity(fixture: dict[str, Any]) -> None:
    proof = fixture.get("proof_identity")
    require(isinstance(proof, dict), "proof_identity must be an object")
    require_exact_keys(proof, {"kind", "id", "full_proof"}, "proof_identity")
    require(proof == {"kind": "synthetic_kernel_slice", "id": fixture["fixture_id"],
                      "full_proof": False}, "tiny fixture proof identity changed")
    from .indexed_fixture import semantic_identity

    expected, _ = semantic_identity()
    identity = fixture.get("semantic_identity")
    require(isinstance(identity, dict), "semantic_identity must be an object")
    require_exact_keys(identity, {"kind", "schema", "scope", "sha256"},
                       "semantic_identity")
    require(identity == expected, "tiny semantic identity differs from the slab descriptor")
    require(fixture.get("full_proof_semantic_hash") is None,
            "tiny kernel slice cannot claim a full-proof semantic hash")


def _validate_payload(payload: Any) -> None:
    require(isinstance(payload, dict), "semantic_payload must be an object")
    require_exact_keys(payload, {"field", "row_count", "inputs", "tables", "expected"},
                       "semantic_payload")
    require(payload.get("field") == "M31", "first slice supports only M31")
    rows = payload.get("row_count")
    require(rows == 32 and not isinstance(rows, bool),
            "reviewed tiny fixture row_count must remain 32")
    inputs, tables, expected = payload.get("inputs"), payload.get("tables"), payload.get("expected")
    require(isinstance(inputs, dict), "inputs must be an object")
    require_exact_keys(inputs, {"segment_start", "enabler", "iota"}, "inputs")
    require(isinstance(tables, dict), "tables must be an object")
    require_exact_keys(tables, {"address_to_id"}, "tables")
    require(isinstance(expected, dict), "expected outputs are required")
    require_exact_keys(expected, {"output_columns", "lookup_words", "sub_words"}, "expected")
    for key in ("segment_start", "enabler", "iota"):
        require_words(inputs[key], rows, key)
    require(all(value == 1 for value in inputs["enabler"]),
            "semantic fixture enabler must be the canonical constant one")
    table = tables["address_to_id"]
    require(isinstance(table, list) and 4 <= len(table) <= 1024 * 1024,
            "canonical address_to_id table must contain at least four values")
    require_words(table, len(table), "address_to_id")
    for row, (segment_start, iota) in enumerate(zip(inputs["segment_start"], inputs["iota"])):
        address = m31_add(segment_start, (3 * iota) % M31_P)
        addresses = [m31_add(address, offset) for offset in range(3)]
        require(all(0 < value < len(table) for value in addresses),
                f"row {row} reaches address zero or leaves address_to_id")
    for label, count in (("output_columns", 3), ("lookup_words", 14), ("sub_words", 6)):
        columns = expected[label]
        require(isinstance(columns, list) and len(columns) == count,
                f"{label} column count mismatch")
        for index, column in enumerate(columns):
            require_words(column, rows, f"{label}[{index}]")


def validate_fixture(path: Path) -> dict[str, Any]:
    require(path.stat().st_size <= 16 * 1024 * 1024,
            "fixture JSON/index exceeds the 16 MiB validation bound")
    fixture = load_json(path)
    if fixture.get("schema_version") == "stwo.gpu-lab.semantic-fixture-index.v2":
        from .indexed_fixture import validate_indexed_fixture

        return validate_indexed_fixture(path, fixture)
    require_exact_keys(
        fixture,
        {"schema_version", "fixture_id", "fixture_class", "proof_identity",
         "semantic_identity", "full_proof_semantic_hash", "boundary", "oracle",
         "semantic_payload"},
        "fixture",
    )
    require(fixture.get("schema_version") == SCHEMA_FIXTURE, "unsupported fixture schema")
    require(fixture.get("fixture_id") == "tiny.witness_pedersen_builtin.v1",
            "inline v1 fixture id changed")
    require(fixture.get("fixture_class") == "tiny", "inline v1 fixture class must be tiny")
    boundary = fixture.get("boundary")
    require(isinstance(boundary, dict), "boundary must be an object")
    require_exact_keys(boundary, {"operation", "entry", "exit"}, "boundary")
    require(all(isinstance(boundary[key], str) and boundary[key].strip()
                for key in ("operation", "entry", "exit")),
            "boundary values must be non-empty strings")
    oracle = fixture.get("oracle")
    require(isinstance(oracle, dict), "oracle must be an object")
    require_exact_keys(oracle, {"implementation", "version", "candidate_independent",
                                "provenance", "reference_closure_sha256",
                                "reference_sources"}, "oracle")
    from .indexed_fixture import reference_evaluator_identity, semantic_identity

    sources, closure = reference_evaluator_identity()
    require(oracle == {
        "implementation": "gpu-lab/tools/lab.py:pedersen_oracle",
        "version": "candidate-free-scalar-v2",
        "candidate_independent": True,
        "provenance": "reviewed candidate-free PedersenBuiltin slab descriptor",
        "reference_closure_sha256": closure,
        "reference_sources": sources,
    }, "tiny oracle identity/reference closure changed")
    _validate_identity(fixture)
    _validate_payload(fixture.get("semantic_payload"))
    require(boundary["operation"] == "cairo.witness.pedersen_builtin",
            "fixture operation does not match the first lab slice")
    _, descriptor = semantic_identity()
    require(boundary["entry"] == descriptor["boundary"]["entry"]
            and boundary["exit"] == descriptor["boundary"]["exit"],
            "fixture boundary differs from the semantic descriptor")
    payload = fixture["semantic_payload"]
    recomputed = pedersen_oracle({key: value for key, value in payload.items() if key != "expected"})
    require(payload["expected"] == recomputed,
            "expected outputs disagree with independent scalar oracle")
    return fixture
