"""Strict validation and bounded replay conversion for indexed fixtures."""

from __future__ import annotations

import hashlib
import os
import struct
import sys
import tempfile
from array import array
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from .common import (
    M31_P,
    REPLAY_MAGIC,
    REPLAY_VERSION,
    WORKSPACE_ROOT,
    canonical_bytes,
    load_json,
    require,
    require_exact_keys,
    require_int,
    require_sha256,
    sha256_bytes,
    sha256_file,
)
from .immutable_output import guard_output, install_output
from .oracle import indexed_production_crosscheck

SCHEMA_INDEXED_FIXTURE = "stwo.gpu-lab.semantic-fixture-index.v2"
SCHEMA_INDEXED_ORACLE = "stwo.gpu-lab.host-oracle-index.v2"
M31_ENCODING = "m31-le-u32-row-major-v1"
INPUT_WORDS = 3
EXPECTED_WORDS = 23
_HEADER_BYTES = struct.calcsize("<8s8I")
_BATCH_ROWS = 4096
_MAX_INDEX_BYTES = 16 * 1024 * 1024
_MAX_ROWS = 1 << 20
_MAX_TABLE_WORDS = 1 << 28
_MAX_CHUNKS = 65_536
_MAX_CHUNK_ROWS = 65_536
_SEMANTIC_DESCRIPTOR = (
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/semantics/"
    "pedersen_builtin.slab-semantics.v1.json"
)
_GOLDEN_EVALUATOR = (
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/pedersen_builtin_semantics.rs"
)
_GOLDEN_SUPPORT = (
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/semantic_support.rs"
)
_SEMANTIC_SCHEMA = "stwo.gpu-lab.slab-semantics.v1"
_SEMANTIC_SCOPE = "cairo.witness.pedersen_builtin.base-trace-and-facts"


def semantic_identity() -> tuple[dict[str, str], dict[str, Any]]:
    path = WORKSPACE_ROOT / _SEMANTIC_DESCRIPTOR
    require(path.is_file() and path.stat().st_size <= _MAX_INDEX_BYTES,
            "slab semantic descriptor is missing or oversized")
    descriptor = load_json(path)
    require_exact_keys(
        descriptor,
        {"boundary", "field", "inputs", "outputs", "preprocessed", "row_semantics",
         "schema", "scope", "valid_domain"},
        "slab semantic descriptor",
    )
    require(descriptor.get("schema") == _SEMANTIC_SCHEMA
            and descriptor.get("scope") == _SEMANTIC_SCOPE,
            "slab semantic descriptor header changed")
    boundary = descriptor.get("boundary")
    require(isinstance(boundary, dict), "slab semantic boundary is missing")
    require_exact_keys(boundary, {"entry", "exit"}, "slab semantic boundary")
    identity = {"kind": "kernel_slice", "schema": _SEMANTIC_SCHEMA,
                "scope": _SEMANTIC_SCOPE,
                "sha256": sha256_bytes(canonical_bytes(descriptor))}
    return identity, descriptor


def reference_evaluator_identity() -> tuple[list[dict[str, str]], str]:
    sources = [
        {"role": "semantic_descriptor", "path": _SEMANTIC_DESCRIPTOR,
         "sha256": sha256_file(WORKSPACE_ROOT / _SEMANTIC_DESCRIPTOR)},
        {"role": "golden_evaluator", "path": _GOLDEN_EVALUATOR,
         "sha256": sha256_file(WORKSPACE_ROOT / _GOLDEN_EVALUATOR)},
        {"role": "golden_support", "path": _GOLDEN_SUPPORT,
         "sha256": sha256_file(WORKSPACE_ROOT / _GOLDEN_SUPPORT)},
    ]
    return sources, sha256_bytes(canonical_bytes(sources))


def _nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.strip(), f"{label} must be non-empty")
    return value


def _content_path(root: Path, identity: dict[str, Any], label: str) -> Path:
    relative = identity.get("path")
    require(isinstance(relative, str) and relative, f"{label}.path must be non-empty")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts and "\\" not in relative,
            f"{label}.path is unsafe")
    digest = require_sha256(identity.get("sha256"), f"{label}.sha256")
    require(relative == f"chunks/sha256/{digest}.m31le",
            f"{label}.path is not content-addressed by its SHA256")
    require(not root.is_symlink(), f"{label} fixture root is a symlink")
    cursor = root
    for part in pure.parts:
        cursor /= part
        require(not cursor.is_symlink(), f"{label}.path traverses a symlink")
    candidate = cursor
    resolved_root, resolved = root.resolve(), candidate.resolve()
    require(resolved.is_relative_to(resolved_root) and candidate.is_file()
            and not candidate.is_symlink(), f"{label} escapes its fixture root or is missing")
    return candidate


def _scan_blob(path: Path, expected_size: int, expected_hash: str, label: str) -> None:
    require(path.stat().st_size == expected_size, f"{label} file size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            require(len(block) % 4 == 0, f"{label} has a partial M31 word")
            digest.update(block)
            require(all(word[0] < M31_P for word in struct.iter_unpack("<I", block)),
                    f"{label} contains a non-canonical M31 word")
    require(digest.hexdigest() == expected_hash, f"{label} SHA256 mismatch")


def _validate_table(root: Path, value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "address_to_id must be an object")
    require_exact_keys(value, {"encoding", "path", "sha256", "element_count", "byte_len"},
                       "address_to_id")
    require(value["encoding"] == M31_ENCODING, "unsupported address_to_id encoding")
    count = require_int(value["element_count"], "address_to_id.element_count", 4)
    require(count <= _MAX_TABLE_WORDS, "address_to_id exceeds the producer bound")
    byte_len = require_int(value["byte_len"], "address_to_id.byte_len", 16)
    require(byte_len == count * 4, "address_to_id byte geometry mismatch")
    path = _content_path(root, value, "address_to_id")
    _scan_blob(path, byte_len, value["sha256"], "address_to_id")
    return value


def _validate_chunks(
    root: Path, values: Any, rows: int, words_per_row: int, label: str,
) -> list[dict[str, Any]]:
    require(isinstance(values, list) and 0 < len(values) <= _MAX_CHUNKS,
            f"{label} count exceeds the producer bound")
    cursor = 0
    for index, chunk in enumerate(values):
        item = f"{label}[{index}]"
        require(isinstance(chunk, dict), f"{item} must be an object")
        require_exact_keys(
            chunk,
            {"encoding", "path", "sha256", "row_start", "row_count",
             "words_per_row", "byte_len"},
            item,
        )
        require(chunk["encoding"] == M31_ENCODING, f"{item} encoding differs")
        require_int(chunk["row_start"], f"{item}.row_start")
        count = require_int(chunk["row_count"], f"{item}.row_count", 1)
        require(count <= _MAX_CHUNK_ROWS, f"{item}.row_count exceeds the producer bound")
        require(chunk["row_start"] == cursor, f"{label} coverage is not contiguous")
        require(chunk["words_per_row"] == words_per_row,
                f"{item}.words_per_row differs")
        expected_bytes = count * words_per_row * 4
        require(chunk["byte_len"] == expected_bytes, f"{item} byte geometry mismatch")
        path = _content_path(root, chunk, item)
        _scan_blob(path, expected_bytes, chunk["sha256"], item)
        cursor += count
    require(cursor == rows, f"{label} does not cover every fixture row")
    return values


def _validate_input_rows(root: Path, chunks: list[dict[str, Any]], table_len: int) -> None:
    first: tuple[int, ...] | None = None
    last: tuple[int, ...] | None = None
    segment: int | None = None
    global_row = 0
    for index, chunk in enumerate(chunks):
        path = _content_path(root, chunk, f"input_chunks[{index}]")
        with path.open("rb") as source:
            while block := source.read(_BATCH_ROWS * INPUT_WORDS * 4):
                require(len(block) % (INPUT_WORDS * 4) == 0,
                        "input chunk splits a logical row")
                for row in struct.iter_unpack("<3I", block):
                    first = row if first is None else first
                    last = row
                    segment_start, enabler, iota = row
                    segment = segment_start if segment is None else segment
                    require(segment_start == segment and enabler == 1 and iota == global_row,
                            "input row differs from the real producer sequence")
                    addresses = tuple((segment_start + 3 * iota + offset) % M31_P
                                      for offset in range(3))
                    require(all(0 < address < table_len for address in addresses),
                            "input row reaches address zero, wraps, or leaves address_to_id")
                    global_row += 1
    require(first is not None and last is not None and first != last,
            "reviewed reverse-row mutation would not vary the boundary rows")


def validate_indexed_fixture(path: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    require_exact_keys(
        fixture,
        {"schema_version", "fixture_id", "fixture_class", "proof_identity",
         "semantic_identity", "full_proof_semantic_hash", "boundary", "oracle",
         "production_crosscheck", "exporter_executable_sha256", "semantic_payload"},
        "indexed fixture",
    )
    require(fixture.get("schema_version") == SCHEMA_INDEXED_FIXTURE,
            "unsupported indexed fixture schema")
    fixture_id = _nonempty_string(fixture.get("fixture_id"), "fixture_id")
    require(fixture.get("fixture_class")
            in {"representative", "stress", "unseen-same-class"},
            "indexed fixture class must be production-shaped")

    proof = fixture.get("proof_identity")
    require(isinstance(proof, dict), "proof_identity must be an object")
    require_exact_keys(
        proof,
        {"kind", "id", "full_proof", "source_prover_input_sha256",
         "source_prover_input_bytes"},
        "proof_identity",
    )
    source_hash = require_sha256(proof.get("source_prover_input_sha256"),
                                 "source prover-input SHA256")
    source_bytes = require_int(proof.get("source_prover_input_bytes"),
                               "source prover-input bytes", 1)
    require(source_bytes <= 256 * 1024 * 1024,
            "source prover-input exceeds the producer bound")
    require(proof == {"kind": "real_prover_input_kernel_slice", "id": fixture_id,
                      "full_proof": False, "source_prover_input_sha256": source_hash,
                      "source_prover_input_bytes": source_bytes},
            "indexed proof identity differs from the reviewed kernel-slice contract")
    require(fixture_id ==
            f"{fixture['fixture_class']}.witness_pedersen_builtin.prover-input-{source_hash}.v1",
            "fixture_id does not exactly bind class and complete source prover-input SHA256")

    expected_semantic_identity, descriptor = semantic_identity()
    identity = fixture.get("semantic_identity")
    require(isinstance(identity, dict), "indexed semantic_identity must be an object")
    require_exact_keys(identity, {"kind", "schema", "scope", "sha256"},
                       "indexed semantic_identity")
    require(identity == expected_semantic_identity,
            "indexed semantic_identity differs from the canonical slab descriptor")
    require(fixture.get("full_proof_semantic_hash") is None,
            "kernel-slice fixture must not claim a full-proof semantic hash")

    boundary = fixture.get("boundary")
    require(boundary == {"operation": "cairo.witness.pedersen_builtin",
                         "entry": descriptor["boundary"]["entry"],
                         "exit": descriptor["boundary"]["exit"]},
            "indexed boundary differs from the reviewed first slice")
    oracle = fixture.get("oracle")
    require(isinstance(oracle, dict), "oracle must be an object")
    require_exact_keys(oracle, {"implementation", "version", "candidate_independent",
                                "provenance", "reference_closure_sha256",
                                "reference_sources"}, "oracle")
    require(oracle.get("implementation") == "stwo-cairo/gpu_benchmarks/lab/fixture-export"
            and oracle.get("version") == "independent-scalar-plus-production-simd-v3"
            and oracle.get("candidate_independent") is True,
            "indexed oracle is not the reviewed candidate-independent exporter")
    require(oracle.get("provenance") ==
            "candidate-free scalar golden cross-checked against production PedersenBuiltin SIMD bytes",
            "indexed oracle provenance changed")
    reference_sources, closure = reference_evaluator_identity()
    require(oracle.get("reference_sources") == reference_sources
            and oracle.get("reference_closure_sha256") == closure,
            "indexed oracle reference evaluator closure changed")
    require(fixture.get("production_crosscheck")
            == indexed_production_crosscheck(WORKSPACE_ROOT),
            "production SIMD cross-check closure differs from the live workspace")
    require_sha256(fixture.get("exporter_executable_sha256"),
                   "exporter_executable_sha256")

    payload = fixture.get("semantic_payload")
    require(isinstance(payload, dict), "semantic_payload must be an object")
    require_exact_keys(payload, {"field", "row_count", "address_to_id", "input_chunks",
                                 "expected_chunks"}, "indexed semantic_payload")
    require(payload.get("field") == "M31", "indexed payload field differs")
    rows = require_int(payload.get("row_count"), "row_count", 16)
    require(rows <= _MAX_ROWS and rows & (rows - 1) == 0,
            "row_count is outside the producer power-of-two domain")
    table = _validate_table(path.parent, payload.get("address_to_id"))
    inputs = _validate_chunks(path.parent, payload.get("input_chunks"), rows, INPUT_WORDS,
                              "input_chunks")
    expected = _validate_chunks(path.parent, payload.get("expected_chunks"), rows,
                                EXPECTED_WORDS, "expected_chunks")
    require([(item["row_start"], item["row_count"]) for item in inputs]
            == [(item["row_start"], item["row_count"]) for item in expected],
            "input and expected chunk boundaries differ")
    estimated_peak = source_bytes * 3 + table["element_count"] * 16 + rows * 192 \
        + 128 * 1024 * 1024
    require(estimated_peak <= 1024 * 1024 * 1024,
            "indexed fixture exceeds the producer peak-memory budget")
    _validate_input_rows(path.parent, inputs, table["element_count"])
    return fixture


def validate_indexed_oracle(
    artifact_path: Path, fixture_path: Path, fixture: dict[str, Any],
) -> None:
    require(artifact_path.stat().st_size <= _MAX_INDEX_BYTES,
            "indexed host-oracle artifact exceeds 16 MiB")
    artifact = load_json(artifact_path)
    require_exact_keys(
        artifact,
        {"schema_version", "fixture_id", "fixture_index_sha256", "semantic_identity",
         "full_proof_semantic_hash", "semantic_operation", "oracle",
         "production_crosscheck", "validation", "expected"},
        "indexed host-oracle artifact",
    )
    require(artifact.get("schema_version") == SCHEMA_INDEXED_ORACLE,
            "unsupported indexed host-oracle artifact")
    require(artifact.get("fixture_index_sha256") == sha256_file(fixture_path)
            and artifact.get("fixture_id") == fixture["fixture_id"]
            and artifact.get("semantic_identity") == fixture["semantic_identity"]
            and artifact.get("full_proof_semantic_hash") is None
            and artifact.get("semantic_operation") == "cairo.witness.pedersen_builtin",
            "indexed host oracle binds another fixture or semantic program")
    require(artifact.get("production_crosscheck") == fixture["production_crosscheck"]
            == indexed_production_crosscheck(WORKSPACE_ROOT),
            "indexed host oracle production cross-check closure differs")
    oracle = artifact.get("oracle")
    require(isinstance(oracle, dict), "indexed host-oracle engine is missing")
    require_exact_keys(oracle, {"engine", "golden_evaluator_path", "candidate_gpu_executed",
                                "candidate_recording_used"}, "indexed host-oracle engine")
    require(oracle.get("engine") == "candidate-free scalar slab semantics"
            and oracle.get("golden_evaluator_path") == _GOLDEN_EVALUATOR
            and oracle.get("candidate_gpu_executed") is False
            and oracle.get("candidate_recording_used") is False,
            "indexed host oracle is not candidate-independent")
    rows = fixture["semantic_payload"]["row_count"]
    validation = artifact.get("validation")
    require(isinstance(validation, dict), "indexed host-oracle validation is missing")
    require_exact_keys(
        validation,
        {"scalar_golden_rows", "production_simd_crosschecked_rows",
         "words_compared_to_checked_fixture", "checked_fixture_match", "peak_chunk_rows",
         "peak_chunk_payload_bytes", "resident_table_words", "max_simd_rows_per_segment",
         "estimated_exporter_peak_bytes", "max_exporter_estimated_peak_bytes"},
        "indexed host-oracle validation",
    )
    require(validation.get("scalar_golden_rows") == rows
            and validation.get("production_simd_crosschecked_rows") == rows
            and validation.get("words_compared_to_checked_fixture") == rows * EXPECTED_WORDS
            and validation.get("checked_fixture_match") is True
            and validation.get("resident_table_words")
            == fixture["semantic_payload"]["address_to_id"]["element_count"],
            "indexed host-oracle validation coverage is incomplete")
    require(0 < require_int(validation.get("peak_chunk_rows"), "peak_chunk_rows") <= rows,
            "peak_chunk_rows exceeds the fixture")
    require_int(validation.get("peak_chunk_payload_bytes"), "peak_chunk_payload_bytes", 1)
    paired_peak = max(chunk["row_count"] for chunk
                      in fixture["semantic_payload"]["input_chunks"]) * (INPUT_WORDS
                                                                          + EXPECTED_WORDS) * 4
    require(validation.get("peak_chunk_payload_bytes") == paired_peak,
            "peak_chunk_payload_bytes disagrees with paired chunk geometry")
    require(validation.get("max_simd_rows_per_segment") == _MAX_ROWS,
            "recorded SIMD segment limit differs from the producer bound")
    peak_estimate = (
        fixture["proof_identity"]["source_prover_input_bytes"] * 3
        + fixture["semantic_payload"]["address_to_id"]["element_count"] * 16
        + rows * 192 + 128 * 1024 * 1024
    )
    require(validation.get("estimated_exporter_peak_bytes") == peak_estimate
            and validation.get("max_exporter_estimated_peak_bytes") == 1024 * 1024 * 1024
            and peak_estimate <= 1024 * 1024 * 1024,
            "exporter peak-resource evidence is inconsistent or over budget")
    expected = artifact.get("expected")
    require(isinstance(expected, dict), "indexed host-oracle expected identity is missing")
    require_exact_keys(expected, {"encoding", "words_per_row", "row_count", "logical_sha256",
                                  "chunks"}, "indexed host-oracle expected identity")
    require(expected.get("encoding") == M31_ENCODING
            and expected.get("words_per_row") == EXPECTED_WORDS
            and expected.get("row_count") == rows
            and expected.get("chunks") == fixture["semantic_payload"]["expected_chunks"],
            "indexed host oracle and fixture expected chunks differ")
    logical = hashlib.sha256()
    for index, chunk in enumerate(expected["chunks"]):
        # Chunks are already exact-equal to the fixture descriptors; consume the
        # staged fixture tree so the active replay path stays pod-local.
        source = _content_path(fixture_path.parent, chunk, f"oracle expected[{index}]")
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                logical.update(block)
    require(expected.get("logical_sha256") == logical.hexdigest(),
            "indexed host-oracle logical expected SHA256 mismatch")


def _little_bytes(words: array[int]) -> bytes:
    if sys.byteorder == "big":
        words = array("I", words)
        words.byteswap()
    return words.tobytes()


def _copy_table(target: BinaryIO, source: Path, offset: int, expected_hash: str) -> None:
    digest = hashlib.sha256()
    target.seek(offset)
    with source.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            target.write(block)
    require(digest.hexdigest() == expected_hash, "address_to_id changed during replay build")


def _transpose_chunks(
    target: BinaryIO, root: Path, chunks: list[dict[str, Any]], rows: int, width: int,
    primary_offset: int, mutated_offset: int, label: str,
) -> None:
    require(array("I").itemsize == 4, "host unsigned-int width cannot encode replay-v2")
    for index, chunk in enumerate(chunks):
        source_path = _content_path(root, chunk, f"{label}[{index}]")
        digest = hashlib.sha256()
        logical_row = chunk["row_start"]
        remaining = chunk["row_count"]
        with source_path.open("rb") as source:
            while remaining:
                batch_rows = min(remaining, _BATCH_ROWS)
                block = source.read(batch_rows * width * 4)
                require(len(block) == batch_rows * width * 4,
                        f"{label}[{index}] changed size during replay build")
                digest.update(block)
                words = array("I")
                words.frombytes(block)
                if sys.byteorder == "big":
                    words.byteswap()
                for column in range(width):
                    column_words = words[column::width]
                    target.seek(primary_offset + (column * rows + logical_row) * 4)
                    target.write(_little_bytes(column_words))
                    column_words.reverse()
                    reverse_row = rows - logical_row - batch_rows
                    target.seek(mutated_offset + (column * rows + reverse_row) * 4)
                    target.write(_little_bytes(column_words))
                logical_row += batch_rows
                remaining -= batch_rows
            require(not source.read(1), f"{label}[{index}] grew during replay build")
        require(digest.hexdigest() == chunk["sha256"],
                f"{label}[{index}] changed during replay build")


def write_indexed_replay(fixture: dict[str, Any], fixture_path: Path, output: Path) -> None:
    payload = fixture["semantic_payload"]
    table_path = _content_path(fixture_path.parent, payload["address_to_id"], "address_to_id")
    sources = [fixture_path, table_path]
    for label in ("input_chunks", "expected_chunks"):
        sources.extend(_content_path(fixture_path.parent, chunk, f"{label}[{index}]")
                       for index, chunk in enumerate(payload[label]))
    prior_output = guard_output(output, sources)
    rows = payload["row_count"]
    table_words = payload["address_to_id"]["element_count"]
    input_offset = _HEADER_BYTES
    table_offset = input_offset + INPUT_WORDS * rows * 4
    expected_offset = table_offset + table_words * 4
    mutated_input_offset = expected_offset + EXPECTED_WORDS * rows * 4
    mutated_expected_offset = mutated_input_offset + INPUT_WORDS * rows * 4
    total_bytes = mutated_expected_offset + EXPECTED_WORDS * rows * 4
    header = struct.pack("<8s8I", REPLAY_MAGIC, REPLAY_VERSION, rows, table_words,
                         INPUT_WORDS, 3, 14, 6, 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w+b") as target:
            target.truncate(total_bytes)
            target.seek(0)
            target.write(header)
            _copy_table(target, table_path, table_offset, payload["address_to_id"]["sha256"])
            _transpose_chunks(target, fixture_path.parent, payload["input_chunks"], rows,
                              INPUT_WORDS, input_offset, mutated_input_offset, "input_chunks")
            _transpose_chunks(target, fixture_path.parent, payload["expected_chunks"], rows,
                              EXPECTED_WORDS, expected_offset, mutated_expected_offset,
                              "expected_chunks")
            target.flush()
            os.fsync(target.fileno())
        if install_output(Path(temporary), output, sources, prior_output):
            directory = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
