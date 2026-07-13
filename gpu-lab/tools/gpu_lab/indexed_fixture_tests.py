"""CPU-only hostile checks for streamed production fixture boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from copy import deepcopy
from pathlib import Path

from .artifacts import encode_replay, verify_canonical_replay, write_replay
from .common import WORKSPACE_ROOT, canonical_bytes, require, sha256_bytes, sha256_file
from .indexed_fixture import M31_ENCODING, semantic_identity, validate_indexed_oracle
from .indexed_fixture import reference_evaluator_identity
from .oracle import indexed_production_crosscheck, validate_indexed_exporter
from .oracle_seal import _while_executable_stable, seal_oracle
from .semantics import validate_fixture


def _row_major(columns: list[list[int]]) -> list[int]:
    return [columns[column][row] for row in range(len(columns[0]))
            for column in range(len(columns))]


def _store(root: Path, words: list[int]) -> dict[str, object]:
    payload = struct.pack(f"<{len(words)}I", *words)
    digest = hashlib.sha256(payload).hexdigest()
    relative = f"chunks/sha256/{digest}.m31le"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"encoding": M31_ENCODING, "path": relative, "sha256": digest,
            "byte_len": len(payload)}


def _make_indexed(root: Path, tiny: dict) -> tuple[Path, dict, Path, dict]:
    payload = tiny["semantic_payload"]
    rows = payload["row_count"]
    table = _store(root, payload["tables"]["address_to_id"])
    inputs = payload["inputs"]
    input_blob = _store(root, _row_major([
        [inputs["segment_start"][0]] * rows, [1] * rows, list(range(rows)),
    ]))
    # Recompute expected words for the exact production-shaped input sequence.
    from .semantics import pedersen_oracle

    production_payload = {
        "row_count": rows,
        "inputs": {"segment_start": [inputs["segment_start"][0]] * rows,
                   "enabler": [1] * rows, "iota": list(range(rows))},
        "tables": payload["tables"],
    }
    expected = pedersen_oracle(production_payload)
    expected_columns = [*expected["output_columns"], *expected["lookup_words"],
                        *expected["sub_words"]]
    expected_blob = _store(root, _row_major(expected_columns))
    source_hash = sha256_bytes(b"cpu-only synthetic ProverInput identity")
    fixture_id = f"representative.witness_pedersen_builtin.prover-input-{source_hash}.v1"
    proof = {"kind": "real_prover_input_kernel_slice", "id": fixture_id,
             "full_proof": False, "source_prover_input_sha256": source_hash,
             "source_prover_input_bytes": len(b"cpu-only synthetic ProverInput identity")}
    slab_identity, descriptor = semantic_identity()
    reference_sources, reference_closure = reference_evaluator_identity()
    production_crosscheck = indexed_production_crosscheck(WORKSPACE_ROOT)
    exporter_hash = sha256_bytes(b"cpu-only synthetic exporter executable")
    fixture = {
        "schema_version": "stwo.gpu-lab.semantic-fixture-index.v2",
        "fixture_id": fixture_id,
        "fixture_class": "representative",
        "proof_identity": proof,
        "semantic_identity": slab_identity,
        "full_proof_semantic_hash": None,
        "boundary": {"operation": "cairo.witness.pedersen_builtin",
                     "entry": descriptor["boundary"]["entry"],
                     "exit": descriptor["boundary"]["exit"]},
        "oracle": {"implementation": "stwo-cairo/gpu_benchmarks/lab/fixture-export",
                   "version": "independent-scalar-plus-production-simd-v3",
                   "candidate_independent": True,
                   "provenance": (
                       "candidate-free scalar golden cross-checked against production "
                       "PedersenBuiltin SIMD bytes"
                   ),
                   "reference_closure_sha256": reference_closure,
                   "reference_sources": reference_sources},
        "production_crosscheck": production_crosscheck,
        "exporter_executable_sha256": exporter_hash,
        "semantic_payload": {
            "field": "M31", "row_count": rows,
            "address_to_id": {**table, "element_count": len(payload["tables"]["address_to_id"])},
            "input_chunks": [{**input_blob, "row_start": 0, "row_count": rows,
                              "words_per_row": 3}],
            "expected_chunks": [{**expected_blob, "row_start": 0, "row_count": rows,
                                 "words_per_row": 23}],
        },
    }
    fixture_path = root / "fixture-index.json"
    fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    artifact = {
        "schema_version": "stwo.gpu-lab.host-oracle-index.v2",
        "fixture_id": fixture_id,
        "fixture_index_sha256": sha256_file(fixture_path),
        "semantic_identity": fixture["semantic_identity"],
        "full_proof_semantic_hash": None,
        "semantic_operation": "cairo.witness.pedersen_builtin",
        "oracle": {"engine": "candidate-free scalar slab semantics",
                   "golden_evaluator_path": reference_sources[1]["path"],
                   "candidate_gpu_executed": False, "candidate_recording_used": False},
        "production_crosscheck": production_crosscheck,
        "validation": {"scalar_golden_rows": rows,
                       "production_simd_crosschecked_rows": rows,
                       "words_compared_to_checked_fixture": rows * 23,
                       "checked_fixture_match": True, "peak_chunk_rows": rows,
                       "peak_chunk_payload_bytes": rows * 26 * 4,
                       "resident_table_words": len(payload["tables"]["address_to_id"]),
                       "max_simd_rows_per_segment": 1 << 20,
                       "estimated_exporter_peak_bytes": (
                           proof["source_prover_input_bytes"] * 3
                           + len(payload["tables"]["address_to_id"]) * 16
                           + rows * 192 + 128 * 1024 * 1024
                       ),
                       "max_exporter_estimated_peak_bytes": 1024 * 1024 * 1024},
        "expected": {"encoding": M31_ENCODING, "words_per_row": 23,
                     "row_count": rows, "logical_sha256": expected_blob["sha256"],
                     "chunks": fixture["semantic_payload"]["expected_chunks"]},
    }
    artifact_path = root / "host-oracle-index.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return fixture_path, fixture, artifact_path, production_payload | {"expected": expected,
                                                                       "field": "M31"}


def _reject(path: Path, fixture: dict, label: str, mutate) -> None:
    hostile = deepcopy(fixture)
    mutate(hostile)
    path.write_text(json.dumps(hostile))
    try:
        validate_fixture(path)
    except ValueError:
        return
    raise ValueError(f"indexed fixture accepted hostile {label}")


def _reject_artifact(path: Path, fixture_path: Path, fixture: dict,
                     artifact: dict, label: str, mutate) -> None:
    hostile = deepcopy(artifact)
    mutate(hostile)
    path.write_text(json.dumps(hostile))
    try:
        validate_indexed_oracle(path, fixture_path, fixture)
    except ValueError:
        return
    raise ValueError(f"indexed host oracle accepted hostile {label}")


def indexed_fixture_self_test(root: Path, tiny: dict) -> None:
    fixture_path, fixture, artifact_path, production_payload = _make_indexed(root, tiny)
    validated = validate_fixture(fixture_path)
    validate_indexed_oracle(artifact_path, fixture_path, validated)
    _test_exporter_binding(validated)
    replay = root / "indexed.replay"
    write_replay(validated, replay, fixture_path)
    equivalent = deepcopy(tiny)
    equivalent["semantic_payload"] = production_payload
    require(replay.read_bytes() == encode_replay(equivalent),
            "bounded indexed replay differs from the legacy canonical encoding")
    verify_canonical_replay(validated, fixture_path, replay)
    immutable_identity = (replay.stat().st_dev, replay.stat().st_ino, replay.stat().st_mtime_ns)
    write_replay(validated, replay, fixture_path)
    require(immutable_identity
            == (replay.stat().st_dev, replay.stat().st_ino, replay.stat().st_mtime_ns),
            "byte-identical replay regeneration replaced immutable evidence")
    _reject_replay_aliases(root, validated, fixture_path)
    corrupted = bytearray(replay.read_bytes())
    corrupted[-1] ^= 1
    replay.write_bytes(corrupted)
    try:
        verify_canonical_replay(validated, fixture_path, replay)
    except ValueError:
        pass
    else:
        raise ValueError("corrupt indexed replay was accepted")
    corrupted_hash = sha256_file(replay)
    _reject_output(validated, fixture_path, replay, replay, "differing existing replay")
    require(sha256_file(replay) == corrupted_hash,
            "differing immutable replay was modified")

    _reject(fixture_path, fixture, "unknown key",
            lambda value: value.update({"candidate_digest": "0" * 64}))
    _reject(fixture_path, fixture, "source/id split",
            lambda value: _mutate_source(value))
    _reject(fixture_path, fixture, "oversized source",
            lambda value: value["proof_identity"].update(
                {"source_prover_input_bytes": 256 * 1024 * 1024 + 1}))
    _reject(fixture_path, fixture, "incomplete golden closure",
            lambda value: value["oracle"]["reference_sources"].pop())
    _reject(fixture_path, fixture, "missing production cross-check",
            lambda value: value.pop("production_crosscheck"))
    _reject(fixture_path, fixture, "mutated production cross-check",
            lambda value: value["production_crosscheck"].update(
                {"closure_sha256": "0" * 64}))
    _reject(fixture_path, fixture, "missing exporter executable identity",
            lambda value: value.pop("exporter_executable_sha256"))
    _reject(fixture_path, fixture, "malformed exporter executable identity",
            lambda value: value.update({"exporter_executable_sha256": "0"}))
    _reject(fixture_path, fixture, "unsafe chunk path",
            lambda value: value["semantic_payload"]["input_chunks"][0].update(
                {"path": "../escape.m31le"}))
    _reject(fixture_path, fixture, "non-production input",
            lambda value: _replace_input_word(root, value, 1, 0))
    _reject(fixture_path, fixture, "non-canonical M31 word",
            lambda value: _replace_input_word(root, value, 0, 2_147_483_647))
    _reject(fixture_path, fixture, "semantic identity drift",
            lambda value: value["semantic_identity"].update({"sha256": "0" * 64}))
    _reject(fixture_path, fixture, "unpaired chunk boundary",
            lambda value: _split_expected(root, value))

    fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    _reject_symlinked_chunk(root, fixture_path)
    artifact = json.loads(artifact_path.read_text())
    _reject_artifact(artifact_path, fixture_path, fixture, artifact, "another fixture",
                     lambda value: value.update({"fixture_index_sha256": "0" * 64}))
    _reject_artifact(artifact_path, fixture_path, fixture, artifact,
                     "missing production cross-check",
                     lambda value: value.pop("production_crosscheck"))
    _reject_artifact(artifact_path, fixture_path, fixture, artifact,
                     "mutated production cross-check",
                     lambda value: value["production_crosscheck"].update(
                         {"closure_sha256": "0" * 64}))
    _test_oracle_sealer(root / "seal-workspace", tiny)


def _mutate_source(value: dict) -> None:
    value["proof_identity"]["source_prover_input_sha256"] = "a" * 64


def _test_exporter_binding(fixture: dict) -> None:
    exporter = {**fixture["production_crosscheck"],
                "executable_sha256": fixture["exporter_executable_sha256"]}
    validate_indexed_exporter(exporter, fixture, WORKSPACE_ROOT)
    for label, mutate in (
        ("missing", lambda value: value.pop("executable_sha256")),
        ("wrong", lambda value: value.update({"executable_sha256": "0" * 64})),
    ):
        hostile = deepcopy(exporter)
        mutate(hostile)
        try:
            validate_indexed_exporter(hostile, fixture, WORKSPACE_ROOT)
        except ValueError:
            continue
        raise ValueError(f"indexed wrapper accepted {label} exporter executable identity")


def _test_oracle_sealer(workspace: Path, tiny: dict) -> None:
    live_closure = indexed_production_crosscheck(WORKSPACE_ROOT)
    for source in live_closure["sources"]:
        destination = workspace / source["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKSPACE_ROOT / source["path"], destination)
    case = workspace / "scratchpad" / "seal-case"
    fixture_path, _, artifact_path, _ = _make_indexed(case, tiny)
    exporter = case / "fixture-export"
    exporter.write_bytes(b"cpu-only synthetic exporter executable")
    exporter.chmod(0o755)
    output = case / "oracle-wrapper.json"
    wrapper = seal_oracle(fixture_path, artifact_path, exporter, output, workspace)
    require(output.read_bytes() == canonical_bytes(wrapper) + b"\n",
            "sealed wrapper encoding is not canonical")
    identity = _stat_identity(output)
    seal_oracle(fixture_path, artifact_path, exporter, output, workspace)
    require(_stat_identity(output) == identity,
            "identical oracle reseal replaced immutable evidence")
    for label, alias in (("fixture", fixture_path), ("artifact", artifact_path),
                         ("exporter", exporter)):
        before = sha256_file(alias)
        try:
            seal_oracle(fixture_path, artifact_path, exporter, alias, workspace)
        except ValueError:
            require(sha256_file(alias) == before, f"oracle seal modified {label}")
        else:
            raise ValueError(f"oracle seal accepted {label} output alias")
    differing = case / "differing-wrapper.json"
    differing.write_bytes(b"immutable-different")
    try:
        seal_oracle(fixture_path, artifact_path, exporter, differing, workspace)
    except ValueError:
        require(differing.read_bytes() == b"immutable-different",
                "oracle seal replaced differing immutable output")
    else:
        raise ValueError("oracle seal replaced differing immutable output")
    outside = workspace.parent / f"{workspace.name}-outside-wrapper.json"
    try:
        seal_oracle(fixture_path, artifact_path, exporter, outside, workspace)
    except ValueError:
        require(not outside.exists(), "outside-workspace wrapper was written")
    else:
        raise ValueError("oracle seal accepted outside-workspace output")
    changing = case / "changing-exporter"
    changing.write_bytes(b"before")
    try:
        _while_executable_stable(changing, lambda _: changing.write_bytes(b"after"))
    except ValueError:
        pass
    else:
        raise ValueError("oracle seal accepted an exporter that changed during sealing")


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    status = path.stat()
    return (status.st_dev, status.st_ino, status.st_size,
            status.st_mtime_ns, status.st_ctime_ns)


def _reject_output(fixture: dict, fixture_path: Path, output: Path,
                   protected: Path, label: str) -> None:
    before = sha256_file(protected)
    try:
        write_replay(fixture, output, fixture_path)
    except ValueError:
        require(sha256_file(protected) == before, f"{label} modified immutable evidence")
        return
    raise ValueError(f"indexed replay accepted {label}")


def _reject_replay_aliases(root: Path, fixture: dict, fixture_path: Path) -> None:
    chunk = root / fixture["semantic_payload"]["input_chunks"][0]["path"]
    _reject_output(fixture, fixture_path, fixture_path, fixture_path, "fixture output alias")
    _reject_output(fixture, fixture_path, chunk, chunk, "chunk output alias")
    hardlink = root / "chunk-hardlink.replay"
    os.link(chunk, hardlink)
    try:
        _reject_output(fixture, fixture_path, hardlink, chunk, "chunk hardlink output alias")
    finally:
        hardlink.unlink()
    symlink = root / "chunk-symlink.replay"
    symlink.symlink_to(chunk)
    try:
        _reject_output(fixture, fixture_path, symlink, chunk, "chunk symlink output alias")
    finally:
        symlink.unlink()


def _reject_symlinked_chunk(root: Path, fixture_path: Path) -> None:
    chunks = root / "chunks"
    storage = root / "chunks-real"
    chunks.rename(storage)
    chunks.symlink_to(storage.name, target_is_directory=True)
    try:
        try:
            validate_fixture(fixture_path)
        except ValueError:
            return
        raise ValueError("indexed fixture accepted a symlinked chunk directory")
    finally:
        chunks.unlink()
        storage.rename(chunks)


def _replace_input_word(root: Path, value: dict, word: int, replacement: int) -> None:
    chunk = value["semantic_payload"]["input_chunks"][0]
    source = root / chunk["path"]
    words = list(struct.unpack(f"<{source.stat().st_size // 4}I", source.read_bytes()))
    words[word] = replacement
    identity = _store(root, words)
    chunk.update(identity)


def _split_expected(root: Path, value: dict) -> None:
    chunk = value["semantic_payload"]["expected_chunks"][0]
    words = list(struct.unpack(
        f"<{chunk['byte_len'] // 4}I", (root / chunk["path"]).read_bytes(),
    ))
    split = value["semantic_payload"]["row_count"] // 2
    first = _store(root, words[:split * 23])
    second = _store(root, words[split * 23:])
    value["semantic_payload"]["expected_chunks"] = [
        {**first, "row_start": 0, "row_count": split, "words_per_row": 23},
        {**second, "row_start": split, "row_count": split, "words_per_row": 23},
    ]
