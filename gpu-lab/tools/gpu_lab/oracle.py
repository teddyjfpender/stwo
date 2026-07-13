"""Sealed independent host-oracle validation for replay preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    FIRST_SLICE_INDEXED_ORACLE_WRAPPER_SHA256,
    FIRST_SLICE_ORACLE_EXPORTER_PATHS,
    FIRST_SLICE_ORACLE_INDEX_SHA256,
    canonical_bytes,
    load_json,
    require,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
    sha256_file,
)

_INDEXED_SEMANTIC_CLOSURE = {
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/Cargo.lock",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/Cargo.toml",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/rust-toolchain.toml",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/semantics/pedersen_builtin.slab-semantics.v1.json",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/artifact_io.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/main.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/model.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/oracle.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/pedersen_builtin_semantics.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/producer.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/semantic_support.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/source_snapshot.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/streaming.rs",
    "stwo-cairo/stwo_cairo_prover/crates/adapter/src/memory.rs",
    "stwo-cairo/stwo_cairo_prover/crates/common/src/builtins.rs",
    "stwo-cairo/stwo_cairo_prover/crates/common/src/preprocessed_columns/preprocessed_trace.rs",
    "stwo-cairo/stwo_cairo_prover/crates/prover/Cargo.toml",
    "stwo-cairo/stwo_cairo_prover/crates/prover/src/witness/components/memory_address_to_id.rs",
    "stwo-cairo/stwo_cairo_prover/crates/prover/src/witness/components/pedersen_aggregator_window_bits_18.rs",
    "stwo-cairo/stwo_cairo_prover/crates/prover/src/witness/components/pedersen_builtin.rs",
    "stwo-cairo/stwo_cairo_prover/crates/prover/src/witness/components/pedersen_builtin/gpu_lab_oracle_bridge.rs",
    "stwo/crates/stwo/src/core/fields/m31.rs",
}


def indexed_production_crosscheck(workspace_root: Path) -> dict[str, Any]:
    """Live identity of every source compiled into the production SIMD cross-check."""
    sources = [
        {"path": path, "sha256": sha256_file(workspace_root / path)}
        for path in sorted(_INDEXED_SEMANTIC_CLOSURE)
    ]
    return {"closure_sha256": sha256_bytes(canonical_bytes(sources)), "sources": sources}


def _validate_legacy_artifact(artifact: dict[str, Any], fixture: dict[str, Any],
                              fixture_hash: str) -> None:
    require_exact_keys(
        artifact,
        {"schema_version", "fixture_id", "fixture_sha256", "semantic_identity",
         "full_proof_semantic_hash", "semantic_operation", "oracle", "validation",
         "expected"},
        "host-oracle artifact",
    )
    require(artifact["schema_version"] == "stwo.gpu-lab.host-oracle.v1",
            "bad host-oracle artifact")
    require(artifact["fixture_sha256"] == fixture_hash,
            "host-oracle artifact binds another fixture")
    require(
        artifact["fixture_id"] == fixture["fixture_id"]
        and artifact["semantic_identity"] == fixture["semantic_identity"]
        and artifact["full_proof_semantic_hash"] is None
        and artifact["semantic_operation"]
        == "cairo.witness.pedersen_builtin",
        "host-oracle artifact semantic identity mismatch",
    )
    require(artifact["expected"] == fixture["semantic_payload"]["expected"],
            "fixture expected words differ from the independent Rust oracle")
    oracle, validation = artifact["oracle"], artifact["validation"]
    require(isinstance(oracle, dict), "host-oracle engine is missing")
    require(oracle == {
        "engine": "candidate-free scalar slab semantics",
        "golden_evaluator_path": (
            "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/"
            "pedersen_builtin_semantics.rs"
        ),
        "candidate_gpu_executed": False,
        "candidate_recording_used": False,
    },
            "host oracle is not candidate-independent")
    rows = fixture["semantic_payload"]["row_count"]
    require(isinstance(validation, dict), "host-oracle validation is missing")
    require_exact_keys(validation, {"scalar_golden_rows", "production_simd_crosschecked_rows",
                                    "words_compared_to_checked_fixture",
                                    "checked_fixture_match"},
                       "host-oracle validation")
    require(isinstance(validation, dict)
            and validation.get("checked_fixture_match") is True
            and validation.get("scalar_golden_rows") == rows
            and validation.get("production_simd_crosschecked_rows") == 0
            and validation.get("words_compared_to_checked_fixture") == rows * 23,
            "host-oracle validation coverage is incomplete")


def _validate_exporter(
    exporter: Any, workspace_root: Path, require_legacy_closure: bool,
) -> tuple[str, str]:
    require(isinstance(exporter, dict), "host-oracle exporter identity is missing")
    require_exact_keys(exporter, {"closure_sha256", "sources", "executable_sha256"},
                       "host-oracle exporter")
    executable_hash = require_sha256(exporter["executable_sha256"],
                                     "host-oracle exporter executable_sha256")
    sources = exporter["sources"]
    require(isinstance(sources, list) and sources, "host-oracle exporter sources are missing")
    paths = [source.get("path") for source in sources if isinstance(source, dict)]
    if require_legacy_closure:
        require(paths == FIRST_SLICE_ORACLE_EXPORTER_PATHS,
                "host-oracle exporter source closure differs")
    else:
        require(all(isinstance(path, str) for path in paths),
                "indexed host-oracle exporter paths must be strings")
        require(paths == sorted(_INDEXED_SEMANTIC_CLOSURE),
                "indexed host-oracle exporter closure is not the exact reviewed source set")
        require(all(path.startswith(("stwo/", "stwo-cairo/")) for path in paths),
                "indexed host-oracle semantic source escapes reviewed repositories")
    for source in sources:
        require(isinstance(source, dict), "host-oracle exporter source must be an object")
        require_exact_keys(source, {"path", "sha256"}, "host-oracle exporter source")
        require_sha256(source["sha256"], "host-oracle exporter source sha256")
        path = (workspace_root / source["path"]).resolve()
        require(path.is_file() and path.is_relative_to(workspace_root.resolve()),
                "host-oracle exporter source escapes the workspace or is missing")
        require(sha256_file(path) == source["sha256"],
                "host-oracle exporter source changed")
    closure_hash = sha256_bytes(canonical_bytes(sources))
    require(exporter["closure_sha256"] == closure_hash,
            "host-oracle exporter closure mismatch")
    return closure_hash, executable_hash


def validate_indexed_exporter(
    exporter: Any, fixture: dict[str, Any], workspace_root: Path,
) -> str:
    closure_hash, executable_hash = _validate_exporter(exporter, workspace_root, False)
    require(fixture["exporter_executable_sha256"] == executable_hash,
            "wrapper binds another exporter executable")
    require({"closure_sha256": closure_hash, "sources": exporter["sources"]}
            == fixture["production_crosscheck"],
            "wrapper production source closure differs from its fixture")
    return closure_hash


def validate_oracle_index(
    index_path: Path,
    fixture_path: Path,
    fixture: dict[str, Any],
    workspace_root: Path,
) -> dict[str, str]:
    """Validate the sealed wrapper, its versioned artifact, and exporter closure."""
    require(index_path.stat().st_size <= 16 * 1024 * 1024,
            "host-oracle wrapper exceeds 16 MiB")
    index = load_json(index_path)
    require_exact_keys(index, {"schema_version", "fixture_sha256", "oracle_artifact",
                               "exporter"}, "host-oracle wrapper")
    require(index["schema_version"] == "stwo.gpu-lab.host-oracle-index.v1",
            "unsupported host-oracle wrapper")
    indexed = fixture.get("schema_version") == "stwo.gpu-lab.semantic-fixture-index.v2"
    index_hash = sha256_file(index_path)
    if indexed:
        require(len(FIRST_SLICE_INDEXED_ORACLE_WRAPPER_SHA256) == 64,
                "indexed host-oracle wrapper has not been reviewed and sealed")
        require(index_hash == FIRST_SLICE_INDEXED_ORACLE_WRAPPER_SHA256,
                "indexed host-oracle wrapper differs from the reviewed seal")
    else:
        require(index_hash == FIRST_SLICE_ORACLE_INDEX_SHA256,
                "host-oracle wrapper differs from the reviewed seal")
    fixture_hash = sha256_file(fixture_path)
    require(index["fixture_sha256"] == fixture_hash, "host oracle binds another fixture")

    artifact_identity = index["oracle_artifact"]
    require(isinstance(artifact_identity, dict), "host-oracle artifact identity is missing")
    require_exact_keys(artifact_identity, {"path", "sha256"}, "host-oracle artifact identity")
    require_sha256(artifact_identity["sha256"], "host-oracle artifact sha256")
    artifact_path = (workspace_root / artifact_identity["path"]).resolve()
    require(artifact_path.is_file() and artifact_path.is_relative_to(workspace_root.resolve()),
            "host-oracle artifact escapes the workspace or is missing")
    require(sha256_file(artifact_path) == artifact_identity["sha256"],
            "host-oracle artifact hash mismatch")
    artifact = load_json(artifact_path)
    if indexed:
        from .indexed_fixture import validate_indexed_oracle

        validate_indexed_oracle(artifact_path, fixture_path, fixture)
    else:
        _validate_legacy_artifact(artifact, fixture, fixture_hash)
    if indexed:
        closure_hash = validate_indexed_exporter(index["exporter"], fixture, workspace_root)
    else:
        closure_hash, _ = _validate_exporter(index["exporter"], workspace_root, True)
    return {
        "index_sha256": index_hash,
        "artifact_sha256": artifact_identity["sha256"],
        "exporter_closure_sha256": closure_hash,
    }
