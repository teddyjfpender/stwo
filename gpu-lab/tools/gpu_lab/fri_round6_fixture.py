"""Strict identity validation for the captured-unsealed FRI round-6 pair."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .common import load_json, require, require_exact_keys, require_int, require_sha256, sha256_file


CAPTURED_SCHEMA = "stwo.gpu-lab.fri-round6-captured-unsealed-index.v1"
CAPTURED_SOURCE = "captured-unsealed-production-simd-observer-claim"
PAYLOAD_BYTES = 1_936
CHUNKS = (
    ("entry_pong", 0, 1_024, 256),
    ("inverse_twiddles", 1_024, 224, 56),
    ("alpha6", 1_248, 16, 4),
    ("entry_state", 1_264, 64, 16),
    ("expected_final_ping", 1_328, 128, 32),
    ("expected_root", 1_456, 32, 8),
    ("expected_exit_state", 1_488, 64, 16),
    ("expected_challenge7", 1_552, 16, 4),
    ("expected_retained", 1_568, 128, 32),
    ("expected_leaves", 1_696, 64, 16),
    ("expected_mix_input", 1_760, 32, 8),
    ("expected_draw_output", 1_792, 16, 4),
    ("expected_boundary_mix", 1_808, 64, 16),
    ("expected_boundary_draw", 1_872, 64, 16),
)
U32_MAX = (1 << 32) - 1
U64_MAX = (1 << 64) - 1
M31_P = (1 << 31) - 1
FNV_PRIME = 0x100000001B3
FNV_MASK = U64_MAX
ABSORB_ROOT7 = bytes.fromhex("1e000100040000001c000100")
DRAW_ALPHA7 = bytes.fromhex("1f000100060000001d000100")
DERIVATION = [
    "commit entry_pong as a canonical packed-leaf log-6 FRI tree",
    "absorb root6 with Blake2s Merkle-channel semantics at cursor32",
    "draw alpha6 with bounded canonical rejection semantics at cursor33",
    "match the two-operation prefix against the complete cursor32-cursor36 replay",
]
ORACLE = {
    "candidate_independent": True,
    "fold": "stwo::prover::backend::cpu::fold_line_cpu",
    "commitment": "CpuBackend PackLeavesOps + MerkleOpsLifted<Blake2sMerkleHasher>",
    "transcript": "stwo_backend_cuda::replay_blake2s_reference plus independent FNV encoding",
}


def _u32(value: Any, label: str, minimum: int = 0) -> int:
    require_int(value, label, minimum)
    require(value <= U32_MAX, f"{label} exceeds u32")
    return value


def _hex(value: Any, digits: int, label: str) -> str:
    require(isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{digits}}}", value) is not None,
            f"{label} must be {digits} lowercase hex digits")
    return value


def _source(value: Any) -> None:
    require(isinstance(value, dict), "FRI fixture source must be an object")
    require_exact_keys(value, {
        "kind", "exporter_executable_sha256", "capture_seed_sha256", "capture",
        "capture_shape", "device_protocol_key", "cairo_schedule_key",
    }, "FRI fixture source")
    require(value["kind"] == CAPTURED_SOURCE, "FRI fixture source is synthetic or unreviewed")
    require_sha256(value["exporter_executable_sha256"], "FRI exporter sha256")
    require_sha256(value["capture_seed_sha256"], "FRI capture seed sha256")
    _hex(value["device_protocol_key"], 16, "FRI device protocol key")
    _hex(value["cairo_schedule_key"], 16, "FRI Cairo schedule key")
    capture = value["capture"]
    require(isinstance(capture, dict), "FRI capture source must be an object")
    require_exact_keys(capture, {
        "observer", "prover_input_sha256", "prover_input_bytes", "observer_proof_shape_id",
    }, "FRI capture source")
    require(capture["observer"] == "stwo-cairo.production-simd-fri-observer.v1",
            "FRI capture did not come from the production SIMD observer")
    require_sha256(capture["prover_input_sha256"], "FRI ProverInput sha256")
    require_int(capture["prover_input_bytes"], "FRI ProverInput bytes", 1)
    require(capture["prover_input_bytes"] <= U64_MAX, "FRI ProverInput bytes exceeds u64")
    require(isinstance(capture["observer_proof_shape_id"], str)
            and capture["observer_proof_shape_id"], "FRI observer proof shape is empty")
    shape = value["capture_shape"]
    require(isinstance(shape, dict), "FRI capture shape must be an object")
    require_exact_keys(shape, {
        "circle_log_size", "claim_enable_felts", "claim_log_size_felts",
        "claim_public_data_felts", "interaction_claim_felts", "oods_sampled_values_felts",
        "interaction_pow_bits", "pcs", "fri_tree_count",
    }, "FRI capture shape")
    for name in (
        "claim_enable_felts", "claim_log_size_felts", "claim_public_data_felts",
        "interaction_claim_felts", "oods_sampled_values_felts", "fri_tree_count",
    ):
        _u32(shape[name], f"FRI capture shape {name}", 1)
    _u32(shape["circle_log_size"], "FRI capture circle log", 1)
    _u32(shape["interaction_pow_bits"], "FRI interaction PoW bits")
    require(shape["circle_log_size"] == 24 and shape["fri_tree_count"] >= 8,
            "FRI capture shape is not the reviewed log-24 proof class")
    pcs = shape["pcs"]
    require(isinstance(pcs, dict), "FRI capture PCS shape must be an object")
    require_exact_keys(pcs, {
        "pow_bits", "log_blowup_factor", "n_queries", "log_last_layer_degree_bound",
        "fold_step", "lifting_log_size",
    }, "FRI capture PCS shape")
    for name in ("pow_bits", "log_blowup_factor", "log_last_layer_degree_bound"):
        _u32(pcs[name], f"FRI PCS {name}")
    _u32(pcs["n_queries"], "FRI query count", 1)
    _u32(pcs["fold_step"], "FRI PCS fold step", 1)
    _u32(pcs["lifting_log_size"], "FRI PCS lifting log", 1)
    require(pcs["log_last_layer_degree_bound"] < 32
            and pcs["fold_step"] == 3 and pcs["lifting_log_size"] == 24,
            "FRI capture PCS shape differs from the round-6 lane")


def _shape(value: Any) -> None:
    require(value == {
        "source_circle_log": 24,
        "predecessor_tree_log": 6,
        "entry_line_log": 6,
        "exit_line_log": 3,
        "fold_count": 3,
        "packed_leaf_log": 2,
        "leaf_count": 2,
        "inverse_twiddle_words": 56,
        "normalized_twiddle_offsets_words": [0, 32, 48],
        "full_twiddle_offsets_words": [8_388_544, 8_388_576, 8_388_592],
        "fold_input_words": [64, 32, 16],
    }, "FRI fixture shape differs from the exact round-6 ABI")


def _transcript(value: Any) -> None:
    require(isinstance(value, dict), "FRI transcript must be an object")
    require_exact_keys(value, {
        "schedule_scope", "protocol_tag", "max_rejection_rounds", "semantic_ids", "chains",
    }, "FRI transcript")
    require(value["schedule_scope"] ==
            "observer-captured-cursor32-through-cursor36-unsealed-prefix"
            and value["protocol_tag"] == "stwo.blake2s.transcript.device.v1"
            and value["max_rejection_rounds"] == 64,
            "FRI transcript scope or policy differs")
    require(value["semantic_ids"] == {
        "cursor32_state_input": 0,
        "root6_input": 0x10018,
        "alpha6_output": 0x10019,
        "round6_root_input": 0x1001C,
        "challenge7_output": 0x1001D,
        "operation_boundaries": [0x1001A, 0x1001B, 0x1001E, 0x1001F],
    }, "FRI transcript semantic IDs differ")
    chains = value["chains"]
    require(isinstance(chains, dict), "FRI transcript chains must be an object")
    require_exact_keys(chains, {"c32", "c33", "c34", "c35", "c36"}, "FRI chains")
    for name, chain in chains.items():
        _hex(chain, 16, f"FRI {name}")


def _chunks(value: Any, payload: bytes) -> None:
    require(isinstance(value, list) and len(value) == len(CHUNKS),
            "FRI chunk count differs")
    for actual, (name, offset, length, words) in zip(value, CHUNKS):
        require(isinstance(actual, dict), f"FRI chunk {name} must be an object")
        require_exact_keys(actual, {"id", "offset_bytes", "byte_length", "word_count", "sha256"},
                           f"FRI chunk {name}")
        require(actual["id"] == name and actual["offset_bytes"] == offset
                and actual["byte_length"] == length and actual["word_count"] == words,
                f"FRI chunk {name} geometry differs")
        digest = hashlib.sha256(payload[offset:offset + length]).hexdigest()
        require(actual["sha256"] == digest, f"FRI chunk {name} sha256 differs")


def _words(payload: bytes, offset: int, length: int) -> list[int]:
    return [int.from_bytes(payload[index:index + 4], "little")
            for index in range(offset, offset + length, 4)]


def _fnv(chain: int, encoded: bytes) -> int:
    for byte in encoded:
        chain = ((chain ^ byte) * FNV_PRIME) & FNV_MASK
    return chain


def _state(words: list[int], cursor: int, label: str) -> int:
    require(len(words) == 16 and words[9] == cursor
            and words[10] == words[11] == words[14] == words[15] == 0,
            f"FRI {label} control state differs")
    return words[12] | words[13] << 32


def _payload_contract(payload: bytes, transcript: dict[str, Any]) -> dict[str, list[int]]:
    chunks = {name: _words(payload, offset, length)
              for name, offset, length, _ in CHUNKS}
    for name in (
        "entry_pong", "inverse_twiddles", "alpha6", "expected_final_ping",
        "expected_challenge7", "expected_retained", "expected_draw_output",
    ):
        require(all(word < M31_P for word in chunks[name]),
                f"FRI {name} contains a non-canonical M31 word")
    require(chunks["expected_final_ping"] == chunks["expected_retained"],
            "FRI final/retained fixture invariant differs")
    require(chunks["expected_root"] == chunks["expected_mix_input"],
            "FRI root/mix fixture invariant differs")
    require(chunks["expected_challenge7"] == chunks["expected_draw_output"],
            "FRI challenge/draw fixture invariant differs")
    require(chunks["expected_exit_state"] == chunks["expected_boundary_draw"],
            "FRI exit/boundary fixture invariant differs")
    chain34 = _state(chunks["entry_state"], 34, "entry")
    chain35 = _state(chunks["expected_boundary_mix"], 35, "boundary-mix")
    chain36 = _state(chunks["expected_boundary_draw"], 36, "boundary-draw")
    require(chain35 == _fnv(chain34, ABSORB_ROOT7)
            and chain36 == _fnv(chain35, DRAW_ALPHA7),
            "FRI payload transcript operation chains differ")
    declared = transcript["chains"]
    require([chain34, chain35, chain36] ==
            [int(declared[name], 16) for name in ("c34", "c35", "c36")],
            "FRI payload state chains differ from the index")
    return chunks


def validate_fixture(
    index_path: Path,
    index_sha256: str,
    payload_path: Path,
    payload_sha256: str,
    case: str,
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    require(case in {"primary", "hostile"}, "FRI fixture case differs")
    require_sha256(index_sha256, f"{case} index sha256")
    require_sha256(payload_sha256, f"{case} payload sha256")
    require(index_path.is_file() and not index_path.is_symlink(), f"{case} index is missing")
    require(payload_path.is_file() and not payload_path.is_symlink(), f"{case} payload is missing")
    require(sha256_file(index_path) == index_sha256, f"{case} index sha256 mismatch")
    require(sha256_file(payload_path) == payload_sha256, f"{case} payload sha256 mismatch")
    index = load_json(index_path)
    require_exact_keys(index, {
        "schema_version", "fixture_id", "fixture_class", "production_admissible",
        "admission_blocker", "source", "payload", "shape", "transcript",
        "predecessor_check", "oracle", "chunks",
    }, f"FRI {case} index")
    require(index["schema_version"] == CAPTURED_SCHEMA,
            "synthetic or non-reviewed FRI fixture schema is forbidden")
    require(index["production_admissible"] is False,
            "captured-unsealed FRI fixture cannot claim production admission")
    blocker = (
        "requires verified reference proof replay through cursor32, PIE-to-ProverInput "
        "adapter seal, and recomputed proof-shape identity"
        if case == "primary" else
        "deliberate hostile mutation; additionally lacks verified reference-proof prefix provenance"
    )
    require(index["admission_blocker"] == blocker, "FRI admission blocker differs")
    require(index["fixture_id"] == f"captured-unsealed.fri.round6.{case}.v1"
            and index["fixture_class"] ==
            ("representative" if case == "primary" else "hostile-mutation"),
            f"FRI {case} identity differs")
    _source(index["source"])
    _shape(index["shape"])
    _transcript(index["transcript"])
    payload = payload_path.read_bytes()
    require(len(payload) == PAYLOAD_BYTES, "FRI payload must be exactly 1,936 bytes")
    declared = index["payload"]
    require(declared == {
        "path": payload_path.name,
        "byte_length": PAYLOAD_BYTES,
        "sha256": payload_sha256,
        "encoding": "headerless-le-u32-v1",
    }, f"FRI {case} payload binding differs")
    require(payload_path.resolve() == (index_path.parent / declared["path"]).resolve(),
            f"FRI {case} payload is not beside its index")
    _chunks(index["chunks"], payload)
    payload_chunks = _payload_contract(payload, index["transcript"])
    predecessor = index["predecessor_check"]
    require(isinstance(predecessor, dict), "FRI predecessor check must be an object")
    require_exact_keys(predecessor, {
        "status", "cursor32_state_words_sha256", "cursor32_digest_blake2s",
        "cursor32_n_draws", "root6_blake2s", "root6_words_sha256", "root6_leaf_count",
        "alpha6_words_sha256", "observed_primary_match", "derivation",
    }, "FRI predecessor check")
    require(predecessor["status"] == "PASS" and predecessor["root6_leaf_count"] == 16
            and predecessor["observed_primary_match"] == ("PASS" if case == "primary" else None),
            "FRI predecessor check did not close")
    _u32(predecessor["cursor32_n_draws"], "FRI cursor32 draw count")
    require(predecessor["derivation"] == DERIVATION,
            "FRI predecessor derivation differs")
    for name in ("cursor32_state_words_sha256", "root6_words_sha256", "alpha6_words_sha256"):
        require_sha256(predecessor[name], f"FRI predecessor {name}")
    _hex(predecessor["cursor32_digest_blake2s"], 64, "FRI cursor32 digest")
    _hex(predecessor["root6_blake2s"], 64, "FRI root6 digest")
    oracle = index["oracle"]
    require(isinstance(oracle, dict), "FRI oracle must be an object")
    require_exact_keys(oracle, {"candidate_independent", "fold", "commitment", "transcript"},
                       "FRI oracle")
    require(oracle == ORACLE, "FRI oracle is not the reviewed candidate-independent oracle")
    return index, payload_chunks


def validate_pair(
    primary: tuple[dict[str, Any], dict[str, list[int]]],
    hostile: tuple[dict[str, Any], dict[str, list[int]]],
) -> None:
    primary_index, primary_chunks = primary
    hostile_index, hostile_chunks = hostile
    for name in ("source", "shape", "transcript", "oracle"):
        require(primary_index[name] == hostile_index[name],
                f"FRI hostile fixture changed {name}")
    require(primary_index["payload"]["sha256"] != hostile_index["payload"]["sha256"],
            "FRI hostile mutation did not change payload identity")
    require(any(primary_chunks[name] != hostile_chunks[name]
                for name in ("entry_pong", "alpha6", "entry_state")),
            "FRI hostile fixture did not mutate a semantic input")
    require(primary_chunks["expected_root"] != hostile_chunks["expected_root"]
            and primary_chunks["expected_challenge7"] !=
            hostile_chunks["expected_challenge7"],
            "FRI hostile mutation did not propagate through root and challenge")


def fixture_identity(primary: tuple[dict[str, Any], dict[str, list[int]]]) -> dict[str, Any]:
    index = primary[0]
    return {
        "schema_version": CAPTURED_SCHEMA,
        "source": index["source"],
        "transcript": index["transcript"],
    }


def validate_fixture_identity(value: Any) -> None:
    require(isinstance(value, dict), "FRI fixture identity must be an object")
    require_exact_keys(value, {"schema_version", "source", "transcript"},
                       "FRI fixture identity")
    require(value["schema_version"] == CAPTURED_SCHEMA,
            "FRI record fixture schema differs")
    _source(value["source"])
    _transcript(value["transcript"])
