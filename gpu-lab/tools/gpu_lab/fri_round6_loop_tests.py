"""Hostile, GPU-free tests for the FRI round-6 orchestration boundary."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from .common import REPO_ROOT, canonical_bytes, load_json, require, sha256_bytes, sha256_file
from .fri_discovery import REQUIRED_REPOSITORY_SOURCES, source_closure
from .fri_module import ABI_RELATIVE, _recipe, validate_fri_abi
from .fri_round6_fixture import (
    ABSORB_ROOT7,
    CHUNKS,
    DERIVATION,
    DRAW_ALPHA7,
    ORACLE,
    _fnv,
    validate_fixture,
)
from .fri_round6_loop import (
    _fixture_bundle,
    _module_bundle,
    _require_semantic_distinct,
    bind_file,
    preflight,
    validate_runner_result,
)


def _expect_rejection(label: str, operation) -> None:
    try:
        operation()
    except (ValueError, TypeError, KeyError, IndexError, OSError):
        return
    raise ValueError(f"hostile FRI loop input was accepted: {label}")


def _toolchain() -> dict:
    digest = "12" * 32
    return {
        "nvcc_path": "/toolchain/nvcc", "nvcc_version": "nvcc test",
        "ptxas_path": "/toolchain/ptxas", "ptxas_version": "ptxas test",
        "cuobjdump_path": "/toolchain/cuobjdump", "cuobjdump_version": "cuobjdump test",
        "host_compiler_path": "/toolchain/c++", "host_compiler_version": "c++ test",
        "environment": {"NVCC_PREPEND_FLAGS": "", "NVCC_APPEND_FLAGS": ""},
        "driver_compatibility_policy": "CUDA enhanced-compatibility; execution binds actual driver",
        "binary_sha256": {name: digest for name in (
            "nvcc_path", "ptxas_path", "cuobjdump_path", "host_compiler_path",
        )},
    }


def _module(directory: Path, sm: int) -> dict[str, object]:
    abi_path = REPO_ROOT / ABI_RELATIVE
    abi = validate_fri_abi(abi_path, REPO_ROOT)
    sources = [REPO_ROOT / path for path in sorted(REQUIRED_REPOSITORY_SOURCES)]
    closure = source_closure(sources, REPO_ROOT)
    recipe = _recipe(sm, abi, abi_path, closure, _toolchain())
    recipe_hash = sha256_bytes(canonical_bytes(recipe))
    recipe_path = directory / f"{recipe_hash}.recipe.json"
    recipe_path.write_bytes(canonical_bytes(recipe))
    module_bytes = b"gpu-free FRI round-6 cubin identity test"
    module_hash = sha256_bytes(module_bytes)
    module_path = directory / f"{module_hash}.cubin"
    module_path.write_bytes(module_bytes)
    (directory / f"{recipe_hash}.module-sha256").write_text(module_hash + "\n")
    index = {
        "schema_version": "stwo.gpu-lab.fri-round6-module-index.v1",
        "module": "fri_round6", "target_sm": sm,
        "build_recipe_hash": recipe_hash, "build_recipe": recipe,
        "build_recipe_path": str(recipe_path),
        "module_content_sha256": module_hash, "module_path": str(module_path),
    }
    index_bytes = canonical_bytes(index)
    index_path = directory / f"{sha256_bytes(index_bytes)}.module-index.json"
    index_path.write_bytes(index_bytes)
    return {
        "index": index_path, "index_sha256": sha256_file(index_path),
        "module_sha256": module_hash, "recipe_sha256": recipe_hash,
    }


def _state(cursor: int, chain: int, salt: int) -> list[int]:
    return [salt + index for index in range(8)] + [0, cursor, 0, 0,
            chain & 0xFFFFFFFF, chain >> 32, 0, 0]


def _payload(case: str, chains: tuple[int, int, int]) -> bytes:
    hostile = case == "hostile"
    entry = list(range(256))
    if hostile:
        entry[0], entry[1] = 313, 271
    final = [1_000 + index + 100 * hostile for index in range(32)]
    root = [0x1234_0000 + index + 0x100 * hostile for index in range(8)]
    challenge = [2_000 + index + 100 * hostile for index in range(4)]
    boundary_draw = _state(36, chains[2], 40)
    words = {
        "entry_pong": entry,
        "inverse_twiddles": [3_000 + index for index in range(56)],
        "alpha6": [4_000 + index for index in range(4)],
        "entry_state": _state(34, chains[0], 20),
        "expected_final_ping": final,
        "expected_root": root,
        "expected_exit_state": boundary_draw,
        "expected_challenge7": challenge,
        "expected_retained": final,
        "expected_leaves": [0x2345_0000 + index for index in range(16)],
        "expected_mix_input": root,
        "expected_draw_output": challenge,
        "expected_boundary_mix": _state(35, chains[1], 30),
        "expected_boundary_draw": boundary_draw,
    }
    return b"".join(word.to_bytes(4, "little")
                    for name, _, _, count in CHUNKS for word in words[name][:count])


def _fixture(directory: Path, case: str, source: dict, transcript: dict,
             chains: tuple[int, int, int]) -> tuple[Path, str, Path, str]:
    payload = _payload(case, chains)
    payload_path = directory / f"fri-round6-{case}.payload.bin"
    payload_path.write_bytes(payload)
    payload_hash = sha256_file(payload_path)
    blocker = (
        "requires verified reference proof replay through cursor32, PIE-to-ProverInput "
        "adapter seal, and recomputed proof-shape identity"
        if case == "primary" else
        "deliberate hostile mutation; additionally lacks verified reference-proof prefix provenance"
    )
    index = {
        "schema_version": "stwo.gpu-lab.fri-round6-captured-unsealed-index.v1",
        "fixture_id": f"captured-unsealed.fri.round6.{case}.v1",
        "fixture_class": "representative" if case == "primary" else "hostile-mutation",
        "production_admissible": False, "admission_blocker": blocker,
        "source": source,
        "payload": {"path": payload_path.name, "byte_length": len(payload),
                    "sha256": payload_hash, "encoding": "headerless-le-u32-v1"},
        "shape": {
            "source_circle_log": 24, "predecessor_tree_log": 6, "entry_line_log": 6,
            "exit_line_log": 3, "fold_count": 3, "packed_leaf_log": 2,
            "leaf_count": 2, "inverse_twiddle_words": 56,
            "normalized_twiddle_offsets_words": [0, 32, 48],
            "full_twiddle_offsets_words": [8_388_544, 8_388_576, 8_388_592],
            "fold_input_words": [64, 32, 16],
        },
        "transcript": transcript,
        "predecessor_check": {
            "status": "PASS", "cursor32_state_words_sha256": "31" * 32,
            "cursor32_digest_blake2s": "32" * 32, "cursor32_n_draws": 0,
            "root6_blake2s": "33" * 32, "root6_words_sha256": "34" * 32,
            "root6_leaf_count": 16, "alpha6_words_sha256": "35" * 32,
            "observed_primary_match": "PASS" if case == "primary" else None,
            "derivation": DERIVATION,
        },
        "oracle": ORACLE,
        "chunks": [{
            "id": name, "offset_bytes": offset, "byte_length": length,
            "word_count": count, "sha256": sha256_bytes(payload[offset:offset + length]),
        } for name, offset, length, count in CHUNKS],
    }
    encoded = json.dumps(index, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    index_path = directory / f"{sha256_bytes(encoded)}.fri-{case}-index.json"
    index_path.write_bytes(encoded)
    return payload_path, payload_hash, index_path, sha256_file(index_path)


def _fixtures(directory: Path) -> dict[str, object]:
    chain34 = 0x0123456789ABCDEF
    chain35 = _fnv(chain34, ABSORB_ROOT7)
    chain36 = _fnv(chain35, DRAW_ALPHA7)
    shape = {
        "circle_log_size": 24, "claim_enable_felts": 2, "claim_log_size_felts": 2,
        "claim_public_data_felts": 3, "interaction_claim_felts": 5,
        "oods_sampled_values_felts": 7, "interaction_pow_bits": 20,
        "pcs": {"pow_bits": 24, "log_blowup_factor": 1, "n_queries": 13,
                "log_last_layer_degree_bound": 3, "fold_step": 3,
                "lifting_log_size": 24},
        "fri_tree_count": 8,
    }
    source = {
        "kind": "captured-unsealed-production-simd-observer-claim",
        "exporter_executable_sha256": "21" * 32, "capture_seed_sha256": "22" * 32,
        "capture": {"observer": "stwo-cairo.production-simd-fri-observer.v1",
                    "prover_input_sha256": "23" * 32, "prover_input_bytes": 1024,
                    "observer_proof_shape_id": "gpu-free-captured-shape"},
        "capture_shape": shape, "device_protocol_key": "1111111111111111",
        "cairo_schedule_key": "2222222222222222",
    }
    transcript = {
        "schedule_scope": "observer-captured-cursor32-through-cursor36-unsealed-prefix",
        "protocol_tag": "stwo.blake2s.transcript.device.v1", "max_rejection_rounds": 64,
        "semantic_ids": {"cursor32_state_input": 0, "root6_input": 0x10018,
                         "alpha6_output": 0x10019, "round6_root_input": 0x1001C,
                         "challenge7_output": 0x1001D,
                         "operation_boundaries": [0x1001A, 0x1001B, 0x1001E, 0x1001F]},
        "chains": {"c32": "3333333333333333", "c33": "4444444444444444",
                   "c34": f"{chain34:016x}", "c35": f"{chain35:016x}",
                   "c36": f"{chain36:016x}"},
    }
    primary = _fixture(directory, "primary", source, transcript, (chain34, chain35, chain36))
    hostile = _fixture(directory, "hostile", source, transcript, (chain34, chain35, chain36))
    return {
        "primary": primary[0], "primary_sha256": primary[1],
        "primary_index": primary[2], "primary_index_sha256": primary[3],
        "hostile": hostile[0], "hostile_sha256": hostile[1],
        "hostile_index": hostile[2], "hostile_index_sha256": hostile[3],
    }


def _arguments(directory: Path, module: dict, fixtures: dict) -> SimpleNamespace:
    runner = directory / "stwo-gpu-lab-fri-round6"
    runner.write_text("gpu-free typed FRI runner identity\n")
    os.chmod(runner, 0o700)
    return SimpleNamespace(
        module_index=module["index"], module_index_sha256=module["index_sha256"],
        module_sha256=module["module_sha256"],
        build_recipe_sha256=module["recipe_sha256"], **fixtures,
        harness=runner, harness_sha256=sha256_file(runner),
        record=directory / "preflight-record.json", target_sm=86, device=0,
        correctness_only_unsealed=True, preflight_only=True,
    )


def _result(args: SimpleNamespace) -> dict:
    passed = {"passed": True, "checked_words": 416, "error": ""}
    return {
        "schema_version": "stwo.gpu-lab.fri-round6-result.v1", "passed": True,
        "standalone_admissible": False, "performance_admissible": False,
        "segment": "GraphSegment::FriLayer(7)/FriRound(6)",
        "module_content_sha256": args.module_sha256,
        "module_index_sha256": args.module_index_sha256,
        "build_recipe_sha256": args.build_recipe_sha256,
        "primary_fixture_sha256": args.primary_sha256,
        "primary_fixture_index_sha256": args.primary_index_sha256,
        "hostile_fixture_sha256": args.hostile_sha256,
        "hostile_fixture_index_sha256": args.hostile_index_sha256,
        "harness_executable_sha256": args.harness_sha256,
        "device": {"name": "gpu-free-test", "uuid": "ab" * 16, "ordinal": 0,
                   "target_sm": 86, "driver_version": 12080},
        "graph_contract": {"kernels": 7, "device_copies": 6, "entry_log": 6,
                           "exit_log": 3, "packed_leaf_log": 2},
        "correctness": {"primary_eager": passed, "primary_graph": passed,
                        "hostile_eager": passed, "hostile_graph": passed,
                        "stale_cursor_status_order": {"passed": True, "error": "rejected"},
                        "reset_replay": passed},
    }


def _generic_index(directory: Path, module: dict) -> Path:
    digest = "55" * 32
    recipe = {
        "schema_version": "stwo.gpu-lab.build-recipe.v2", "source_sha256": digest,
        "transitive_headers": [], "abi_sha256": digest, "cache_key": "11" * 8,
        "semantic_ir_hash": "22" * 8, "exported_symbols": ["generic_kernel"],
        "generator_sources": [{"repository": "stwo", "path": "x/y.rs", "sha256": digest}],
        "build_tool_sha256": digest, "nvcc_path": "/tools/nvcc", "nvcc_version": "test",
        "host_compiler_path": "/tools/c++", "host_compiler_version": "test",
        "ptxas_path": "/tools/ptxas", "ptxas_version": "test",
        "cuobjdump_path": "/tools/cuobjdump", "cuobjdump_version": "test",
        "environment": {"NVCC_PREPEND_FLAGS": "", "NVCC_APPEND_FLAGS": ""},
        "driver_compatibility_policy": "test", "source_staging": {
            "policy": "private-content-addressed-canonical-path-v1",
            "root": "/tmp/stwo-gpu-lab-aot-v1", "directory": digest,
            "source_name": "source.cu", "mtime_ns": 0}, "target_sm": 86,
        "normalized_command": ["nvcc", "-cubin", "-O3", "--std=c++17",
                               "--expt-relaxed-constexpr", "-lineinfo", "-arch=sm_86",
                               "-ccbin", "/tools/c++", "source.cu", "-o", "<output>"],
    }
    document = {
        "schema_version": "stwo.gpu-lab.module-index.v1", "build_recipe_hash": digest,
        "build_recipe": recipe, "module_content_sha256": module["module_sha256"],
        "module_path": "/modules/generic.cubin", "target_sm": 86,
        "abi_path": "/manifests/generic.json",
    }
    schema = load_json(REPO_ROOT / "gpu-lab/schemas/module-index.schema.json")
    require(set(document) == set(schema["required"]), "generic module-index test drifted")
    payload = canonical_bytes(document)
    path = directory / f"{sha256_bytes(payload)}.module-index.json"
    path.write_bytes(payload)
    return path


def fri_round6_loop_self_test(lab_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix=".fri-round6-loop-test-", dir=lab_root) as name:
        directory = Path(name).resolve()
        module = _module(directory, 86)
        fixtures = _fixtures(directory)
        args = _arguments(directory, module, fixtures)
        record = preflight(args)
        require(record == preflight(args) and record["correctness_admissible"] is False,
                "FRI preflight record is not immutable/non-admissible")

        generic = _generic_index(directory, module)
        generic_args = SimpleNamespace(**vars(args))
        generic_args.module_index = generic
        generic_args.module_index_sha256 = sha256_file(generic)
        _expect_rejection("generic module-index used as FRI authority",
                          lambda: _module_bundle(generic_args))

        production = SimpleNamespace(**vars(args))
        production.correctness_only_unsealed = False
        _expect_rejection("implicit production fixture admission",
                          lambda: _fixture_bundle(production))
        mutated = load_json(args.primary_index)
        mutated["schema_version"] = "stwo.gpu-lab.fri-round6-synthetic-index.v1"
        synthetic = directory / "synthetic-index.json"
        synthetic.write_bytes(canonical_bytes(mutated))
        _expect_rejection("synthetic fixture schema", lambda: validate_fixture(
            synthetic, sha256_file(synthetic), args.primary, args.primary_sha256, "primary"))
        mutated = load_json(args.primary_index)
        mutated["source"]["capture_shape"]["pcs"]["log_last_layer_degree_bound"] = 32
        invalid_pcs = directory / "invalid-pcs-index.json"
        invalid_pcs.write_bytes(canonical_bytes(mutated))
        _expect_rejection("out-of-range captured PCS field", lambda: validate_fixture(
            invalid_pcs, sha256_file(invalid_pcs), args.primary, args.primary_sha256, "primary"))

        link = directory / "artifact-link"
        link.symlink_to(directory, target_is_directory=True)
        _expect_rejection("symlinked artifact ancestor", lambda: bind_file(
            link / args.primary.name, args.primary_sha256, "symlink-hostile"))
        semantic = [bind_file(Path(value["path"]), value["sha256"], role,
                              executable=role == "FRI runner")
                    for role, value in record["artifacts"].items()]
        primary = next(bound for bound in semantic if bound.role == "primary FRI payload")
        aliased = [bound for bound in semantic if bound.role != "hostile FRI payload"]
        aliased.append(replace(primary, role="hostile FRI payload"))
        _expect_rejection("semantic inode alias", lambda: _require_semantic_distinct(aliased))
        alias_args = SimpleNamespace(**vars(args))
        alias_args.record = args.primary
        _expect_rejection("record/input alias", lambda: preflight(alias_args))

        expected = vars(args)
        result = _result(args)
        validate_runner_result(result, expected)
        for label, mutation in (
            ("raw standalone admission", lambda value: value.__setitem__(
                "standalone_admissible", True)),
            ("result hash substitution", lambda value: value.__setitem__(
                "module_index_sha256", "00" * 32)),
            ("partial correctness", lambda value: value["correctness"][
                "primary_graph"].__setitem__("checked_words", 415)),
            ("extra result key", lambda value: value.__setitem__("unreviewed", True)),
        ):
            hostile = deepcopy(result)
            mutation(hostile)
            _expect_rejection(label, lambda hostile=hostile: validate_runner_result(
                hostile, expected))


if __name__ == "__main__":
    fri_round6_loop_self_test(REPO_ROOT / "gpu-lab")
    print("FRI round-6 loop GPU-free hostile tests: PASS")
