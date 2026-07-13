"""Hostile local checks for baseline acceptance boundaries."""

from __future__ import annotations

import copy
import json

from .baselines import _acceptance_ready, _comparison_key, _safe_component
from .baseline_validation import (
    TOP_LEVEL_SNAPSHOTS,
    validate_baseline_comparison,
    validate_baseline_envelope,
)
from .common import canonical_bytes, require, sha256_bytes
from .loop import ENVIRONMENT_QUERIES


def _expect_rejection(call, label: str) -> None:
    try:
        call()
    except ValueError:
        return
    raise ValueError(f"baseline hostile mutation was accepted: {label}")


def _digest(character: str) -> str:
    return character * 64


def _series() -> dict:
    return {
        "warmup_iterations": 3, "warmup_stable": True, "iterations": 30,
        "p5_gpu_ms": 1.0, "p50_gpu_ms": 1.0, "p95_gpu_ms": 1.0,
        "stddev_gpu_ms": 0.0, "p50_submit_us": 0.0, "p95_submit_us": 0.0,
        "p50_wall_ms": 1.0, "p95_wall_ms": 1.0, "p95_exploratory": False,
        "raw_gpu_ms": [1.0] * 30, "raw_submit_us": [0.0] * 30, "raw_wall_ms": [1.0] * 30,
    }


def _accepted_envelope(environment: dict) -> dict:
    timing = {
        "post_benchmark_passed": True, "performance_admissible": True,
        "budget_ms": 30.0, "budget_used_ms": 1.0,
        "budget_remaining_ms": 29.0, "graph_capture_ms": 0.1, "graph_instantiate_ms": 0.1,
        "eager": _series(), "graph": _series(),
    }
    shape = {
        "fixture_class": "tiny", "row_count": 32,
        "semantic_operation": "cairo.witness.pedersen_builtin",
        "launch": {"block": [256, 1, 1]}, "physical_layout": {"output_columns": 3},
    }
    shape = {"sha256": sha256_bytes(canonical_bytes(shape)), **shape}
    toolchain = {
        "nvcc_path": "/cuda/nvcc", "nvcc_version": "nvcc 1",
        "host_compiler_path": "/usr/bin/g++", "host_compiler_version": "g++ 1",
        "ptxas_path": "/cuda/ptxas", "ptxas_version": "ptxas 1",
        "cuobjdump_path": "/cuda/cuobjdump", "cuobjdump_version": "cuobjdump 1",
        "environment": {"NVCC_PREPEND_FLAGS": "", "NVCC_APPEND_FLAGS": ""},
        "driver_compatibility_policy": "test", "target_sm": 86,
        "normalized_flags": ["-cubin", "-O3", "-arch=sm_86"],
    }
    toolchain["sha256"] = sha256_bytes(canonical_bytes(toolchain))
    top = {
        name: {"sha256": _digest(format(index + 1, "x")[-1]), "bytes": 1}
        for index, name in enumerate(sorted(TOP_LEVEL_SNAPSHOTS))
    }
    roles = {
        "abi": _digest("a"), "cuda_source": _digest("f"),
        "host_oracle": _digest("b"), "module": _digest("c"),
        "orchestrator": _digest("d"), "slo_checker": _digest("e"),
    }
    transitive = [
        {"path": f"/{role}", "roles": [role], "sha256": digest, "bytes": 1}
        for role, digest in sorted(roles.items())
    ]
    top["build_commands"]["sha256"] = _digest("1")
    top["harness"]["sha256"] = _digest("2")
    top["loop_record"]["sha256"] = _digest("3")
    top["result"]["sha256"] = _digest("4")
    top["fixture"]["sha256"] = _digest("5")
    top["oracle_index"]["sha256"] = _digest("7")
    environment_sha = sha256_bytes(
        json.dumps(environment, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    )
    top["environment_before"]["sha256"] = environment_sha
    top["environment_after"]["sha256"] = environment_sha
    snapshot = {"top_level": top, "transitive": transitive}
    snapshot["sha256"] = sha256_bytes(canonical_bytes(snapshot))
    return {
        "schema_version": "stwo.gpu-lab.baseline.v1",
        "identity": {
            "device": {"uuid": "00112233445566778899aabbccddeeff", "name": "name",
                       "target_sm": 86, "driver_version": 12000, "ecc_enabled": False},
            "fixture": {"id": "tiny.test", "class": "tiny", "sha256": _digest("5"),
                        "proof_semantic_hash": _digest("6"), "rows": 32, "words_per_row": 23},
            "oracle": {"index_sha256": _digest("7"), "artifact_sha256": roles["host_oracle"],
                       "exporter_closure_sha256": _digest("8")},
            "abi_sha256": roles["abi"], "shape": shape, "toolchain": toolchain,
        },
        "module": {"content_sha256": roles["module"], "build_recipe_hash": _digest("9"),
                   "source_sha256": roles["cuda_source"]},
        "measurement": {
            "result_sha256": top["result"]["sha256"],
            "resources": {"registers_per_thread": 16, "static_shared_bytes": 0,
                          "local_bytes_per_thread": 0},
            "timing": timing,
            "derived": {
                topology: {
                    "semantic_rows_per_s_p50": 32_000.0,
                    "semantic_words_per_s_p50": 736_000.0,
                    "semantic_rows_per_s_p95": 32_000.0,
                    "semantic_words_per_s_p95": 736_000.0,
                }
                for topology in ("eager", "graph")
            },
        },
        "environment": {"before": environment, "after": environment},
        "audit": {
            "loop_record_sha256": top["loop_record"]["sha256"],
            "correctness_result_sha256": top["correctness_result"]["sha256"],
            "comparison_sha256": top["comparison"]["sha256"],
            "build": {"commands_sha256": top["build_commands"]["sha256"], "command_count": 1,
                      "commands": ["nvcc -cubin source.cu"], "forbidden_hits": [],
                      "cargo_builds": 0, "scope": "test"},
            "runtime": {"harness_sha256": top["harness"]["sha256"],
                        "orchestrator_sha256": roles["orchestrator"],
                        "slo_checker_sha256": roles["slo_checker"],
                        "invocations": ["correctness", "benchmark"],
                        "benchmark_exit_code": 0, "performance_admissible": True,
                        "full_proofs": 0,
                        "scope": "test"},
        },
        "snapshot": snapshot,
    }


def baseline_self_test() -> None:
    fields = {
        name: {"status": "available", "value": "GPU-00112233-4455-6677-8899-aabbccddeeff"}
        if name == "uuid" else {"status": "available", "value": name}
        for name in ENVIRONMENT_QUERIES
    }
    environment = {
        "schema_version": "stwo.gpu-lab.environment.v1", "device_selector": "0", "fields": fields,
    }
    result = {
        "device": {"uuid": "00112233445566778899aabbccddeeff"},
        "timing": {"performance_admissible": True, **{
            topology: {"warmup_stable": True, "iterations": 30, "p95_exploratory": False}
            for topology in ("eager", "graph")
        }},
    }
    _acceptance_ready(result, environment, environment)
    require(_safe_component("tiny.fixture-v1", "fixture_id") == "tiny.fixture-v1",
            "safe fixture component changed")
    for hostile in ("../escape", ".", "a/b"):
        _expect_rejection(lambda value=hostile: _safe_component(value, "fixture_id"), hostile)

    short = copy.deepcopy(result)
    short["timing"]["graph"]["iterations"] = 29
    _expect_rejection(lambda: _acceptance_ready(short, environment, environment), "29 samples")
    incomplete = copy.deepcopy(environment)
    incomplete["fields"]["power_draw_w"] = {"status": "unavailable", "reason": "unsupported"}
    _expect_rejection(
        lambda: _acceptance_ready(result, incomplete, incomplete), "missing power evidence"
    )

    envelope = {
        "identity": {
            "device": {"uuid": result["device"]["uuid"], "target_sm": 86,
                       "driver_version": 12000, "ecc_enabled": False},
            "fixture": {"id": "tiny", "class": "tiny", "sha256": "1", "proof_semantic_hash": "2",
                        "rows": 32, "words_per_row": 23},
            "toolchain": {"sha256": "3"}, "abi_sha256": "4", "shape": {"sha256": "5"},
        },
        "module": {"content_sha256": "before"},
        "environment": {"before": environment, "after": environment},
    }
    candidate = copy.deepcopy(envelope)
    candidate["module"]["content_sha256"] = "after"
    require(_comparison_key(envelope) == _comparison_key(candidate),
            "a changed module incorrectly invalidates comparison identity")
    candidate["identity"]["shape"]["sha256"] = "changed"
    require(_comparison_key(envelope) != _comparison_key(candidate),
            "a changed shape remained comparable")
    candidate = copy.deepcopy(envelope)
    candidate["identity"]["device"]["driver_version"] += 1
    require(_comparison_key(envelope) != _comparison_key(candidate),
            "a changed driver remained comparable")
    candidate = copy.deepcopy(envelope)
    candidate["environment"]["before"]["fields"]["power_limit_w"]["value"] = "changed"
    require(_comparison_key(envelope) != _comparison_key(candidate),
            "a changed power limit remained comparable")

    diagnostic = {
        "schema_version": "stwo.gpu-lab.baseline-comparison.v1",
        "status": "non_admissible", "baseline_path": "/baseline.json",
        "baseline_sha256": None, "matching_identity": None, "before_module_sha256": None,
        "after_module_sha256": _digest("1"), "metrics": None,
    }
    validate_baseline_comparison(diagnostic)
    hostile_diagnostic = copy.deepcopy(diagnostic)
    hostile_diagnostic["metrics"] = {}
    _expect_rejection(
        lambda: validate_baseline_comparison(hostile_diagnostic),
        "non-admissible metrics",
    )

    accepted = _accepted_envelope(environment)
    validate_baseline_envelope(accepted)
    for label, mutate in (
        ("toolchain hash", lambda value: value["identity"]["toolchain"]["normalized_flags"].append("-G")),
        ("derived rate", lambda value: value["measurement"]["derived"]["eager"].update(
            {"semantic_rows_per_s_p50": 1.0}
        )),
        ("orchestrator hash", lambda value: value["audit"]["runtime"].update(
            {"orchestrator_sha256": _digest("0")}
        )),
        ("fixture cross-link", lambda value: value["identity"]["fixture"].update(
            {"sha256": _digest("0")}
        )),
        ("source cross-link", lambda value: value["module"].update(
            {"source_sha256": _digest("0")}
        )),
        ("correctness cross-link", lambda value: value["audit"].update(
            {"correctness_result_sha256": _digest("0")}
        )),
    ):
        hostile = copy.deepcopy(accepted)
        mutate(hostile)
        _expect_rejection(lambda value=hostile: validate_baseline_envelope(value), label)
