"""Fail-closed result validation and raw metric reconciliation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from .artifacts import encode_plan, execution_document, verify_canonical_replay
from .common import (
    REPO_ROOT,
    WORKSPACE_ROOT,
    load_json,
    percentile,
    require,
    require_exact_keys,
    require_int,
    require_number,
    sha256_bytes,
    sha256_file,
)
from .identity import validate_abi, validate_module_index
from .oracle import validate_oracle_index
from .semantics import validate_fixture


def _validate_harness_identity(result: dict[str, Any], harness: Path) -> None:
    require(harness.is_file(), "trusted harness executable is missing")
    identity = result.get("harness_executable_sha256")
    require(
        isinstance(identity, str)
        and len(identity) == 64
        and all(character in "0123456789abcdef" for character in identity),
        "result harness executable identity is not a SHA256",
    )
    require(identity == sha256_file(harness),
            "result harness executable identity mismatch")


def _validate_device(result: dict[str, Any], target_sm: int) -> None:
    device = result.get("device")
    require(isinstance(device, dict) and isinstance(device.get("name"), str) and device["name"],
            "result device identity is missing")
    require_exact_keys(
        device,
        {
            "name", "uuid", "ordinal", "sm_major", "sm_minor", "target_sm",
            "driver_version", "total_memory_bytes", "multiprocessor_count",
            "clock_rate_khz", "memory_clock_rate_khz", "memory_bus_width_bits",
            "ecc_enabled",
        },
        "result device",
    )
    require(
        isinstance(device["uuid"], str)
        and len(device["uuid"]) == 32
        and all(character in "0123456789abcdef" for character in device["uuid"]),
        "device.uuid must be 32 lowercase hex digits",
    )
    for key, minimum in (("ordinal", 0), ("sm_major", 5), ("sm_minor", 0),
                         ("target_sm", 50), ("driver_version", 1)):
        require_int(device[key], f"device.{key}", minimum)
    for key in ("total_memory_bytes", "multiprocessor_count", "clock_rate_khz",
                "memory_clock_rate_khz", "memory_bus_width_bits"):
        require_int(device[key], f"device.{key}", 1)
    require(isinstance(device["ecc_enabled"], bool), "device.ecc_enabled must be boolean")
    require(device["sm_minor"] <= 9, "device.sm_minor exceeds one decimal digit")
    require(device["sm_major"] * 10 + device["sm_minor"] == target_sm,
            "result device SM differs from execution target")
    require(device["target_sm"] == target_sm,
            "result target SM differs from execution target")


def _validate_correctness(
    result: dict[str, Any], mode: str, expected_words: int,
    performance_failed: bool = False,
) -> None:
    correctness = result.get("correctness", {})
    base_keys = {
        "eager_passed", "graph_passed", "mutated_eager_passed",
        "mutated_graph_passed", "eager_checked_words", "graph_checked_words",
        "mutated_eager_checked_words", "mutated_graph_checked_words",
        "eager_error", "graph_error", "mutated_eager_error", "mutated_graph_error",
    }
    post_keys = {"post_eager_passed", "post_graph_passed", "post_eager_checked_words",
                 "post_graph_checked_words", "post_eager_error", "post_graph_error"}
    require(isinstance(correctness, dict), "result correctness is missing")
    require_exact_keys(correctness, base_keys | (post_keys if mode == "benchmark" else set()),
                       "result correctness")
    for prefix in ("eager", "graph", "mutated_eager", "mutated_graph"):
        require(correctness.get(f"{prefix}_passed") is True,
                f"{prefix.replace('_', '-')} correctness is not green")
        require(correctness.get(f"{prefix}_checked_words") == expected_words,
                f"{prefix.replace('_', '-')} validation coverage is incomplete")
        require(correctness.get(f"{prefix}_error") == "",
                f"green {prefix.replace('_', '-')} result contains an error")
    if mode == "benchmark":
        for prefix in ("post_eager", "post_graph"):
            passed = correctness[f"{prefix}_passed"]
            checked = require_int(
                correctness[f"{prefix}_checked_words"],
                f"correctness.{prefix}_checked_words",
            )
            error = correctness[f"{prefix}_error"]
            require(isinstance(passed, bool), f"correctness.{prefix}_passed must be boolean")
            require(isinstance(error, str), f"correctness.{prefix}_error must be a string")
            if passed:
                require(checked == expected_words and error == "",
                        f"green {prefix.replace('_', '-')} validation is inconsistent")
            else:
                require(performance_failed, f"{prefix.replace('_', '-')} correctness is not green")
                require(checked in (0, expected_words) and error,
                        f"failed {prefix.replace('_', '-')} validation is incomplete")
        if performance_failed:
            require(not (correctness["post_eager_passed"] and
                         correctness["post_graph_passed"]),
                    "failed performance result has green post-benchmark validation")


def _reconcile_timing_series(samples: Any, topology: str) -> None:
    require(isinstance(samples, dict), f"{topology} timing series is missing")
    require_exact_keys(
        samples,
        {"warmup_iterations", "warmup_stable", "iterations", "p5_gpu_ms", "p50_gpu_ms",
         "p95_gpu_ms", "stddev_gpu_ms", "p50_submit_us", "p95_submit_us", "p50_wall_ms",
         "p95_wall_ms", "p95_exploratory", "raw_gpu_ms", "raw_submit_us", "raw_wall_ms"},
        f"{topology} timing series",
    )
    iterations = require_int(samples["iterations"], f"{topology}.iterations", 5)
    require_int(samples["warmup_iterations"], f"{topology}.warmup_iterations", 1)
    require(isinstance(samples["warmup_stable"], bool),
            f"{topology}.warmup_stable must be boolean")
    require(samples["p95_exploratory"] == (iterations < 30),
            f"{topology}.p95_exploratory disagrees with sample count")
    raw_series: dict[str, list[float]] = {}
    for key, exclusive in (("raw_gpu_ms", True), ("raw_submit_us", False),
                           ("raw_wall_ms", True)):
        raw = samples[key]
        require(isinstance(raw, list) and len(raw) == iterations,
                f"{topology}.{key} sample count mismatch")
        raw_series[key] = [
            require_number(value, f"{topology}.{key}[{index}]", exclusive=exclusive)
            for index, value in enumerate(raw)
        ]
    gpu = raw_series["raw_gpu_ms"]
    mean_gpu = sum(gpu) / iterations
    summaries = {
        "p5_gpu_ms": percentile(gpu, 0.05),
        "p50_gpu_ms": percentile(gpu, 0.50),
        "p95_gpu_ms": percentile(gpu, 0.95),
        "stddev_gpu_ms": math.sqrt(sum((value - mean_gpu) ** 2 for value in gpu) / iterations),
        "p50_submit_us": percentile(raw_series["raw_submit_us"], 0.50),
        "p95_submit_us": percentile(raw_series["raw_submit_us"], 0.95),
        "p50_wall_ms": percentile(raw_series["raw_wall_ms"], 0.50),
        "p95_wall_ms": percentile(raw_series["raw_wall_ms"], 0.95),
    }
    for key, expected in summaries.items():
        actual = require_number(
            samples[key], f"{topology}.{key}",
            exclusive=key != "stddev_gpu_ms" and not key.endswith("submit_us"),
        )
        require(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6),
                f"{topology}.{key} disagrees with raw samples")


def _validate_timing(
    result: dict[str, Any], mode: str, performance_failed: bool = False,
) -> None:
    timing = result.get("timing")
    require(isinstance(timing, dict), "result timing must be an object")
    if mode == "correctness":
        require(timing == {}, "correctness mode must not emit benchmark timing")
        return
    if performance_failed:
        require_exact_keys(
            timing, {"post_benchmark_passed", "performance_admissible", "error"},
            "failed benchmark timing",
        )
        require(timing["post_benchmark_passed"] is False and
                timing["performance_admissible"] is False,
                "failed benchmark timing is marked admissible")
        require(isinstance(timing["error"], str) and timing["error"],
                "failed benchmark timing lacks an error")
        return
    require_exact_keys(timing, {"post_benchmark_passed", "performance_admissible",
                                "budget_ms", "budget_used_ms", "budget_remaining_ms",
                                "graph_capture_ms", "graph_instantiate_ms", "eager", "graph"},
                       "benchmark timing")
    require(timing.get("post_benchmark_passed") is True,
            "post-benchmark output validation is not green")
    require(timing.get("performance_admissible") is True,
            "successful benchmark timing is not performance-admissible")
    budget = require_number(timing["budget_ms"], "timing.budget_ms", exclusive=True)
    used = require_number(timing["budget_used_ms"], "timing.budget_used_ms", exclusive=True)
    remaining = require_number(timing["budget_remaining_ms"],
                               "timing.budget_remaining_ms")
    require(used <= budget and math.isclose(used + remaining, budget, abs_tol=2e-6),
            "timing shared-budget accounting is inconsistent")
    require_number(timing["graph_capture_ms"], "timing.graph_capture_ms")
    require_number(timing["graph_instantiate_ms"], "timing.graph_instantiate_ms")
    for topology in ("eager", "graph"):
        _reconcile_timing_series(timing.get(topology), topology)


def validate_result(args: argparse.Namespace) -> dict[str, Any]:
    result = load_json(args.result)
    module = load_json(args.module_index)
    fixture = validate_fixture(args.fixture)
    execution = load_json(args.execution)
    require_exact_keys(
        result,
        {"schema_version", "passed", "post_benchmark_passed", "mode",
         "semantic_fixture_sha256", "module_content_sha256", "build_recipe_hash",
         "host_oracle_index_sha256", "execution_manifest_sha256", "plan_sha256",
         "replay_sha256", "harness_executable_sha256", "device", "launch",
         "correctness", "resources", "timing"},
        "result",
    )
    require(result.get("schema_version") == "stwo.gpu-lab.result.v1", "bad result schema")
    require(result.get("mode") == args.mode, "result mode mismatch")
    allow_failure = bool(getattr(args, "allow_performance_failure", False))
    passed = result.get("passed")
    post_benchmark_passed = result.get("post_benchmark_passed")
    require(isinstance(passed, bool), "result passed flag must be boolean")
    require(isinstance(post_benchmark_passed, bool),
            "result post-benchmark flag must be boolean")
    performance_failed = (
        args.mode == "benchmark" and passed is False and post_benchmark_passed is False
    )
    require(passed is True or (allow_failure and performance_failed),
            "candidate result did not pass")
    fixture_hash, plan_hash, replay_hash = (
        sha256_file(args.fixture), sha256_file(args.plan), sha256_file(args.replay)
    )
    verify_canonical_replay(fixture, args.fixture, args.replay)
    abi_path = Path(module.get("abi_path", "")).resolve()
    abi = load_json(abi_path)
    validate_abi(abi, abi_path, REPO_ROOT)
    module_path = validate_module_index(module, abi, abi_path, REPO_ROOT)
    oracle_identity = validate_oracle_index(
        args.oracle_index, args.fixture, fixture, WORKSPACE_ROOT
    )
    expected_execution = execution_document(
        fixture, fixture_hash, replay_hash, module, module_path, abi, abi_path, REPO_ROOT,
        module["target_sm"],
        oracle_identity,
    )
    expected_plan = encode_plan(expected_execution)
    require(args.plan.read_bytes() == expected_plan,
            "launch-plan bytes are not the canonical execution encoding")
    expected_execution["artifacts"]["launch_plan_sha256"] = sha256_bytes(expected_plan)
    require(execution == expected_execution,
            "execution manifest is not the canonical fixture/module/ABI binding")
    require(result.get("semantic_fixture_sha256") == fixture_hash,
            "result fixture identity mismatch")
    require(result.get("module_content_sha256") == module.get("module_content_sha256"),
            "result module identity mismatch")
    require(result.get("build_recipe_hash") == module.get("build_recipe_hash"),
            "result build recipe identity mismatch")
    require(result.get("host_oracle_index_sha256") == oracle_identity["index_sha256"],
            "result host-oracle identity mismatch")
    require(result.get("execution_manifest_sha256") == sha256_file(args.execution),
            "result execution identity mismatch")
    require(result.get("plan_sha256") == plan_hash,
            "result launch-plan identity mismatch")
    require(result.get("replay_sha256") == replay_hash,
            "result replay identity mismatch")
    _validate_harness_identity(result, args.harness)
    target_sm = execution["target"]["sm"]
    _validate_device(result, target_sm)
    launch = result.get("launch")
    require(isinstance(launch, dict), "result launch is missing")
    require_exact_keys(launch, {"symbol", "grid", "block", "dynamic_shared_bytes"},
                       "result launch")
    rows = fixture["semantic_payload"]["row_count"]
    block = execution["kernel"]["launch"]["block"]
    require(launch == {
        "symbol": execution["kernel"]["symbol"],
        "grid": [(rows + block[0] - 1) // block[0], 1, 1],
        "block": block,
        "dynamic_shared_bytes": execution["kernel"]["launch"]["dynamic_shared_bytes"],
    }, "result launch differs from the authenticated execution")
    _validate_correctness(
        result, args.mode, rows * (3 + 14 + 6), performance_failed,
    )
    resources = result.get("resources")
    require(isinstance(resources, dict), "kernel resource evidence is missing")
    require_exact_keys(resources, {"registers_per_thread", "static_shared_bytes",
                                   "local_bytes_per_thread"}, "result resources")
    for key in resources:
        require_int(resources[key], f"resources.{key}")
    require(post_benchmark_passed is (not performance_failed),
            "top-level post-benchmark state is inconsistent")
    _validate_timing(result, args.mode, performance_failed)
    return result
