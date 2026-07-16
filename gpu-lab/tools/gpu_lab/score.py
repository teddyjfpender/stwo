"""Fail-closed, same-identity objective vector for quick-loop sweeps."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from .baseline_validation import (
    MATCH_FIELDS,
    validate_baseline_comparison,
    validate_baseline_envelope,
)
from .baselines import _comparison_key, _core_envelope, _metric_comparison
from .common import (
    canonical_bytes,
    load_json,
    require,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
    sha256_file,
)
from .loop import SLO_BUDGETS, check_loop_slo, validate_loop_record
from .results import validate_result


OBJECTIVES = (
    "graph_p50_speedup",
    "graph_p95_speedup",
    "eager_p50_speedup",
    "eager_p95_speedup",
    "hbm_passes_reduction",
    "hbm_bytes_reduction",
    "launch_count_reduction",
    "registers_per_thread_reduction",
    "local_bytes_per_thread_reduction",
    "spill_bytes_reduction",
    "ordinary_loop_slo_headroom",
)
AVAILABLE_RANKS = frozenset({1, 2, 3, 4, 8, 9, 11})
COMPARISON_POLICY = (
    "lexicographic only; require equal comparison_identity, baseline_identity, and objective "
    "availability; candidate_identity is evidence, not a comparability field"
)
TIMING_REGRESSION_LIMIT = 1.02
TIMING_CREDIT_THRESHOLD = 1.0 / 0.98
MAX_REGISTERS_PER_THREAD = 255
REGISTER_FILE_WORDS_PER_SM_FLOOR = 65536
MIN_REGISTER_LIMITED_BLOCKS_PER_SM = 1
MAX_LOCAL_BYTES_PER_THREAD = 16384
MAX_TOTAL_SHARED_BYTES_PER_BLOCK = 98304
P95_MIN_ITERATIONS = 30
PROMOTION_POLICY = {
    "timing_regression_limit_percent": 2.0,
    "timing_credit_threshold_percent": 2.0,
    "max_registers_per_thread": MAX_REGISTERS_PER_THREAD,
    "register_file_words_per_sm_floor": REGISTER_FILE_WORDS_PER_SM_FLOOR,
    "min_register_limited_blocks_per_sm": MIN_REGISTER_LIMITED_BLOCKS_PER_SM,
    "max_local_bytes_per_thread": MAX_LOCAL_BYTES_PER_THREAD,
    "max_total_shared_bytes_per_block": MAX_TOTAL_SHARED_BYTES_PER_BLOCK,
    "resource_increase_requires_graph_p50_credit": True,
    "p95_min_iterations": P95_MIN_ITERATIONS,
    "p95_requires_stable_warmup": True,
    "p95_requires_non_exploratory": True,
    "loop_slo": "ordinary",
}
UNAVAILABLE_REASON = "validated quick-loop artifacts do not emit this counter"


def _available(rank: int, name: str, unit: str, before: float, after: float,
               raw_value: float, value: float) -> dict[str, Any]:
    require(all(math.isfinite(item) for item in (before, after, raw_value, value)),
            f"non-finite objective: {name}")
    return {
        "rank": rank,
        "name": name,
        "status": "available",
        "direction": "maximize",
        "unit": unit,
        "baseline": before,
        "candidate": after,
        "raw_value": raw_value,
        "value": value,
    }


def _unavailable(rank: int, name: str, reason: str) -> dict[str, Any]:
    return {
        "rank": rank,
        "name": name,
        "status": "unavailable",
        "direction": "maximize",
        "reason": reason,
    }


def _objectives(
    baseline: dict[str, Any], candidate: dict[str, Any], loop: dict[str, Any]
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for topology, percentile in (
        ("graph", "p50"), ("graph", "p95"),
        ("eager", "p50"), ("eager", "p95"),
    ):
        before = baseline["measurement"]["timing"][topology][f"{percentile}_gpu_ms"]
        after = candidate["timing"][topology][f"{percentile}_gpu_ms"]
        raw_speedup = before / after
        values.append(_available(
            len(values) + 1, f"{topology}_{percentile}_speedup", "ratio",
            before, after, raw_speedup,
            raw_speedup if raw_speedup >= TIMING_CREDIT_THRESHOLD else 1.0,
        ))

    for name in ("hbm_passes_reduction", "hbm_bytes_reduction", "launch_count_reduction"):
        values.append(_unavailable(len(values) + 1, name, UNAVAILABLE_REASON))

    before_resources = baseline["measurement"]["resources"]
    after_resources = candidate["resources"]
    for name, field in (
        ("registers_per_thread_reduction", "registers_per_thread"),
        ("local_bytes_per_thread_reduction", "local_bytes_per_thread"),
    ):
        before, after = before_resources[field], after_resources[field]
        values.append(_available(
            len(values) + 1, name, "count", before, after, before - after, before - after,
        ))
    values.append(_unavailable(
        len(values) + 1, "spill_bytes_reduction", UNAVAILABLE_REASON,
    ))

    measured = {
        "build_s": loop["phases_s"]["configure_build"],
        "run_s": sum(loop["phases_s"][name]
                     for name in ("prepare", "correctness", "benchmark")),
        "end_to_end_s": loop["end_to_end_s"],
    }
    utilization = max(
        measured[name] / ceiling
        for name, ceiling in SLO_BUDGETS["ordinary"].items()
    )
    headroom = 1.0 - utilization
    values.append(_available(
        len(values) + 1, "ordinary_loop_slo_headroom", "ratio", 1.0,
        utilization, headroom, headroom,
    ))
    return values


def _require_qualified_p95(candidate: dict[str, Any]) -> None:
    for topology in ("graph", "eager"):
        series = candidate["timing"][topology]
        require(series["warmup_stable"] is True,
                f"{topology} p95 is unavailable because warmup was unstable")
        require(series["iterations"] >= P95_MIN_ITERATIONS,
                f"{topology} p95 is unavailable below {P95_MIN_ITERATIONS} samples")
        require(series["p95_exploratory"] is False,
                f"{topology} p95 is exploratory")


def _require_non_regression(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> None:
    for topology in ("graph", "eager"):
        for percentile in ("p50", "p95"):
            name = f"{percentile}_gpu_ms"
            before = baseline["measurement"]["timing"][topology][name]
            after = candidate["timing"][topology][name]
            require(after <= before * TIMING_REGRESSION_LIMIT,
                    f"{topology} {percentile} exceeds the 2% non-regression budget")


def _require_resource_budget(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> None:
    before = baseline["measurement"]["resources"]
    after = candidate["resources"]
    block = candidate["launch"]["block"]
    threads = block[0] * block[1] * block[2]
    registers = after["registers_per_thread"]
    require(registers <= MAX_REGISTERS_PER_THREAD,
            "register count exceeds the per-thread hardware ceiling")
    require(registers * threads * MIN_REGISTER_LIMITED_BLOCKS_PER_SM <=
            REGISTER_FILE_WORDS_PER_SM_FLOOR,
            "register allocation cannot retain one block per SM")
    require(after["local_bytes_per_thread"] <= MAX_LOCAL_BYTES_PER_THREAD,
            "local bytes exceed the bounded tradeoff ceiling")
    total_shared = after["static_shared_bytes"] + candidate["launch"]["dynamic_shared_bytes"]
    require(total_shared <= MAX_TOTAL_SHARED_BYTES_PER_BLOCK,
            "shared bytes exceed the registered-device ceiling")
    increased = any(after[name] > before[name] for name in (
        "registers_per_thread", "local_bytes_per_thread", "static_shared_bytes",
    ))
    graph_before = baseline["measurement"]["timing"]["graph"]["p50_gpu_ms"]
    graph_after = candidate["timing"]["graph"]["p50_gpu_ms"]
    require(not increased or graph_before / graph_after >= TIMING_CREDIT_THRESHOLD,
            "resource increase lacks material graph-p50 credit")


def _result_args(args: argparse.Namespace, result: Path, mode: str) -> argparse.Namespace:
    values = {**vars(args), "result": result, "mode": mode,
              "allow_performance_failure": False}
    return argparse.Namespace(**values)


def _bind_candidate(correctness: dict[str, Any], benchmark: dict[str, Any]) -> None:
    for field in (
        "semantic_fixture_sha256", "module_content_sha256", "build_recipe_hash",
        "host_oracle_index_sha256", "execution_manifest_sha256", "plan_sha256",
        "replay_sha256", "harness_executable_sha256",
    ):
        require(correctness[field] == benchmark[field],
                f"correctness and benchmark differ at {field}")
    for field in ("uuid", "target_sm", "driver_version", "ecc_enabled"):
        require(correctness["device"][field] == benchmark["device"][field],
                f"correctness and benchmark devices differ at {field}")


def _validate_environment_field(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    status = value.get("status")
    keys = {"status", "value"} if status == "available" else {"status", "reason"}
    require_exact_keys(value, keys, label)
    require(status in {"available", "not_supported", "unavailable"},
            f"{label} status is invalid")
    detail = value["value"] if status == "available" else value["reason"]
    require(isinstance(detail, str) and detail, f"{label} detail is empty")


def validate_score(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "objective score must be an object")
    require_exact_keys(
        value,
        {"schema_version", "eligible", "gates", "inputs", "comparison_identity",
         "baseline_identity", "candidate_identity", "promotion_policy", "objective_vector", "scalar_score",
         "comparison_policy"},
        "objective score",
    )
    require(value["schema_version"] == "stwo.gpu-lab.objective-score.v1",
            "unsupported objective-score schema")
    require(value["eligible"] is True and value["scalar_score"] is None,
            "objective score is not eligible or exposed an opaque scalar")
    require(value["gates"] == {
        "correctness": "passed", "performance": "admissible",
        "identity": "comparable", "loop_record": "valid", "loop_slo": "passed",
        "p95_qualification": "passed", "non_regression": "passed",
        "resource_budget": "passed",
    }, "objective score gates are not all closed")
    require_exact_keys(value["inputs"], {
        "correctness_result_sha256", "benchmark_result_sha256",
        "baseline_comparison_sha256", "loop_record_sha256",
    }, "objective score inputs")
    for name, digest in value["inputs"].items():
        require_sha256(digest, f"objective score {name}")

    identity = value["comparison_identity"]
    require_exact_keys(identity, {
        "device_uuid", "target_sm", "driver_version", "fixture_sha256",
        "toolchain_sha256", "abi_sha256", "shape_sha256",
        "stable_gpu_configuration", "stable_gpu_configuration_sha256", "harness_sha256",
        "orchestrator_sha256", "slo_checker_sha256",
    }, "objective comparison identity")
    require(isinstance(identity["device_uuid"], str) and len(identity["device_uuid"]) == 32
            and all(character in "0123456789abcdef"
                    for character in identity["device_uuid"]),
            "objective score device UUID is invalid")
    require(isinstance(identity["target_sm"], int) and not isinstance(identity["target_sm"], bool)
            and identity["target_sm"] >= 50, "objective score target SM is invalid")
    require(isinstance(identity["driver_version"], int)
            and not isinstance(identity["driver_version"], bool)
            and identity["driver_version"] > 0, "objective score driver is invalid")
    for name in (
        "fixture_sha256", "toolchain_sha256", "abi_sha256", "shape_sha256",
        "stable_gpu_configuration_sha256", "harness_sha256", "orchestrator_sha256",
        "slo_checker_sha256",
    ):
        require_sha256(identity[name], f"objective score {name}")
    configuration = identity["stable_gpu_configuration"]
    require(isinstance(configuration, dict), "stable GPU configuration must be an object")
    require_exact_keys(configuration, {"cuda_ecc_enabled", "nvidia_smi"},
                       "stable GPU configuration")
    require(isinstance(configuration["cuda_ecc_enabled"], bool),
            "stable CUDA ECC state must be boolean")
    nvidia_smi = configuration["nvidia_smi"]
    require(isinstance(nvidia_smi, dict), "stable nvidia-smi configuration must be an object")
    require_exact_keys(nvidia_smi, {"mig_mode", "ecc_mode", "persistence_mode", "power_limit_w"},
                       "stable nvidia-smi configuration")
    for name, field in nvidia_smi.items():
        _validate_environment_field(field, f"stable nvidia-smi {name}")
    require(identity["stable_gpu_configuration_sha256"] ==
            sha256_bytes(canonical_bytes(configuration)),
            "stable GPU configuration hash mismatch")

    baseline_identity = value["baseline_identity"]
    require_exact_keys(baseline_identity, {
        "envelope_sha256", "module_sha256",
    }, "objective baseline identity")
    candidate_identity = value["candidate_identity"]
    require_exact_keys(candidate_identity, {
        "module_sha256", "build_recipe_hash",
    }, "objective candidate identity")
    for label, identities in (
        ("baseline", baseline_identity), ("candidate", candidate_identity),
    ):
        for name, digest in identities.items():
            require_sha256(digest, f"objective score {label} {name}")
    require(value["promotion_policy"] == PROMOTION_POLICY,
            "objective promotion policy changed")

    objectives = value["objective_vector"]
    require(isinstance(objectives, list) and len(objectives) == len(OBJECTIVES),
            "objective vector length mismatch")
    for rank, (entry, name) in enumerate(zip(objectives, OBJECTIVES), 1):
        require(isinstance(entry, dict), f"objective {rank} must be an object")
        require(entry.get("rank") == rank and entry.get("name") == name
                and entry.get("direction") == "maximize",
                f"objective {rank} order or direction mismatch")
        expected_status = "available" if rank in AVAILABLE_RANKS else "unavailable"
        require(entry.get("status") == expected_status,
                f"objective {rank} availability mismatch")
        if expected_status == "unavailable":
            require_exact_keys(entry, {"rank", "name", "status", "direction", "reason"},
                               f"objective {rank}")
            require(entry["reason"] == UNAVAILABLE_REASON,
                    f"objective {rank} unavailable reason changed")
            continue
        require_exact_keys(entry, {
            "rank", "name", "status", "direction", "unit", "baseline", "candidate",
            "raw_value", "value",
        }, f"objective {rank}")
        expected_unit = "count" if rank in {8, 9} else "ratio"
        require(entry["unit"] == expected_unit, f"objective {rank} unit mismatch")
        require(all(isinstance(entry[key], (int, float)) and not isinstance(entry[key], bool)
                    and math.isfinite(entry[key])
                    for key in ("baseline", "candidate", "raw_value", "value")),
                f"objective {rank} contains a non-finite value")
        before, after, raw, objective = (
            entry["baseline"], entry["candidate"], entry["raw_value"], entry["value"]
        )
        if rank <= 4:
            require(before > 0 and after > 0,
                    f"objective {rank} timing inputs must be positive")
            expected_raw = before / after
            expected_value = expected_raw if expected_raw >= TIMING_CREDIT_THRESHOLD else 1.0
            require(math.isclose(raw, expected_raw,
                    rel_tol=1e-12, abs_tol=1e-12) and math.isclose(objective, expected_value,
                    rel_tol=1e-12, abs_tol=1e-12),
                    f"objective {rank} speedup formula mismatch")
        elif rank in {8, 9}:
            require(before >= 0 and after >= 0 and raw == before - after and objective == raw,
                    f"objective {rank} reduction formula mismatch")
        else:
            require(before == 1.0 and 0 <= after <= 1.0 and raw == 1.0 - after
                    and objective == raw, "loop-SLO objective formula mismatch")
    require(value["comparison_policy"] == COMPARISON_POLICY,
            "objective comparison policy changed")
    return value


def score(args: argparse.Namespace) -> dict[str, Any]:
    require(args.output.name == "objective-score.json"
            and args.output.parent.resolve() == args.loop.parent.resolve(),
            "objective output must be objective-score.json beside loop.json")
    protected = (
        args.result, args.correctness_result, args.fixture, args.oracle_index,
        args.module_index, args.execution, args.replay, args.plan, args.harness,
        args.comparison, args.loop,
    )
    require(args.output.resolve() not in {path.resolve() for path in protected},
            "objective output aliases an input artifact")
    # Delete the old reward before parsing any candidate-controlled artifact.
    args.output.unlink(missing_ok=True)
    raw_comparison = load_json(args.comparison)
    correctness = validate_result(_result_args(args, args.correctness_result, "correctness"))
    benchmark = validate_result(_result_args(args, args.result, "benchmark"))
    _bind_candidate(correctness, benchmark)

    comparison = validate_baseline_comparison(raw_comparison)
    require(comparison["status"] == "compared",
            "objective score requires a comparable accepted baseline")
    baseline_path = Path(comparison["baseline_path"])
    require(sha256_file(baseline_path) == comparison["baseline_sha256"],
            "objective baseline bytes changed after comparison")
    baseline = validate_baseline_envelope(load_json(baseline_path))
    require(sha256_file(baseline_path) == comparison["baseline_sha256"],
            "objective baseline bytes changed during validation")
    loop = validate_loop_record(load_json(args.loop))
    require(loop["runtime_audit"]["performance_admissible"] is True,
            "objective loop is performance-non-admissible")

    bound_paths = {
        "fixture": args.fixture,
        "oracle_index": args.oracle_index,
        "module_index": args.module_index,
        "execution": args.execution,
        "replay": args.replay,
        "plan": args.plan,
        "harness": args.harness,
        "correctness_result": args.correctness_result,
        "benchmark_result": args.result,
        "baseline_comparison": args.comparison,
    }
    for name, path in bound_paths.items():
        require(loop["artifacts"][name]["sha256"] == sha256_file(path),
                f"objective input differs from loop artifact: {name}")

    fixture, module, execution = (
        load_json(args.fixture), load_json(args.module_index), load_json(args.execution)
    )
    current = _core_envelope(
        fixture, module, execution, benchmark,
        loop["environment"]["before"], loop["environment"]["after"],
    )
    current["measurement"]["result_sha256"] = sha256_file(args.result)
    baseline_key, current_key = _comparison_key(baseline), _comparison_key(current)
    matching = {name: baseline_key[name] == current_key[name] for name in MATCH_FIELDS}
    require(all(matching.values()) and comparison["matching_identity"] == matching,
            "objective baseline identity is not exactly comparable")
    require(comparison["before_module_sha256"] == baseline["module"]["content_sha256"]
            and comparison["after_module_sha256"] == benchmark["module_content_sha256"],
            "objective comparison module identity mismatch")
    require(comparison["metrics"] == _metric_comparison(baseline, current),
            "objective comparison metrics do not bind baseline and candidate")
    baseline_runtime = baseline["audit"]["runtime"]
    candidate_runtime = loop["runtime_audit"]
    require(baseline_runtime["harness_sha256"] == benchmark["harness_executable_sha256"]
            == candidate_runtime["harness_sha256"],
            "measurement harness differs from the accepted baseline")
    for name in ("orchestrator_sha256", "slo_checker_sha256"):
        require(baseline_runtime[name] == candidate_runtime[name],
                f"measurement {name} differs from the accepted baseline")
    _require_qualified_p95(benchmark)
    check_loop_slo(loop, "ordinary")
    _require_non_regression(baseline, benchmark)
    _require_resource_budget(baseline, benchmark)

    identity = baseline["identity"]
    stable_configuration = baseline_key["configuration"]
    value = {
        "schema_version": "stwo.gpu-lab.objective-score.v1",
        "eligible": True,
        "gates": {
            "correctness": "passed", "performance": "admissible",
            "identity": "comparable", "loop_record": "valid", "loop_slo": "passed",
            "p95_qualification": "passed", "non_regression": "passed",
            "resource_budget": "passed",
        },
        "inputs": {
            "correctness_result_sha256": sha256_file(args.correctness_result),
            "benchmark_result_sha256": sha256_file(args.result),
            "baseline_comparison_sha256": sha256_file(args.comparison),
            "loop_record_sha256": sha256_file(args.loop),
        },
        "comparison_identity": {
            "device_uuid": identity["device"]["uuid"],
            "target_sm": identity["device"]["target_sm"],
            "driver_version": identity["device"]["driver_version"],
            "fixture_sha256": identity["fixture"]["sha256"],
            "toolchain_sha256": identity["toolchain"]["sha256"],
            "abi_sha256": identity["abi_sha256"],
            "shape_sha256": identity["shape"]["sha256"],
            "stable_gpu_configuration": stable_configuration,
            "stable_gpu_configuration_sha256": sha256_bytes(
                canonical_bytes(stable_configuration)
            ),
            "harness_sha256": baseline_runtime["harness_sha256"],
            "orchestrator_sha256": baseline_runtime["orchestrator_sha256"],
            "slo_checker_sha256": baseline_runtime["slo_checker_sha256"],
        },
        "baseline_identity": {
            "envelope_sha256": comparison["baseline_sha256"],
            "module_sha256": baseline["module"]["content_sha256"],
        },
        "candidate_identity": {
            "module_sha256": benchmark["module_content_sha256"],
            "build_recipe_hash": benchmark["build_recipe_hash"],
        },
        "promotion_policy": PROMOTION_POLICY,
        "objective_vector": _objectives(baseline, benchmark, loop),
        "scalar_score": None,
        "comparison_policy": COMPARISON_POLICY,
    }
    validate_score(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return value
