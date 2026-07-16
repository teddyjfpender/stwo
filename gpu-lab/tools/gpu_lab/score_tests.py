"""Mutation and real-validator integration checks for objective scoring."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from .baseline_tests import _accepted_envelope
from .baseline_validation import MATCH_FIELDS
from .baselines import _comparison_key, _metric_comparison
from .common import LAB_ROOT, canonical_bytes, require, sha256_bytes, sha256_file
from .loop import ARTIFACTS, ENVIRONMENT_QUERIES, PHASES, write_loop_record
from .score import (
    OBJECTIVES,
    _objectives,
    _require_non_regression,
    _require_qualified_p95,
    _require_resource_budget,
    score,
    validate_score,
)
from .staging import seal_staging, stage_fixture


def _timing(graph: tuple[float, float], eager: tuple[float, float]) -> dict:
    return {
        "graph": {"p50_gpu_ms": graph[0], "p95_gpu_ms": graph[1]},
        "eager": {"p50_gpu_ms": eager[0], "p95_gpu_ms": eager[1]},
    }


def _baseline() -> dict:
    return {
        "measurement": {
            "timing": _timing((4.0, 6.0), (5.0, 8.0)),
            "resources": {
                "registers_per_thread": 64,
                "static_shared_bytes": 0,
                "local_bytes_per_thread": 8,
            },
        },
    }


def _candidate() -> dict:
    return {
        "timing": _timing((2.0, 3.0), (4.0, 4.0)),
        "resources": {
            "registers_per_thread": 60,
            "static_shared_bytes": 0,
            "local_bytes_per_thread": 0,
        },
        "launch": {"block": [256, 1, 1], "dynamic_shared_bytes": 0},
    }


def _environment() -> dict:
    fields = {
        name: {
            "status": "available",
            "value": "GPU-00112233-4455-6677-8899-aabbccddeeff"
            if name == "uuid" else name,
        }
        for name in ENVIRONMENT_QUERIES
    }
    return {
        "schema_version": "stwo.gpu-lab.environment.v1",
        "device_selector": "0",
        "fields": fields,
    }


def _candidate_identity(timing: dict, resources: dict) -> dict:
    return {
        **{name: "6" * 64 for name in (
            "semantic_fixture_sha256", "build_recipe_hash", "host_oracle_index_sha256",
            "execution_manifest_sha256", "plan_sha256", "replay_sha256",
            "harness_executable_sha256",
        )},
        "module_content_sha256": "b" * 64,
        "device": {
            "uuid": "00112233445566778899aabbccddeeff",
            "target_sm": 86,
            "driver_version": 12000,
            "ecc_enabled": False,
        },
        "timing": timing,
        "resources": resources,
        "launch": {"block": [256, 1, 1], "dynamic_shared_bytes": 0},
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _integration_case(root: Path) -> tuple[argparse.Namespace, dict, dict, dict]:
    root.mkdir(parents=True)
    environment = _environment()
    environment_path = root / "environment.json"
    _write_json(environment_path, environment)
    baseline = _accepted_envelope(environment)
    baseline_path = root / "baseline.json"

    timing = copy.deepcopy(baseline["measurement"]["timing"])
    resources = copy.deepcopy(baseline["measurement"]["resources"])
    resources["registers_per_thread"] -= 1
    benchmark = _candidate_identity(timing, resources)
    correctness = {
        name: copy.deepcopy(benchmark[name])
        for name in (
            "semantic_fixture_sha256", "module_content_sha256", "build_recipe_hash",
            "host_oracle_index_sha256", "execution_manifest_sha256", "plan_sha256",
            "replay_sha256", "harness_executable_sha256", "device",
        )
    }
    current = copy.deepcopy(baseline)
    current["module"]["content_sha256"] = benchmark["module_content_sha256"]
    current["measurement"]["timing"] = benchmark["timing"]
    current["measurement"]["resources"] = benchmark["resources"]

    baseline_key, current_key = _comparison_key(baseline), _comparison_key(current)
    matching = {name: baseline_key[name] == current_key[name] for name in MATCH_FIELDS}
    comparison = {
        "schema_version": "stwo.gpu-lab.baseline-comparison.v1",
        "status": "compared",
        "baseline_path": str(baseline_path),
        "baseline_sha256": None,
        "matching_identity": matching,
        "before_module_sha256": baseline["module"]["content_sha256"],
        "after_module_sha256": benchmark["module_content_sha256"],
        "metrics": _metric_comparison(baseline, current),
    }

    artifact_paths = {name: root / name for name in ARTIFACTS}
    for name, path in artifact_paths.items():
        if name == "build_commands":
            path.write_text("nvcc -cubin source.cu -o module.cubin\n")
        elif name == "baseline_comparison":
            continue
        elif name in {"fixture", "oracle_index", "module_index", "execution",
                      "correctness_result", "benchmark_result"}:
            _write_json(path, {})
        else:
            path.write_text(name)
    runtime_hashes = {
        "harness_sha256": sha256_file(artifact_paths["harness"]),
        "orchestrator_sha256": sha256_file(LAB_ROOT / "tools/quick-loop"),
        "slo_checker_sha256": sha256_file(LAB_ROOT / "tools/check-loop-slo"),
    }
    baseline["audit"]["runtime"].update(runtime_hashes)
    baseline["snapshot"]["top_level"]["harness"]["sha256"] = runtime_hashes["harness_sha256"]
    for dependency in baseline["snapshot"]["transitive"]:
        for role in ("orchestrator", "slo_checker"):
            if role in dependency["roles"]:
                dependency["sha256"] = runtime_hashes[f"{role}_sha256"]
    baseline["snapshot"]["sha256"] = sha256_bytes(canonical_bytes({
        "top_level": baseline["snapshot"]["top_level"],
        "transitive": baseline["snapshot"]["transitive"],
    }))
    _write_json(baseline_path, baseline)
    benchmark["harness_executable_sha256"] = runtime_hashes["harness_sha256"]
    correctness["harness_executable_sha256"] = runtime_hashes["harness_sha256"]
    comparison["baseline_sha256"] = sha256_file(baseline_path)
    _write_json(artifact_paths["baseline_comparison"], comparison)
    module = root / "module.cubin"
    module.write_text("module")
    seed = stage_fixture(artifact_paths["fixture"], None, "", root)
    staging = seal_staging(seed, root, {
        "build_commands": artifact_paths["build_commands"],
        "execution": artifact_paths["execution"],
        "harness": artifact_paths["harness"],
        "module": module,
        "module_index": artifact_paths["module_index"],
        "plan": artifact_paths["plan"],
        "replay": artifact_paths["replay"],
    })
    staging_path = root / "staging.json"
    _write_json(staging_path, staging)
    loop_path = root / "loop.json"
    loop_args = argparse.Namespace(
        phase=[f"{name}=1:2" for name in PHASES],
        artifact=[f"{name}={path}" for name, path in artifact_paths.items()],
        total="1:5",
        environment_before=environment_path,
        environment_after=environment_path,
        harness=artifact_paths["harness"],
        runtime_mode=["correctness", "benchmark"],
        benchmark_exit_code=0,
        performance_status="admissible",
        orchestrator=LAB_ROOT / "tools/quick-loop",
        slo_checker=LAB_ROOT / "tools/check-loop-slo",
        staging_record=staging_path,
        output=loop_path,
    )
    write_loop_record(loop_args)
    args = argparse.Namespace(
        result=artifact_paths["benchmark_result"],
        correctness_result=artifact_paths["correctness_result"],
        fixture=artifact_paths["fixture"],
        oracle_index=artifact_paths["oracle_index"],
        module_index=artifact_paths["module_index"],
        execution=artifact_paths["execution"],
        replay=artifact_paths["replay"],
        plan=artifact_paths["plan"],
        harness=artifact_paths["harness"],
        comparison=artifact_paths["baseline_comparison"],
        loop=loop_path,
        output=root / "objective-score.json",
    )
    return args, correctness, benchmark, current


def _run_integration(args: argparse.Namespace, correctness: dict,
                     benchmark: dict, current: dict) -> dict:
    with (
        patch("gpu_lab.score.validate_result", side_effect=[correctness, benchmark]),
        patch("gpu_lab.score._core_envelope", return_value=current),
    ):
        return score(args)


def _expect_failure(call, label: str) -> None:
    try:
        call()
    except ValueError:
        return
    raise ValueError(f"objective score accepted {label}")


def score_self_test() -> None:
    loop = {
        "phases_s": {"configure_build": 1.0, "prepare": 1.0,
                     "correctness": 1.0, "benchmark": 1.0},
        "end_to_end_s": 5.0,
    }
    objectives = _objectives(_baseline(), _candidate(), loop)
    require([entry["name"] for entry in objectives] == list(OBJECTIVES),
            "objective order drifted")
    require([entry["value"] for entry in objectives[:4]] == [2.0, 2.0, 1.25, 2.0],
            "timing speedups are incorrect")
    require([entry["status"] for entry in objectives[4:7]] == ["unavailable"] * 3,
            "missing structural counters were assigned reward")
    require(objectives[7]["value"] == 4 and objectives[8]["value"] == 8,
            "resource reductions are incorrect")
    require(objectives[9]["status"] == "unavailable" and 0 < objectives[10]["value"] < 1,
            "spill or loop-SLO objective is incorrect")

    epsilon = _candidate()
    epsilon["timing"] = _timing((3.98, 6.0), (5.0, 8.0))
    epsilon_values = _objectives(_baseline(), epsilon, loop)
    require(epsilon_values[0]["raw_value"] > 1 and epsilon_values[0]["value"] == 1.0,
            "sub-threshold p50 noise received lexicographic credit")

    with tempfile.TemporaryDirectory() as temporary_name:
        root = Path(temporary_name)
        args, correctness, benchmark, current = _integration_case(root / "valid")
        value = _run_integration(args, correctness, benchmark, current)
        validate_score(json.loads(args.output.read_text()))
        require(value["baseline_identity"]["module_sha256"] !=
                value["candidate_identity"]["module_sha256"],
                "candidate cubin was incorrectly made a comparability field")

        hostile = copy.deepcopy(value)
        hostile["objective_vector"][0]["value"] = 1.5
        _expect_failure(lambda: validate_score(hostile), "an incorrect objective formula")

        args.comparison.write_text("{")
        args.output.write_text("stale reward\n")
        _expect_failure(lambda: score(args), "malformed comparison JSON")
        require(not args.output.exists(), "malformed input left stale reward visible")

        args, correctness, benchmark, current = _integration_case(root / "gates")
        wrong_harness = copy.deepcopy(benchmark)
        wrong_harness["harness_executable_sha256"] = "f" * 64
        wrong_correctness = copy.deepcopy(correctness)
        wrong_correctness["harness_executable_sha256"] = "f" * 64
        _expect_failure(
            lambda: _run_integration(args, wrong_correctness, wrong_harness, current),
            "a changed measurement harness",
        )
        require(not args.output.exists(), "measurement-method mismatch wrote reward")

        exploratory = copy.deepcopy(benchmark)
        exploratory["timing"]["graph"]["iterations"] = 5
        exploratory["timing"]["graph"]["p95_exploratory"] = True
        exploratory_current = copy.deepcopy(current)
        exploratory_current["measurement"]["timing"] = exploratory["timing"]
        _expect_failure(
            lambda: _run_integration(args, correctness, exploratory, exploratory_current),
            "an exploratory p95",
        )
        require(not args.output.exists(), "exploratory p95 wrote reward")

        regressed = copy.deepcopy(benchmark)
        regressed["resources"]["registers_per_thread"] += 2
        regressed_current = copy.deepcopy(current)
        regressed_current["measurement"]["resources"] = regressed["resources"]
        _expect_failure(
            lambda: _run_integration(args, correctness, regressed, regressed_current),
            "a register regression",
        )
        require(not args.output.exists(), "resource regression wrote reward")

        _expect_failure(
            lambda: _require_non_regression(
                _baseline(), {**_candidate(), "timing": _timing((4.2, 6.0), (5.0, 8.0))}
            ),
            "a latency regression",
        )
        qualified = copy.deepcopy(benchmark)
        qualified["timing"]["eager"]["warmup_stable"] = False
        _expect_failure(lambda: _require_qualified_p95(qualified), "unstable warmup p95")
        profitable = _candidate()
        profitable["resources"]["registers_per_thread"] = 65
        _require_resource_budget(_baseline(), profitable)

        slo_args, slo_correctness, slo_benchmark, slo_current = _integration_case(root / "slo")
        slo_record = json.loads(slo_args.loop.read_text())
        slo_record["end_to_end_s"] = 121.0
        slo_record["record_sha256"] = sha256_bytes(canonical_bytes({
            name: payload for name, payload in slo_record.items() if name != "record_sha256"
        }))
        _write_json(slo_args.loop, slo_record)
        _expect_failure(
            lambda: _run_integration(
                slo_args, slo_correctness, slo_benchmark, slo_current
            ),
            "an ordinary-loop SLO regression",
        )
        require(not slo_args.output.exists(), "loop-SLO failure wrote reward")
