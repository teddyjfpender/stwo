"""Fail-closed validation for accepted baseline and comparison records."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .common import canonical_bytes, require, require_exact_keys, require_sha256, sha256_bytes
from .loop import (
    normalized_gpu_uuid,
    validate_build_audit,
    validate_environment_pair,
    validate_runtime_audit,
)
from .results import _validate_timing


MATCH_FIELDS = (
    "device_uuid", "target_sm", "driver_version", "configuration",
    "fixture", "toolchain", "abi", "shape",
)
TOP_LEVEL_SNAPSHOTS = {
    "result", "fixture", "oracle_index", "module_index", "execution", "replay", "plan",
    "harness", "correctness_result", "build_commands", "environment_before", "environment_after",
    "loop_record", "comparison",
}


def _exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    require_exact_keys(value, keys, label)
    return value


def _artifact(value: Any, label: str) -> dict[str, Any]:
    value = _exact_object(value, {"sha256", "bytes"}, label)
    require_sha256(value["sha256"], f"{label}.sha256")
    require(isinstance(value["bytes"], int) and not isinstance(value["bytes"], bool)
            and value["bytes"] >= 0, f"{label}.bytes is invalid")
    return value


def _identity(value: Any) -> dict[str, Any]:
    value = _exact_object(
        value, {"device", "fixture", "oracle", "abi_sha256", "shape", "toolchain"},
        "baseline identity",
    )
    device = _exact_object(
        value["device"], {"uuid", "name", "target_sm", "driver_version", "ecc_enabled"},
        "baseline device",
    )
    require(re.fullmatch(r"[0-9a-f]{32}", device["uuid"]) is not None,
            "baseline device UUID is invalid")
    require(isinstance(device["name"], str) and device["name"], "baseline device name is empty")
    for name in ("target_sm", "driver_version"):
        require(isinstance(device[name], int) and not isinstance(device[name], bool)
                and device[name] > 0, f"baseline device {name} is invalid")
    require(isinstance(device["ecc_enabled"], bool), "baseline CUDA ECC state is invalid")

    fixture = _exact_object(
        value["fixture"],
        {"id", "class", "sha256", "proof_semantic_hash", "rows", "words_per_row"},
        "baseline fixture",
    )
    require(isinstance(fixture["id"], str)
            and len(fixture["id"]) <= 128
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", fixture["id"]) is not None,
            "baseline fixture id is unsafe")
    require(isinstance(fixture["class"], str) and fixture["class"], "baseline fixture class is empty")
    require_sha256(fixture["sha256"], "baseline fixture sha256")
    require_sha256(fixture["proof_semantic_hash"], "baseline proof semantic hash")
    for name in ("rows", "words_per_row"):
        require(isinstance(fixture[name], int) and not isinstance(fixture[name], bool)
                and fixture[name] > 0, f"baseline fixture {name} is invalid")

    oracle = _exact_object(
        value["oracle"], {"index_sha256", "artifact_sha256", "exporter_closure_sha256"},
        "baseline oracle",
    )
    for name, digest in oracle.items():
        require_sha256(digest, f"baseline oracle {name}")
    require_sha256(value["abi_sha256"], "baseline ABI sha256")

    shape = _exact_object(
        value["shape"],
        {"sha256", "fixture_class", "row_count", "semantic_operation", "launch", "physical_layout"},
        "baseline shape",
    )
    expected_shape = sha256_bytes(canonical_bytes({
        name: payload for name, payload in shape.items() if name != "sha256"
    }))
    require(shape["sha256"] == expected_shape, "baseline shape hash mismatch")
    require(shape["row_count"] == fixture["rows"]
            and shape["fixture_class"] == fixture["class"], "baseline shape differs from fixture")

    toolchain_keys = {
        "nvcc_path", "nvcc_version", "host_compiler_path", "host_compiler_version",
        "ptxas_path", "ptxas_version", "cuobjdump_path", "cuobjdump_version", "environment",
        "driver_compatibility_policy", "target_sm", "normalized_flags", "sha256",
    }
    toolchain = _exact_object(value["toolchain"], toolchain_keys, "baseline toolchain")
    expected_toolchain = sha256_bytes(canonical_bytes({
        name: payload for name, payload in toolchain.items() if name != "sha256"
    }))
    require(toolchain["sha256"] == expected_toolchain, "baseline toolchain hash mismatch")
    require(toolchain["target_sm"] == device["target_sm"], "baseline toolchain SM mismatch")
    require(isinstance(toolchain["normalized_flags"], list) and toolchain["normalized_flags"],
            "baseline normalized flags are missing")
    require(all(isinstance(flag, str) and flag for flag in toolchain["normalized_flags"]),
            "baseline normalized flags contain an invalid token")
    return value


def _derived(timing: dict[str, Any], rows: int, words_per_row: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for topology in ("eager", "graph"):
        result[topology] = {}
        for percentile in ("p50", "p95"):
            milliseconds = timing[topology][f"{percentile}_gpu_ms"]
            result[topology][f"semantic_rows_per_s_{percentile}"] = rows * 1000.0 / milliseconds
            result[topology][f"semantic_words_per_s_{percentile}"] = (
                rows * words_per_row * 1000.0 / milliseconds
            )
    return result


def _measurement(value: Any, fixture: dict[str, Any]) -> dict[str, Any]:
    value = _exact_object(
        value, {"result_sha256", "resources", "timing", "derived"}, "baseline measurement"
    )
    require_sha256(value["result_sha256"], "baseline result sha256")
    resources = _exact_object(
        value["resources"], {"registers_per_thread", "static_shared_bytes", "local_bytes_per_thread"},
        "baseline resources",
    )
    require(all(isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in resources.values()), "baseline resources are invalid")
    _validate_timing({"timing": value["timing"]}, "benchmark")
    expected = _derived(value["timing"], fixture["rows"], fixture["words_per_row"])
    derived = _exact_object(value["derived"], {"eager", "graph"}, "baseline derived metrics")
    for topology in ("eager", "graph"):
        _exact_object(derived[topology], set(expected[topology]), f"baseline {topology} derived metrics")
        for name, expected_value in expected[topology].items():
            actual = derived[topology][name]
            require(isinstance(actual, (int, float)) and math.isfinite(actual)
                    and math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-12),
                    f"baseline derived metric mismatch: {topology}.{name}")
        series = value["timing"][topology]
        require(series["warmup_stable"] is True, f"{topology} baseline warmup was unstable")
        require(series["iterations"] >= 30, f"{topology} baseline has fewer than 30 samples")
        require(series["p95_exploratory"] is False, f"{topology} baseline p95 is exploratory")
    return value


def _snapshot(value: Any) -> dict[str, Any]:
    value = _exact_object(value, {"top_level", "transitive", "sha256"}, "baseline snapshot")
    require_sha256(value["sha256"], "baseline snapshot sha256")
    top = _exact_object(value["top_level"], TOP_LEVEL_SNAPSHOTS, "baseline top-level snapshot")
    for name, artifact in top.items():
        _artifact(artifact, f"baseline top-level {name}")
    transitive = value["transitive"]
    require(isinstance(transitive, list) and transitive, "baseline transitive snapshot is empty")
    paths = []
    for index, dependency in enumerate(transitive):
        dependency = _exact_object(
            dependency, {"path", "roles", "sha256", "bytes"},
            f"baseline transitive dependency {index}",
        )
        require(Path(dependency["path"]).is_absolute(), "baseline dependency path is not absolute")
        require(isinstance(dependency["roles"], list) and dependency["roles"]
                and all(isinstance(role, str) and role for role in dependency["roles"])
                and dependency["roles"] == sorted(set(dependency["roles"])),
                "baseline dependency roles are invalid")
        _artifact({"sha256": dependency["sha256"], "bytes": dependency["bytes"]},
                  f"baseline transitive dependency {index}")
        paths.append(dependency["path"])
    require(paths == sorted(set(paths)), "baseline transitive dependencies are not canonical")
    expected = sha256_bytes(canonical_bytes({"top_level": top, "transitive": transitive}))
    require(value["sha256"] == expected, "baseline snapshot aggregate hash mismatch")
    return value


def _role_hash(snapshot: dict[str, Any], role: str) -> str:
    matches = [item["sha256"] for item in snapshot["transitive"] if role in item["roles"]]
    require(len(matches) == 1, f"baseline snapshot must contain exactly one {role}")
    return matches[0]


def validate_baseline_envelope(value: Any) -> dict[str, Any]:
    value = _exact_object(
        value,
        {"schema_version", "identity", "module", "measurement", "environment", "audit", "snapshot"},
        "baseline envelope",
    )
    require(value["schema_version"] == "stwo.gpu-lab.baseline.v1",
            "unsupported baseline schema")
    identity = _identity(value["identity"])
    module = _exact_object(
        value["module"], {"content_sha256", "build_recipe_hash", "source_sha256"},
        "baseline module",
    )
    for name, digest in module.items():
        require_sha256(digest, f"baseline module {name}")
    measurement = _measurement(value["measurement"], identity["fixture"])
    environment = _exact_object(value["environment"], {"before", "after"}, "baseline environment")
    validate_environment_pair(environment["before"], environment["after"], True)
    fields = environment["before"]["fields"]
    require(normalized_gpu_uuid(fields["uuid"]["value"]) == identity["device"]["uuid"],
            "baseline environment belongs to another GPU")
    require(fields["name"]["value"] == identity["device"]["name"],
            "baseline environment GPU name mismatch")
    audit = _exact_object(value["audit"],
                          {"loop_record_sha256", "correctness_result_sha256",
                           "comparison_sha256", "build", "runtime"},
                          "baseline audit")
    for name in ("loop_record_sha256", "correctness_result_sha256", "comparison_sha256"):
        require_sha256(audit[name], f"baseline {name}")
    validate_build_audit(audit["build"])
    validate_runtime_audit(audit["runtime"])
    snapshot = _snapshot(value["snapshot"])
    top = snapshot["top_level"]
    for name, document in (("environment_before", environment["before"]),
                           ("environment_after", environment["after"])):
        encoded = json.dumps(document, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
        require(sha256_bytes(encoded) == top[name]["sha256"],
                f"baseline {name} snapshot mismatch")
    require(measurement["result_sha256"] == top["result"]["sha256"],
            "baseline result snapshot mismatch")
    require(identity["fixture"]["sha256"] == top["fixture"]["sha256"],
            "baseline fixture snapshot mismatch")
    require(identity["oracle"]["index_sha256"] == top["oracle_index"]["sha256"],
            "baseline oracle-index snapshot mismatch")
    require(audit["loop_record_sha256"] == top["loop_record"]["sha256"],
            "baseline loop snapshot mismatch")
    require(audit["correctness_result_sha256"] == top["correctness_result"]["sha256"],
            "baseline correctness snapshot mismatch")
    require(audit["comparison_sha256"] == top["comparison"]["sha256"],
            "baseline comparison snapshot mismatch")
    require(audit["build"]["commands_sha256"] == top["build_commands"]["sha256"],
            "baseline build-command snapshot mismatch")
    require(audit["runtime"]["harness_sha256"] == top["harness"]["sha256"],
            "baseline harness snapshot mismatch")
    require(module["content_sha256"] == _role_hash(snapshot, "module"),
            "baseline module dependency mismatch")
    require(module["source_sha256"] == _role_hash(snapshot, "cuda_source"),
            "baseline CUDA source dependency mismatch")
    require(identity["abi_sha256"] == _role_hash(snapshot, "abi"),
            "baseline ABI dependency mismatch")
    require(identity["oracle"]["artifact_sha256"] == _role_hash(snapshot, "host_oracle"),
            "baseline oracle dependency mismatch")
    require(audit["runtime"]["orchestrator_sha256"] == _role_hash(snapshot, "orchestrator"),
            "baseline orchestrator dependency mismatch")
    require(audit["runtime"]["slo_checker_sha256"] == _role_hash(snapshot, "slo_checker"),
            "baseline SLO checker dependency mismatch")
    return value


def validate_baseline_comparison(value: Any) -> dict[str, Any]:
    value = _exact_object(
        value,
        {"schema_version", "status", "baseline_path", "baseline_sha256", "matching_identity",
         "before_module_sha256", "after_module_sha256", "metrics"},
        "baseline comparison",
    )
    require(value["schema_version"] == "stwo.gpu-lab.baseline-comparison.v1",
            "unsupported baseline comparison schema")
    require(value["status"] in {"missing", "non_admissible", "incomparable", "compared"},
            "invalid baseline comparison status")
    require(isinstance(value["baseline_path"], str) and value["baseline_path"],
            "baseline comparison path is empty")
    require_sha256(value["after_module_sha256"], "candidate module sha256")
    if value["status"] == "missing":
        require(all(value[name] is None for name in
                    ("baseline_sha256", "matching_identity", "before_module_sha256", "metrics")),
                "missing baseline comparison contains prior evidence")
        return value
    if value["status"] == "non_admissible" and value["baseline_sha256"] is None:
        require(value["matching_identity"] is None and value["before_module_sha256"] is None
                and value["metrics"] is None,
                "non-admissible missing-baseline result contains prior evidence")
        return value
    require_sha256(value["baseline_sha256"], "prior baseline sha256")
    require_sha256(value["before_module_sha256"], "prior module sha256")
    matching = _exact_object(value["matching_identity"], set(MATCH_FIELDS),
                             "baseline matching identity")
    require(all(isinstance(item, bool) for item in matching.values()),
            "baseline matching identity contains a non-boolean")
    if value["status"] == "non_admissible":
        require(value["metrics"] is None,
                "non-admissible benchmark exposed comparison metrics")
    elif value["status"] == "compared":
        require(all(matching.values()) and isinstance(value["metrics"], dict),
                "compared baseline lacks matching identity or metrics")
        metrics = _exact_object(value["metrics"], {"eager", "graph"},
                                "baseline comparison metrics")
        for topology in ("eager", "graph"):
            expected = {
                "p50_gpu_ms": "lower", "p95_gpu_ms": "lower",
                "semantic_rows_per_s_p50": "higher", "semantic_words_per_s_p50": "higher",
            }
            series = _exact_object(metrics[topology], set(expected),
                                   f"baseline comparison {topology}")
            for name, direction in expected.items():
                metric = _exact_object(
                    series[name], {"before", "after", "delta_percent", "better_when"},
                    f"baseline comparison {topology}.{name}",
                )
                before, after = metric["before"], metric["after"]
                require(isinstance(before, (int, float)) and before > 0
                        and isinstance(after, (int, float)) and after > 0,
                        f"baseline comparison {topology}.{name} is invalid")
                expected_delta = (after / before - 1.0) * 100.0
                require(metric["better_when"] == direction
                        and math.isclose(metric["delta_percent"], expected_delta,
                                         rel_tol=1e-12, abs_tol=1e-12),
                        f"baseline comparison {topology}.{name} is inconsistent")
    else:
        require(not all(matching.values()) and value["metrics"] is None,
                "incomparable baseline has admissible metrics")
    return value
