"""Same-device baseline comparison and explicit immutable promotion."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .baseline_validation import (
    MATCH_FIELDS,
    validate_baseline_comparison,
    validate_baseline_envelope,
)
from .common import (
    LAB_ROOT,
    REPO_ROOT,
    WORKSPACE_ROOT,
    canonical_bytes,
    lab_tool_source_paths,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
)
from .loop import normalized_gpu_uuid, validate_environment_pair, validate_loop_record
from .results import validate_result


BASELINE_ROOT = WORKSPACE_ROOT / "stwo-cairo" / "gpu_benchmarks" / "lab" / "baselines"
RESULT_ARGUMENTS = (
    "result", "fixture", "oracle_index", "module_index", "execution", "replay", "plan",
)


def add_result_arguments(parser: argparse.ArgumentParser) -> None:
    for name in RESULT_ARGUMENTS:
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def _safe_component(value: str, label: str) -> str:
    require(
        value not in {".", ".."}
        and len(value) <= 128
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is not None,
        f"unsafe baseline {label}",
    )
    return value


def _destination(root: Path, identity: dict[str, Any]) -> Path:
    fixture_id = _safe_component(identity["fixture"]["id"], "fixture_id")
    uuid = _safe_component(identity["device"]["uuid"], "device UUID")
    target = root / fixture_id / f"sm{identity['device']['target_sm']}" / f"{uuid}.json"
    require(target.resolve().is_relative_to(root.resolve()), "baseline destination escapes root")
    return target


def _normalized_flags(recipe: dict[str, Any]) -> list[str]:
    command = recipe.get("normalized_command")
    require(isinstance(command, list) and all(isinstance(item, str) for item in command),
            "build recipe normalized command is missing")
    source_positions = [index for index, item in enumerate(command) if item.endswith(".cu")]
    require(len(source_positions) == 1, "normalized command must contain one CUDA source")
    source = source_positions[0]
    require(source > 0 and command[0] == "nvcc", "normalized command must begin with nvcc")
    return command[1:source]


def _identity(fixture: dict[str, Any], module: dict[str, Any], execution: dict[str, Any],
              result: dict[str, Any]) -> dict[str, Any]:
    recipe = module["build_recipe"]
    toolchain = {
        key: recipe[key]
        for key in (
            "nvcc_path", "nvcc_version", "host_compiler_path", "host_compiler_version",
            "ptxas_path", "ptxas_version", "cuobjdump_path", "cuobjdump_version",
            "environment", "driver_compatibility_policy", "target_sm",
        )
    }
    toolchain["normalized_flags"] = _normalized_flags(recipe)
    toolchain["sha256"] = sha256_bytes(canonical_bytes(toolchain))
    shape_material = {
        "fixture_class": fixture["fixture_class"],
        "row_count": fixture["semantic_payload"]["row_count"],
        "semantic_operation": execution["kernel"]["semantic_operation"],
        "launch": execution["kernel"]["launch"],
        "physical_layout": execution["physical_layout"],
    }
    rows = shape_material["row_count"]
    words_per_row = (
        execution["physical_layout"]["output_columns"]
        + execution["physical_layout"]["lookup_words_per_row"]
        + execution["physical_layout"]["sub_words_per_row"]
    )
    return {
        "device": {
            "uuid": result["device"]["uuid"],
            "name": result["device"]["name"],
            "target_sm": result["device"]["target_sm"],
            "driver_version": result["device"]["driver_version"],
            "ecc_enabled": result["device"]["ecc_enabled"],
        },
        "fixture": {
            "id": fixture["fixture_id"],
            "class": fixture["fixture_class"],
            "sha256": result["semantic_fixture_sha256"],
            # Envelope v1 retains its historical field name; indexed slabs bind
            # the candidate-independent semantic descriptor SHA here.
            "proof_semantic_hash": (
                fixture["semantic_identity"]["sha256"]
                if "semantic_identity" in fixture else fixture["proof_semantic_hash"]
            ),
            "rows": rows,
            "words_per_row": words_per_row,
        },
        "oracle": execution["host_oracle"],
        "abi_sha256": execution["kernel"]["abi_sha256"],
        "shape": {"sha256": sha256_bytes(canonical_bytes(shape_material)), **shape_material},
        "toolchain": toolchain,
    }


def _derived(result: dict[str, Any], rows: int, words_per_row: int) -> dict[str, Any]:
    derived: dict[str, Any] = {}
    for topology in ("eager", "graph"):
        series = result["timing"][topology]
        derived[topology] = {}
        for percentile in ("p50", "p95"):
            milliseconds = series[f"{percentile}_gpu_ms"]
            derived[topology][f"semantic_rows_per_s_{percentile}"] = rows * 1000.0 / milliseconds
            derived[topology][f"semantic_words_per_s_{percentile}"] = (
                rows * words_per_row * 1000.0 / milliseconds
            )
    return derived


def _core_envelope(fixture: dict[str, Any], module: dict[str, Any], execution: dict[str, Any],
                   result: dict[str, Any], before: dict[str, Any],
                   after: dict[str, Any]) -> dict[str, Any]:
    validate_environment_pair(before, after)
    identity = _identity(fixture, module, execution, result)
    nvidia_uuid = normalized_gpu_uuid(before["fields"]["uuid"].get("value", "")) \
        if before["fields"]["uuid"]["status"] == "available" else None
    require(nvidia_uuid is None or nvidia_uuid == identity["device"]["uuid"],
            "nvidia-smi and CUDA Driver UUIDs differ")
    rows, words = identity["fixture"]["rows"], identity["fixture"]["words_per_row"]
    return {
        "identity": identity,
        "module": {
            "content_sha256": result["module_content_sha256"],
            "build_recipe_hash": result["build_recipe_hash"],
            "source_sha256": module["build_recipe"]["source_sha256"],
        },
        "measurement": {
            "result_sha256": None,
            "resources": result["resources"],
            "timing": result["timing"],
            "derived": _derived(result, rows, words),
        },
        "environment": {"before": before, "after": after},
    }


def _comparison_key(envelope: dict[str, Any]) -> dict[str, Any]:
    identity = envelope["identity"]
    fields = envelope["environment"]["before"]["fields"]
    return {
        "device_uuid": identity["device"]["uuid"],
        "target_sm": identity["device"]["target_sm"],
        "driver_version": identity["device"]["driver_version"],
        "configuration": {
            "cuda_ecc_enabled": identity["device"]["ecc_enabled"],
            "nvidia_smi": {
                name: fields[name]
                for name in ("mig_mode", "ecc_mode", "persistence_mode", "power_limit_w")
            },
        },
        "fixture": {
            key: identity["fixture"][key]
            for key in ("id", "class", "sha256", "proof_semantic_hash", "rows", "words_per_row")
        },
        "toolchain": identity["toolchain"]["sha256"],
        "abi": identity["abi_sha256"],
        "shape": identity["shape"]["sha256"],
    }


def _delta(before: float, after: float, better_when: str) -> dict[str, Any]:
    require(before > 0 and after > 0, "comparison metric must be positive")
    return {
        "before": before,
        "after": after,
        "delta_percent": (after / before - 1.0) * 100.0,
        "better_when": better_when,
    }


def _metric_comparison(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    compared: dict[str, Any] = {}
    for topology in ("eager", "graph"):
        compared[topology] = {
            metric: _delta(
                before["measurement"]["timing"][topology][metric],
                after["measurement"]["timing"][topology][metric],
                "lower",
            )
            for metric in ("p50_gpu_ms", "p95_gpu_ms")
        }
        for metric in ("semantic_rows_per_s_p50", "semantic_words_per_s_p50"):
            compared[topology][metric] = _delta(
                before["measurement"]["derived"][topology][metric],
                after["measurement"]["derived"][topology][metric],
                "higher",
            )
    return compared


def compare_baseline(args: argparse.Namespace) -> dict[str, Any]:
    result = validate_result(args)
    fixture, module, execution = (
        load_json(args.fixture), load_json(args.module_index), load_json(args.execution)
    )
    before, after = load_json(args.environment_before), load_json(args.environment_after)
    validate_environment_pair(before, after)
    identity = _identity(fixture, module, execution, result)
    nvidia_uuid = normalized_gpu_uuid(before["fields"]["uuid"].get("value", "")) \
        if before["fields"]["uuid"]["status"] == "available" else None
    require(nvidia_uuid is None or nvidia_uuid == identity["device"]["uuid"],
            "nvidia-smi and CUDA Driver UUIDs differ")
    target = _destination(args.baseline_root, identity)
    performance_admissible = result["timing"].get("performance_admissible")
    require(isinstance(performance_admissible, bool),
            "benchmark result lacks an explicit performance-admissible state")
    comparison: dict[str, Any] = {
        "schema_version": "stwo.gpu-lab.baseline-comparison.v1",
        "status": "missing" if performance_admissible else "non_admissible",
        "baseline_path": str(target),
        "baseline_sha256": None,
        "matching_identity": None,
        "before_module_sha256": None,
        "after_module_sha256": result["module_content_sha256"],
        "metrics": None,
    }
    current = None
    if performance_admissible:
        current = _core_envelope(fixture, module, execution, result, before, after)
        current["measurement"]["result_sha256"] = sha256_file(args.result)
    if target.exists():
        baseline = validate_baseline_envelope(load_json(target))
        matching = {
            key: _comparison_key(baseline)[key]
            == _comparison_key({"identity": identity, "environment": {"before": before}})[key]
            for key in MATCH_FIELDS
        }
        comparison.update({
            "status": (
                "non_admissible" if not performance_admissible
                else "compared" if all(matching.values()) else "incomparable"
            ),
            "baseline_sha256": sha256_file(target),
            "matching_identity": matching,
            "before_module_sha256": baseline["module"]["content_sha256"],
            "metrics": _metric_comparison(baseline, current)
            if performance_admissible and all(matching.values()) else None,
        })
    validate_baseline_comparison(comparison)
    _write_json(args.output, comparison)
    print(f"baseline comparison: {comparison['status']} ({target})")
    return comparison


def _dependency_paths(module_path: Path, oracle_path: Path) -> list[tuple[str, Path]]:
    module, oracle = load_json(module_path), load_json(oracle_path)
    abi_path = Path(module["abi_path"]).resolve()
    abi = load_json(abi_path)
    dependencies: list[tuple[str, Path]] = [
        ("module", Path(module["module_path"]).resolve()),
        ("abi", abi_path),
        ("cuda_source", (REPO_ROOT / abi["source"]).resolve()),
        ("host_oracle", (WORKSPACE_ROOT / oracle["oracle_artifact"]["path"]).resolve()),
    ]
    recipe = module["build_recipe"]
    staging = recipe.get("source_staging")
    if isinstance(staging, dict):
        dependencies.append((
            "staged_cuda_source",
            Path(staging["root"]) / staging["directory"] / staging["source_name"],
        ))
    for identity in recipe["generator_sources"] + recipe["transitive_headers"]:
        dependencies.append((
            "generator_or_header", WORKSPACE_ROOT / identity["repository"] / identity["path"]
        ))
    for source in oracle["exporter"]["sources"]:
        dependencies.append(("oracle_exporter", WORKSPACE_ROOT / source["path"]))
    dependencies.extend(("lab_tool", path) for path in lab_tool_source_paths())
    dependencies.extend((role, LAB_ROOT / path) for role, path in (
        ("orchestrator", "tools/quick-loop"),
        ("slo_checker", "tools/check-loop-slo"),
    ))
    return dependencies


def _capture_dependencies(module_path: Path, oracle_path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for role, path in _dependency_paths(module_path, oracle_path):
        resolved = path.resolve()
        require(resolved.is_file(), f"transitive baseline dependency is missing: {resolved}")
        key = str(resolved)
        grouped.setdefault(key, {
            "path": key,
            "roles": [],
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
        })["roles"].append(role)
    for item in grouped.values():
        item["roles"].sort()
    return [grouped[key] for key in sorted(grouped)]


def _snapshot_sources(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    names = [
        *RESULT_ARGUMENTS, "harness", "correctness_result", "build_commands",
        "environment_before", "environment_after", "loop_record", "comparison",
    ]
    snapshots: dict[str, Path] = {}
    for name in names:
        source = getattr(args, name)
        target = root / name
        target.write_bytes(source.read_bytes())
        snapshots[name] = target
    return snapshots


def _snapshot_namespace(paths: dict[str, Path]) -> argparse.Namespace:
    return argparse.Namespace(
        **{name: paths[name] for name in RESULT_ARGUMENTS},
        harness=paths["harness"],
        mode="benchmark",
    )


def _verify_loop(loop: dict[str, Any], snapshots: dict[str, Path]) -> None:
    bindings = {
        "fixture": "fixture", "oracle_index": "oracle_index", "module_index": "module_index",
        "execution": "execution", "replay": "replay", "plan": "plan",
        "harness": "harness", "correctness_result": "correctness_result", "benchmark_result": "result",
        "baseline_comparison": "comparison", "build_commands": "build_commands",
    }
    for artifact, snapshot in bindings.items():
        require(loop["artifacts"][artifact]["sha256"] == sha256_file(snapshots[snapshot]),
                f"loop record does not bind {artifact}")


def _acceptance_ready(result: dict[str, Any], before: dict[str, Any],
                      after: dict[str, Any]) -> None:
    validate_environment_pair(before, after, True)
    require(result["timing"].get("performance_admissible") is True,
            "performance-non-admissible result cannot become a baseline")
    require(normalized_gpu_uuid(before["fields"]["uuid"]["value"]) == result["device"]["uuid"],
            "required GPU environment belongs to another device")
    for topology in ("eager", "graph"):
        series = result["timing"][topology]
        require(series["warmup_stable"] is True, f"{topology} warmup did not stabilize")
        require(series["iterations"] >= 30, f"{topology} needs at least 30 samples")
        require(series["p95_exploratory"] is False, f"{topology} p95 remains exploratory")


def _atomic_write(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _token(payload: bytes, target: Path) -> str:
    identity = {
        "destination": str(target.relative_to(WORKSPACE_ROOT)),
        "prior_sha256": sha256_file(target) if target.exists() else None,
        "envelope_sha256": sha256_bytes(payload),
    }
    return "ACCEPT-" + sha256_bytes(canonical_bytes(identity))[:16]


def accept_baseline(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="gpu-lab-baseline-") as temporary:
        snapshots = _snapshot_sources(args, Path(temporary))
        dependencies_before = _capture_dependencies(snapshots["module_index"],
                                                    snapshots["oracle_index"])
        result = validate_result(_snapshot_namespace(snapshots))
        fixture, module, execution = (
            load_json(snapshots["fixture"]), load_json(snapshots["module_index"]),
            load_json(snapshots["execution"]),
        )
        before, after = load_json(snapshots["environment_before"]), \
            load_json(snapshots["environment_after"])
        _acceptance_ready(result, before, after)
        loop = validate_loop_record(load_json(snapshots["loop_record"]))
        require(loop["environment"] == {"before": before, "after": after},
                "loop environment differs from promotion snapshots")
        _verify_loop(loop, snapshots)
        comparison = validate_baseline_comparison(load_json(snapshots["comparison"]))
        require(comparison["status"] != "non_admissible",
                "non-admissible comparison cannot become a baseline")
        require(comparison.get("after_module_sha256") == result["module_content_sha256"],
                "baseline comparison belongs to another candidate module")
        dependencies_after = _capture_dependencies(snapshots["module_index"],
                                                   snapshots["oracle_index"])
        require(dependencies_before == dependencies_after,
                "transitive identity changed during baseline validation")
        envelope = _core_envelope(fixture, module, execution, result, before, after)
        envelope["schema_version"] = "stwo.gpu-lab.baseline.v1"
        envelope["measurement"]["result_sha256"] = sha256_file(snapshots["result"])
        envelope["audit"] = {
            "loop_record_sha256": sha256_file(snapshots["loop_record"]),
            "correctness_result_sha256": sha256_file(snapshots["correctness_result"]),
            "comparison_sha256": sha256_file(snapshots["comparison"]),
            "build": loop["build_audit"],
            "runtime": loop["runtime_audit"],
        }
        envelope["snapshot"] = {
            "top_level": {
                name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for name, path in sorted(snapshots.items())
            },
            "transitive": dependencies_before,
        }
        envelope["snapshot"]["sha256"] = sha256_bytes(canonical_bytes(envelope["snapshot"]))
        validate_baseline_envelope(envelope)
        payload = json.dumps(envelope, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
        target = _destination(args.baseline_root, envelope["identity"])
        args.baseline_root.mkdir(parents=True, exist_ok=True)
        with (args.baseline_root / ".accept.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            confirmation = _token(payload, target)
            if args.confirm != confirmation:
                print(f"NO BASELINE CHANGED. Re-run with: --confirm {confirmation}")
                print(f"destination: {target}")
                return 2
            _atomic_write(target, payload)
            print(f"ACCEPTED {target} sha256={sha256_file(target)}")
    return 0


def accept_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_result_arguments(parser)
    parser.add_argument("--environment-before", required=True, type=Path)
    parser.add_argument("--environment-after", required=True, type=Path)
    parser.add_argument("--correctness-result", required=True, type=Path)
    parser.add_argument("--build-commands", required=True, type=Path)
    parser.add_argument("--harness", required=True, type=Path)
    parser.add_argument("--loop-record", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--confirm")
    return parser


def accept_main() -> int:
    return accept_baseline(accept_parser().parse_args())
