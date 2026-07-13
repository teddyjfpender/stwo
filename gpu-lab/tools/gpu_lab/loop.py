"""GPU environment capture and auditable fast-loop records."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from .common import (
    LAB_ROOT,
    canonical_bytes,
    load_json,
    require,
    require_exact_keys,
    sha256_bytes,
    sha256_file,
)
from .staging import seal_staging, stage_fixture, validate_staging, verify_staging


ENVIRONMENT_QUERIES = {
    "uuid": "uuid",
    "name": "name",
    "pci_bus_id": "pci.bus_id",
    "persistence_mode": "persistence_mode",
    "performance_state": "pstate",
    "power_draw_w": "power.draw",
    "power_limit_w": "power.limit",
    "temperature_c": "temperature.gpu",
    "graphics_clock_mhz": "clocks.current.graphics",
    "sm_clock_mhz": "clocks.current.sm",
    "memory_clock_mhz": "clocks.current.memory",
    "mig_mode": "mig.mode.current",
    "ecc_mode": "ecc.mode.current",
}
UNSUPPORTED_CAPABILITIES = frozenset({"mig_mode", "ecc_mode"})
STABLE_ENVIRONMENT_FIELDS = frozenset({"uuid", "name", "pci_bus_id", "mig_mode", "ecc_mode"})
STABLE_ENVIRONMENT_FIELDS |= frozenset({"persistence_mode", "power_limit_w"})
PHASES = ("configure_build", "prepare", "correctness", "benchmark")
ARTIFACTS = (
    "fixture", "oracle_index", "module_index", "execution", "replay", "plan",
    "harness", "correctness_result", "benchmark_result", "baseline_comparison", "build_commands",
)
SLO_BUDGETS = {
    "noop": {"build_s": 1.0},
    "ordinary": {"build_s": 45.0, "run_s": 45.0, "end_to_end_s": 120.0},
    "large": {"build_s": 180.0, "end_to_end_s": 300.0},
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")


def capture_environment(device: str, output: Path) -> dict[str, Any]:
    names, queries = tuple(ENVIRONMENT_QUERIES), tuple(ENVIRONMENT_QUERIES.values())
    command = [
        "nvidia-smi", f"--query-gpu={','.join(queries)}", "--format=csv,noheader,nounits",
        "-i", device,
    ]
    fields: dict[str, dict[str, str]]
    try:
        run = subprocess.run(command, text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        reason = str(error)[:240]
        fields = {name: {"status": "unavailable", "reason": reason} for name in names}
    else:
        rows = list(csv.reader(io.StringIO(run.stdout), skipinitialspace=True))
        if run.returncode or len(rows) != 1 or len(rows[0]) != len(names):
            reason = (run.stderr or run.stdout or "batched nvidia-smi query malformed").strip()[:240]
            fields = {name: {"status": "unavailable", "reason": reason} for name in names}
        else:
            fields = {}
            for name, query, raw in zip(names, queries, rows[0]):
                value = raw.strip()
                if value.upper() in {"N/A", "[N/A]"}:
                    fields[name] = {
                        "status": "not_supported",
                        "reason": f"nvidia-smi reports {query}=N/A",
                    }
                elif not value:
                    fields[name] = {"status": "unavailable", "reason": f"empty {query}"}
                else:
                    fields[name] = {"status": "available", "value": value}
    record = {
        "schema_version": "stwo.gpu-lab.environment.v1",
        "device_selector": device,
        "fields": fields,
    }
    _write_json(output, record)
    return record


def validate_environment(value: Any, required: bool = False) -> dict[str, Any]:
    require(isinstance(value, dict), "GPU environment must be an object")
    require_exact_keys(value, {"schema_version", "device_selector", "fields"}, "GPU environment")
    require(value["schema_version"] == "stwo.gpu-lab.environment.v1",
            "unsupported GPU environment schema")
    require(isinstance(value["device_selector"], str) and value["device_selector"],
            "GPU device selector is missing")
    fields = value["fields"]
    require(isinstance(fields, dict), "GPU environment fields are missing")
    require_exact_keys(fields, set(ENVIRONMENT_QUERIES), "GPU environment fields")
    for name, field in fields.items():
        require(isinstance(field, dict), f"GPU environment {name} is not an object")
        status = field.get("status")
        require(status in {"available", "not_supported", "unavailable"},
                f"GPU environment {name} has an invalid status")
        expected = {"status", "value"} if status == "available" else {"status", "reason"}
        require_exact_keys(field, expected, f"GPU environment {name}")
        payload = field.get("value") if status == "available" else field.get("reason")
        require(isinstance(payload, str) and payload, f"GPU environment {name} is empty")
        if required:
            require(status == "available" or (status == "not_supported"
                                               and name in UNSUPPORTED_CAPABILITIES),
                    f"GPU environment {name} is unavailable: {payload}")
    return value


def validate_environment_pair(before: Any, after: Any, required: bool = False) -> None:
    before = validate_environment(before, required)
    after = validate_environment(after, required)
    require(before["device_selector"] == after["device_selector"],
            "GPU environment device selector changed")
    for name in STABLE_ENVIRONMENT_FIELDS:
        left, right = before["fields"][name], after["fields"][name]
        if left["status"] == right["status"] == "unavailable":
            continue
        require(left == right, f"stable GPU environment field changed: {name}")


def normalized_gpu_uuid(value: str) -> str:
    normalized = value.strip().lower().removeprefix("gpu-").replace("-", "")
    require(re.fullmatch(r"[0-9a-f]{32}", normalized) is not None,
            "nvidia-smi UUID is not a canonical GPU UUID")
    return normalized


def _parse_assignments(values: list[str], expected: set[str], label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        name, separator, payload = value.partition("=")
        require(separator == "=" and name in expected and name not in parsed and payload,
                f"invalid or duplicate {label}: {value}")
        parsed[name] = payload
    require(set(parsed) == expected,
            f"{label} keys differ: missing={sorted(expected - set(parsed))}")
    return parsed


def _phase_seconds(value: str, name: str) -> float:
    start, separator, end = value.partition(":")
    require(separator == ":" and start.isdigit() and end.isdigit(), f"bad {name} timestamps")
    require(int(end) >= int(start), f"negative {name} duration")
    return round((int(end) - int(start)) / 1e9, 6)


def _audit_build_commands(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    lines = [line for line in content.decode("utf-8").splitlines() if line.strip()]
    forbidden = _forbidden_commands(lines)
    require(not forbidden, f"fast-loop build graph contains forbidden commands: {forbidden}")
    return {
        "commands_sha256": sha256_bytes(content),
        "command_count": len(lines),
        "commands": lines,
        "forbidden_hits": forbidden,
        "cargo_builds": 0,
        "scope": "configured Ninja target command graph",
    }


def _forbidden_commands(lines: list[str]) -> list[dict[str, Any]]:
    forbidden = []
    patterns = {
        "cargo": re.compile(r"(?:^|[\s;&|])(?:[^\s;&|]*/)?cargo(?:[\s;&|]|$)"),
        "full_prover": re.compile(
            r"(?:^|[\s;&|])(?:[^\s;&|]*/)?(?:gpu_bench|stwo-cairo-prover)(?:[\s;&|]|$)"
        ),
    }
    for number, line in enumerate(lines, 1):
        try:
            normalized = " ".join(shlex.split(line))
        except ValueError:
            forbidden.append({"line": number, "executable": "unparseable_shell"})
            continue
        for executable, pattern in patterns.items():
            if pattern.search(normalized):
                forbidden.append({"line": number, "executable": executable})
    return forbidden


def validate_build_audit(build: Any) -> dict[str, Any]:
    require(isinstance(build, dict), "loop build audit must be an object")
    require_exact_keys(
        build,
        {"commands_sha256", "command_count", "commands", "forbidden_hits", "cargo_builds", "scope"},
        "loop build audit",
    )
    require(re.fullmatch(r"[0-9a-f]{64}", build["commands_sha256"]) is not None,
            "loop build command hash is invalid")
    require(isinstance(build["commands"], list)
            and build["command_count"] == len(build["commands"]) > 0
            and all(isinstance(command, str) and command for command in build["commands"]),
            "loop build command count mismatch")
    require(build["forbidden_hits"] == _forbidden_commands(build["commands"]) == [],
            "loop build audit is not clean")
    require(build["cargo_builds"] == 0, "loop build audit recorded Cargo")
    require(isinstance(build["scope"], str) and build["scope"], "loop build audit scope is empty")
    return build


def validate_runtime_audit(runtime: Any) -> dict[str, Any]:
    require(isinstance(runtime, dict), "loop runtime audit must be an object")
    require_exact_keys(runtime, {"harness_sha256", "orchestrator_sha256", "slo_checker_sha256",
                                 "invocations", "benchmark_exit_code", "performance_admissible",
                                 "full_proofs", "scope"},
                       "loop runtime audit")
    for name in ("harness_sha256", "orchestrator_sha256", "slo_checker_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", runtime[name]) is not None,
                f"loop runtime {name} is invalid")
    require(runtime["full_proofs"] == 0
            and runtime["invocations"] == ["correctness", "benchmark"],
            "loop runtime audit is not clean")
    require((runtime["performance_admissible"] is True and runtime["benchmark_exit_code"] == 0)
            or (runtime["performance_admissible"] is False
                and runtime["benchmark_exit_code"] == 2),
            "loop performance status is inconsistent")
    require(isinstance(runtime["scope"], str) and runtime["scope"],
            "loop runtime audit scope is empty")
    return runtime


def write_loop_record(args: argparse.Namespace) -> dict[str, Any]:
    phases = _parse_assignments(args.phase, set(PHASES), "phase")
    artifact_paths = {
        name: Path(path) for name, path in _parse_assignments(
            args.artifact, set(ARTIFACTS), "artifact"
        ).items()
    }
    artifacts = {
        name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in sorted(artifact_paths.items())
    }
    before = validate_environment(load_json(args.environment_before))
    after = validate_environment(load_json(args.environment_after))
    validate_environment_pair(before, after)
    build_audit = _audit_build_commands(artifact_paths["build_commands"])
    require(build_audit["commands_sha256"] == artifacts["build_commands"]["sha256"],
            "build command audit hash mismatch")
    require(args.runtime_mode == ["correctness", "benchmark"],
            "fast loop runtime modes must be correctness then benchmark")
    performance_admissible = args.performance_status == "admissible"
    require((performance_admissible and args.benchmark_exit_code == 0)
            or (not performance_admissible and args.benchmark_exit_code == 2),
            "benchmark exit code disagrees with performance status")
    orchestrator = (LAB_ROOT / "tools/quick-loop").resolve()
    slo_checker = (LAB_ROOT / "tools/check-loop-slo").resolve()
    require(args.orchestrator.resolve() == orchestrator and args.slo_checker.resolve() == slo_checker,
            "loop audit must bind the canonical orchestrator and SLO checker")
    staging = validate_staging(load_json(args.staging_record), sealed=True)
    verify_staging(staging)
    staged = {entry["role"]: entry for entry in staging["entries"]}
    for role in ("fixture", "module_index", "execution", "replay", "plan", "harness",
                 "build_commands"):
        require(staged[role]["destination_sha256"] == artifacts[role]["sha256"],
                f"loop artifact differs from staged {role}")
    record = {
        "schema_version": "stwo.gpu-lab.loop-result.v3",
        "phases_s": {name: _phase_seconds(phases[name], name) for name in PHASES},
        "end_to_end_s": _phase_seconds(args.total, "total"),
        "artifacts": artifacts,
        "build_audit": build_audit,
        "runtime_audit": {
            "harness_sha256": sha256_file(args.harness),
            "orchestrator_sha256": sha256_file(orchestrator),
            "slo_checker_sha256": sha256_file(slo_checker),
            "invocations": args.runtime_mode,
            "benchmark_exit_code": args.benchmark_exit_code,
            "performance_admissible": performance_admissible,
            "full_proofs": 0,
            "scope": "quick-loop orchestrated runtime invocations",
        },
        "environment": {"before": before, "after": after},
        "staging": staging,
    }
    record["record_sha256"] = sha256_bytes(canonical_bytes(record))
    require(record["runtime_audit"]["harness_sha256"] == artifacts["harness"]["sha256"],
            "runtime harness differs from the artifact binding")
    _write_json(args.output, record)
    return record


def validate_loop_record(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "loop record must be an object")
    require_exact_keys(
        value,
        {"schema_version", "phases_s", "end_to_end_s", "artifacts", "build_audit",
         "runtime_audit", "environment", "staging", "record_sha256"},
        "loop record",
    )
    require(value["schema_version"] == "stwo.gpu-lab.loop-result.v3",
            "unsupported loop-result schema")
    expected_hash = sha256_bytes(canonical_bytes({
        key: payload for key, payload in value.items() if key != "record_sha256"
    }))
    require(value["record_sha256"] == expected_hash, "loop record self-hash mismatch")
    require_exact_keys(value["phases_s"], set(PHASES), "loop phases")
    require_exact_keys(value["artifacts"], set(ARTIFACTS), "loop artifacts")
    for name, artifact in value["artifacts"].items():
        require(isinstance(artifact, dict), f"loop artifact {name} must be an object")
        require_exact_keys(artifact, {"sha256", "bytes"}, f"loop artifact {name}")
        require(re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is not None,
                f"loop artifact {name} hash is invalid")
        require(isinstance(artifact["bytes"], int) and not isinstance(artifact["bytes"], bool)
                and artifact["bytes"] >= 0, f"loop artifact {name} size is invalid")
    for name, seconds in {**value["phases_s"], "end_to_end": value["end_to_end_s"]}.items():
        require(isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                and 0 <= seconds < float("inf"), f"invalid loop duration: {name}")
    validate_build_audit(value["build_audit"])
    validate_runtime_audit(value["runtime_audit"])
    require(value["build_audit"]["commands_sha256"]
            == value["artifacts"]["build_commands"]["sha256"],
            "loop build command artifact mismatch")
    require(value["runtime_audit"]["harness_sha256"] == value["artifacts"]["harness"]["sha256"],
            "loop harness artifact mismatch")
    require(value["runtime_audit"]["orchestrator_sha256"]
            == sha256_file(LAB_ROOT / "tools/quick-loop"), "loop orchestrator source changed")
    require(value["runtime_audit"]["slo_checker_sha256"]
            == sha256_file(LAB_ROOT / "tools/check-loop-slo"), "loop SLO checker source changed")
    environment = value["environment"]
    require(isinstance(environment, dict), "loop environment must be an object")
    require_exact_keys(environment, {"before", "after"}, "loop environment")
    validate_environment_pair(environment["before"], environment["after"])
    staging = validate_staging(value["staging"], sealed=True)
    staged = {entry["role"]: entry for entry in staging["entries"]}
    for role in ("fixture", "module_index", "execution", "replay", "plan", "harness",
                 "build_commands"):
        require(staged[role]["destination_sha256"] == value["artifacts"][role]["sha256"],
                f"loop artifact differs from staged {role}")
    require(sum(value["phases_s"].values()) <= value["end_to_end_s"] + 1e-6,
            "loop phases exceed end-to-end time")
    return value


def check_loop_slo(value: Any, kind: str) -> None:
    require(kind in SLO_BUDGETS, f"unknown feedback SLO: {kind}")
    record = validate_loop_record(value)
    measured = {
        "build_s": record["phases_s"]["configure_build"],
        "run_s": sum(record["phases_s"][name]
                     for name in ("prepare", "correctness", "benchmark")),
        "end_to_end_s": record["end_to_end_s"],
    }
    for metric, ceiling in SLO_BUDGETS[kind].items():
        actual = measured[metric]
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and 0 <= actual <= ceiling,
                f"{kind} {metric} {actual!r}s exceeds {ceiling:.6f}s")
    require(record["runtime_audit"]["performance_admissible"] is True,
            "benchmark diagnostics retained, but performance is non-admissible")


def loop_self_test() -> None:
    available = {
        "schema_version": "stwo.gpu-lab.environment.v1",
        "device_selector": "0",
        "fields": {
            name: {"status": "available", "value": "GPU-00112233-4455-6677-8899-aabbccddeeff"}
            if name == "uuid" else {"status": "available", "value": name}
            for name in ENVIRONMENT_QUERIES
        },
    }
    with tempfile.TemporaryDirectory() as temporary_name:
        root = Path(temporary_name)
        query_values = [
            "GPU-00112233-4455-6677-8899-aabbccddeeff", "test", "0000:01:00.0", "Enabled",
            "P0", "100", "250", "50", "1500", "1500", "5000", "N/A", "N/A",
        ]
        captured = root / "captured.json"
        with patch("gpu_lab.loop.subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout=", ".join(query_values) + "\n", stderr="",
        )) as run:
            capture_environment("0", captured)
            require(run.call_count == 1, "GPU environment capture used more than one subprocess")
        captured_value = load_json(captured)
        validate_environment(captured_value, True)
        require(captured_value["fields"]["mig_mode"]["status"] == "not_supported"
                and captured_value["fields"]["ecc_mode"]["status"] == "not_supported",
                "unsupported GPU capabilities were not explicit")
        environment = root / "environment.json"
        _write_json(environment, available)
        commands = root / "commands"
        commands.write_text("nvcc -cubin source.cu -o module.cubin\n")
        artifact_paths = {name: root / name for name in ARTIFACTS}
        for name, path in artifact_paths.items():
            if name == "build_commands":
                path.write_bytes(commands.read_bytes())
            elif name == "fixture":
                path.write_text("{}\n")
            else:
                path.write_text(name)
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
        args = argparse.Namespace(
            phase=[f"{name}=1:2" for name in PHASES],
            artifact=[f"{name}={path}" for name, path in artifact_paths.items()],
            total="1:5",
            environment_before=environment,
            environment_after=environment,
            harness=artifact_paths["harness"],
            runtime_mode=["correctness", "benchmark"],
            benchmark_exit_code=0,
            performance_status="admissible",
            orchestrator=LAB_ROOT / "tools/quick-loop",
            slo_checker=LAB_ROOT / "tools/check-loop-slo",
            staging_record=staging_path,
            output=root / "loop.json",
        )
        record = write_loop_record(args)
        validate_loop_record(record)
        check_loop_slo(record, "ordinary")
        diagnostic = json.loads(json.dumps(record))
        diagnostic["runtime_audit"].update({
            "benchmark_exit_code": 2, "performance_admissible": False,
        })
        diagnostic["record_sha256"] = sha256_bytes(canonical_bytes({
            name: payload for name, payload in diagnostic.items() if name != "record_sha256"
        }))
        validate_loop_record(diagnostic)
        try:
            check_loop_slo(diagnostic, "ordinary")
        except ValueError:
            pass
        else:
            raise ValueError("non-admissible benchmark passed the feedback SLO")
        mutated = json.loads(json.dumps(record))
        mutated["build_audit"]["cargo_builds"] = 1
        try:
            validate_loop_record(mutated)
        except ValueError:
            pass
        else:
            raise ValueError("mutated loop audit was accepted")
        for hostile in ("cargo build\n", '"/usr/bin/cargo" build\n',
                        "bash -c 'cargo build'\n", "bash -c 'unterminated\n"):
            commands.write_text(hostile)
            try:
                _audit_build_commands(commands)
            except ValueError:
                pass
            else:
                raise ValueError(f"hostile command escaped the fast-loop audit: {hostile!r}")
