"""Linux-only execution boundary for the sealed FRI round-6 runner."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import canonical_bytes, require, require_exact_keys, require_int
from .common import require_sha256, sha256_bytes
from .fri_round6_fixture import validate_fixture_identity
from .fri_round6_loop import (
    EVIDENCE_MODE,
    SEMANTIC_ROLES,
    BoundFile,
    _artifact_records,
    _fixture_bundle,
    _identity,
    _loop_tool_bounds,
    _module_bundle,
    _reject_symlink_components,
    _require_semantic_distinct,
    bind_file,
    observe_file,
    recheck,
    validate_runner_result,
)
from .fri_round6_loop_identity import loop_tool_sha256, loop_tool_sources, validate_loop_tool
from .fri_round6_process import (
    MAX_RESULT_BYTES,
    RUNNER_TIMEOUT_SECONDS,
    reviewed_environment,
    run_linux,
    validate_environment_contract,
)
from .immutable_output import guard_output, write_immutable_bytes


EXECUTION_RECORD_SCHEMA = "stwo.gpu-lab.fri-round6-execution-record.v1"
RAW_RESULT_ROLE = "FRI raw runner result"
EMPTY_SHA256 = sha256_bytes(b"")


@dataclass(frozen=True)
class OpenBoundFile:
    bound: BoundFile
    descriptor: int

    @property
    def proc_path(self) -> str:
        return f"/proc/self/fd/{self.descriptor}"

    def record(self) -> dict[str, Any]:
        return {
            "fd": self.descriptor,
            "proc_path": self.proc_path,
            "artifact_path": str(self.bound.path),
            "sha256": self.bound.sha256,
            "bytes": self.bound.byte_length,
        }


def _descriptor_bytes(descriptor: int, length: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < length:
        try:
            chunk = os.pread(descriptor, min(1024 * 1024, length - offset), offset)
        except InterruptedError:
            continue
        require(chunk != b"", "sealed FRI descriptor ended early")
        chunks.append(chunk)
        offset += len(chunk)
    require(os.pread(descriptor, 1, length) == b"", "sealed FRI descriptor grew")
    return b"".join(chunks)


def _descriptor_sha256(descriptor: int, length: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < length:
        try:
            chunk = os.pread(descriptor, min(1024 * 1024, length - offset), offset)
        except InterruptedError:
            continue
        require(chunk != b"", "sealed FRI descriptor ended early")
        digest.update(chunk)
        offset += len(chunk)
    require(os.pread(descriptor, 1, length) == b"", "sealed FRI descriptor grew")
    return digest.hexdigest()


def _verify_opened(opened: OpenBoundFile) -> None:
    before = os.fstat(opened.descriptor)
    require(stat.S_ISREG(before.st_mode) and _identity(before) == opened.bound.identity,
            f"{opened.bound.role} descriptor identity differs from BoundFile")
    require(_descriptor_sha256(opened.descriptor, opened.bound.byte_length)
            == opened.bound.sha256,
            f"{opened.bound.role} descriptor sha256 differs from BoundFile")
    require(_identity(os.fstat(opened.descriptor)) == opened.bound.identity,
            f"{opened.bound.role} descriptor changed while it was hashed")


def _open_bound(bound: BoundFile) -> OpenBoundFile:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(bound.path, flags)
    if descriptor < 3:
        duplicated = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 3)
        os.close(descriptor)
        descriptor = duplicated
    opened = OpenBoundFile(bound, descriptor)
    try:
        _verify_opened(opened)
        recheck(bound)
    except BaseException:
        os.close(descriptor)
        raise
    return opened


def _open_roles(roles: dict[str, BoundFile]) -> dict[str, OpenBoundFile]:
    opened: dict[str, OpenBoundFile] = {}
    try:
        for role in sorted(roles):
            opened[role] = _open_bound(roles[role])
    except BaseException:
        for item in opened.values():
            os.close(item.descriptor)
        raise
    return opened


def _fresh_output(path: Path, role: str, sources: list[Path]) -> Path:
    require(isinstance(path, Path), f"{role} path is missing")
    absolute = _reject_symlink_components(path, role)
    require(absolute.name not in {"", ".", ".."}, f"{role} filename is invalid")
    parent = absolute.parent.resolve(strict=True)
    require(parent.is_dir(), f"{role} output directory is missing")
    normalized = parent / absolute.name
    require(guard_output(normalized, sources) is None, f"{role} path is not fresh")
    return normalized


def _roles(bounds: list[BoundFile]) -> dict[str, BoundFile]:
    selected = {bound.role: bound for bound in bounds if bound.role in SEMANTIC_ROLES}
    require(set(selected) == SEMANTIC_ROLES, "FRI launch artifact roles are incomplete")
    return selected


def _expected(roles: dict[str, BoundFile], target_sm: int, device: int) -> dict[str, Any]:
    return {
        "module_sha256": roles["FRI cubin"].sha256,
        "module_index_sha256": roles["FRI module index"].sha256,
        "build_recipe_sha256": roles["FRI build recipe"].sha256,
        "primary_sha256": roles["primary FRI payload"].sha256,
        "primary_index_sha256": roles["primary FRI index"].sha256,
        "hostile_sha256": roles["hostile FRI payload"].sha256,
        "hostile_index_sha256": roles["hostile FRI index"].sha256,
        "harness_sha256": roles["FRI runner"].sha256,
        "target_sm": target_sm,
        "device": device,
    }


def _command(
    roles: dict[str, BoundFile], launch_paths: dict[str, str], raw_result: Path,
    target_sm: int, device: int,
) -> list[str]:
    def artifact(role: str) -> str:
        return launch_paths[role]

    expected = _expected(roles, target_sm, device)
    return [
        artifact("FRI runner"),
        "--module", artifact("FRI cubin"),
        "--module-sha256", expected["module_sha256"],
        "--module-index", artifact("FRI module index"),
        "--module-index-sha256", expected["module_index_sha256"],
        "--build-recipe", artifact("FRI build recipe"),
        "--build-recipe-sha256", expected["build_recipe_sha256"],
        "--primary", artifact("primary FRI payload"),
        "--primary-sha256", expected["primary_sha256"],
        "--primary-index", artifact("primary FRI index"),
        "--primary-index-sha256", expected["primary_index_sha256"],
        "--hostile", artifact("hostile FRI payload"),
        "--hostile-sha256", expected["hostile_sha256"],
        "--hostile-index", artifact("hostile FRI index"),
        "--hostile-index-sha256", expected["hostile_index_sha256"],
        "--harness-sha256", expected["harness_sha256"],
        "--result", str(raw_result),
        "--device", str(device),
        "--target-sm", str(target_sm),
    ]


def _parse_result(payload: bytes) -> dict[str, Any]:
    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            require(key not in value, f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=object_without_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"FRI raw result is not strict UTF-8 JSON: {error}") from error
    require(isinstance(value, dict), "FRI raw result top level must be an object")
    return value


def _recheck_all(bounds: list[BoundFile], sources: list[dict[str, str]], digest: str) -> None:
    for bound in bounds:
        recheck(bound)
    require(loop_tool_sources() == sources and loop_tool_sha256() == digest,
            "FRI loop-tool closure changed across runner execution")


def _validate_artifacts(value: Any) -> dict[str, dict[str, Any]]:
    require(isinstance(value, dict), "FRI execution artifacts must be an object")
    require(set(value) == SEMANTIC_ROLES, "FRI execution artifact roles differ")
    for role, artifact in value.items():
        require(isinstance(artifact, dict), f"FRI execution {role} must be an object")
        require_exact_keys(artifact, {"path", "sha256", "bytes"}, f"FRI execution {role}")
        require(isinstance(artifact["path"], str) and Path(artifact["path"]).is_absolute(),
                f"FRI execution {role} path is not absolute")
        require_sha256(artifact["sha256"], f"FRI execution {role} sha256")
        require_int(artifact["bytes"], f"FRI execution {role} bytes", 1)
    paths = [artifact["path"] for artifact in value.values()]
    require(len(paths) == len(set(paths)), "FRI execution artifacts contain a path alias")
    return value


def _records_as_roles(artifacts: dict[str, dict[str, Any]]) -> dict[str, BoundFile]:
    return {
        role: BoundFile(role, Path(artifact["path"]), artifact["sha256"], artifact["bytes"],
                        (0, 0, artifact["bytes"], 0, 0, 0, 0))
        for role, artifact in artifacts.items()
    }


def validate_execution_record_structure(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "FRI execution record must be an object")
    require_exact_keys(value, {
        "schema_version", "evidence_mode", "preflight_only", "passed",
        "production_admissible", "correctness_admissible", "performance_admissible",
        "target_sm", "device", "artifacts", "module_identity", "fixture_identity",
        "loop_tool_sources", "loop_tool_sha256", "execution_contract",
        "raw_result", "raw_runner_admission", "runner_result",
    }, "FRI execution record")
    require(value["schema_version"] == EXECUTION_RECORD_SCHEMA
            and value["evidence_mode"] == EVIDENCE_MODE
            and value["preflight_only"] is False and value["passed"] is True
            and value["production_admissible"] is False
            and value["correctness_admissible"] is False
            and value["performance_admissible"] is False,
            "FRI execution record overclaims its evidence")
    target_sm = require_int(value["target_sm"], "FRI execution target SM", 50)
    require(target_sm <= 999, "FRI execution target SM exceeds the supported range")
    device = require_int(value["device"], "FRI execution device ordinal")
    artifacts = _validate_artifacts(value["artifacts"])
    roles = _records_as_roles(artifacts)
    identity = value["module_identity"]
    require(isinstance(identity, dict), "FRI execution module identity must be an object")
    require_exact_keys(identity, {
        "abi_sha256", "source_closure_sha256", "build_tool_sha256", "toolchain_sha256",
    }, "FRI execution module identity")
    for name, digest in identity.items():
        require_sha256(digest, f"FRI execution module identity {name}")
    validate_fixture_identity(value["fixture_identity"])
    validate_loop_tool(value["loop_tool_sources"], value["loop_tool_sha256"])
    raw = value["raw_result"]
    require(isinstance(raw, dict), "FRI execution raw result identity must be an object")
    require_exact_keys(raw, {"path", "sha256", "bytes"}, "FRI execution raw result identity")
    require(isinstance(raw["path"], str) and Path(raw["path"]).is_absolute(),
            "FRI execution raw result path is not absolute")
    require_sha256(raw["sha256"], "FRI execution raw result sha256")
    raw_bytes = require_int(raw["bytes"], "FRI execution raw result bytes", 1)
    require(raw_bytes <= MAX_RESULT_BYTES, "FRI execution raw result is too large")
    require(raw["path"] not in {artifact["path"] for artifact in artifacts.values()},
            "FRI execution raw result aliases an input")
    admission = value["raw_runner_admission"]
    admission_names = {
        "production_admissible", "correctness_admissible", "performance_admissible",
    }
    require(isinstance(admission, dict), "FRI raw runner admission must be an object")
    require_exact_keys(admission, admission_names, "FRI raw runner admission")
    require(all(admission[name] is False for name in admission_names),
            "FRI raw runner admission overclaims captured-unsealed evidence")
    expected = _expected(roles, target_sm, device)
    validate_runner_result(value["runner_result"], expected)
    contract = value["execution_contract"]
    require(isinstance(contract, dict), "FRI execution contract must be an object")
    require_exact_keys(contract, {
        "contract", "shell", "stdin", "working_directory", "argv", "exit_status",
        "environment", "fd_bindings", "timeout_seconds", "stdout_sha256", "stdout_bytes",
        "stderr_sha256", "stderr_bytes",
    }, "FRI execution contract")
    environment = validate_environment_contract(contract["environment"])
    require(contract["shell"] is False, "FRI execution shell flag is not false")
    exit_status = require_int(contract["exit_status"], "FRI execution exit status")
    timeout = require_int(contract["timeout_seconds"], "FRI execution timeout", 1)
    stdout_bytes = require_int(contract["stdout_bytes"], "FRI execution stdout bytes", 1)
    stderr_bytes = require_int(contract["stderr_bytes"], "FRI execution stderr bytes")
    bindings = contract["fd_bindings"]
    require(isinstance(bindings, dict) and set(bindings) == SEMANTIC_ROLES,
            "FRI execution fd-binding roles differ")
    descriptors: list[int] = []
    launch_paths: dict[str, str] = {}
    for role, binding in bindings.items():
        require(isinstance(binding, dict), f"FRI execution fd binding {role} is not an object")
        require_exact_keys(binding, {"fd", "proc_path", "artifact_path", "sha256", "bytes"},
                           f"FRI execution fd binding {role}")
        descriptor = require_int(binding["fd"], f"FRI execution fd binding {role}", 3)
        binding_bytes = require_int(binding["bytes"], f"FRI execution fd binding {role} bytes", 1)
        expected_binding = {
            "fd": descriptor,
            "proc_path": f"/proc/self/fd/{descriptor}",
            "artifact_path": artifacts[role]["path"],
            "sha256": artifacts[role]["sha256"],
            "bytes": binding_bytes,
        }
        require(binding_bytes == artifacts[role]["bytes"] and binding == expected_binding,
                f"FRI execution fd binding {role} differs")
        descriptors.append(descriptor)
        launch_paths[role] = binding["proc_path"]
    require(len(descriptors) == len(set(descriptors)),
            "FRI execution fd bindings reuse a descriptor")
    expected_command = _command(roles, launch_paths, Path(raw["path"]), target_sm, device)
    require(contract == {
        "contract": "linux-direct-argv-no-shell-v1",
        "shell": False,
        "stdin": "devnull",
        "working_directory": str(Path(raw["path"]).parent),
        "argv": expected_command,
        "exit_status": exit_status,
        "environment": environment,
        "fd_bindings": bindings,
        "timeout_seconds": timeout,
        "stdout_sha256": raw["sha256"],
        "stdout_bytes": stdout_bytes,
        "stderr_sha256": EMPTY_SHA256,
        "stderr_bytes": stderr_bytes,
    }, "FRI Linux execution contract differs")
    require(exit_status == 0 and timeout == RUNNER_TIMEOUT_SECONDS
            and stdout_bytes == raw_bytes and stderr_bytes == 0,
            "FRI Linux execution numeric contract differs")
    return value


def _install_execution_record(
    path: Path, record: dict[str, Any], sources: list[Path],
) -> tuple[dict[str, Any], BoundFile]:
    payload = canonical_bytes(record) + b"\n"
    write_immutable_bytes(path, payload, sources, fresh_only=True)
    bound = observe_file(path, "FRI execution record", read_only=True)
    require(bound.sha256 == sha256_bytes(payload) and bound.byte_length == len(payload),
            "installed FRI execution record differs")
    opened = _open_bound(bound)
    try:
        installed = validate_execution_record_structure(
            _parse_result(_descriptor_bytes(opened.descriptor, bound.byte_length))
        )
        _verify_opened(opened)
    finally:
        os.close(opened.descriptor)
    require(installed == record, "installed FRI execution record content differs")
    recheck(bound)
    return installed, bound


def execute(args: Any) -> dict[str, Any]:
    require(not args.preflight_only, "FRI execution cannot use --preflight-only")
    require_int(args.target_sm, "FRI target SM", 50)
    require(args.target_sm <= 999, "FRI target SM exceeds the supported range")
    require_int(args.device, "FRI device ordinal")
    environment = reviewed_environment()
    module_identity, module_bounds = _module_bundle(args)
    fixture_identity, fixture_bounds = _fixture_bundle(args)
    harness = bind_file(args.harness, args.harness_sha256, "FRI runner", executable=True)
    require(harness.path.name == "stwo-gpu-lab-fri-round6", "FRI runner filename differs")
    tool_sources = loop_tool_sources()
    tool_digest = loop_tool_sha256()
    validate_loop_tool(tool_sources, tool_digest)
    tool_bounds = _loop_tool_bounds(tool_sources)
    bounds = [*module_bounds, *fixture_bounds, harness, *tool_bounds]
    _require_semantic_distinct(bounds)
    input_paths = [bound.path for bound in bounds]
    raw_path = _fresh_output(args.raw_result, RAW_RESULT_ROLE, input_paths)
    record_path = _fresh_output(args.record, "FRI execution record", input_paths)
    require(raw_path != record_path, "FRI raw result and execution record paths alias")
    roles = _roles(bounds)
    opened = _open_roles(roles)
    launch_paths = {role: item.proc_path for role, item in opened.items()}
    fd_bindings = {role: item.record() for role, item in opened.items()}
    command = _command(roles, launch_paths, raw_path, args.target_sm, args.device)
    try:
        _recheck_all(bounds, tool_sources, tool_digest)
        for item in opened.values():
            _verify_opened(item)
        run = run_linux(command, raw_path.parent,
                        tuple(item.descriptor for item in opened.values()), environment)
        for item in opened.values():
            _verify_opened(item)
    finally:
        for item in opened.values():
            os.close(item.descriptor)
    _recheck_all(bounds, tool_sources, tool_digest)
    require(run.returncode == 0, f"FRI runner exit status differs: {run.returncode}")
    require(run.stderr == b"", "FRI runner emitted unexpected stderr")
    require(len(run.stdout) <= MAX_RESULT_BYTES, "FRI runner stdout is too large")
    raw_bound = observe_file(raw_path, RAW_RESULT_ROLE, read_only=True)
    require(raw_bound.byte_length <= MAX_RESULT_BYTES, "FRI raw result is too large")
    raw_opened = _open_bound(raw_bound)
    try:
        raw_bytes = _descriptor_bytes(raw_opened.descriptor, raw_bound.byte_length)
        _verify_opened(raw_opened)
        require(run.stdout == raw_bytes, "FRI runner stdout differs from its result file")
        runner_result = validate_runner_result(_parse_result(raw_bytes),
                                               _expected(roles, args.target_sm, args.device))
        _recheck_all([*bounds, raw_bound], tool_sources, tool_digest)
        record = {
            "schema_version": EXECUTION_RECORD_SCHEMA,
            "evidence_mode": EVIDENCE_MODE,
            "preflight_only": False,
            "passed": True,
            "production_admissible": False,
            "correctness_admissible": False,
            "performance_admissible": False,
            "target_sm": args.target_sm,
            "device": args.device,
            "artifacts": _artifact_records(bounds),
            "module_identity": module_identity,
            "fixture_identity": fixture_identity,
            "loop_tool_sources": tool_sources,
            "loop_tool_sha256": tool_digest,
            "execution_contract": {
                "contract": "linux-direct-argv-no-shell-v1",
                "shell": False,
                "stdin": "devnull",
                "working_directory": str(raw_path.parent),
                "argv": command,
                "exit_status": 0,
                "environment": environment,
                "fd_bindings": fd_bindings,
                "timeout_seconds": RUNNER_TIMEOUT_SECONDS,
                "stdout_sha256": raw_bound.sha256,
                "stdout_bytes": raw_bound.byte_length,
                "stderr_sha256": EMPTY_SHA256,
                "stderr_bytes": 0,
            },
            "raw_result": raw_bound.record(),
            "raw_runner_admission": {
                "production_admissible": False,
                "correctness_admissible": False,
                "performance_admissible": False,
            },
            "runner_result": runner_result,
        }
        validate_execution_record_structure(record)
        installed, record_bound = _install_execution_record(
            record_path, record, [*input_paths, raw_bound.path],
        )
        _verify_opened(raw_opened)
    finally:
        os.close(raw_opened.descriptor)
    _recheck_all([*bounds, raw_bound, record_bound], tool_sources, tool_digest)
    return installed
