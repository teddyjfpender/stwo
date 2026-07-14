"""GPU-free hostile tests for the FRI round-6 process boundary."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import immutable_output as immutable_output_module
from .common import REPO_ROOT, load_json, require, sha256_file
from .fri_round6_execution import execute, validate_execution_record_structure
from .fri_round6_loop_tests import (
    _arguments,
    _expect_rejection,
    _fixtures,
    _module,
)
from .fri_round6_process import (
    MAX_RESULT_BYTES,
    reviewed_environment,
    run_bounded_process,
    validate_environment_contract,
)
from .immutable_output import write_immutable_bytes
from .sealed_process import run_bounded_child


def _runner_source(behavior: str) -> str:
    return f'''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

behavior = {behavior!r}
arguments = sys.argv[1:]
if len(arguments) % 2:
    raise SystemExit(91)
options = {{arguments[index][2:]: arguments[index + 1]
           for index in range(0, len(arguments), 2)}}
for inherited in filter(None, options.get("test-fds", "").split(",")):
    descriptor = int(inherited)
    if os.fstat(descriptor).st_size <= 0 or not os.pread(descriptor, 1, 0):
        raise SystemExit(92)
passed = {{"passed": True, "checked_words": 416, "error": ""}}
result = {{
    "schema_version": "stwo.gpu-lab.fri-round6-result.v1",
    "passed": True,
    "standalone_admissible": False,
    "performance_admissible": False,
    "segment": "GraphSegment::FriLayer(7)/FriRound(6)",
    "module_content_sha256": options["module-sha256"],
    "module_index_sha256": options["module-index-sha256"],
    "build_recipe_sha256": options["build-recipe-sha256"],
    "primary_fixture_sha256": options["primary-sha256"],
    "primary_fixture_index_sha256": options["primary-index-sha256"],
    "hostile_fixture_sha256": options["hostile-sha256"],
    "hostile_fixture_index_sha256": options["hostile-index-sha256"],
    "harness_executable_sha256": options["harness-sha256"],
    "device": {{"name": "gpu-free-test", "uuid": "ab" * 16,
               "ordinal": int(options["device"]),
               "target_sm": int(options["target-sm"]), "driver_version": 12080}},
    "graph_contract": {{"kernels": 7, "device_copies": 6, "entry_log": 6,
                        "exit_log": 3, "packed_leaf_log": 2}},
    "correctness": {{"primary_eager": passed, "primary_graph": passed,
                    "hostile_eager": passed, "hostile_graph": passed,
                    "stale_cursor_status_order": {{"passed": True, "error": "rejected"}},
                    "reset_replay": passed}},
}}
if behavior == "hash-substitution":
    result["module_index_sha256"] = "00" * 32
elif behavior == "wrong-status":
    result["passed"] = False
elif behavior == "extra-key":
    result["unreviewed"] = True
elif behavior == "missing-key":
    del result["segment"]
elif behavior == "numeric-float":
    result["device"]["ordinal"] = float(result["device"]["ordinal"])
payload = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\\n").encode()
destination = Path(options["result"])
destination.write_bytes(payload)
os.chmod(destination, 0o400)
sys.stdout.buffer.write(b"{{}}\\n" if behavior == "stdout-file-mismatch" else payload)
raise SystemExit(2 if behavior == "wrong-exit" else 0)
'''


def _configure(args: SimpleNamespace, directory: Path, behavior: str) -> None:
    args.harness.write_text(_runner_source(behavior))
    os.chmod(args.harness, 0o700)
    args.harness_sha256 = sha256_file(args.harness)
    args.raw_result = directory / f"raw-{behavior}.json"
    args.record = directory / f"execution-{behavior}.json"
    args.preflight_only = False


def _execute_portable(args: SimpleNamespace, behavior: str) -> dict:
    def launch(command: list[str], working_directory: Path,
               descriptors: tuple[int, ...],
               environment: dict) -> subprocess.CompletedProcess[bytes]:
        require(set(descriptors) and command[0].startswith("/proc/self/fd/"),
                "FRI test launch was not descriptor-bound")
        validate_environment_contract(environment)
        for descriptor in descriptors:
            metadata = os.fstat(descriptor)
            require(metadata.st_size > 0 and os.pread(descriptor, 1, 0),
                    "FRI semantic descriptor was not inherited/readable")
        harness_descriptor = int(command[0].rsplit("/", 1)[1])
        require(harness_descriptor in descriptors, "FRI harness descriptor was not inherited")
        backup = args.harness.with_name(args.harness.name + ".transient-original")
        marker = working_directory / "hostile-runner-executed"
        transient = behavior == "transient-executable-substitution"
        if transient:
            args.harness.rename(backup)
            args.harness.write_text(
                f"#!/usr/bin/env python3\nfrom pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('hostile')\nraise SystemExit(97)\n"
            )
            os.chmod(args.harness, 0o700)
        try:
            length = os.fstat(harness_descriptor).st_size
            source = os.pread(harness_descriptor, length, 0).decode("utf-8")
            inherited = ",".join(str(descriptor) for descriptor in descriptors)
            run = subprocess.run(
                [sys.executable, "-c", source, *command[1:], "--test-fds", inherited],
                cwd=working_directory, stdin=subprocess.DEVNULL,
                capture_output=True, text=False, shell=False, check=False,
                close_fds=True, pass_fds=descriptors,
                timeout=5, env=environment["variables"],
            )
        finally:
            if transient:
                args.harness.unlink()
                backup.rename(args.harness)
        require(not marker.exists(), "transient hostile runner pathname was executed")
        if behavior == "post-run-replacement":
            backup = args.primary.with_name(args.primary.name + ".runner-original")
            args.primary.rename(backup)
            args.primary.write_bytes(backup.read_bytes())
        return run

    with patch("gpu_lab.fri_round6_execution.run_linux", side_effect=launch):
        return execute(args)


def _restore_primary(args: SimpleNamespace) -> None:
    backup = args.primary.with_name(args.primary.name + ".runner-original")
    if backup.exists():
        args.primary.unlink(missing_ok=True)
        backup.rename(args.primary)


def _publisher_substitution_test(directory: Path) -> None:
    source = directory / "publisher-source"
    source.write_bytes(b"sealed source")
    output = directory / "substituted-record"

    def substitute(temporary: Path) -> None:
        temporary.unlink()
        temporary.write_bytes(b"substituted bytes")
        os.chmod(temporary, 0o400)

    _expect_rejection("immutable publisher temporary substitution", lambda: write_immutable_bytes(
        output, b"intended record", [source], fresh_only=True,
        _before_publish=substitute,
    ))
    require(not output.exists() or output.read_bytes() != b"intended record",
            "substituted publisher output was accepted as the intended record")

    early = directory / "early-failure-record"
    before = set(directory.glob(f".{early.name}.*"))
    with patch("gpu_lab.immutable_output.os.fchmod", side_effect=OSError("injected fchmod")):
        _expect_rejection("immutable publisher early failure", lambda: write_immutable_bytes(
            early, b"intended record", [source], fresh_only=True,
        ))
    require(not early.exists() and set(directory.glob(f".{early.name}.*")) == before,
            "immutable publisher leaked its bound temporary after an early failure")

    durability = directory / "directory-fsync-failure-record"
    real_fsync = os.fsync

    def fail_directory_sync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory fsync")
        real_fsync(descriptor)

    with patch("gpu_lab.immutable_output.os.fsync", side_effect=fail_directory_sync):
        _expect_rejection("immutable publisher directory fsync failure",
                          lambda: write_immutable_bytes(
                              durability, b"intended record", [source], fresh_only=True,
                          ))
    require(not durability.exists(),
            "directory-fsync failure left a published PASS artifact")

    final = directory / "final-verification-failure-record"
    real_verify = immutable_output_module._verify_installed
    calls = 0

    def fail_final_verification(path: Path, identity, mode: int, length: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected final verification failure")
        real_verify(path, identity, mode, length)

    with patch("gpu_lab.immutable_output._verify_installed",
               side_effect=fail_final_verification):
        _expect_rejection("immutable publisher final verification failure",
                          lambda: write_immutable_bytes(
                              final, b"intended record", [source], fresh_only=True,
                          ))
    require(calls == 2 and not final.exists(),
            "final-verification failure left a published PASS artifact")


def _process_boundary_test(directory: Path) -> None:
    environment = reviewed_environment({})
    validate_environment_contract(environment)
    _expect_rejection("loader injection", lambda: reviewed_environment(
        {"LD_PRELOAD": "/tmp/hostile.so"}
    ))
    sanitized = reviewed_environment({
        "PYTHONPATH": "/tmp/hostile", "CUDA_LAUNCH_BLOCKING": "1",
        "CUDA_VISIBLE_DEVICES": "GPU-01234567-89ab-cdef-0123-456789abcdef",
    })
    require("PYTHONPATH" not in sanitized["variables"]
            and sanitized["variables"]["CUDA_LAUNCH_BLOCKING"] == "0",
            "FRI process inherited an unreviewed parent binding")

    def run(source: str, timeout: float = 2.0) -> None:
        run_bounded_process(
            [sys.executable, "-c", source], directory, (), environment,
            timeout_seconds=timeout,
        )

    def reject_exact(expected: str, operation) -> None:
        try:
            operation()
        except ValueError as error:
            require(str(error) == expected,
                    f"sealed process error wording differs: {error}")
            return
        raise ValueError(f"hostile sealed process input was accepted: {expected}")

    inherited_path = directory / "sealed-process-input"
    inherited_path.write_bytes(b"sealed")
    marker = directory / "shell-must-not-run"
    literal_argument = f"; touch {marker}"
    descriptor = os.open(inherited_path, os.O_RDONLY)
    try:
        direct = run_bounded_process(
            [sys.executable, "-c", (
                "import os,sys; os.write(1, os.pread(int(sys.argv[1]), 6, 0)"
                " + sys.argv[2].encode())"
            ), str(descriptor), literal_argument],
            directory, (descriptor,), environment, timeout_seconds=2.0,
        )
    finally:
        os.close(descriptor)
    require(direct.stdout == b"sealed" + literal_argument.encode() and not marker.exists(),
            "sealed process did not preserve direct argv and inherited descriptors")

    reject_exact("FRI runner stdout exceeded the 1 MiB bound", lambda: run(
        f"import os; os.write(1, b'x' * {MAX_RESULT_BYTES + 1})"
    ))
    reject_exact("FRI runner emitted unexpected stderr", lambda: run(
        "import os; os.write(2, b'x')"
    ))
    reject_exact("FRI runner exceeded 0.05s execution timeout",
                 lambda: run("while True: pass", 0.05))

    marker = directory / "sealed-process-child-setup"
    marker.write_bytes(b"")
    descriptor = os.open(marker, os.O_RDWR)
    try:
        accepted = run_bounded_child(
            [sys.executable, "-c", "import os;os.write(2,b'accepted')"],
            directory, (descriptor,), {}, timeout_seconds=2, max_stdout_bytes=0,
            process_name="adapter test", stdout_bound_name="the zero-byte bound",
            max_stderr_bytes=8, stderr_bound_name="8 bytes",
            child_setup=lambda: os.write(descriptor, b"setup"),
        )
    finally:
        os.close(descriptor)
    require(accepted.stdout == b"" and accepted.stderr == b"accepted"
            and marker.read_bytes() == b"setup",
            "sealed process did not return bounded stderr or run child setup")
    reject_exact("adapter test stderr exceeded 8 bytes", lambda: run_bounded_child(
        [sys.executable, "-c", "import os;os.write(2,b'123456789')"],
        directory, (), {}, timeout_seconds=2, max_stdout_bytes=0,
        process_name="adapter test", stdout_bound_name="the zero-byte bound",
        max_stderr_bytes=8, stderr_bound_name="8 bytes",
    ))


def _record_type_mutation_test(record: dict) -> None:
    binding_role = next(iter(record["execution_contract"]["fd_bindings"]))
    binding_bytes = record["execution_contract"]["fd_bindings"][binding_role]["bytes"]
    stdout_bytes = record["execution_contract"]["stdout_bytes"]
    cases = (
        ("fd binding bytes", lambda value, hostile: value["execution_contract"]
         ["fd_bindings"][binding_role].__setitem__("bytes", hostile),
         (True, float(binding_bytes))),
        ("shell flag", lambda value, hostile: value["execution_contract"].__setitem__(
            "shell", hostile), (0, 0.0)),
        ("exit status", lambda value, hostile: value["execution_contract"].__setitem__(
            "exit_status", hostile), (False, 0.0)),
        ("timeout", lambda value, hostile: value["execution_contract"].__setitem__(
            "timeout_seconds", hostile), (True, 120.0)),
        ("stdout bytes", lambda value, hostile: value["execution_contract"].__setitem__(
            "stdout_bytes", hostile), (True, float(stdout_bytes))),
        ("stderr bytes", lambda value, hostile: value["execution_contract"].__setitem__(
            "stderr_bytes", hostile), (False, 0.0)),
        ("raw runner admission", lambda value, hostile: value["raw_runner_admission"].__setitem__(
            "production_admissible", hostile), (0, 0.0)),
    )
    for label, mutate, hostile_values in cases:
        for hostile_value in hostile_values:
            hostile = deepcopy(record)
            mutate(hostile, hostile_value)
            _expect_rejection(
                f"{label} {type(hostile_value).__name__}",
                lambda hostile=hostile: validate_execution_record_structure(hostile),
            )


def fri_round6_execution_self_test(lab_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix=".fri-round6-execution-test-", dir=lab_root) as name:
        directory = Path(name).resolve()
        args = _arguments(directory, _module(directory, 86), _fixtures(directory))

        _configure(args, directory, "success")
        record = _execute_portable(args, "success")
        validate_execution_record_structure(record)
        require(record["raw_runner_admission"] == {
            "production_admissible": False,
            "correctness_admissible": False,
            "performance_admissible": False,
        }, "FRI fake-runner record admitted captured-unsealed evidence")
        require((args.record.stat().st_mode & 0o777) == 0o400,
                "FRI execution record is not owner-read-only")

        for behavior in (
            "hash-substitution", "wrong-exit", "wrong-status", "stdout-file-mismatch",
            "post-run-replacement", "extra-key", "missing-key", "numeric-float",
        ):
            _configure(args, directory, behavior)
            try:
                _expect_rejection(behavior, lambda: _execute_portable(args, behavior))
            finally:
                _restore_primary(args)

        _configure(args, directory, "transient-executable-substitution")
        _expect_rejection(
            "transient executable pathname substitution",
            lambda: _execute_portable(args, "transient-executable-substitution"),
        )
        require(args.raw_result.is_file() and load_json(args.raw_result)["passed"] is True,
                "sealed runner bytes did not execute before transient substitution rejection")

        _configure(args, directory, "stale-raw")
        args.raw_result.write_bytes(b"stale raw result")
        _expect_rejection("stale raw result path", lambda: _execute_portable(args, "stale-raw"))

        _configure(args, directory, "stale-record")
        args.record.write_bytes(b"stale execution record")
        _expect_rejection("stale execution record path",
                          lambda: _execute_portable(args, "stale-record"))

        hostile = deepcopy(record)
        hostile["unreviewed"] = True
        _expect_rejection("extra execution-record key",
                          lambda: validate_execution_record_structure(hostile))
        hostile = deepcopy(record)
        del hostile["execution_contract"]
        _expect_rejection("missing execution-record key",
                          lambda: validate_execution_record_structure(hostile))
        _record_type_mutation_test(record)
        _publisher_substitution_test(directory)
        _process_boundary_test(directory)

        if sys.platform == "linux" and Path("/proc/self/fd").is_dir():
            _configure(args, directory, "linux-fd-exec")
            require(execute(args)["passed"] is True, "real Linux fd-exec smoke failed")


if __name__ == "__main__":
    fri_round6_execution_self_test(REPO_ROOT / "gpu-lab")
    print("FRI round-6 execution GPU-free hostile tests: PASS")
