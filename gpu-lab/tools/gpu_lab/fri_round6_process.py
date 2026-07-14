"""Reviewed Linux environment and bounded process I/O for FRI round-6."""

from __future__ import annotations

import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .common import require, require_exact_keys


MAX_RESULT_BYTES = 1 << 20
RUNNER_TIMEOUT_SECONDS = 120
ENVIRONMENT_POLICY = "stwo.gpu-lab.linux-cuda-environment.v1"
FIXED_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C",
    "LC_ALL": "C",
    "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_MODULE_LOADING": "EAGER",
    "CUDA_CACHE_DISABLE": "1",
    "CUDA_DISABLE_PTX_JIT": "1",
    "CUDA_LAUNCH_BLOCKING": "0",
}


def reviewed_environment(source: Mapping[str, str] | None = None) -> dict[str, Any]:
    parent = os.environ if source is None else source
    injection = sorted(
        name for name in parent
        if ((name.startswith(("LD_", "DYLD_")) and name != "LD_LIBRARY_PATH")
            or name == "GLIBC_TUNABLES")
    )
    require(not injection, f"loader-injection environment variables are forbidden: {injection}")
    variables = dict(FIXED_ENVIRONMENT)
    visible = parent.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        require(0 < len(visible) <= 256 and re.fullmatch(r"[A-Za-z0-9,._:/-]+", visible),
                "CUDA_VISIBLE_DEVICES is not a bounded canonical selector")
        variables["CUDA_VISIBLE_DEVICES"] = visible
    contract = {"policy": ENVIRONMENT_POLICY, "variables": dict(sorted(variables.items()))}
    return validate_environment_contract(contract)


def validate_environment_contract(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "FRI process environment must be an object")
    require_exact_keys(value, {"policy", "variables"}, "FRI process environment")
    require(value["policy"] == ENVIRONMENT_POLICY, "FRI process environment policy differs")
    variables = value["variables"]
    require(isinstance(variables, dict), "FRI process environment variables must be an object")
    allowed = set(FIXED_ENVIRONMENT) | {"CUDA_VISIBLE_DEVICES"}
    require(set(FIXED_ENVIRONMENT) <= set(variables) <= allowed,
            "FRI process environment variable names differ")
    require(all(isinstance(name, str) and isinstance(item, str)
                for name, item in variables.items()),
            "FRI process environment contains a non-string binding")
    for name, expected in FIXED_ENVIRONMENT.items():
        require(variables[name] == expected, f"FRI process environment {name} differs")
    visible = variables.get("CUDA_VISIBLE_DEVICES")
    require(visible is None or (0 < len(visible) <= 256
            and re.fullmatch(r"[A-Za-z0-9,._:/-]+", visible) is not None),
            "FRI process CUDA visibility differs")
    return value


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_bounded_process(
    command: list[str], working_directory: Path, descriptors: tuple[int, ...],
    environment: dict[str, Any], *, timeout_seconds: float = RUNNER_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    contract = validate_environment_contract(environment)
    require(timeout_seconds > 0, "FRI runner timeout must be positive")
    process = subprocess.Popen(
        command,
        executable=command[0],
        cwd=working_directory,
        env=contract["variables"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        shell=False,
        close_fds=True,
        pass_fds=descriptors,
        start_new_session=True,
        bufsize=0,
    )
    require(process.stdout is not None and process.stderr is not None,
            "FRI runner pipes are unavailable")
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    deadline = time.monotonic() + timeout_seconds
    try:
        for stream, role in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, role)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError(f"FRI runner exceeded {timeout_seconds}s execution timeout")
            for key, _ in selector.select(min(remaining, 0.1)):
                try:
                    chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                elif key.data == "stderr":
                    raise ValueError("FRI runner emitted unexpected stderr")
                elif len(stdout) + len(chunk) > MAX_RESULT_BYTES:
                    raise ValueError("FRI runner stdout exceeded the 1 MiB bound")
                else:
                    stdout.extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f"FRI runner exceeded {timeout_seconds}s execution timeout")
        try:
            status = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise ValueError(f"FRI runner exceeded {timeout_seconds}s execution timeout") from error
    except BaseException:
        _kill_group(process)
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return subprocess.CompletedProcess(command, status, bytes(stdout), b"")


def run_linux(
    command: list[str], working_directory: Path, descriptors: tuple[int, ...],
    environment: dict[str, Any],
) -> subprocess.CompletedProcess[bytes]:
    require(sys.platform == "linux", "FRI runner execution requires Linux")
    require(Path("/proc/self/fd").is_dir(), "Linux /proc/self/fd is unavailable")
    for descriptor in descriptors:
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode)
                and Path(f"/proc/self/fd/{descriptor}").exists(),
                "FRI inherited descriptor is unavailable through /proc")
    return run_bounded_process(command, working_directory, descriptors, environment)
