"""Bounded direct execution of a sealed child process."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from pathlib import Path
from typing import Mapping


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


def run_bounded_child(
    command: list[str], working_directory: Path, descriptors: tuple[int, ...],
    environment: Mapping[str, str], *, timeout_seconds: float, max_stdout_bytes: int,
    process_name: str, stdout_bound_name: str,
) -> subprocess.CompletedProcess[bytes]:
    """Run one direct child with bounded output, time, and inherited descriptors."""
    process = subprocess.Popen(
        command,
        executable=command[0],
        cwd=working_directory,
        env=environment,
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
    if process.stdout is None or process.stderr is None:
        _kill_group(process)
        raise ValueError(f"{process_name} pipes are unavailable")
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
                raise ValueError(
                    f"{process_name} exceeded {timeout_seconds}s execution timeout"
                )
            for key, _ in selector.select(min(remaining, 0.1)):
                try:
                    chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                elif key.data == "stderr":
                    raise ValueError(f"{process_name} emitted unexpected stderr")
                elif len(stdout) + len(chunk) > max_stdout_bytes:
                    raise ValueError(f"{process_name} stdout exceeded {stdout_bound_name}")
                else:
                    stdout.extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f"{process_name} exceeded {timeout_seconds}s execution timeout")
        try:
            status = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise ValueError(
                f"{process_name} exceeded {timeout_seconds}s execution timeout"
            ) from error
    except BaseException:
        _kill_group(process)
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return subprocess.CompletedProcess(command, status, bytes(stdout), b"")
