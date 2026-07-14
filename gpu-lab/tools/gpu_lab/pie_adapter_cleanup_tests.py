"""Focused cleanup fault checks for authenticated PIE adapter execution."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from . import pie_adapter_execution as execution_module
from .common import LAB_ROOT, require
from .pie_adapter_execution import _execute as execute
from .pie_adapter_execution_tests import EXPECTED, _fixture, _runner
from .retained_fs import RetainedDirectory


def _open_fds() -> set[int]:
    return {int(name) for name in os.listdir("/dev/fd") if name.isdigit()}


def _reject(operation: Callable[[], Any]) -> BaseException:
    try:
        operation()
    except (OSError, ValueError) as error:
        return error
    raise ValueError("injected PIE adapter cleanup fault was accepted")


def _require_rolled_back(args: Any, baseline: set[int], label: str) -> None:
    require(not args.observed_prover_input.exists(), f"{label} left observed output")
    require(not args.record.exists(), f"{label} left a PASS record")
    require(not [path for path in args.record.parent.iterdir() if path.name.startswith(".")],
            f"{label} left a hidden record temporary")
    require(_open_fds() == baseline, f"{label} leaked or double-closed a descriptor")


def _fail_first_created_fstat(parent: Path, *, record_temporary: bool) -> None:
    label = "record temporary fstat" if record_temporary else "output fstat"
    args, tools = _fixture(parent / label.replace(" ", "-"))
    baseline = _open_fds()
    real_open, real_fstat = RetainedDirectory.open, os.fstat
    state: dict[str, int | bool | None] = {"descriptor": None, "failed": False}

    def tracked_open(directory, name, flags, open_label, mode=0o777):
        descriptor = real_open(directory, name, flags, open_label, mode)
        is_created = flags & os.O_CREAT and flags & os.O_EXCL
        is_target = (name.startswith(".") if record_temporary
                     else name == args.observed_prover_input.name)
        if is_created and is_target:
            state["descriptor"] = descriptor
        return descriptor

    def fail_once(descriptor: int):
        if descriptor == state["descriptor"] and not state["failed"]:
            state["failed"] = True
            raise OSError(f"injected first {label} failure")
        return real_fstat(descriptor)

    with patch.object(RetainedDirectory, "open", new=tracked_open), patch.object(
            os, "fstat", new=fail_once):
        _reject(lambda: execute(args, _runner=_runner(args, "success"), _tool_paths=tools))
    require(state["failed"] is True, f"{label} injection did not reach its target")
    _require_rolled_back(args, baseline, label)


def _output_fchmod_failure(parent: Path) -> None:
    args, tools = _fixture(parent / "output-fchmod")
    baseline = _open_fds()
    real_open, real_fchmod = RetainedDirectory.open, os.fchmod
    output_descriptor: int | None = None
    fired = False

    def tracked_open(directory, name, flags, open_label, mode=0o777):
        nonlocal output_descriptor
        descriptor = real_open(directory, name, flags, open_label, mode)
        if (name == args.observed_prover_input.name
                and flags & os.O_CREAT and flags & os.O_EXCL):
            output_descriptor = descriptor
        return descriptor

    def fail_fchmod(descriptor: int, mode: int) -> None:
        nonlocal fired
        if descriptor == output_descriptor and mode == 0o600 and not fired:
            fired = True
            raise OSError("injected output fchmod failure")
        real_fchmod(descriptor, mode)

    with patch.object(RetainedDirectory, "open", new=tracked_open), patch.object(
            os, "fchmod", new=fail_fchmod):
        _reject(lambda: execute(args, _runner=_runner(args, "success"), _tool_paths=tools))
    require(fired, "output fchmod injection did not reach its target")
    _require_rolled_back(args, baseline, "output fchmod failure")


def _record_close_after_publication(parent: Path) -> None:
    args, tools = _fixture(parent / "record-close-after-publication")
    baseline = _open_fds()
    real_close, real_fstat = os.close, os.fstat
    fired = False

    def close_then_fail(descriptor: int) -> None:
        nonlocal fired
        target = False
        if not fired and args.record.exists():
            try:
                opened = real_fstat(descriptor)
                installed = args.record.stat()
                target = (stat.S_ISREG(opened.st_mode)
                          and (opened.st_dev, opened.st_ino)
                          == (installed.st_dev, installed.st_ino))
            except OSError:
                pass
        real_close(descriptor)
        if target:
            fired = True
            raise OSError("injected published record descriptor close failure")

    with patch.object(os, "close", new=close_then_fail):
        _reject(lambda: execute(args, _runner=_runner(args, "success"), _tool_paths=tools))
    require(fired, "published record descriptor close injection did not fire")
    _require_rolled_back(args, baseline, "published record descriptor close failure")


def _outer_output_close_after_commit(parent: Path) -> None:
    args, tools = _fixture(parent / "outer-output-close-after-commit")
    baseline = _open_fds()
    real_close, real_fstat = os.close, os.fstat
    fired = False

    def close_then_fail(descriptor: int) -> None:
        nonlocal fired
        target = False
        if not fired and args.record.exists() and args.observed_prover_input.exists():
            try:
                opened = real_fstat(descriptor)
                output = args.observed_prover_input.stat()
                target = ((opened.st_dev, opened.st_ino)
                          == (output.st_dev, output.st_ino))
            except OSError:
                pass
        real_close(descriptor)
        if target:
            fired = True
            raise OSError("injected post-commit output descriptor close failure")

    with patch.object(os, "close", new=close_then_fail):
        record = execute(args, _runner=_runner(args, "success"), _tool_paths=tools)
    require(fired and record["passed"] is True,
            "post-commit output close fault changed PASS status")
    require(args.observed_prover_input.read_bytes() == EXPECTED,
            "post-commit output close fault changed exact bytes")
    require(stat.S_IMODE(args.observed_prover_input.stat().st_mode) == 0o400
            and stat.S_IMODE(args.record.stat().st_mode) == 0o400,
            "post-commit output close fault changed immutable modes")
    require(not [path for path in args.record.parent.iterdir() if path.name.startswith(".")],
            "post-commit output close fault left a record temporary")
    require(_open_fds() == baseline, "post-commit output close fault drifted descriptors")


def _cleanup_directory_fsync(parent: Path) -> None:
    args, tools = _fixture(parent / "cleanup-directory-fsync")
    baseline = _open_fds()
    real_fsync, real_fstat = os.fsync, os.fstat
    fired = False

    def sync_then_fail(descriptor: int) -> None:
        nonlocal fired
        metadata = real_fstat(descriptor)
        real_fsync(descriptor)
        if not fired and stat.S_ISDIR(metadata.st_mode):
            fired = True
            raise OSError("injected cleanup directory fsync failure")

    with patch.object(os, "fsync", new=sync_then_fail):
        error = _reject(lambda: execute(
            args, _runner=_runner(args, "timeout"), _tool_paths=tools
        ))
    require(fired, "cleanup directory fsync injection did not fire")
    require("exceeded 300s wall timeout" in str(error),
            "cleanup fsync failure masked the primary adapter failure")
    _require_rolled_back(args, baseline, "cleanup directory fsync failure")


def _restrictive_umask_success(parent: Path) -> None:
    args, tools = _fixture(parent / "restrictive-umask")
    baseline = _open_fds()
    previous = os.umask(0o777)
    try:
        record = execute(args, _runner=_runner(args, "success"), _tool_paths=tools)
    finally:
        os.umask(previous)
    require(record["passed"] is True, "restrictive umask replay did not PASS")
    require(args.observed_prover_input.read_bytes() == EXPECTED,
            "restrictive umask replay changed output bytes")
    require(stat.S_IMODE(args.observed_prover_input.stat().st_mode) == 0o400,
            "restrictive umask replay did not publish mode 0400 output")
    require(stat.S_IMODE(args.record.stat().st_mode) == 0o400,
            "restrictive umask replay did not publish mode 0400 record")
    require(not [path for path in args.record.parent.iterdir() if path.name.startswith(".")],
            "restrictive umask replay left a hidden record temporary")
    require(_open_fds() == baseline, "restrictive umask replay leaked a descriptor")


def _long_record_leaf_success(parent: Path) -> None:
    args, tools = _fixture(parent / "long-record-leaf")
    require(os.pathconf(args.record.parent, "PC_NAME_MAX") >= 240,
            "host filename bound cannot exercise a 240-byte record leaf")
    args.record = args.record.with_name("r" * 240)
    baseline = _open_fds()
    record = execute(args, _runner=_runner(args, "success"), _tool_paths=tools)
    require(record["passed"] is True and args.record.exists(),
            "valid 240-byte record leaf did not PASS")
    require(stat.S_IMODE(args.record.stat().st_mode) == 0o400
            and stat.S_IMODE(args.observed_prover_input.stat().st_mode) == 0o400,
            "long record leaf did not publish mode 0400 artifacts")
    require(not [path for path in args.record.parent.iterdir() if path.name.startswith(".")],
            "long record leaf replay leaked a hidden temporary")
    require(_open_fds() == baseline, "long record leaf replay leaked a descriptor")


def pie_adapter_cleanup_self_test(lab_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix=".pie-adapter-cleanup-test-", dir=lab_root) as name:
        parent = Path(name).resolve()
        _fail_first_created_fstat(parent, record_temporary=False)
        _fail_first_created_fstat(parent, record_temporary=True)
        _output_fchmod_failure(parent)
        _record_close_after_publication(parent)
        _outer_output_close_after_commit(parent)
        _cleanup_directory_fsync(parent)
        _restrictive_umask_success(parent)
        _long_record_leaf_success(parent)


if __name__ == "__main__":
    pie_adapter_cleanup_self_test(LAB_ROOT)
    print("pie_adapter_cleanup_tests: PASS")
