"""Canonical source-tree staging for path-stable FRI cubins with line information."""

from __future__ import annotations

import fcntl
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .common import canonical_bytes, require, require_sha256, sha256_bytes, sha256_file
from .immutable_output import write_immutable_bytes


STAGE_ROOT = Path("/tmp/stwo-gpu-lab-fri-v1").resolve()
STAGE_POLICY = "private-content-addressed-source-tree-v1"


def _private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    metadata = path.lstat()
    require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)
            and metadata.st_uid == os.geteuid() and stat.S_IMODE(metadata.st_mode) == 0o700,
            f"canonical FRI staging directory is foreign or unsafe: {path}")


def _private_tree(path: Path) -> None:
    require(path == STAGE_ROOT or path.is_relative_to(STAGE_ROOT),
            "canonical FRI staging path escapes its private root")
    missing: list[Path] = []
    cursor = path
    while cursor != STAGE_ROOT and not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    _private_directory(STAGE_ROOT)
    for directory in reversed(missing):
        _private_directory(directory)
    cursor = path
    while cursor != STAGE_ROOT:
        _private_directory(cursor)
        cursor = cursor.parent


def _require_private_tree(path: Path) -> None:
    require(path.is_relative_to(STAGE_ROOT), "canonical FRI stage uses a foreign path")
    cursor = path
    while True:
        require(cursor.is_dir(), f"canonical FRI staging directory is missing: {cursor}")
        metadata = cursor.lstat()
        require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)
                and metadata.st_uid == os.geteuid() and stat.S_IMODE(metadata.st_mode) == 0o700,
                f"canonical FRI staging directory is foreign or unsafe: {cursor}")
        if cursor == STAGE_ROOT:
            return
        cursor = cursor.parent


@contextmanager
def _stage_lock(closure_sha256: str):
    _private_directory(STAGE_ROOT)
    lock_path = STAGE_ROOT / f".{closure_sha256}.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.geteuid()
                and stat.S_IMODE(metadata.st_mode) == 0o600 and metadata.st_nlink == 1,
                "canonical FRI staging lock is foreign or unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def staging_identity(closure_sha256: str) -> dict[str, str]:
    require_sha256(closure_sha256, "FRI closure sha256")
    working_directory = STAGE_ROOT / closure_sha256 / "repo"
    return {
        "policy": STAGE_POLICY,
        "closure_sha256": closure_sha256,
        "working_directory": str(working_directory),
        "source_mode": "0400",
        "source_mtime_ns": "0",
    }


def stage_repository(closure: list[dict[str, str]], repo_root: Path,
                     closure_sha256: str) -> Path:
    """Install repository members of a hashed nvcc closure at one canonical path."""
    require(closure_sha256 == sha256_bytes(canonical_bytes(closure)),
            "FRI staging key does not bind the exact source closure")
    working_directory = Path(staging_identity(closure_sha256)["working_directory"])
    with _stage_lock(closure_sha256):
        if working_directory.parent.exists() or working_directory.parent.is_symlink():
            validate_staging(closure, working_directory)
            return working_directory
        temporary: Path | None = Path(
            tempfile.mkdtemp(prefix=f".{closure_sha256}.", dir=STAGE_ROOT)
        )
        try:
            _private_directory(temporary)
            temporary_working = temporary / "repo"
            _private_tree(temporary_working)
            _install_sources(closure, repo_root, temporary_working)
            _normalize_directories(closure, temporary_working)
            validate_staging_at(closure, temporary_working)
            os.rename(temporary, working_directory.parent)
            temporary = None
            directory = os.open(STAGE_ROOT, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)
        validate_staging(closure, working_directory)
        return working_directory


def _install_sources(closure: list[dict[str, str]], repo_root: Path,
                     working_directory: Path) -> None:
    for identity in closure:
        if identity["scope"] != "repository":
            continue
        source = (repo_root / identity["path"]).resolve()
        require(source.is_relative_to(repo_root) and source.is_file(),
                "FRI staged source escapes/is missing")
        require(sha256_file(source) == identity["sha256"],
                "FRI source changed before canonical staging")
        require(not Path(identity["path"]).is_absolute(), "FRI staged source path is absolute")
        destination = (working_directory / identity["path"]).resolve(strict=False)
        require(destination.is_relative_to(working_directory), "FRI staged source path escapes")
        _private_tree(destination.parent)
        write_immutable_bytes(destination, source.read_bytes(), [source])
        os.chmod(destination, 0o400, follow_symlinks=False)
        os.utime(destination, ns=(0, 0), follow_symlinks=False)


def _normalize_directories(closure: list[dict[str, str]], working_directory: Path) -> None:
    directories = {working_directory, working_directory.parent}
    for identity in closure:
        if identity["scope"] != "repository":
            continue
        cursor = (working_directory / identity["path"]).parent
        while cursor != working_directory.parent:
            directories.add(cursor)
            cursor = cursor.parent
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        os.utime(directory, ns=(0, 0), follow_symlinks=False)


def validate_staging(closure: list[dict[str, str]], working_directory: Path) -> None:
    closure_sha256 = sha256_bytes(canonical_bytes(closure))
    expected = Path(staging_identity(closure_sha256)["working_directory"])
    require(working_directory == expected,
            "canonical FRI stage path does not bind the exact source closure")
    validate_staging_at(closure, working_directory)


def validate_staging_at(closure: list[dict[str, str]], working_directory: Path) -> None:
    """Validate a private candidate tree before its atomic canonical rename."""
    _require_private_tree(working_directory)
    expected_files = {
        working_directory / identity["path"]
        for identity in closure if identity["scope"] == "repository"
    }
    expected_directories = {working_directory}
    for path in expected_files:
        cursor = path.parent
        while cursor != working_directory.parent:
            expected_directories.add(cursor)
            cursor = cursor.parent
    actual_files: set[Path] = set()
    actual_directories = {working_directory}
    for path in working_directory.rglob("*"):
        require(not path.is_symlink(), f"canonical FRI stage contains a symlink: {path}")
        if path.is_dir():
            metadata = path.stat()
            require(metadata.st_uid == os.geteuid() and stat.S_IMODE(metadata.st_mode) == 0o700,
                    f"canonical FRI staged directory is foreign or unsafe: {path}")
            actual_directories.add(path)
        else:
            actual_files.add(path)
    require(actual_files == expected_files and actual_directories == expected_directories,
            "canonical FRI stage differs from the exact repository closure")
    require(set(working_directory.parent.iterdir()) == {working_directory},
            "canonical FRI content directory contains a foreign entry")
    require(all(path.stat().st_mtime_ns == 0
                for path in expected_directories | {working_directory.parent}),
            "canonical FRI staging directory timestamp changed")
    for identity in closure:
        if identity["scope"] != "repository":
            continue
        staged = working_directory / identity["path"]
        metadata = staged.lstat()
        require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)
                and metadata.st_uid == os.geteuid() and stat.S_IMODE(metadata.st_mode) == 0o400
                and metadata.st_mtime_ns == 0 and sha256_file(staged) == identity["sha256"],
                f"canonical FRI staged source changed: {staged}")


def original_dependencies(paths: list[Path], closure: list[dict[str, str]],
                          working_directory: Path, repo_root: Path) -> list[Path]:
    """Map compile-depfile staging paths back to the reviewed repository paths."""
    closure_sha256 = sha256_bytes(canonical_bytes(closure))
    expected = Path(staging_identity(closure_sha256)["working_directory"])
    require(working_directory == expected,
            "FRI dependency mapping uses a foreign staging directory")
    original: list[Path] = []
    for path in paths:
        if path.is_relative_to(working_directory):
            relative = path.relative_to(working_directory)
            mapped = (repo_root / relative).resolve()
            require(mapped.is_relative_to(repo_root), "staged FRI dependency escapes repository")
            original.append(mapped)
        else:
            original.append(path)
    require(len(original) == len(set(original)), "mapped FRI dependency closure has duplicates")
    return original
