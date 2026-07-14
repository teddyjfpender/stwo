"""Fail-closed installation for generated artifacts with immutable inputs."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Callable, Iterable

from .common import require

FileIdentity = tuple[int, int, int, int, int]


def _identity(path: Path) -> FileIdentity | None:
    if path.is_symlink():
        raise ValueError(f"output path is a symlink: {path}")
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    require(path.is_file(), f"output path is not a regular file: {path}")
    return (status.st_dev, status.st_ino, status.st_size,
            status.st_mtime_ns, status.st_ctime_ns)


def guard_output(output: Path, sources: Iterable[Path]) -> FileIdentity | None:
    """Reject canonical-path and inode aliases before a generator writes anything."""
    source_paths = list(sources)
    output_identity = _identity(output)
    output_canonical = output.resolve(strict=False)
    for source in source_paths:
        require(output_canonical != source.resolve(strict=True),
                f"output aliases immutable source: {source}")
        if output_identity is not None:
            require(not output.samefile(source),
                    f"output hardlinks immutable source: {source}")
    return output_identity


def _equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as lhs, right.open("rb") as rhs:
        while left_block := lhs.read(1024 * 1024):
            if left_block != rhs.read(len(left_block)):
                return False
        return not rhs.read(1)


def _read_descriptor(descriptor: int, length: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < length:
        try:
            chunk = os.pread(descriptor, min(1024 * 1024, length - offset), offset)
        except InterruptedError:
            continue
        require(chunk != b"", "immutable output descriptor ended early")
        chunks.append(chunk)
        offset += len(chunk)
    require(os.pread(descriptor, 1, length) == b"", "immutable output descriptor grew")
    return b"".join(chunks)


def _same_inode(path: Path, status: os.stat_result) -> bool:
    try:
        candidate = path.lstat()
    except FileNotFoundError:
        return False
    return candidate.st_dev == status.st_dev and candidate.st_ino == status.st_ino


def _verify_installed(path: Path, status: os.stat_result, mode: int, length: int) -> None:
    installed = path.lstat()
    require(
        stat.S_ISREG(installed.st_mode)
        and installed.st_dev == status.st_dev
        and installed.st_ino == status.st_ino
        and installed.st_size == length
        and stat.S_IMODE(installed.st_mode) == mode,
        "published immutable output differs from its written inode",
    )


def _sync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _existing_matches(output: Path, prior: FileIdentity, payload: bytes, mode: int) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(output, flags)
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode),
            "existing immutable output is not a regular file",
        )
        require(
            stat.S_IMODE(before.st_mode) == mode and before.st_nlink == 1,
            "legacy/mutable output is not mode 0400 with one link; regenerate at a fresh path",
        )
        require(
            (before.st_dev, before.st_ino, before.st_size,
                 before.st_mtime_ns, before.st_ctime_ns) == prior,
            "existing immutable output identity changed before comparison",
        )
        matches = _read_descriptor(descriptor, before.st_size) == payload
        after = os.fstat(descriptor)
        require(
            (after.st_dev, after.st_ino, after.st_size,
             after.st_mtime_ns, after.st_ctime_ns) == prior,
            "existing immutable output changed during comparison",
        )
    finally:
        os.close(descriptor)
    require(_identity(output) == prior, "existing immutable output path was rebound")
    return matches


def install_output(
    temporary: Path, output: Path, sources: Iterable[Path], prior: FileIdentity | None,
) -> bool:
    """Legacy path-based install; callers own stronger evidence-identity checks."""
    current = guard_output(output, sources)
    require(current == prior, "output identity changed while the artifact was generated")
    if current is not None:
        require(_equal(temporary, output),
                "existing immutable output differs from the generated artifact")
        require(guard_output(output, sources) == current,
                "output identity changed while its bytes were compared")
        temporary.unlink()
        return False
    try:
        os.link(temporary, output, follow_symlinks=False)
    except FileExistsError as error:
        raise ValueError("output appeared while the artifact was generated") from error
    temporary.unlink()
    return True


def write_immutable_bytes(
    output: Path, payload: bytes, sources: Iterable[Path], *, fresh_only: bool = False,
    _before_publish: Callable[[Path], None] | None = None,
) -> None:
    """Durably link the exact written inode once; never replace a published path."""
    source_paths = list(sources)
    prior = guard_output(output, source_paths)
    require(not fresh_only or prior is None, f"fresh output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        created = os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    written: os.stat_result | None = None
    published = False
    try:
        mode = 0o400
        os.fchmod(descriptor, mode)
        remaining = memoryview(payload)
        while remaining:
            try:
                count = os.write(descriptor, remaining)
            except InterruptedError:
                continue
            require(count > 0, "cannot write complete immutable output")
            remaining = remaining[count:]
        os.fsync(descriptor)
        written = os.fstat(descriptor)
        require(
            stat.S_ISREG(written.st_mode)
            and stat.S_IMODE(written.st_mode) == mode
            and written.st_size == len(payload)
            and written.st_nlink == 1
            and _read_descriptor(descriptor, len(payload)) == payload,
            "immutable output temporary identity or content differs",
        )
        if prior is not None:
            require(_existing_matches(output, prior, payload, mode),
                    "existing immutable output differs from generated bytes")
            require(_same_inode(temporary, written),
                    "immutable output temporary was substituted")
            temporary.unlink()
            return
        if _before_publish is not None:
            _before_publish(temporary)
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as error:
            raise ValueError("output appeared while the artifact was generated") from error
        published = True
        _verify_installed(output, written, mode, len(payload))
        require(_same_inode(temporary, written), "immutable output temporary was substituted")
        temporary.unlink()
        linked = os.fstat(descriptor)
        require(linked.st_dev == written.st_dev and linked.st_ino == written.st_ino
                and linked.st_nlink == 1,
                "published immutable output link count differs")
        _sync_directory(output.parent)
        _verify_installed(output, written, mode, len(payload))
        require(_read_descriptor(descriptor, len(payload)) == payload,
                "published immutable output content changed")
    except BaseException:
        rollback_error: OSError | None = None
        if published and _same_inode(output, created):
            try:
                output.unlink()
                _sync_directory(output.parent)
            except OSError as error:
                rollback_error = error
        if _same_inode(temporary, created):
            temporary.unlink()
        if rollback_error is not None:
            raise ValueError(
                "failed to durably roll back rejected immutable output"
            ) from rollback_error
        raise
    finally:
        os.close(descriptor)
