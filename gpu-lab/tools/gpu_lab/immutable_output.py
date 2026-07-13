"""Fail-closed installation for generated artifacts with immutable inputs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

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


def install_output(
    temporary: Path, output: Path, sources: Iterable[Path], prior: FileIdentity | None,
) -> bool:
    """Install once, or accept an existing byte-identical immutable artifact."""
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


def write_immutable_bytes(output: Path, payload: bytes, sources: Iterable[Path]) -> None:
    """Durably create a byte artifact once; never replace differing evidence."""
    source_paths = list(sources)
    prior = guard_output(output, source_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        if install_output(temporary, output, source_paths, prior):
            directory = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
