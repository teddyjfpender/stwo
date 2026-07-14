"""Symlink-free, directory-fd-anchored filesystem operations.

Ancestor replacement is detected by full-chain rewalks. Conditional leaf cleanup is a
point-in-time check and assumes no concurrent writer inside the retained parent between
its anchored ``stat`` and ``unlink``; adapter records remain explicitly non-admitting.
"""

from __future__ import annotations

import fcntl
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .common import require


FileId = tuple[int, int]


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _require_capabilities() -> None:
    require(all(hasattr(os, name) for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")),
            "retained filesystem requires CLOEXEC, DIRECTORY, and NOFOLLOW flags")
    require(hasattr(fcntl, "F_DUPFD_CLOEXEC"),
            "retained filesystem requires CLOEXEC descriptor duplication")
    require(all(function in os.supports_dir_fd
                for function in (os.open, os.stat, os.link, os.unlink)),
            "retained filesystem requires openat/statat/linkat/unlinkat support")
    require(os.stat in os.supports_follow_symlinks and os.link in os.supports_follow_symlinks,
            "retained filesystem requires no-follow stat/link support")
    require(os.stat in os.supports_fd,
            "retained filesystem requires descriptor stat support")


def _file_id(metadata: os.stat_result) -> FileId:
    return (metadata.st_dev, metadata.st_ino)


def _retained_descriptor(descriptor: int) -> int:
    if descriptor >= 3:
        return descriptor
    duplicate = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 3)
    try:
        os.close(descriptor)
    except BaseException as primary:
        try:
            os.close(duplicate)
        except BaseException as cleanup:
            raise primary from cleanup
        raise
    return duplicate


def _attempt(first_error: BaseException | None,
             operation: Callable[[], object]) -> BaseException | None:
    try:
        operation()
    except BaseException as error:
        return first_error or error
    return first_error


def _require_leaf_name(name: str, label: str) -> None:
    require(isinstance(name, str) and name not in {"", ".", ".."}
            and Path(name).name == name and "/" not in name,
            f"{label} is not a single leaf name")


@dataclass(frozen=True)
class RetainedDirectory:
    """A root-to-directory walk whose every component remains open and pinned."""

    path: Path
    components: tuple[str, ...]
    descriptors: tuple[int, ...]
    identities: tuple[FileId, ...]

    @classmethod
    def bind(cls, path: Path, label: str) -> RetainedDirectory:
        _require_capabilities()
        require(isinstance(path, Path) and path.anchor == "/"
                and path == Path(os.path.abspath(path))
                and not ({".", ".."} & set(path.parts)),
                f"{label} directory path is not normalized absolute")
        components = path.relative_to("/").parts
        descriptors: list[int] = []
        identities: list[FileId] = []
        try:
            for component in ("/", *components):
                descriptor = (os.open(component, _directory_flags()) if not descriptors
                              else os.open(component, _directory_flags(),
                                           dir_fd=descriptors[-1]))
                descriptor = _retained_descriptor(descriptor)
                descriptors.append(descriptor)
                metadata = os.fstat(descriptor)
                require(stat.S_ISDIR(metadata.st_mode),
                        f"{label} path component is not a directory")
                require(fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC,
                        f"{label} retained directory descriptor is inheritable")
                require(descriptor >= 3, f"{label} retained a standard-I/O descriptor")
                identities.append(_file_id(metadata))
            result = cls(path, components, tuple(descriptors), tuple(identities))
            result.verify(label)
            return result
        except BaseException as primary:
            cleanup: BaseException | None = None
            for descriptor in reversed(descriptors):
                cleanup = _attempt(cleanup, lambda descriptor=descriptor: os.close(descriptor))
            if cleanup is not None:
                raise primary from cleanup
            raise

    @property
    def descriptor(self) -> int:
        return self.descriptors[-1]

    def verify(self, label: str) -> None:
        """Rewalk the textual path and match every retained directory identity."""
        current: int | None = None
        try:
            for index, component in enumerate(("/", *self.components)):
                following = (os.open(component, _directory_flags()) if current is None
                             else os.open(component, _directory_flags(), dir_fd=current))
                previous, current = current, following
                if previous is not None:
                    os.close(previous)
                observed = os.fstat(current)
                retained = os.fstat(self.descriptors[index])
                require(stat.S_ISDIR(observed.st_mode) and stat.S_ISDIR(retained.st_mode)
                        and _file_id(observed) == self.identities[index]
                        and _file_id(retained) == self.identities[index],
                        f"{label} directory chain was rebound")
        except BaseException as primary:
            if current is not None:
                try:
                    os.close(current)
                except BaseException as cleanup:
                    raise primary from cleanup
            raise
        else:
            if current is not None:
                os.close(current)

    def open(self, name: str, flags: int, label: str, mode: int = 0o777) -> int:
        _require_leaf_name(name, label)
        self.verify(label)
        descriptor = os.open(name, flags | os.O_CLOEXEC | os.O_NOFOLLOW, mode,
                             dir_fd=self.descriptor)
        try:
            self.verify(label)
            return descriptor
        except BaseException as primary:
            cleanup: BaseException | None = None
            if flags & os.O_CREAT and flags & os.O_EXCL:
                created: os.stat_result | None = None
                try:
                    created = os.fstat(descriptor)
                except BaseException as error:
                    cleanup = cleanup or error
                    try:
                        created = os.stat(descriptor)
                    except BaseException as fallback:
                        cleanup = cleanup or fallback
                if created is None:
                    def remove_owned_name() -> None:
                        try:
                            os.unlink(name, dir_fd=self.descriptor)
                        except FileNotFoundError:
                            pass
                    cleanup = _attempt(cleanup, remove_owned_name)
                else:
                    cleanup = _attempt(
                        cleanup, lambda: self.unlink_if_same(name, created)
                    )
                cleanup = _attempt(cleanup, self.sync)
            cleanup = _attempt(cleanup, lambda: os.close(descriptor))
            if cleanup is not None:
                raise primary from cleanup
            raise

    def stat(self, name: str) -> os.stat_result:
        _require_leaf_name(name, "retained filesystem stat name")
        return os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)

    def same(self, name: str, expected: os.stat_result) -> bool:
        try:
            return _file_id(self.stat(name)) == _file_id(expected)
        except FileNotFoundError:
            return False

    def link(self, source: str, destination: str) -> None:
        _require_leaf_name(source, "retained filesystem link source")
        _require_leaf_name(destination, "retained filesystem link destination")
        self.verify("retained filesystem link")
        os.link(source, destination, src_dir_fd=self.descriptor,
                dst_dir_fd=self.descriptor, follow_symlinks=False)
        self.verify("retained filesystem link")

    def unlink_if_same(self, name: str, expected: os.stat_result) -> bool:
        _require_leaf_name(name, "retained filesystem unlink name")
        if not self.same(name, expected):
            return False
        os.unlink(name, dir_fd=self.descriptor)
        return True

    def sync(self) -> None:
        os.fsync(self.descriptor)

    def close(self) -> None:
        first_error: BaseException | None = None
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except BaseException as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error


@dataclass(frozen=True)
class RetainedLeaf:
    """A leaf name anchored to a retained, symlink-free directory walk."""

    path: Path
    directory: RetainedDirectory
    label: str

    @classmethod
    def bind(cls, path: Path, label: str, *, fresh: bool = False) -> RetainedLeaf:
        require(isinstance(path, Path), f"{label} path is missing")
        absolute = Path(os.path.abspath(path))
        require(absolute.anchor == "/" and absolute.name not in {"", ".", ".."},
                f"{label} path is not a valid absolute file path")
        directory = RetainedDirectory.bind(absolute.parent, label)
        result = cls(absolute, directory, label)
        try:
            if fresh:
                try:
                    directory.stat(absolute.name)
                except FileNotFoundError:
                    pass
                else:
                    raise ValueError(f"{label} path is not fresh")
            directory.verify(label)
            return result
        except BaseException as primary:
            try:
                directory.close()
            except BaseException as cleanup:
                raise primary from cleanup
            raise

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def key(self) -> tuple[int, int, str]:
        parent = os.fstat(self.directory.descriptor)
        return (parent.st_dev, parent.st_ino, self.name)

    def open(self, flags: int, mode: int = 0o777) -> int:
        return self.directory.open(self.name, flags, self.label, mode)

    def stat(self) -> os.stat_result:
        return self.directory.stat(self.name)

    def same(self, expected: os.stat_result) -> bool:
        return self.directory.same(self.name, expected)

    def unlink_if_same(self, expected: os.stat_result) -> bool:
        return self.directory.unlink_if_same(self.name, expected)

    def verify(self) -> None:
        self.directory.verify(self.label)

    def close(self) -> None:
        self.directory.close()
