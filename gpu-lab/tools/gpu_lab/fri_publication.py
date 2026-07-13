"""Fail-closed publication primitives for the exact FRI module index."""

from __future__ import annotations

import fcntl
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .common import (
    canonical_bytes,
    load_json,
    require,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
    sha256_file,
)
from .immutable_output import write_immutable_bytes


LOCATOR_SCHEMA = "stwo.gpu-lab.fri-round6-module-locator.non-evidence.v1"
LOCATOR_POLICY = "non-evidence-locator-only"
MODULE_NAME = "fri_round6"


@contextmanager
def output_index_lock(index_path: Path):
    """Serialize one module index through publication and final validation."""
    require(index_path.is_absolute(), "FRI module index lock path is not absolute")
    require(index_path.parent.is_dir() and not index_path.parent.is_symlink(),
            "FRI module index directory is missing or unsafe")
    lock_path = index_path.with_name(f".{index_path.name}.lock")
    descriptor = os.open(
        lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.geteuid()
                and stat.S_IMODE(metadata.st_mode) == 0o600 and metadata.st_nlink == 1,
                "FRI module index lock is foreign or unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        installed = lock_path.lstat()
        require(installed.st_dev == metadata.st_dev and installed.st_ino == metadata.st_ino,
                "FRI module index lock was substituted")
        try:
            yield
        finally:
            final = lock_path.lstat()
            require(final.st_dev == metadata.st_dev and final.st_ino == metadata.st_ino
                    and stat.S_ISREG(final.st_mode) and final.st_uid == os.geteuid()
                    and stat.S_IMODE(final.st_mode) == 0o600 and final.st_nlink == 1,
                    "FRI module index lock changed while held")
    finally:
        os.close(descriptor)


def atomic_text(path: Path, text: str) -> None:
    require(not path.is_symlink(), f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text() == text:
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def make_escape(path: Path) -> str:
    value = str(path)
    return (value.replace("\\", "\\\\").replace("$", "$$").replace("#", "\\#")
            .replace(" ", "\\ ").replace(":", "\\:"))


def write_depfile(path: Path, stamp: Path, sources: list[Path]) -> None:
    dependencies = " \\\n  ".join(make_escape(source) for source in sources)
    atomic_text(path, f"{make_escape(stamp)}: {dependencies}\n")


def require_exact_index(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    require(canonical_bytes(actual) == canonical_bytes(expected),
            "a concurrent FRI build installed a different module index")


def locator_document(index_sha256: str, sm: int) -> dict[str, Any]:
    require_sha256(index_sha256, "FRI immutable index sha256")
    return {
        "schema_version": LOCATOR_SCHEMA,
        "evidence_policy": LOCATOR_POLICY,
        "module": MODULE_NAME,
        "target_sm": sm,
        "index_content_sha256": index_sha256,
        "index_path": f"{index_sha256}.module-index.json",
    }


def install_recipe(output_dir: Path, recipe: dict[str, Any], recipe_hash: str,
                   module_hash: str, module: Path) -> Path:
    recipe_path = output_dir / f"{recipe_hash}.recipe.json"
    write_immutable_bytes(recipe_path, canonical_bytes(recipe), [module])
    write_immutable_bytes(output_dir / f"{recipe_hash}.module-sha256",
                          (module_hash + "\n").encode(), [module])
    return recipe_path


def install_immutable_index(output_dir: Path, index: dict[str, Any]) -> tuple[Path, str]:
    payload = canonical_bytes(index)
    digest = sha256_bytes(payload)
    path = output_dir / f"{digest}.module-index.json"
    sources = [
        Path(index["module_path"]), Path(index["build_recipe_path"]),
        output_dir / f"{index['build_recipe_hash']}.module-sha256",
    ]
    write_immutable_bytes(path, payload, sources)
    require(path.read_bytes() == payload and sha256_file(path) == digest,
            "immutable FRI module index content differs")
    return path, digest


def validate_locator_document(locator: Any, output_dir: Path,
                              sm: int) -> tuple[Path, dict[str, Any]]:
    require(isinstance(locator, dict), "FRI non-evidence locator must be an object")
    require_exact_keys(locator, {
        "schema_version", "evidence_policy", "module", "target_sm",
        "index_content_sha256", "index_path",
    }, "FRI non-evidence locator")
    require(locator["schema_version"] == LOCATOR_SCHEMA
            and locator["evidence_policy"] == LOCATOR_POLICY
            and locator["module"] == MODULE_NAME and locator["target_sm"] == sm,
            "unsupported FRI non-evidence locator")
    digest = locator["index_content_sha256"]
    require_sha256(digest, "FRI immutable index sha256")
    relative = Path(locator["index_path"])
    require(not relative.is_absolute() and relative == Path(f"{digest}.module-index.json"),
            "FRI locator does not name its exact sibling immutable index")
    path = output_dir / relative
    require(path.is_file() and not path.is_symlink(), "immutable FRI module index is missing")
    payload = path.read_bytes()
    require(sha256_bytes(payload) == digest, "immutable FRI module index hash differs")
    index = load_json(path)
    require(payload == canonical_bytes(index), "immutable FRI module index is non-canonical")
    return path, index
