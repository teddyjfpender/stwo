"""Exact nvcc dependency discovery and source-closure validation for FRI."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .common import require, require_exact_keys, require_sha256, sha256_file


SOURCE_RELATIVE = "gpu-lab/kernels/fri_round6.cu"
CUDA_ROOT_RELATIVE = "crates/backend-cuda-kernels/cuda"
REQUIRED_REPOSITORY_SOURCES = {
    SOURCE_RELATIVE,
    f"{CUDA_ROOT_RELATIVE}/fields.cu",
    f"{CUDA_ROOT_RELATIVE}/fold_line.cu",
    f"{CUDA_ROOT_RELATIVE}/blake2s.cu",
    f"{CUDA_ROOT_RELATIVE}/device_transcript.cu",
}


def parse_nvcc_depfile(text: str, cwd: Path, expected_target: Path | None = None) -> list[Path]:
    require(text and "\0" not in text, "nvcc dependency file is empty or contains NUL")
    logical = re.sub(r"\\\r?\n", " ", text)
    lines = [line.strip() for line in logical.splitlines() if line.strip()]
    require(len(lines) == 1 and ":" in lines[0], "nvcc dependency file must contain one rule")
    target_text, dependency_text = lines[0].split(":", 1)
    targets = shlex.split(target_text, comments=False, posix=True)
    dependencies = shlex.split(dependency_text, comments=False, posix=True)
    require(len(targets) == 1 and dependencies, "nvcc dependency rule is malformed")
    target = Path(targets[0])
    target = (cwd / target).resolve() if not target.is_absolute() else target.resolve()
    if expected_target is not None:
        require(target == expected_target.resolve(), "nvcc dependency target differs from cubin")
    resolved = [((cwd / Path(item)).resolve() if not Path(item).is_absolute()
                 else Path(item).resolve()) for item in dependencies]
    require(len(resolved) == len(set(resolved)), "nvcc dependency closure contains duplicates")
    require(all(path.is_file() and not path.is_symlink() for path in resolved),
            "nvcc dependency closure contains a missing/non-regular file")
    return resolved


def _source_identity(path: Path, repo_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if resolved.is_relative_to(repo_root):
        return {"scope": "repository", "path": str(resolved.relative_to(repo_root)),
                "sha256": sha256_file(resolved)}
    return {"scope": "toolchain", "path": str(resolved), "sha256": sha256_file(resolved)}


def source_closure(paths: list[Path], repo_root: Path) -> list[dict[str, str]]:
    identities = sorted((_source_identity(path, repo_root) for path in paths),
                        key=lambda item: (item["scope"], item["path"]))
    repository_sources = {item["path"] for item in identities if item["scope"] == "repository"}
    require(REQUIRED_REPOSITORY_SOURCES <= repository_sources,
            "nvcc closure omits a required FRI production source")
    return identities


def discovery_command(toolchain: dict[str, Any], sm: int, depfile: Path) -> list[str]:
    return [
        toolchain["nvcc_path"], "-M", "-O3", "--std=c++17", "--expt-relaxed-constexpr",
        "-lineinfo", f"-arch=sm_{sm}", "-ccbin", toolchain["host_compiler_path"],
        "-I", CUDA_ROOT_RELATIVE, "-MF", str(depfile), SOURCE_RELATIVE,
    ]


def discover_source_closure(toolchain: dict[str, Any], sm: int, repo_root: Path,
                            depfile: Path) -> tuple[list[Path], list[dict[str, str]]]:
    subprocess.run(discovery_command(toolchain, sm, depfile), cwd=repo_root, check=True)
    paths = parse_nvcc_depfile(depfile.read_text(), repo_root)
    return paths, source_closure(paths, repo_root)


def require_sealed_closure(sealed: list[dict[str, str]],
                           discovered: list[dict[str, str]]) -> None:
    require(discovered == sealed, "discovered FRI source closure differs from sealed recipe")


def _identity_path(identity: dict[str, str], repo_root: Path) -> Path:
    if identity["scope"] == "repository":
        path = (repo_root / identity["path"]).resolve()
        require(path.is_relative_to(repo_root), "FRI source identity escapes repository")
        return path
    require(identity["scope"] == "toolchain" and Path(identity["path"]).is_absolute(),
            "FRI external source identity is invalid")
    return Path(identity["path"])


def validate_closure(closure: Any, repo_root: Path, *, check_bytes: bool) -> list[Path]:
    require(isinstance(closure, list) and closure, "FRI source closure is missing")
    order: list[tuple[str, str]] = []
    paths: list[Path] = []
    for identity in closure:
        require(isinstance(identity, dict), "FRI source identity must be an object")
        require_exact_keys(identity, {"scope", "path", "sha256"}, "FRI source identity")
        require(isinstance(identity["path"], str) and identity["path"],
                "FRI source identity path is empty")
        require_sha256(identity["sha256"], "FRI source sha256")
        order.append((identity["scope"], identity["path"]))
        path = _identity_path(identity, repo_root)
        if check_bytes:
            require(path.is_file() and not path.is_symlink(), f"FRI source is missing: {path}")
            require(sha256_file(path) == identity["sha256"], f"FRI source changed: {path}")
        paths.append(path)
    require(order == sorted(order) and len(order) == len(set(order)),
            "FRI source closure order/uniqueness differs")
    repository_sources = {path for scope, path in order if scope == "repository"}
    require(REQUIRED_REPOSITORY_SOURCES <= repository_sources,
            "FRI source closure omits a required production source")
    return paths
