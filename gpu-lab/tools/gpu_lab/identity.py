"""Immutable CUDA toolchain, ABI, generator, recipe, and cubin identity."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .common import (
    FIRST_SLICE_ABI_SHA256,
    FIRST_SLICE_GENERATORS,
    KERNEL_ENTRY_SCHEMA,
    canonical_bytes,
    lab_tool_sha256,
    load_json,
    require,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
    sha256_file,
)


CANONICAL_STAGE_ROOT = Path("/tmp/stwo-gpu-lab-aot-v1")
CANONICAL_STAGE_SOURCE = "source.cu"


def canonical_staging_identity(source_sha256: str) -> dict[str, Any]:
    return {
        "policy": "private-content-addressed-canonical-path-v1",
        "root": str(CANONICAL_STAGE_ROOT),
        "directory": source_sha256,
        "source_name": CANONICAL_STAGE_SOURCE,
        "mtime_ns": 0,
    }


def _require_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    metadata = path.lstat()
    require(
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and stat.S_IMODE(metadata.st_mode) == 0o700,
        f"canonical CUDA staging directory is foreign or unsafe: {path}",
    )


def stage_source(source: Path, source_sha256: str) -> Path:
    """Install source at a path/mtime that cannot leak a workspace mount root."""
    _require_private_directory(CANONICAL_STAGE_ROOT)
    stage_dir = CANONICAL_STAGE_ROOT / source_sha256
    _require_private_directory(stage_dir)
    staged = stage_dir / CANONICAL_STAGE_SOURCE
    source_bytes = source.read_bytes()
    require(sha256_bytes(source_bytes) == source_sha256,
            "CUDA source changed while entering canonical staging")

    descriptor, temporary_name = tempfile.mkstemp(prefix=".candidate-", dir=CANONICAL_STAGE_ROOT)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(source_bytes)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o400)
        os.utime(temporary, ns=(0, 0), follow_symlinks=False)
        try:
            os.link(temporary, staged, follow_symlinks=False)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)

    os.utime(staged, ns=(0, 0), follow_symlinks=False)
    os.utime(stage_dir, ns=(0, 0), follow_symlinks=False)
    validate_staged_source(staged, source_sha256)
    return staged


def validate_staged_source(staged: Path, source_sha256: str) -> None:
    expected = CANONICAL_STAGE_ROOT / source_sha256 / CANONICAL_STAGE_SOURCE
    require(staged == expected, "CUDA staged source path differs from canonical policy")
    _require_private_directory(CANONICAL_STAGE_ROOT)
    _require_private_directory(staged.parent)
    metadata = staged.lstat()
    require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and stat.S_IMODE(metadata.st_mode) == 0o400
        and metadata.st_mtime_ns == 0
        and sha256_file(staged) == source_sha256,
        "canonical CUDA staged source is foreign or has changed",
    )
    require(staged.parent.lstat().st_mtime_ns == 0,
            "canonical CUDA staging directory mtime changed")
    require(list(staged.parent.iterdir()) == [staged],
            "canonical CUDA staging directory contains a foreign file")


def workspace_source_identity(path: Path, workspace_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    for repository in ("stwo", "stwo-cairo"):
        repository_root = (workspace_root / repository).resolve()
        if resolved.is_relative_to(repository_root):
            return {
                "repository": repository,
                "path": str(resolved.relative_to(repository_root)),
                "sha256": sha256_file(resolved),
            }
    raise ValueError(f"generator identity escapes the reviewed workspace repositories: {resolved}")


def nvcc_version(nvcc: str) -> str:
    result = subprocess.run([nvcc, "--version"], text=True, capture_output=True, check=True)
    return result.stdout.strip()


def tool_version(executable: str) -> str:
    result = subprocess.run([executable, "--version"], text=True, capture_output=True, check=True)
    return (result.stdout + result.stderr).strip()


def toolchain_identity(nvcc: str, host_compiler: str) -> dict[str, Any]:
    require(not os.environ.get("NVCC_PREPEND_FLAGS") and not os.environ.get("NVCC_APPEND_FLAGS"),
            "NVCC_PREPEND_FLAGS/NVCC_APPEND_FLAGS are forbidden; flags must be explicit")
    nvcc_path = str(Path(nvcc).resolve())
    host_path = str(Path(host_compiler).resolve())
    ptxas_path = str(Path(nvcc_path).parent / "ptxas")
    cuobjdump_path = str(Path(nvcc_path).parent / "cuobjdump")
    require(all(Path(path).is_file() for path in
                (nvcc_path, host_path, ptxas_path, cuobjdump_path)),
            "nvcc, ptxas, cuobjdump, or host compiler is missing")
    return {
        "nvcc_path": nvcc_path,
        "nvcc_version": nvcc_version(nvcc_path),
        "ptxas_path": ptxas_path,
        "ptxas_version": tool_version(ptxas_path),
        "cuobjdump_path": cuobjdump_path,
        "cuobjdump_version": tool_version(cuobjdump_path),
        "host_compiler_path": host_path,
        "host_compiler_version": tool_version(host_path),
        "environment": {"NVCC_PREPEND_FLAGS": "", "NVCC_APPEND_FLAGS": ""},
        "driver_compatibility_policy": "CUDA enhanced-compatibility; execution binds actual driver",
    }


def write_toolchain_identity(args: argparse.Namespace) -> None:
    encoded = json.dumps(toolchain_identity(args.nvcc, args.host_compiler),
                         indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != encoded:
        temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
        temporary.write_text(encoded)
        os.replace(temporary, args.output)


def validate_abi(abi: dict[str, Any], abi_path: Path, repo_root: Path) -> None:
    require(abi.get("schema_version") == KERNEL_ENTRY_SCHEMA, "unsupported kernel ABI schema")
    require_exact_keys(
        abi,
        {"schema_version", "semantic_operation", "source", "cache_key", "semantic_ir_hash",
         "symbol", "abi_version", "arguments", "launch", "module_globals", "effects"},
        "kernel ABI",
    )
    require(sha256_file(abi_path) == FIRST_SLICE_ABI_SHA256,
            "first-slice ABI differs from the reviewed manifest")
    require(isinstance(abi["cache_key"], str) and len(abi["cache_key"]) == 16,
            "cache_key must be a 64-bit lowercase hex string")
    require(isinstance(abi["semantic_ir_hash"], str) and len(abi["semantic_ir_hash"]) == 16,
            "semantic_ir_hash must be a 64-bit lowercase hex string")
    require(all(character in "0123456789abcdef"
                for character in abi["cache_key"] + abi["semantic_ir_hash"]),
            "kernel 64-bit hashes must be lowercase hex")
    source = (repo_root / abi["source"]).resolve()
    require(source.is_file() and source.is_relative_to(repo_root), "ABI source escapes/is missing")
    require(
        re.search(r"^\s*#\s*include(?:_next)?\b", source.read_text(), re.MULTILINE) is None,
        "self-contained first-slice source gained an unbound transitive header",
    )
    require(source.stem.endswith("_" + abi["cache_key"]), "cache key does not match source")
    require(abi["symbol"].endswith("_" + abi["semantic_ir_hash"]),
            "semantic IR hash does not match exported symbol")
    require(isinstance(abi["arguments"], list) and len(abi["arguments"]) == 8,
            "first witness ABI must have eight arguments")
    require(abi["module_globals"] == [], "first slice must not require module globals")
    require(abi["launch"] == {
        "grid_x": "ceil_div(row_count,256)",
        "grid_y": 1,
        "grid_z": 1,
        "block": [256, 1, 1],
        "dynamic_shared_bytes": 0,
        "cooperative": False,
        "graph_capture": True,
        "graph_parameter_policy": "row_count_change_requires_recapture",
        "parameter_buffer_size_bytes": 60,
        "launch_bounds": {"max_threads_per_block": 256, "min_blocks_per_sm": None},
        "required_function_attributes": {
            "max_dynamic_shared_bytes": 0,
            "preferred_shared_memory_carveout": None,
        },
        "supported_shape": {
            "field": "M31",
            "row_count_min": 1,
            "row_count_max": 4_294_967_040,
            "input_columns": 3,
            "output_columns": 3,
            "table_pointer_slots": 37,
        },
    }, "first-slice launch contract changed")


def validate_generator_sources(generators: Any, workspace_root: Path) -> None:
    require(isinstance(generators, list) and generators, "generator identity is missing")
    require([{key: generator.get(key) for key in ("repository", "path")}
             for generator in generators] == FIRST_SLICE_GENERATORS,
            "first-slice generator identity set/order differs")
    for generator in generators:
        require_exact_keys(generator, {"repository", "path", "sha256"}, "generator identity")
        repository_root = (workspace_root / generator["repository"]).resolve()
        generator_path = (repository_root / generator["path"]).resolve()
        require(generator_path.is_file() and generator_path.is_relative_to(repository_root),
                "generator path escapes/is missing")
        require(generator["sha256"] == sha256_file(generator_path),
                "generator source hash mismatch")


def validate_module_index(module: dict[str, Any], abi: dict[str, Any], abi_path: Path,
                          repo_root: Path) -> Path:
    require_exact_keys(
        module,
        {"schema_version", "build_recipe_hash", "build_recipe", "module_content_sha256",
         "module_path", "target_sm", "abi_path"},
        "module index",
    )
    require(module.get("schema_version") == "stwo.gpu-lab.module-index.v1", "bad module index")
    require(isinstance(module.get("target_sm"), int) and not isinstance(module["target_sm"], bool)
            and 50 <= module["target_sm"] <= 999, "module target SM is invalid")
    recipe = module.get("build_recipe")
    require(isinstance(recipe, dict), "module build_recipe must be an object")
    require_exact_keys(
        recipe,
        {"schema_version", "source_sha256", "transitive_headers", "abi_sha256", "cache_key",
         "semantic_ir_hash", "exported_symbols", "generator_sources", "build_tool_sha256",
         "nvcc_path", "nvcc_version", "ptxas_path", "ptxas_version", "cuobjdump_path",
         "cuobjdump_version", "host_compiler_path", "host_compiler_version", "environment",
         "driver_compatibility_policy", "source_staging", "target_sm", "normalized_command"},
        "build recipe",
    )
    require(recipe["schema_version"] == "stwo.gpu-lab.build-recipe.v2",
            "unsupported build recipe")
    require(recipe["transitive_headers"] == [],
            "self-contained first-slice source unexpectedly declares headers")
    require_sha256(module.get("build_recipe_hash"), "build_recipe_hash")
    require(module["build_recipe_hash"] == sha256_bytes(canonical_bytes(recipe)),
            "build_recipe_hash does not bind build_recipe")
    require_sha256(module.get("module_content_sha256"), "module_content_sha256")
    require(Path(module.get("abi_path", "")).resolve() == abi_path,
            "module index points at another ABI manifest")
    require(recipe.get("abi_sha256") == sha256_file(abi_path),
            "module recipe is bound to another ABI")
    require(recipe.get("cache_key") == abi["cache_key"], "module recipe cache key mismatch")
    require(recipe.get("semantic_ir_hash") == abi["semantic_ir_hash"],
            "module recipe semantic hash mismatch")
    require(recipe.get("exported_symbols") == [abi["symbol"]], "module recipe symbol mismatch")
    require(recipe.get("target_sm") == module.get("target_sm"), "module target mismatch")
    source = (repo_root / abi["source"]).resolve()
    require(recipe.get("source_sha256") == sha256_file(source), "module source hash mismatch")
    require(recipe.get("build_tool_sha256") == lab_tool_sha256(),
            "module was built by another lab tool revision")
    require(
        recipe.get("source_staging")
        == canonical_staging_identity(recipe["source_sha256"]),
        "module recipe uses another CUDA source-staging policy",
    )
    validate_generator_sources(recipe.get("generator_sources"), repo_root.parent)
    host_compiler = recipe.get("host_compiler_path")
    expected_command = ["nvcc", "-cubin", "-O3", "--std=c++17", "--expt-relaxed-constexpr",
                        "-lineinfo", f"-arch=sm_{module['target_sm']}", "-ccbin", host_compiler,
                        CANONICAL_STAGE_SOURCE, "-o", "<output>"]
    require(recipe.get("normalized_command") == expected_command,
            "normalized compiler command mismatch")
    for path_key, version_key in (("nvcc_path", "nvcc_version"),
                                  ("ptxas_path", "ptxas_version"),
                                  ("cuobjdump_path", "cuobjdump_version"),
                                  ("host_compiler_path", "host_compiler_version")):
        tool = recipe.get(path_key)
        require(isinstance(tool, str) and Path(tool).is_file(), f"{path_key} is missing")
        actual = nvcc_version(tool) if path_key == "nvcc_path" else tool_version(tool)
        require(recipe.get(version_key) == actual, f"{version_key} changed")
    require(recipe.get("environment") == {"NVCC_PREPEND_FLAGS": "", "NVCC_APPEND_FLAGS": ""},
            "implicit nvcc flags are not accepted")
    module_path = Path(module["module_path"])
    require(module_path.is_file(), "indexed cubin is missing")
    require(sha256_file(module_path) == module["module_content_sha256"],
            "cubin content does not match immutable identity")
    require(module_path.name == module["module_content_sha256"] + ".cubin",
            "cubin path is not content addressed")
    return module_path


def build_module(args: argparse.Namespace) -> None:
    source, abi = args.source.resolve(), args.abi.resolve()
    repo_root, workspace_root = args.repo_root.resolve(), args.workspace_root.resolve()
    require(source.is_file(), f"missing source: {source}")
    require(repo_root.is_dir(), f"missing repository root: {repo_root}")
    require(repo_root == workspace_root / "stwo" and (workspace_root / "stwo-cairo").is_dir(),
            "workspace root must contain sibling stwo and stwo-cairo repositories")
    abi_doc = load_json(abi)
    validate_abi(abi_doc, abi, repo_root)
    require((repo_root / abi_doc["source"]).resolve() == source,
            "ABI source does not match build source")
    require(50 <= args.sm <= 999, "invalid SM target")
    toolchain = toolchain_identity(args.nvcc, args.host_compiler)
    flags = ["-cubin", "-O3", "--std=c++17", "--expt-relaxed-constexpr", "-lineinfo",
             f"-arch=sm_{args.sm}", "-ccbin", toolchain["host_compiler_path"]]
    generators = [path.resolve() for path in args.generator]
    require(all(path.is_file() for path in generators), "a generator identity input is missing")
    generator_ids = [workspace_source_identity(path, workspace_root) for path in generators]
    require([{key: identity[key] for key in ("repository", "path")}
             for identity in generator_ids] == FIRST_SLICE_GENERATORS,
            "first-slice build requires the complete reviewed generator identity set")
    source_sha256 = sha256_file(source)
    recipe = {
        "schema_version": "stwo.gpu-lab.build-recipe.v2",
        "source_sha256": source_sha256,
        "transitive_headers": [],
        "abi_sha256": sha256_file(abi),
        "cache_key": abi_doc["cache_key"],
        "semantic_ir_hash": abi_doc["semantic_ir_hash"],
        "exported_symbols": [abi_doc["symbol"]],
        "generator_sources": generator_ids,
        "build_tool_sha256": lab_tool_sha256(),
        **toolchain,
        "source_staging": canonical_staging_identity(source_sha256),
        "target_sm": args.sm,
        "normalized_command": ["nvcc", *flags, CANONICAL_STAGE_SOURCE, "-o", "<output>"],
    }
    recipe_hash = sha256_bytes(canonical_bytes(recipe))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_source = stage_source(source, source_sha256)
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        candidate = (Path(temporary) / "candidate.cubin").resolve()
        command = [toolchain["nvcc_path"], *flags, staged_source.name, "-o", str(candidate)]
        subprocess.run(command, check=True, cwd=staged_source.parent)
        validate_staged_source(staged_source, source_sha256)
        module_hash = sha256_file(candidate)
        symbols = subprocess.run(
            [toolchain["cuobjdump_path"], "--dump-elf-symbols", str(candidate)],
            text=True, capture_output=True, check=True,
        ).stdout
        exported = [line.split()[-1] for line in symbols.splitlines()
                    if "STB_GLOBAL" in line and "STO_ENTRY" in line]
        require(exported == [abi_doc["symbol"]],
                f"cubin exported kernel symbols differ: {exported}")
        if args.repro_check:
            second = (Path(temporary) / "repro.cubin").resolve()
            subprocess.run(
                [toolchain["nvcc_path"], *flags, staged_source.name, "-o", str(second)],
                check=True,
                cwd=staged_source.parent,
            )
            validate_staged_source(staged_source, source_sha256)
            require(sha256_file(second) == module_hash,
                    "clean same-recipe rebuild emitted different cubin bytes")
        if args.index.exists():
            prior = load_json(args.index)
            if prior.get("build_recipe_hash") == recipe_hash:
                require(prior.get("module_content_sha256") == module_hash,
                        "identical build recipe emitted different cubin bytes")
        destination = output_dir / f"{module_hash}.cubin"
        if destination.exists():
            require(sha256_file(destination) == module_hash, "immutable module path was mutated")
        else:
            os.replace(candidate, destination)
    _bind_recipe(output_dir, recipe_hash, module_hash)
    index = {
        "schema_version": "stwo.gpu-lab.module-index.v1",
        "build_recipe_hash": recipe_hash,
        "build_recipe": recipe,
        "module_content_sha256": module_hash,
        "module_path": str(destination.resolve()),
        "target_sm": args.sm,
        "abi_path": str(abi),
    }
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    args.stamp.parent.mkdir(parents=True, exist_ok=True)
    args.stamp.write_text(f"{recipe_hash} {module_hash}\n")


def _bind_recipe(output_dir: Path, recipe_hash: str, module_hash: str) -> None:
    binding_path = output_dir / f"{recipe_hash}.module-sha256"
    try:
        descriptor = os.open(binding_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        require(binding_path.read_text().strip() == module_hash,
                "persistent recipe cache maps one recipe to multiple cubins")
    else:
        with os.fdopen(descriptor, "w") as binding:
            binding.write(module_hash + "\n")
