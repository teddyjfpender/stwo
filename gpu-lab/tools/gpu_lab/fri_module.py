"""Exact, content-addressed builder for the reviewed FRI round-6 CUDA module."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile
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
from .fri_staging import (
    original_dependencies,
    stage_repository,
    staging_identity,
    validate_staging,
)
from .fri_tool_identity import (
    fri_build_tool_sha256,
    fri_build_tool_sources,
    validate_fri_build_tool,
)
from .identity import toolchain_identity
from .immutable_output import guard_output, install_output, write_immutable_bytes


ABI_SCHEMA = "stwo.gpu-lab.fri-round6-abi.v1"
ABI_SHA256 = "b232d2338e290570692011280a89232409f3f610401b9e0464a32bf9564f7776"
ABI_RELATIVE = "gpu-lab/manifests/fri_round6.abi.json"
SOURCE_RELATIVE = "gpu-lab/kernels/fri_round6.cu"
RECIPE_SCHEMA = "stwo.gpu-lab.fri-round6-build-recipe.v1"
INDEX_SCHEMA = "stwo.gpu-lab.fri-round6-module-index.v1"
MODULE_NAME = "fri_round6"
CUDA_ROOT_RELATIVE = "crates/backend-cuda-kernels/cuda"
FIXED_FLAGS = ["--cubin", "-O3", "--std=c++17", "--expt-relaxed-constexpr", "-lineinfo"]
REQUIRED_REPOSITORY_SOURCES = {
    SOURCE_RELATIVE,
    f"{CUDA_ROOT_RELATIVE}/fields.cu",
    f"{CUDA_ROOT_RELATIVE}/fold_line.cu",
    f"{CUDA_ROOT_RELATIVE}/blake2s.cu",
    f"{CUDA_ROOT_RELATIVE}/device_transcript.cu",
}


def _atomic_text(path: Path, text: str) -> None:
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


def _sealed_toolchain(nvcc: str, host_compiler: str) -> dict[str, Any]:
    identity = toolchain_identity(nvcc, host_compiler)
    identity["binary_sha256"] = {
        key: sha256_file(Path(identity[key]))
        for key in ("nvcc_path", "ptxas_path", "cuobjdump_path", "host_compiler_path")
    }
    return identity


def validate_fri_abi(abi_path: Path, repo_root: Path) -> dict[str, Any]:
    expected = (repo_root / ABI_RELATIVE).resolve()
    require(abi_path.resolve() == expected, "FRI ABI path differs from the reviewed manifest")
    require(sha256_file(expected) == ABI_SHA256, "FRI ABI manifest seal differs")
    abi = load_json(expected)
    require_exact_keys(abi, {"schema_version", "module", "source", "driver_entries"}, "FRI ABI")
    require(abi["schema_version"] == ABI_SCHEMA and abi["module"] == MODULE_NAME,
            "unsupported FRI ABI identity")
    require(abi["source"] == SOURCE_RELATIVE, "FRI ABI points at another source")
    entries = abi["driver_entries"]
    require(isinstance(entries, list) and len(entries) == 5, "FRI ABI must declare five entries")
    symbols: list[str] = []
    for entry in entries:
        require_exact_keys(entry, {"symbol", "arguments", "launch"}, "FRI ABI entry")
        require(isinstance(entry["symbol"], str) and entry["symbol"].startswith("stwo_gpu_lab_"),
                "FRI ABI symbol is not a stable lab entry")
        require(isinstance(entry["arguments"], list) and entry["arguments"],
                "FRI ABI entry has no arguments")
        for argument in entry["arguments"]:
            require_exact_keys(argument, {"name", "type", "size_bytes"}, "FRI ABI argument")
        require(entry["launch"] in ({"block": [256, 1, 1], "dynamic_shared_bytes": 0},
                                    {"block": [1, 1, 1], "dynamic_shared_bytes": 0}),
                "FRI ABI launch contract changed")
        symbols.append(entry["symbol"])
    require(len(set(symbols)) == len(symbols), "FRI ABI repeats a driver entry")
    return abi


def _compile_flags(sm: int, host_compiler: str) -> list[str]:
    require(isinstance(sm, int) and not isinstance(sm, bool) and 50 <= sm <= 999,
            "invalid FRI module SM")
    return [*FIXED_FLAGS, f"-arch=sm_{sm}", "-ccbin", host_compiler, "-MD"]


def _normalized_command(sm: int, host_compiler: str) -> list[str]:
    return [
        "nvcc", *_compile_flags(sm, host_compiler), "-MF", "<depfile>",
        "-I", CUDA_ROOT_RELATIVE, SOURCE_RELATIVE, "-o", "<output>",
    ]


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


def _identity_path(identity: dict[str, str], repo_root: Path) -> Path:
    if identity["scope"] == "repository":
        path = (repo_root / identity["path"]).resolve()
        require(path.is_relative_to(repo_root), "FRI source identity escapes repository")
        return path
    require(identity["scope"] == "toolchain" and Path(identity["path"]).is_absolute(),
            "FRI external source identity is invalid")
    return Path(identity["path"])


def _validate_closure(closure: Any, repo_root: Path, *, check_bytes: bool) -> list[Path]:
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


def _recipe(sm: int, abi: dict[str, Any], abi_path: Path, closure: list[dict[str, str]],
            toolchain: dict[str, Any]) -> dict[str, Any]:
    flags = _compile_flags(sm, toolchain["host_compiler_path"])
    symbols = [entry["symbol"] for entry in abi["driver_entries"]]
    closure_sha256 = sha256_bytes(canonical_bytes(closure))
    build_tool_sources = fri_build_tool_sources()
    return {
        "schema_version": RECIPE_SCHEMA,
        "module": MODULE_NAME,
        "target_sm": sm,
        "abi_path": ABI_RELATIVE,
        "abi_sha256": sha256_file(abi_path),
        "driver_entries": symbols,
        "compile_flags": flags,
        "compile_flags_sha256": sha256_bytes(canonical_bytes(flags)),
        "normalized_command": _normalized_command(sm, toolchain["host_compiler_path"]),
        "source_closure": closure,
        "source_closure_sha256": closure_sha256,
        "source_staging": staging_identity(closure_sha256),
        "build_tool_sources": build_tool_sources,
        "build_tool_sha256": sha256_bytes(canonical_bytes(build_tool_sources)),
        "toolchain": toolchain,
        "toolchain_sha256": sha256_bytes(canonical_bytes(toolchain)),
    }


def _validate_recipe(recipe: Any, abi: dict[str, Any], repo_root: Path) -> None:
    require(isinstance(recipe, dict), "FRI build recipe must be an object")
    require_exact_keys(recipe, {
        "schema_version", "module", "target_sm", "abi_path", "abi_sha256",
        "driver_entries", "compile_flags", "compile_flags_sha256", "normalized_command",
        "source_closure", "source_closure_sha256", "source_staging", "build_tool_sources",
        "build_tool_sha256", "toolchain", "toolchain_sha256",
    }, "FRI build recipe")
    require(recipe["schema_version"] == RECIPE_SCHEMA and recipe["module"] == MODULE_NAME,
            "unsupported FRI build recipe")
    require(recipe["abi_path"] == ABI_RELATIVE and recipe["abi_sha256"] == ABI_SHA256,
            "FRI build recipe is bound to another ABI")
    symbols = [entry["symbol"] for entry in abi["driver_entries"]]
    require(recipe["driver_entries"] == symbols, "FRI build recipe entry set differs")
    toolchain = recipe["toolchain"]
    require(isinstance(toolchain, dict) and isinstance(toolchain.get("binary_sha256"), dict),
            "FRI build recipe toolchain is malformed")
    require_exact_keys(toolchain, {
        "nvcc_path", "nvcc_version", "ptxas_path", "ptxas_version", "cuobjdump_path",
        "cuobjdump_version", "host_compiler_path", "host_compiler_version", "environment",
        "driver_compatibility_policy", "binary_sha256",
    }, "FRI toolchain identity")
    require_exact_keys(toolchain["binary_sha256"],
                       {"nvcc_path", "ptxas_path", "cuobjdump_path", "host_compiler_path"},
                       "FRI tool binary identity")
    require_sha256(recipe["toolchain_sha256"], "FRI toolchain sha256")
    require(recipe["toolchain_sha256"] == sha256_bytes(canonical_bytes(toolchain)),
            "FRI toolchain hash does not bind its identity")
    for digest in toolchain["binary_sha256"].values():
        require_sha256(digest, "FRI tool binary sha256")
    flags = _compile_flags(recipe["target_sm"], toolchain["host_compiler_path"])
    require(recipe["compile_flags"] == flags, "FRI fixed compiler flags differ")
    require(recipe["normalized_command"] ==
            _normalized_command(recipe["target_sm"], toolchain["host_compiler_path"]),
            "FRI normalized compiler command differs")
    require(recipe["compile_flags_sha256"] == sha256_bytes(canonical_bytes(flags)),
            "FRI compiler flag hash differs")
    validate_fri_build_tool(recipe["build_tool_sources"], recipe["build_tool_sha256"])
    closure = recipe["source_closure"]
    _validate_closure(closure, repo_root, check_bytes=False)
    require(recipe["source_closure_sha256"] == sha256_bytes(canonical_bytes(closure)),
            "FRI source closure hash differs")
    require(recipe["source_staging"] == staging_identity(recipe["source_closure_sha256"]),
            "FRI canonical source-staging identity differs")


def _validate_index_document(index: dict[str, Any], abi: dict[str, Any], repo_root: Path,
                             output_dir: Path, sm: int) -> tuple[Path, dict[str, Any]]:
    require_exact_keys(index, {"schema_version", "module", "target_sm", "build_recipe_hash",
                               "build_recipe", "build_recipe_path", "module_content_sha256",
                               "module_path"},
                       "FRI module index")
    require(index["schema_version"] == INDEX_SCHEMA and index["module"] == MODULE_NAME,
            "unsupported FRI module index")
    require(index["target_sm"] == sm, "FRI module index target SM differs")
    recipe = index["build_recipe"]
    _validate_recipe(recipe, abi, repo_root)
    require(recipe["target_sm"] == sm, "FRI recipe target SM differs")
    require_sha256(index["build_recipe_hash"], "FRI build recipe hash")
    require(index["build_recipe_hash"] == sha256_bytes(canonical_bytes(recipe)),
            "FRI build recipe hash does not bind recipe")
    recipe_path = Path(index["build_recipe_path"])
    require(recipe_path.is_absolute() and recipe_path.parent == output_dir
            and recipe_path.name == index["build_recipe_hash"] + ".recipe.json"
            and recipe_path.is_file() and not recipe_path.is_symlink(),
            "FRI recipe artifact path differs")
    require(sha256_file(recipe_path) == index["build_recipe_hash"]
            and recipe_path.read_bytes() == canonical_bytes(recipe),
            "FRI recipe artifact content differs")
    require_sha256(index["module_content_sha256"], "FRI module content sha256")
    module_path = Path(index["module_path"])
    require(module_path.is_absolute() and module_path.parent == output_dir,
            "FRI module path escapes the content-addressed directory")
    require(module_path.name == index["module_content_sha256"] + ".cubin",
            "FRI module path is not content addressed")
    require(module_path.is_file() and not module_path.is_symlink(), "indexed FRI cubin is missing")
    require(sha256_file(module_path) == index["module_content_sha256"],
            "indexed FRI cubin content differs")
    binding = output_dir / f"{index['build_recipe_hash']}.module-sha256"
    require(binding.is_file() and not binding.is_symlink()
            and binding.read_text().strip() == index["module_content_sha256"],
            "FRI recipe-to-module immutable binding differs")
    return module_path, recipe


def _validate_cubin(path: Path, toolchain: dict[str, Any], sm: int, symbols: list[str]) -> None:
    cuobjdump = toolchain["cuobjdump_path"]
    elf = subprocess.run([cuobjdump, "--dump-elf", str(path)], text=True,
                         capture_output=True, check=True).stdout
    architectures = re.findall(r"\bsm=(\d+)\b", elf)
    require(architectures == [str(sm)], f"FRI cubin SM differs: {architectures}")
    table = subprocess.run([cuobjdump, "--dump-elf-symbols", str(path)], text=True,
                           capture_output=True, check=True).stdout
    exported = [line.split()[-1] for line in table.splitlines()
                if "STB_GLOBAL" in line and "STO_ENTRY" in line]
    stable = [symbol for symbol in exported if symbol.startswith("stwo_gpu_lab_")]
    require(sorted(stable) == sorted(symbols) and len(stable) == len(symbols),
            f"FRI cubin stable entries differ: {stable}")


def _make_escape(path: Path) -> str:
    value = str(path)
    return (value.replace("\\", "\\\\").replace("$", "$$").replace("#", "\\#")
            .replace(" ", "\\ ").replace(":", "\\:"))


def _write_depfile(path: Path, stamp: Path, sources: list[Path]) -> None:
    dependencies = " \\\n  ".join(_make_escape(source) for source in sources)
    _atomic_text(path, f"{_make_escape(stamp)}: {dependencies}\n")


def _require_exact_index(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    require(canonical_bytes(actual) == canonical_bytes(expected),
            "a concurrent FRI build installed a different module index")

def _inputs_match(recipe: dict[str, Any], closure: list[dict[str, str]],
                  toolchain: dict[str, Any]) -> bool:
    return (recipe["source_closure"] == closure
            and recipe["toolchain"] == toolchain
            and recipe["build_tool_sources"] == fri_build_tool_sources()
            and recipe["build_tool_sha256"] == fri_build_tool_sha256())

def _install_recipe(output_dir: Path, recipe: dict[str, Any], recipe_hash: str,
                    module_hash: str, module: Path) -> Path:
    recipe_path = output_dir / f"{recipe_hash}.recipe.json"
    write_immutable_bytes(recipe_path, canonical_bytes(recipe), [module])
    write_immutable_bytes(output_dir / f"{recipe_hash}.module-sha256",
                          (module_hash + "\n").encode(), [module])
    return recipe_path

def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path]:
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    source, abi = args.source.resolve(), args.abi.resolve()
    index, stamp, depfile = args.index.absolute(), args.stamp.absolute(), args.depfile.absolute()
    require(repo_root.is_dir() and source.is_file() and abi.is_file(), "FRI build input is missing")
    require(source == (repo_root / SOURCE_RELATIVE).resolve(), "FRI build source differs")
    require(all(path.parent.resolve() == output_dir for path in (index, stamp, depfile)),
            "FRI index/stamp/depfile must live in the module directory")
    require(index.name == f"fri_round6.sm{args.sm}.module.json"
            and stamp.name == f"fri_round6.sm{args.sm}.stamp"
            and depfile.name == f"fri_round6.sm{args.sm}.d", "FRI output names differ")
    return repo_root, output_dir, source, abi, index, stamp


def build_fri_module(args: argparse.Namespace) -> None:
    repo_root, output_dir, source, abi_path, index_path, stamp = _paths(args)
    depfile = args.depfile.absolute()
    abi = validate_fri_abi(abi_path, repo_root)
    toolchain = _sealed_toolchain(args.nvcc, args.host_compiler)
    output_dir.mkdir(parents=True, exist_ok=True)

    if index_path.is_file():
        prior = load_json(index_path)
        module, recipe = _validate_index_document(prior, abi, repo_root, output_dir, args.sm)
        prior_paths = _validate_closure(recipe["source_closure"], repo_root, check_bytes=False)
        current = (source_closure(prior_paths, repo_root)
                   if all(path.is_file() and not path.is_symlink() for path in prior_paths)
                   else [])
        if current and _inputs_match(recipe, current, toolchain):
            _validate_cubin(module, toolchain, args.sm, recipe["driver_entries"])
            _write_depfile(depfile, stamp, prior_paths)
            _atomic_text(stamp, f"{prior['build_recipe_hash']} {prior['module_content_sha256']}\n")
            installed = load_json(index_path)
            _require_exact_index(installed, prior)
            _validate_index_document(installed, abi, repo_root, output_dir, args.sm)
            require(_sealed_toolchain(args.nvcc, args.host_compiler) == toolchain,
                    "FRI toolchain changed during cache validation")
            print(f"FRI round-6 module cache hit: {module}")
            return

    with tempfile.TemporaryDirectory(dir=output_dir) as temporary_name:
        temporary = Path(temporary_name)
        discovery = temporary / "discovery.d"
        discover = [toolchain["nvcc_path"], "-M", "-O3", "--std=c++17",
                    "--expt-relaxed-constexpr", "-lineinfo", f"-arch=sm_{args.sm}", "-ccbin",
                    toolchain["host_compiler_path"], "-I", CUDA_ROOT_RELATIVE,
                    "-MF", str(discovery), SOURCE_RELATIVE]
        subprocess.run(discover, cwd=repo_root, check=True)
        before_paths = parse_nvcc_depfile(discovery.read_text(), repo_root)
        before = source_closure(before_paths, repo_root)
        closure_sha256 = sha256_bytes(canonical_bytes(before))
        working_directory = stage_repository(before, repo_root, closure_sha256)

        candidate = temporary / "candidate.cubin"
        compile_depfile = temporary / "compile.d"
        command = [toolchain["nvcc_path"], *_compile_flags(args.sm, toolchain["host_compiler_path"]),
                   "-MF", str(compile_depfile), "-I", CUDA_ROOT_RELATIVE,
                   SOURCE_RELATIVE, "-o", str(candidate)]
        subprocess.run(command, cwd=working_directory, check=True)
        staged_paths = parse_nvcc_depfile(
            compile_depfile.read_text(), working_directory, candidate,
        )
        compiled_paths = original_dependencies(staged_paths, before, working_directory, repo_root)
        after = source_closure(compiled_paths, repo_root)
        require(after == before, "FRI source closure changed while nvcc compiled")
        validate_staging(before, working_directory)
        require(sha256_file(abi_path) == ABI_SHA256 and
                _sealed_toolchain(args.nvcc, args.host_compiler) == toolchain and
                fri_build_tool_sha256() ==
                _recipe(args.sm, abi, abi_path, after, toolchain)["build_tool_sha256"],
                "FRI ABI/toolchain/build tool changed while nvcc compiled")

        recipe = _recipe(args.sm, abi, abi_path, after, toolchain)
        recipe_hash = sha256_bytes(canonical_bytes(recipe))
        module_hash = sha256_file(candidate)
        destination = output_dir / f"{module_hash}.cubin"
        _validate_cubin(candidate, toolchain, args.sm, recipe["driver_entries"])
        prior_identity = guard_output(destination, [abi_path, *compiled_paths])
        install_output(candidate, destination, [abi_path, *compiled_paths], prior_identity)

    recipe_path = _install_recipe(output_dir, recipe, recipe_hash, module_hash, destination)
    index = {
        "schema_version": INDEX_SCHEMA,
        "module": MODULE_NAME,
        "target_sm": args.sm,
        "build_recipe_hash": recipe_hash,
        "build_recipe": recipe,
        "build_recipe_path": str(recipe_path),
        "module_content_sha256": module_hash,
        "module_path": str(destination),
    }
    require(source_closure(compiled_paths, repo_root) == after,
            "FRI source closure changed before index installation")
    validate_staging(after, working_directory)
    require(sha256_file(abi_path) == ABI_SHA256
            and _sealed_toolchain(args.nvcc, args.host_compiler) == toolchain
            and fri_build_tool_sha256() == recipe["build_tool_sha256"],
            "FRI inputs changed before index installation")
    _validate_cubin(destination, toolchain, args.sm, recipe["driver_entries"])
    _validate_index_document(index, abi, repo_root, output_dir, args.sm)
    _atomic_text(index_path, json.dumps(index, indent=2, sort_keys=True) + "\n")
    _require_exact_index(load_json(index_path), index)
    _write_depfile(depfile, stamp, compiled_paths)
    _atomic_text(stamp, f"{recipe_hash} {module_hash}\n")
    installed_module, installed_recipe = _validate_index_document(
        load_json(index_path), abi, repo_root, output_dir, args.sm,
    )
    _validate_closure(installed_recipe["source_closure"], repo_root, check_bytes=True)
    require(_sealed_toolchain(args.nvcc, args.host_compiler) == toolchain
            and fri_build_tool_sha256() == installed_recipe["build_tool_sha256"],
            "FRI identities changed after index installation")
    _validate_cubin(installed_module, toolchain, args.sm, installed_recipe["driver_entries"])
    final_index = load_json(index_path)
    _require_exact_index(final_index, index)
    _validate_index_document(final_index, abi, repo_root, output_dir, args.sm)
    require(source_closure(compiled_paths, repo_root) == after
            and _sealed_toolchain(args.nvcc, args.host_compiler) == toolchain
            and fri_build_tool_sha256() == installed_recipe["build_tool_sha256"],
            "FRI identities changed during final validation")
    validate_staging(after, working_directory)
    print(f"FRI round-6 module built: {destination}")


def validate_fri_module(args: argparse.Namespace) -> None:
    repo_root, output_dir, _, abi_path, index_path, _ = _paths(args)
    abi = validate_fri_abi(abi_path, repo_root)
    toolchain = _sealed_toolchain(args.nvcc, args.host_compiler)
    index = load_json(index_path)
    module, recipe = _validate_index_document(index, abi, repo_root, output_dir, args.sm)
    _validate_closure(recipe["source_closure"], repo_root, check_bytes=True)
    require(recipe["toolchain"] == toolchain, "FRI module toolchain identity changed")
    require(recipe["build_tool_sha256"] == fri_build_tool_sha256(),
            "FRI module build-tool identity changed")
    require(recipe["build_tool_sources"] == fri_build_tool_sources(),
            "FRI module build-tool source closure changed")
    _validate_cubin(module, toolchain, args.sm, recipe["driver_entries"])
    installed = load_json(index_path)
    _require_exact_index(installed, index)
    _validate_index_document(installed, abi, repo_root, output_dir, args.sm)
    require(_sealed_toolchain(args.nvcc, args.host_compiler) == toolchain,
            "FRI toolchain changed during module validation")
    print(f"FRI round-6 module valid: {module}")
