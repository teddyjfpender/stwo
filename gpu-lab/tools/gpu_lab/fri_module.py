"""Exact, content-addressed builder for the reviewed FRI round-6 CUDA module."""

from __future__ import annotations

import argparse
import json
import re
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
from .fri_discovery import (
    CUDA_ROOT_RELATIVE,
    SOURCE_RELATIVE,
    discover_source_closure as _discover_source_closure,
    parse_nvcc_depfile,
    require_sealed_closure as _require_sealed_closure,
    source_closure,
    validate_closure as _validate_closure,
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
from .immutable_output import guard_output, install_output
from .fri_publication import (
    atomic_text,
    install_immutable_index,
    install_recipe,
    locator_document,
    output_index_lock,
    require_exact_index,
    validate_locator_document,
    write_depfile,
)


ABI_SCHEMA = "stwo.gpu-lab.fri-round6-abi.v1"
ABI_SHA256 = "b232d2338e290570692011280a89232409f3f610401b9e0464a32bf9564f7776"
ABI_RELATIVE = "gpu-lab/manifests/fri_round6.abi.json"
RECIPE_SCHEMA = "stwo.gpu-lab.fri-round6-build-recipe.v1"
INDEX_SCHEMA = "stwo.gpu-lab.fri-round6-module-index.v1"
MODULE_NAME = "fri_round6"
FIXED_FLAGS = ["--cubin", "-O3", "--std=c++17", "--expt-relaxed-constexpr", "-lineinfo"]


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


def _inputs_match(recipe: dict[str, Any], closure: list[dict[str, str]],
                  toolchain: dict[str, Any]) -> bool:
    return (recipe["source_closure"] == closure
            and recipe["toolchain"] == toolchain
            and recipe["build_tool_sources"] == fri_build_tool_sources()
            and recipe["build_tool_sha256"] == fri_build_tool_sha256())

def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path]:
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    source, abi = args.source.resolve(), args.abi.resolve()
    index, stamp, depfile = args.index.absolute(), args.stamp.absolute(), args.depfile.absolute()
    require(repo_root.is_dir() and source.is_file() and abi.is_file(), "FRI build input is missing")
    require(source == (repo_root / SOURCE_RELATIVE).resolve(), "FRI build source differs")
    require(all(path.parent.resolve() == output_dir for path in (index, stamp, depfile)),
            "FRI index/stamp/depfile must live in the module directory")
    require(index.name == f"fri_round6.sm{args.sm}.locator.non-evidence.json"
            and stamp.name == f"fri_round6.sm{args.sm}.stamp"
            and depfile.name == f"fri_round6.sm{args.sm}.d", "FRI output names differ")
    return repo_root, output_dir, source, abi, index, stamp


def _build_fri_module_locked(args: argparse.Namespace) -> None:
    repo_root, output_dir, source, abi_path, index_path, stamp = _paths(args)
    depfile = args.depfile.absolute()
    abi = validate_fri_abi(abi_path, repo_root)
    toolchain = _sealed_toolchain(args.nvcc, args.host_compiler)
    output_dir.mkdir(parents=True, exist_ok=True)

    if index_path.exists() or index_path.is_symlink():
        require(index_path.is_file() and not index_path.is_symlink(),
                "FRI non-evidence locator is not a regular file")
        prior_locator = load_json(index_path)
        prior_index_path, prior = validate_locator_document(
            prior_locator, output_dir, args.sm,
        )
        module, recipe = _validate_index_document(prior, abi, repo_root, output_dir, args.sm)
        with tempfile.TemporaryDirectory(dir=output_dir) as discovery_directory:
            prior_paths, current = _discover_source_closure(
                toolchain, args.sm, repo_root, Path(discovery_directory) / "discovery.d",
            )
        if current and _inputs_match(recipe, current, toolchain):
            _require_sealed_closure(recipe["source_closure"], current)
            _validate_cubin(module, toolchain, args.sm, recipe["driver_entries"])
            write_depfile(depfile, stamp, prior_paths)
            atomic_text(
                stamp, f"{prior['build_recipe_hash']} {prior['module_content_sha256']} "
                f"{prior_locator['index_content_sha256']} {prior_index_path.name}\n",
            )
            installed = load_json(prior_index_path)
            require_exact_index(installed, prior)
            _validate_index_document(installed, abi, repo_root, output_dir, args.sm)
            require_exact_index(load_json(index_path), prior_locator)
            require(_sealed_toolchain(args.nvcc, args.host_compiler) == toolchain,
                    "FRI toolchain changed during cache validation")
            print(f"FRI round-6 module cache hit: {prior_index_path}")
            return

    with tempfile.TemporaryDirectory(dir=output_dir) as temporary_name:
        temporary = Path(temporary_name)
        discovery = temporary / "discovery.d"
        before_paths, before = _discover_source_closure(
            toolchain, args.sm, repo_root, discovery,
        )
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

    recipe_path = install_recipe(output_dir, recipe, recipe_hash, module_hash, destination)
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
    immutable_index_path, immutable_index_sha256 = install_immutable_index(output_dir, index)
    locator = locator_document(immutable_index_sha256, args.sm)
    atomic_text(index_path, json.dumps(locator, indent=2, sort_keys=True) + "\n")
    require_exact_index(load_json(index_path), locator)
    write_depfile(depfile, stamp, compiled_paths)
    atomic_text(stamp, f"{recipe_hash} {module_hash} {immutable_index_sha256} "
                f"{immutable_index_path.name}\n")
    installed_index_path, installed = validate_locator_document(
        load_json(index_path), output_dir, args.sm,
    )
    require(installed_index_path == immutable_index_path,
            "FRI non-evidence locator advanced during publication")
    installed_module, installed_recipe = _validate_index_document(
        installed, abi, repo_root, output_dir, args.sm,
    )
    _validate_closure(installed_recipe["source_closure"], repo_root, check_bytes=True)
    require(_sealed_toolchain(args.nvcc, args.host_compiler) == toolchain
            and fri_build_tool_sha256() == installed_recipe["build_tool_sha256"],
            "FRI identities changed after index installation")
    _validate_cubin(installed_module, toolchain, args.sm, installed_recipe["driver_entries"])
    final_index = load_json(immutable_index_path)
    require_exact_index(final_index, index)
    _validate_index_document(final_index, abi, repo_root, output_dir, args.sm)
    require(source_closure(compiled_paths, repo_root) == after
            and _sealed_toolchain(args.nvcc, args.host_compiler) == toolchain
            and fri_build_tool_sha256() == installed_recipe["build_tool_sha256"],
            "FRI identities changed during final validation")
    validate_staging(after, working_directory)
    require_exact_index(load_json(index_path), locator)
    print(f"FRI round-6 module built: {immutable_index_path}")


def build_fri_module(args: argparse.Namespace) -> None:
    _, output_dir, _, _, index_path, _, = _paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_index_lock(index_path):
        _build_fri_module_locked(args)


def _validate_fri_module_locked(args: argparse.Namespace) -> None:
    repo_root, output_dir, _, abi_path, index_path, _ = _paths(args)
    abi = validate_fri_abi(abi_path, repo_root)
    toolchain = _sealed_toolchain(args.nvcc, args.host_compiler)
    locator = load_json(index_path)
    immutable_index_path, index = validate_locator_document(locator, output_dir, args.sm)
    module, recipe = _validate_index_document(index, abi, repo_root, output_dir, args.sm)
    _validate_closure(recipe["source_closure"], repo_root, check_bytes=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as discovery_directory:
        _, discovered = _discover_source_closure(
            toolchain, args.sm, repo_root, Path(discovery_directory) / "discovery.d",
        )
    _require_sealed_closure(recipe["source_closure"], discovered)
    require(recipe["toolchain"] == toolchain, "FRI module toolchain identity changed")
    require(recipe["build_tool_sha256"] == fri_build_tool_sha256(),
            "FRI module build-tool identity changed")
    require(recipe["build_tool_sources"] == fri_build_tool_sources(),
            "FRI module build-tool source closure changed")
    _validate_cubin(module, toolchain, args.sm, recipe["driver_entries"])
    installed = load_json(immutable_index_path)
    require_exact_index(installed, index)
    _validate_index_document(installed, abi, repo_root, output_dir, args.sm)
    require(_sealed_toolchain(args.nvcc, args.host_compiler) == toolchain,
            "FRI toolchain changed during module validation")
    require_exact_index(load_json(index_path), locator)
    print(f"FRI round-6 module valid: {immutable_index_path}")


def validate_fri_module(args: argparse.Namespace) -> None:
    _, _, _, _, index_path, _ = _paths(args)
    with output_index_lock(index_path):
        _validate_fri_module_locked(args)
