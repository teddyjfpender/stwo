"""Hostile, GPU-free tests for the exact FRI module identity boundary."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

from .common import canonical_bytes, load_json, require, sha256_bytes, sha256_file
from .fri_module import (
    ABI_RELATIVE,
    _atomic_text,
    _make_escape,
    _recipe,
    _require_exact_index,
    _validate_closure,
    _validate_cubin,
    _validate_index_document,
    parse_nvcc_depfile,
    source_closure,
    validate_fri_abi,
)
from .fri_staging import (
    original_dependencies,
    stage_repository,
    staging_identity,
    validate_staging,
)
from .fri_tool_identity import FRI_BUILD_TOOL_PATHS, fri_build_tool_sha256


def _expect_rejection(label: str, operation) -> None:
    try:
        operation()
    except (ValueError, KeyError, OSError):
        return
    raise ValueError(f"hostile FRI identity was accepted: {label}")


def _fake_toolchain() -> dict:
    digest = "12" * 32
    return {
        "nvcc_path": "/toolchain/nvcc",
        "nvcc_version": "nvcc test",
        "ptxas_path": "/toolchain/ptxas",
        "ptxas_version": "ptxas test",
        "cuobjdump_path": "/toolchain/cuobjdump",
        "cuobjdump_version": "cuobjdump test",
        "host_compiler_path": "/toolchain/c++",
        "host_compiler_version": "c++ test",
        "environment": {"NVCC_PREPEND_FLAGS": "", "NVCC_APPEND_FLAGS": ""},
        "driver_compatibility_policy": "CUDA enhanced-compatibility; execution binds actual driver",
        "binary_sha256": {
            "nvcc_path": digest,
            "ptxas_path": digest,
            "cuobjdump_path": digest,
            "host_compiler_path": digest,
        },
    }


def _rehash(index: dict) -> None:
    index["build_recipe_hash"] = sha256_bytes(canonical_bytes(index["build_recipe"]))


def fri_module_self_test(lab_root: Path) -> None:
    repo_root = lab_root.parent.resolve()
    abi_path = repo_root / ABI_RELATIVE
    abi = validate_fri_abi(abi_path, repo_root)
    general_cli = (lab_root / "tools/gpu_lab/cli.py").read_text()
    require("build-fri-round6-module" not in general_cli
            and "validate-fri-round6-module" not in general_cli
            and "from .fri_module import" not in general_cli,
            "general lab CLI exposes an unsealed FRI module build path")
    require("tools/gpu_lab/results.py" not in FRI_BUILD_TOOL_PATHS,
            "unrelated result tooling entered the FRI compiler closure")
    tool_hash = fri_build_tool_sha256()
    real_sha256_file = sha256_file
    results_path = lab_root / "tools/gpu_lab/results.py"
    with patch("gpu_lab.fri_tool_identity.sha256_file", side_effect=lambda path:
               "00" * 32 if path == results_path else real_sha256_file(path)):
        require(fri_build_tool_sha256() == tool_hash,
                "unrelated result-tool mutation invalidated the FRI compiler")
    init_path = lab_root / "tools/gpu_lab/__init__.py"
    with patch("gpu_lab.fri_tool_identity.sha256_file", side_effect=lambda path:
               "00" * 32 if path == init_path else real_sha256_file(path)):
        require(fri_build_tool_sha256() != tool_hash,
                "package initializer mutation did not invalidate the FRI compiler")
    required = [
        repo_root / "gpu-lab/kernels/fri_round6.cu",
        repo_root / "crates/backend-cuda-kernels/cuda/fields.cu",
        repo_root / "crates/backend-cuda-kernels/cuda/fold_line.cu",
        repo_root / "crates/backend-cuda-kernels/cuda/blake2s.cu",
        repo_root / "crates/backend-cuda-kernels/cuda/device_transcript.cu",
    ]
    closure = source_closure(required, repo_root)
    _validate_closure(closure, repo_root, check_bytes=True)

    with tempfile.TemporaryDirectory() as temporary_name:
        temporary = Path(temporary_name).resolve()
        target = temporary / "candidate.cubin"
        first = temporary / "source one.cu"
        second = temporary / "source#two.cuh"
        first.write_text("one")
        second.write_text("two")
        depfile = f"{_make_escape(target)}: {_make_escape(first)} \\\n  {_make_escape(second)}\n"
        require(parse_nvcc_depfile(depfile, temporary, target) == [first, second],
                "nvcc dependency continuation/escaping round trip failed")
        _expect_rejection(
            "duplicate depfile source",
            lambda: parse_nvcc_depfile(
                f"{target}: {_make_escape(first)} {_make_escape(first)}\n", temporary, target),
        )
        _expect_rejection(
            "wrong depfile target",
            lambda: parse_nvcc_depfile(f"{target}: {first}\n", temporary, temporary / "other"),
        )
        _expect_rejection(
            "multiple depfile rules",
            lambda: parse_nvcc_depfile(f"{target}: {first}\nother: {second}\n", temporary),
        )
        staging_repo = temporary / "staging-repo"
        staged_source = staging_repo / "src/input.cu"
        staged_source.parent.mkdir(parents=True)
        staged_source.write_text(f"unique stage {temporary_name}")
        staged_closure = [{
            "scope": "repository", "path": "src/input.cu", "sha256": sha256_file(staged_source),
        }]
        staged_hash = sha256_bytes(canonical_bytes(staged_closure))
        expected_stage = Path(staging_identity(staged_hash)["working_directory"])
        foreign_stage = temporary / "foreign-stage"
        foreign_stage.mkdir()
        _expect_rejection(
            "foreign canonical stage path",
            lambda: validate_staging(staged_closure, foreign_stage),
        )
        _expect_rejection(
            "foreign dependency-mapping stage",
            lambda: original_dependencies([], staged_closure, foreign_stage, staging_repo),
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(stage_repository, staged_closure, staging_repo, staged_hash)
                for _ in range(2)
            ]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except ValueError:
                pass
        require(len(outcomes) == 2 and all(path == expected_stage for path in outcomes),
                "concurrent canonical FRI staging did not fail closed")
        validate_staging(staged_closure, expected_stage)
        _expect_rejection(
            "closure substitution under one staging key",
            lambda: stage_repository([{**staged_closure[0], "sha256": "00" * 32}],
                                     staging_repo, staged_hash),
        )
        staged_copy = expected_stage / "src/input.cu"
        os.chmod(staged_copy, 0o600)
        staged_copy.write_text("substituted after staging")
        _expect_rejection(
            "staged source substituted after install",
            lambda: validate_staging(staged_closure, expected_stage),
        )
        shutil.rmtree(expected_stage.parent)
        captured = source_closure([*required, first], repo_root)
        first.write_text("changed after dependency capture")
        _expect_rejection(
            "source changed after capture",
            lambda: _validate_closure(captured, repo_root, check_bytes=True),
        )

        output_dir = temporary / "modules"
        output_dir.mkdir()
        module = output_dir / ("34" * 32 + ".cubin")
        module.write_bytes(b"reviewed cubin")
        module_hash = sha256_file(module)
        canonical_module = output_dir / f"{module_hash}.cubin"
        module.rename(canonical_module)
        recipe = _recipe(86, abi, abi_path, closure, _fake_toolchain())
        recipe_hash = sha256_bytes(canonical_bytes(recipe))
        recipe_path = output_dir / f"{recipe_hash}.recipe.json"
        recipe_path.write_bytes(canonical_bytes(recipe))
        (output_dir / f"{recipe_hash}.module-sha256").write_text(module_hash + "\n")
        index = {
            "schema_version": "stwo.gpu-lab.fri-round6-module-index.v1",
            "module": "fri_round6",
            "target_sm": 86,
            "build_recipe_hash": recipe_hash,
            "build_recipe": recipe,
            "build_recipe_path": str(recipe_path),
            "module_content_sha256": module_hash,
            "module_path": str(canonical_module),
        }
        _validate_index_document(index, abi, repo_root, output_dir, 86)
        recipe_path.write_bytes(canonical_bytes(recipe) + b"\n")
        _expect_rejection(
            "non-canonical recipe artifact",
            lambda: _validate_index_document(index, abi, repo_root, output_dir, 86),
        )
        recipe_path.write_bytes(canonical_bytes(recipe))
        race_path = temporary / "module-index-race.json"
        barrier = Barrier(2)

        def race(document: dict) -> bool:
            _atomic_text(race_path, json.dumps(document))
            barrier.wait()
            try:
                _require_exact_index(load_json(race_path), document)
                return True
            except ValueError:
                return False

        with ThreadPoolExecutor(max_workers=2) as executor:
            accepted = list(executor.map(race, ({"index": 1}, {"index": 2})))
        require(accepted.count(False) == 1,
                "concurrent different module indexes were both accepted")
        symbols = index["build_recipe"]["driver_entries"]
        with patch("gpu_lab.fri_module.subprocess.run",
                   return_value=SimpleNamespace(stdout="64bit elf: sm=90\n")):
            _expect_rejection(
                "wrong cubin SM",
                lambda: _validate_cubin(canonical_module, _fake_toolchain(), 86, symbols),
            )
        with patch("gpu_lab.fri_module.subprocess.run", side_effect=[
            SimpleNamespace(stdout="64bit elf: sm=86\n"), SimpleNamespace(stdout=""),
        ]):
            _expect_rejection(
                "cubin without stable entries",
                lambda: _validate_cubin(canonical_module, _fake_toolchain(), 86, symbols),
            )

        mutated = deepcopy(index)
        mutated["build_recipe"]["compile_flags"][1] = "-O0"
        mutated["build_recipe"]["compile_flags_sha256"] = sha256_bytes(
            canonical_bytes(mutated["build_recipe"]["compile_flags"])
        )
        _rehash(mutated)
        _expect_rejection(
            "compiler flag rewrite",
            lambda: _validate_index_document(mutated, abi, repo_root, output_dir, 86),
        )
        mutated = deepcopy(index)
        mutated["build_recipe"]["normalized_command"][-3] = str(
            repo_root / "gpu-lab/kernels/fri_round6.cu"
        )
        _rehash(mutated)
        _expect_rejection(
            "absolute workspace path in recipe",
            lambda: _validate_index_document(mutated, abi, repo_root, output_dir, 86),
        )
        mutated = deepcopy(index)
        mutated["build_recipe"]["driver_entries"][0] = "attacker_kernel"
        _rehash(mutated)
        _expect_rejection(
            "driver entry rewrite",
            lambda: _validate_index_document(mutated, abi, repo_root, output_dir, 86),
        )
        mutated = deepcopy(index)
        mutated["build_recipe"]["abi_sha256"] = "00" * 32
        _rehash(mutated)
        _expect_rejection(
            "ABI rewrite", lambda: _validate_index_document(mutated, abi, repo_root, output_dir, 86)
        )
        mutated = deepcopy(index)
        mutated["module_path"] = str(temporary / canonical_module.name)
        _expect_rejection(
            "hash-paired module outside cache",
            lambda: _validate_index_document(mutated, abi, repo_root, output_dir, 86),
        )
        mutated = deepcopy(closure)
        mutated[0]["sha256"] = "00" * 32
        _expect_rejection(
            "source content rewrite", lambda: _validate_closure(mutated, repo_root, check_bytes=True)
        )
        original = canonical_module.read_bytes()
        canonical_module.write_bytes(original + b"x")
        _expect_rejection(
            "mutated cubin", lambda: _validate_index_document(index, abi, repo_root, output_dir, 86)
        )
        canonical_module.write_bytes(original)

    require(load_json(abi_path)["source"] == "gpu-lab/kernels/fri_round6.cu",
            "FRI ABI source contract changed")
