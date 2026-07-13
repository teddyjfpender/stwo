"""Hostile, GPU-free tests for the exact FRI module identity boundary."""

from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from .common import canonical_bytes, load_json, require, sha256_bytes, sha256_file
from .fri_discovery import (
    discover_source_closure,
    discovery_command,
    parse_nvcc_depfile,
    require_sealed_closure,
    source_closure,
    validate_closure,
)
from .fri_module import (
    ABI_RELATIVE,
    _recipe,
    _validate_cubin,
    _validate_index_document,
    build_fri_module,
    validate_fri_module,
    validate_fri_abi,
)
from .fri_publication import (
    atomic_text,
    install_immutable_index,
    locator_document,
    make_escape,
    output_index_lock,
    require_exact_index,
    validate_locator_document,
    write_depfile,
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
    validate_closure(closure, repo_root, check_bytes=True)

    with tempfile.TemporaryDirectory() as temporary_name:
        temporary = Path(temporary_name).resolve()
        target = temporary / "candidate.cubin"
        first = temporary / "source one.cu"
        second = temporary / "source#two.cuh"
        first.write_text("one")
        second.write_text("two")
        depfile = f"{make_escape(target)}: {make_escape(first)} \\\n  {make_escape(second)}\n"
        require(parse_nvcc_depfile(depfile, temporary, target) == [first, second],
                "nvcc dependency continuation/escaping round trip failed")
        written_depfile = temporary / "written.d"
        write_depfile(written_depfile, target, [first, second])
        require(written_depfile.read_text() == depfile,
                "FRI depfile writer bytes differ from the exact make rule")
        _expect_rejection(
            "duplicate depfile source",
            lambda: parse_nvcc_depfile(
                f"{target}: {make_escape(first)} {make_escape(first)}\n", temporary, target),
        )
        _expect_rejection(
            "wrong depfile target",
            lambda: parse_nvcc_depfile(f"{target}: {first}\n", temporary, temporary / "other"),
        )
        _expect_rejection(
            "multiple depfile rules",
            lambda: parse_nvcc_depfile(f"{target}: {first}\nother: {second}\n", temporary),
        )
        discovery_calls: list[list[str]] = []
        discovery_depfile = temporary / "discovery.d"

        def discover(command: list[str], *, cwd: Path, check: bool):
            require(cwd == repo_root and check, "mock discovery invocation differs")
            discovery_calls.append(command)
            output = Path(command[command.index("-MF") + 1])
            dependencies = " ".join(make_escape(path) for path in required)
            output.write_text(f"ignored: {dependencies}\n")
            return SimpleNamespace()

        with patch("gpu_lab.fri_discovery.subprocess.run", side_effect=discover):
            discovered_paths, discovered = discover_source_closure(
                _fake_toolchain(), 86, repo_root, discovery_depfile,
            )
        require(discovery_calls == [discovery_command(
                    _fake_toolchain(), 86, discovery_depfile,
                )] and discovered_paths == required,
                "FRI dependency discovery command/path closure differs")
        require_sealed_closure(closure, discovered)
        _expect_rejection(
            "rediscovered dependency closure differs from seal",
            lambda: require_sealed_closure(closure, discovered[:-1]),
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
        hostile_root_target = temporary / "hostile-stage-root-target"
        hostile_root_target.mkdir()
        hostile_root = temporary / "hostile-stage-root"
        hostile_root.symlink_to(hostile_root_target, target_is_directory=True)
        with patch("gpu_lab.fri_staging.STAGE_ROOT", hostile_root):
            _expect_rejection(
                "pre-existing canonical stage-root symlink",
                lambda: stage_repository(staged_closure, staging_repo, staged_hash),
            )
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
            lambda: validate_closure(captured, repo_root, check_bytes=True),
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
        second_recipe = deepcopy(recipe)
        second_recipe["toolchain"]["environment"]["NVCC_APPEND_FLAGS"] = "-DSECOND_BUILD=1"
        second_recipe["toolchain_sha256"] = sha256_bytes(
            canonical_bytes(second_recipe["toolchain"])
        )
        second_recipe_hash = sha256_bytes(canonical_bytes(second_recipe))
        second_recipe_path = output_dir / f"{second_recipe_hash}.recipe.json"
        second_recipe_path.write_bytes(canonical_bytes(second_recipe))
        (output_dir / f"{second_recipe_hash}.module-sha256").write_text(module_hash + "\n")
        second_index = {
            **index,
            "build_recipe_hash": second_recipe_hash,
            "build_recipe": second_recipe,
            "build_recipe_path": str(second_recipe_path),
        }
        _validate_index_document(second_index, abi, repo_root, output_dir, 86)
        immutable_first, first_hash = install_immutable_index(output_dir, index)
        immutable_second, second_hash = install_immutable_index(output_dir, second_index)
        first_locator = locator_document(first_hash, 86)
        second_locator = locator_document(second_hash, 86)
        hostile_locator = {**first_locator, "index_path": "../external.module-index.json"}
        _expect_rejection(
            "locator smuggles external immutable-index path",
            lambda: validate_locator_document(hostile_locator, output_dir, 86),
        )
        race_path = output_dir / "fri_round6.sm86.locator.non-evidence.json"
        first_published, second_attempting, second_entered = Event(), Event(), Event()
        observed: list[Path] = []

        def first_builder() -> None:
            with output_index_lock(race_path):
                atomic_text(race_path, canonical_bytes(first_locator).decode())
                path, document = validate_locator_document(load_json(race_path), output_dir, 86)
                require(path == immutable_first, "first builder published another immutable index")
                _validate_index_document(document, abi, repo_root, output_dir, 86)
                first_published.set()
                require(second_attempting.wait(2), "hostile second builder did not attempt lock")
                require(not second_entered.wait(0.05),
                        "second builder entered before first final validation")
                require_exact_index(load_json(path), index)

        def second_builder() -> None:
            require(first_published.wait(2), "first builder did not publish")
            second_attempting.set()
            with output_index_lock(race_path):
                second_entered.set()
                observed.append(validate_locator_document(
                    load_json(race_path), output_dir, 86,
                )[0])
                atomic_text(race_path, canonical_bytes(second_locator).decode())
                path, document = validate_locator_document(load_json(race_path), output_dir, 86)
                require(path == immutable_second, "second builder published another immutable index")
                _validate_index_document(document, abi, repo_root, output_dir, 86)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(first_builder), executor.submit(second_builder)]
            for future in futures:
                future.result()
        require(observed == [immutable_first],
                "output-index lock did not serialize publication through final validation")
        current_path, current_index = validate_locator_document(
            load_json(race_path), output_dir, 86,
        )
        require(current_path == immutable_second, "non-evidence locator did not advance")
        _validate_index_document(current_index, abi, repo_root, output_dir, 86)
        _validate_index_document(
            load_json(immutable_first), abi, repo_root, output_dir, 86,
        )
        atomic_text(race_path, canonical_bytes(first_locator).decode())
        module_args = SimpleNamespace(
            nvcc="/toolchain/nvcc",
            host_compiler="/toolchain/c++",
            sm=86,
            source=repo_root / "gpu-lab/kernels/fri_round6.cu",
            abi=abi_path,
            repo_root=repo_root,
            output_dir=output_dir,
            index=race_path,
            stamp=output_dir / "fri_round6.sm86.stamp",
            depfile=output_dir / "fri_round6.sm86.d",
        )
        rediscoveries: list[tuple[dict, int, Path]] = []

        def rediscover(toolchain: dict, sm: int, root: Path, depfile_path: Path):
            require(root == repo_root, "module route rediscovered from another repository")
            rediscoveries.append((toolchain, sm, depfile_path))
            return required, closure

        with patch("gpu_lab.fri_module._sealed_toolchain", return_value=_fake_toolchain()), \
                patch("gpu_lab.fri_module._discover_source_closure",
                      side_effect=rediscover), \
                patch("gpu_lab.fri_module._validate_cubin"):
            validate_fri_module(module_args)
            build_fri_module(module_args)
        require(len(rediscoveries) == 2 and all(sm == 86 for _, sm, _ in rediscoveries),
                "cache-hit/validate routes did not both rediscover the exact source closure")
        with patch("gpu_lab.fri_module._sealed_toolchain", return_value=_fake_toolchain()), \
                patch("gpu_lab.fri_module._discover_source_closure",
                      return_value=(required[:-1], closure[:-1])), \
                patch("gpu_lab.fri_module._validate_cubin"):
            _expect_rejection(
                "validation rediscovery differs from sealed closure",
                lambda: validate_fri_module(module_args),
            )
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
            "source content rewrite", lambda: validate_closure(mutated, repo_root, check_bytes=True)
        )
        original = canonical_module.read_bytes()
        canonical_module.write_bytes(original + b"x")
        _expect_rejection(
            "mutated cubin", lambda: _validate_index_document(index, abi, repo_root, output_dir, 86)
        )
        canonical_module.write_bytes(original)

    require(load_json(abi_path)["source"] == "gpu-lab/kernels/fri_round6.cu",
            "FRI ABI source contract changed")
