"""Hostile local tests for the sealed PIE-adapter build transaction."""

from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .common import canonical_bytes, require, sha256_bytes
from .pie_adapter_build_execution import (
    CACHE_SCHEMA,
    MAX_STDOUT_BYTES,
    _linker_environment_name,
    execute,
    parse_record_bytes,
    validate_record,
)
from .pie_adapter_inventory import write_inventory
from .pie_adapter_inventory_tests import TARGET, _executable, _target, _workspace
from .pie_adapter_receipt import _closure_document, _source_body
from .pie_adapter_receipt_tests import _commit


IMAGE_ID = "sha256:" + "a" * 64
IMAGE_DIGEST = "sha256:" + "b" * 64


def _write(path: Path, payload: bytes, mode: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return path


def _expect(label: str, operation: Callable[[], object], fragment: str | None = None) -> None:
    try:
        operation()
    except (OSError, ValueError) as error:
        require(fragment is None or fragment in str(error),
                f"{label} failed for the wrong reason: {error}")
        return
    raise ValueError(f"{label} was accepted")


@dataclass
class Case:
    root: Path
    args: SimpleNamespace
    target: Path
    executable: Path
    source: Path
    environment: dict[str, str]

    @property
    def context(self) -> dict[str, Any]:
        return {
            "inventory_sha256": self.args.inventory_sha256,
            "closure_sha256": self.args.source_closure_sha256,
            "cache_sha256": self.args.cache_identity_sha256,
            "cargo_sha256": self.args.cargo_sha256,
            "rustc_sha256": self.args.rustc_sha256,
            "native_compiler_linker_sha256": self.args.native_compiler_linker_sha256,
            "native_archiver_sha256": self.args.native_archiver_sha256,
            "builder_image_id": IMAGE_ID,
            "builder_image_digest": IMAGE_DIGEST,
            "workspace_root": self.root,
            "commit_reader": _commit,
        }


def _case(parent: Path) -> Case:
    parent = parent.resolve()
    root = parent / "workspace"
    _workspace(root)
    inventory_path = parent / "inventory.json"
    inventory, inventory_sha = write_inventory(
        inventory_path, TARGET, _target(), _executable(), root,
    )
    body, _ = _source_body(inventory, sha256_bytes(canonical_bytes(inventory)), root, _commit)
    closure = _closure_document(body)
    closure_path = _write(
        parent / "source-closure.json",
        json.dumps(closure, indent=2, sort_keys=True).encode() + b"\n",
        0o400,
    )
    cargo = _write(parent / "toolchain/cargo", b"cargo-tool\n", 0o555)
    rustc = _write(parent / "toolchain/rustc", b"rustc-tool\n", 0o555)
    compiler_linker = _write(
        parent / "native/aarch64-linux-gnu-gcc-12", b"compiler-linker-tool\n", 0o555)
    archiver = _write(parent / "native/aarch64-linux-gnu-ar", b"archiver-tool\n", 0o555)
    cache = {
        "schema_version": CACHE_SCHEMA,
        "kind": "builder-image-baked-cargo-home-declared-v1",
        "path": "/opt/stwo-builder/cargo-home",
        "builder_image_id": IMAGE_ID,
        "builder_image_digest": IMAGE_DIGEST,
        "storage": "builder-image-layer",
        "external_mount": False,
        "read_only": True,
        "runtime_observed": False,
    }
    cache_path = _write(parent / "cache-identity.json",
                        canonical_bytes(cache) + b"\n", 0o400)
    target = root / "stwo-cairo" / _target()["path"]
    executable = root / "stwo-cairo" / _executable()["path"]
    linker_key = _linker_environment_name(TARGET)
    environment = {
        "CARGO_HOME": cache["path"],
        "CARGO_TARGET_DIR": str(target),
        "CARGO_NET_OFFLINE": "true",
        "LANG": "C",
        "LC_ALL": "C",
        "RUSTC": str(rustc),
        "CC": str(compiler_linker),
        "AR": str(archiver),
        linker_key: str(compiler_linker),
    }
    args = SimpleNamespace(
        inventory=inventory_path,
        inventory_sha256=inventory_sha,
        source_closure=closure_path,
        source_closure_sha256=sha256_bytes(closure_path.read_bytes()),
        cache_identity=cache_path,
        cache_identity_sha256=sha256_bytes(cache_path.read_bytes()),
        cargo=cargo,
        cargo_sha256=sha256_bytes(cargo.read_bytes()),
        rustc=rustc,
        rustc_sha256=sha256_bytes(rustc.read_bytes()),
        native_compiler_linker=compiler_linker,
        native_compiler_linker_sha256=sha256_bytes(compiler_linker.read_bytes()),
        native_archiver=archiver,
        native_archiver_sha256=sha256_bytes(archiver.read_bytes()),
        builder_image_id=IMAGE_ID,
        builder_image_digest=IMAGE_DIGEST,
        environment_json=json.dumps(environment, sort_keys=True),
        build_receipt=parent / "build-receipt.json",
        record=parent / "build-execution.json",
    )
    return Case(root, args, target, executable, root / "stwo/Cargo.toml", environment)


def _successful_runner(case: Case, *, hardlink: bool = True,
                       stdout: bytes = b"cargo build complete\n") -> Callable:
    def run(argv: list[str], cwd: Path,
            environment: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
        require(argv[0] == str(case.args.cargo)
                and argv[1:] == ["build", "--manifest-path", "Cargo.toml", "--release",
                                  "--locked", "--offline", "--target", TARGET],
                "test runner received a substituted Cargo invocation")
        require(cwd == case.root / "stwo-cairo/gpu_benchmarks/lab/pie-adapter"
                and environment == dict(sorted(case.environment.items()))
                and "PATH" not in environment,
                "test runner received an unsealed environment or cwd")
        cargo_output = _write(case.target / "cargo-output", b"adapter executable\n", 0o755)
        case.executable.parent.mkdir(parents=True, exist_ok=True)
        if hardlink:
            os.link(cargo_output, case.executable)
        else:
            _write(case.executable, cargo_output.read_bytes(), 0o755)
        return subprocess.CompletedProcess(argv, 0, stdout, b"cargo diagnostic\n")
    return run


def _schema_check(record: dict[str, Any]) -> None:
    schema_path = Path(__file__).resolve().parents[2]
    schema_path /= "schemas/pie-adapter-build-execution-record-v1.schema.json"
    schema = json.loads(schema_path.read_text())
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        require(schema["$schema"].endswith("2020-12/schema")
                and schema["properties"]["builder_runtime_attested"]["const"] is False,
                "PIE adapter build execution schema boundary differs")
        return
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    validator.validate(record)
    for label, mutate in (
        ("runtime overclaim", lambda value: value.update({"builder_runtime_attested": True})),
        ("admission overclaim", lambda value: value.update({"production_admissible": True})),
        ("cache observation overclaim", lambda value: value["builder"]["cache_identity"].update(
            {"runtime_observed": True})),
    ):
        hostile = copy.deepcopy(record)
        mutate(hostile)
        require(not validator.is_valid(hostile), f"build execution schema accepted {label}")


def _successful_case(parent: Path) -> tuple[Case, dict[str, Any]]:
    case = _case(parent)
    record = execute(case.args, runner=_successful_runner(case),
                     workspace_root=case.root, commit_reader=_commit)
    require(record["build_execution_attested"] is True
            and record["builder_runtime_attested"] is False
            and all(record[key] is False for key in
                    ("production_admissible", "correctness_admissible",
                     "performance_admissible")),
            "build transaction admission or attestation flags differ")
    require(record["builder"]["tools"]["cargo"]["path"] == str(case.args.cargo)
            and record["builder"]["tools"]["rustc"]["path"] == str(case.args.rustc)
            and record["builder"]["tools"]["native_compiler_linker"]["path"]
            == str(case.args.native_compiler_linker)
            and record["builder"]["tools"]["native_archiver"]["path"]
            == str(case.args.native_archiver),
            "build transaction omitted an exact tool identity")
    publication = record["executable_publication"]
    require(publication["source_link_count"] == 2 and publication["detached"] is True
            and publication["source_sha256"] == publication["final_sha256"]
            and case.executable.stat().st_nlink == 1,
            "Cargo hardlink was not byte-preservingly detached")
    require(stat.S_IMODE(case.args.record.stat().st_mode) == 0o400
            and stat.S_IMODE(case.args.build_receipt.stat().st_mode) == 0o400,
            "build artifacts are not immutable")
    require(parse_record_bytes(case.args.record.read_bytes(), **case.context) == record,
            "installed build execution record does not revalidate")
    _schema_check(record)
    return case, record


def _mutation_checks(case: Case, record: dict[str, Any]) -> None:
    mutations = (
        ("runtime overclaim", lambda value: value.update({"builder_runtime_attested": True})),
        ("admission overclaim", lambda value: value.update({"correctness_admissible": True})),
        ("image substitution", lambda value: value["builder"].update(
            {"image_digest": "sha256:" + "c" * 64})),
        ("rustc substitution", lambda value: value["invocation"]["environment"].update(
            {"RUSTC": str(case.args.cargo)})),
        ("CC substitution", lambda value: value["invocation"]["environment"].update(
            {"CC": str(case.args.native_archiver)})),
        ("AR substitution", lambda value: value["invocation"]["environment"].update(
            {"AR": str(case.args.native_compiler_linker)})),
        ("tool hash substitution", lambda value: value["builder"]["tools"]["native_archiver"].update(
            {"sha256": "0" * 64})),
        ("publication substitution", lambda value: value["executable_publication"].update(
            {"source_sha256": "0" * 64})),
        ("receipt upgrade overclaim", lambda value: value["receipt_binding"].update(
            {"v2_build_execution_attested": True})),
        ("child failure rewrite", lambda value: value["child_result"].update(
            {"exit_status": 7})),
    )
    for label, mutate in mutations:
        hostile = copy.deepcopy(record)
        mutate(hostile)
        _expect(label, lambda hostile=hostile: validate_record(hostile, **case.context))
    duplicate = case.args.record.read_bytes().replace(
        b'"passed":true', b'"passed":true,"passed":true', 1,
    )
    _expect("duplicate record key", lambda: parse_record_bytes(duplicate, **case.context),
            "duplicate")


def _failure_cases() -> None:
    with tempfile.TemporaryDirectory(prefix="gpu-lab-build-existing-") as temporary:
        case = _case(Path(temporary))
        case.target.mkdir(parents=True)
        called = False

        def forbidden(*_: object) -> subprocess.CompletedProcess[bytes]:
            nonlocal called
            called = True
            raise AssertionError("runner called")

        _expect("preexisting target", lambda: execute(
            case.args, runner=forbidden, workspace_root=case.root, commit_reader=_commit), "fresh")
        require(case.target.is_dir() and not called, "preexisting target was modified or executed")

    def rejected(label: str, runner_factory: Callable[[Case], Callable],
                 fragment: str | None = None) -> None:
        with tempfile.TemporaryDirectory(prefix=f"gpu-lab-build-{label}-") as temporary:
            case = _case(Path(temporary))
            _expect(label, lambda: execute(case.args, runner=runner_factory(case),
                    workspace_root=case.root, commit_reader=_commit), fragment)
            require(not case.target.exists() and not case.target.is_symlink()
                    and not case.args.build_receipt.exists() and not case.args.record.exists(),
                    f"{label} left fresh transaction outputs")

    def nonzero(case: Case) -> Callable:
        success = _successful_runner(case)
        return lambda argv, cwd, env: subprocess.CompletedProcess(
            argv, 7, success(argv, cwd, env).stdout, b"failed\n")

    def missing(case: Case) -> Callable:
        def run(argv: list[str], _: Path, __: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
            case.target.mkdir(parents=True)
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        return run

    def symlink(case: Case) -> Callable:
        def run(argv: list[str], _: Path, __: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
            outside = _write(case.target.parent / "outside", b"outside\n", 0o755)
            case.executable.parent.mkdir(parents=True)
            case.executable.symlink_to(outside)
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        return run

    def mutate_source(case: Case) -> Callable:
        success = _successful_runner(case)
        def run(argv: list[str], cwd: Path, env: dict[str, str]):
            result = success(argv, cwd, env)
            case.source.write_bytes(b"mutated source\n")
            return result
        return run

    def mutate_rustc(case: Case) -> Callable:
        success = _successful_runner(case)
        def run(argv: list[str], cwd: Path, env: dict[str, str]):
            result = success(argv, cwd, env)
            case.args.rustc.chmod(0o755)
            case.args.rustc.write_bytes(b"mutated rustc\n")
            return result
        return run

    def mutate_cache_declaration(case: Case) -> Callable:
        success = _successful_runner(case)
        def run(argv: list[str], cwd: Path, env: dict[str, str]):
            result = success(argv, cwd, env)
            case.args.cache_identity.chmod(0o600)
            case.args.cache_identity.write_bytes(b"{}\n")
            return result
        return run

    rejected("nonzero", nonzero, "exit status")
    rejected("missing-output", missing)
    rejected("symlink-output", symlink, "symlink")
    rejected("oversized-stdout",
             lambda case: _successful_runner(case, stdout=b"x" * (MAX_STDOUT_BYTES + 1)),
             "stdout")
    rejected("source-mutation", mutate_source, "source")
    rejected("rustc-mutation", mutate_rustc, "out-of-band")
    rejected("cache-declaration-mutation", mutate_cache_declaration, "mode")

    with tempfile.TemporaryDirectory(prefix="gpu-lab-build-env-") as temporary:
        case = _case(Path(temporary))
        hostile = dict(case.environment)
        hostile["PATH"] = "/usr/bin"
        case.args.environment_json = json.dumps(hostile)
        _expect("PATH environment injection", lambda: execute(
            case.args, runner=_successful_runner(case), workspace_root=case.root,
            commit_reader=_commit), "exact sealed")
        require(not case.target.exists(), "environment rejection launched the builder")

    with tempfile.TemporaryDirectory(prefix="gpu-lab-build-cache-root-") as temporary:
        case = _case(Path(temporary))
        cache = json.loads(case.args.cache_identity.read_text())
        cache["external_mount"] = True
        case.args.cache_identity.chmod(0o600)
        case.args.cache_identity.write_bytes(canonical_bytes(cache) + b"\n")
        case.args.cache_identity.chmod(0o400)
        case.args.cache_identity_sha256 = sha256_bytes(case.args.cache_identity.read_bytes())
        _expect("external cache mount declaration", lambda: execute(
            case.args, runner=_successful_runner(case), workspace_root=case.root,
            commit_reader=_commit), "status differs")
        require(not case.target.exists(), "external cache declaration launched the builder")


def pie_adapter_build_execution_self_test(_: Path | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="gpu-lab-build-success-") as temporary:
        case, record = _successful_case(Path(temporary))
        _mutation_checks(case, record)
    _failure_cases()


if __name__ == "__main__":
    pie_adapter_build_execution_self_test()
    print("pie_adapter_build_execution_tests: PASS")
