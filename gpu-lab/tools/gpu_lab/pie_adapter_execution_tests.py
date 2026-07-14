"""CPU-only hostile tests for authenticated PIE adapter execution."""

from __future__ import annotations

import copy
import json
import os
import resource
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

from . import pie_adapter_execution as execution_module
from .common import LAB_ROOT, canonical_bytes, require, sha256_bytes, sha256_file
from .pie_adapter_contract import (ARTIFACT_SPECS, EXECUTION_HONESTY, INVOCATION_SCHEMA,
                                   OUTPUT_ENCODING, PROTOCOL, invocation_bytes)
from .pie_adapter_contract import validate_execution_record
from .pie_adapter_execution import _execute as execute
from .pie_adapter_execution import _install_limits
from .retained_fs import RetainedDirectory
from .sealed_process import run_bounded_child


EXPECTED = b"exact-prover-input\x00\x01"


def _write(path: Path, payload: bytes, mode: int = 0o600) -> Path:
    path.write_bytes(payload)
    os.chmod(path, mode)
    return path


def _swap_parent(path: Path) -> None:
    moved = path.parent.with_name(f"{path.parent.name}.bound")
    path.parent.rename(moved)
    path.parent.symlink_to(moved.name, target_is_directory=True)


def _replace_parent(path: Path) -> tuple[Path, Path]:
    moved = path.parent.with_name(f"{path.parent.name}.bound")
    path.parent.rename(moved)
    path.parent.mkdir()
    return moved, _write(path.parent / "attacker-sentinel", b"untouched")


def _identity(role: str, path: Path) -> dict[str, Any]:
    return {"kind": ARTIFACT_SPECS[role][0], "path": str(path.resolve()),
            "byte_length": path.stat().st_size, "sha256": sha256_file(path)}


def _seal(role: str, path: Path) -> dict[str, Any]:
    value = _identity(role, path)
    value["path"] = path.name
    return value


def _unused_seal(kind: str, name: str) -> dict[str, Any]:
    return {"kind": kind, "path": name, "byte_length": 1, "sha256": sha256_bytes(name.encode())}


def _proof_shape() -> dict[str, Any]:
    shape = {
        "schema_version": "stwo.gpu-lab.cairo-proof-shape.v1",
        "pcs": {"pow_bits": 26, "log_blowup_factor": 1,
                "log_last_layer_degree_bound": 0, "n_queries": 3,
                "fold_step": 1, "lifting_log_size": None},
        "channel_salt": 0, "preprocessed_trace_variant_sha256": "1" * 64,
        "component_slots": 1, "component_enable_bits_sha256": "2" * 64,
        "component_log_sizes": [1], "trace_column_log_sizes": [[1]],
        "public_data_word_counts": [0, 0, 0], "interaction_claim_felts": 0,
        "commitment_trees": 1, "sampled_value_counts": [[1]],
        "decommitment_trees": 1, "queried_value_counts": [[1]],
        "fri_inner_layers": 1, "fri_witness_counts": [1],
        "fri_last_layer_coefficients": 1, "unsorted_query_locations": 1,
    }
    return {"sha256": sha256_bytes(canonical_bytes(shape)), "shape": shape}


def _fake_executable(payload: bytes = EXPECTED) -> bytes:
    literal = "".join(f"\\{byte:03o}" for byte in payload)
    return (
        "#!/bin/sh\n"
        f"printf '{literal}' > \"$STWO_DUMP_INPUT\"\n"
        f"printf 'prover input dumped: {len(payload)} bytes -> %s\\n' "
        '"$STWO_DUMP_INPUT" >&2\n'
    ).encode()


def _fixture(root: Path) -> tuple[SimpleNamespace, tuple[Path, ...]]:
    root.mkdir()
    source = _write(root / "source.zip", b"sealed PIE")
    bootloader = _write(root / "bootloader.json", b'{"program":"sealed"}\n')
    expected = _write(root / "expected.bin", EXPECTED)
    executable = _write(root / "adapter", _fake_executable(), 0o700)
    source_closure = _write(root / "adapter-sources.json", b'{"sealed":true}\n')
    bootloader_identity = _identity("bootloader_program", bootloader)
    invocation_value = {
        "schema_version": INVOCATION_SCHEMA, "protocol": PROTOCOL,
        "bootloader_program": bootloader_identity, "pie_copies": 1,
        "backend": "simd", "engine": "legacy", "output_encoding": OUTPUT_ENCODING,
    }
    invocation = _write(root / "invocation.json", invocation_bytes(invocation_value))
    legacy = {"kind": "adapter-run-json-v1", "path": "legacy-run.json",
              "byte_length": 17, "sha256": sha256_bytes(b"legacy")}
    manifest = {
        "schema_version": "stwo.gpu-lab.fri-round6-provenance.v1",
        "status": "captured-unsealed", "production_admissible": False,
        "source_pie": _seal("source_pie", source),
        "adapted_prover_input": _seal("expected_prover_input", expected),
        "adapter": {"run_record": legacy, "invocation": _seal("adapter_invocation", invocation),
                    "executable": _seal("adapter_executable", executable),
                    "source_closure": _seal("adapter_source_closure", source_closure)},
        "extended_cairo_proof_bincode": _unused_seal(
            "extended-cairo-proof-bincode-v1", "proof.bin"),
        "canonical_cairo_transport": _unused_seal(
            "canonical-cairo-proof-felts-be32-v1", "transport.bin"),
        "verifier_source_closure": _unused_seal(
            "verifier-source-closure-json-v1", "verifier-sources.json"),
        "proof_shape": _proof_shape(),
    }
    provenance = _write(root / "provenance.json", canonical_bytes(manifest) + b"\n")
    paths = {
        "provenance_manifest": provenance, "source_pie": source,
        "bootloader_program": bootloader, "expected_prover_input": expected,
        "adapter_executable": executable, "adapter_invocation": invocation,
        "adapter_source_closure": source_closure,
    }
    args = SimpleNamespace(observed_prover_input=root / "observed.bin",
                           record=root / "execution.json")
    for role, path in paths.items():
        setattr(args, role, path)
        setattr(args, f"{role}_sha256", sha256_file(path))
    tool_dir = root / "tool-sources"
    tool_dir.mkdir()
    tools = tuple(_write(tool_dir / name, f"source {name}\n".encode())
                  for name in ("launcher.py", "contract.py", "execution.py"))
    return args, tools


def _expect_rejection(label: str, operation: Callable[[], Any]) -> None:
    try:
        operation()
    except (OSError, ValueError):
        return
    raise ValueError(f"hostile PIE adapter case was accepted: {label}")


def _runner(args: SimpleNamespace, behavior: str) -> Callable[..., subprocess.CompletedProcess]:
    def run(command: list[str], cwd: Path, descriptors: tuple[int, ...],
            environment: dict[str, str], file_size: int) -> subprocess.CompletedProcess[bytes]:
        require(cwd == Path("/"), "adapter child working directory is not fixed root")
        expected_environment = {
            "LANG": "C", "LC_ALL": "C", "RAYON_NUM_THREADS": "8", "RUST_BACKTRACE": "0",
            "STWO_BOOTLOADER_JSON": command[0].replace(
                str(int(command[0].rsplit("/", 1)[1])),
                str(int(environment["STWO_BOOTLOADER_JSON"].rsplit("/", 1)[1]))),
            "STWO_DUMP_INPUT": environment["STWO_DUMP_INPUT"],
        }
        require(environment == expected_environment, "adapter environment was not cleared/fixed")
        require(file_size == len(EXPECTED), "adapter output-sized file limit differs")
        inherited = {int(command[0].rsplit("/", 1)[1]),
                     int(command[2].rsplit("/", 1)[1]),
                     int(environment["STWO_BOOTLOADER_JSON"].rsplit("/", 1)[1]),
                     int(environment["STWO_DUMP_INPUT"].rsplit("/", 1)[1])}
        require(set(descriptors) == inherited and len(descriptors) == 4,
                "adapter inherited undeclared descriptors or its golden")
        output_fd = int(environment["STWO_DUMP_INPUT"].rsplit("/", 1)[1])
        initial = os.fstat(output_fd)
        require(initial.st_size == 0 and initial.st_nlink == 1
                and stat.S_IMODE(initial.st_mode) == 0o600,
                "adapter output fd was not prebound empty and single-link")
        payload = EXPECTED
        if behavior == "wrong-byte":
            payload = bytes([EXPECTED[0] ^ 1]) + EXPECTED[1:]
        elif behavior == "truncated":
            payload = EXPECTED[:-1]
        elif behavior == "oversized":
            payload = EXPECTED + b"x"
        os.ftruncate(output_fd, 0)
        os.pwrite(output_fd, payload, 0)
        if behavior.endswith("-rebind") and behavior != "output-rebind":
            role = {"source": "source_pie", "bootloader": "bootloader_program",
                    "executable": "adapter_executable", "invocation": "adapter_invocation",
                    "closure": "adapter_source_closure", "manifest": "provenance_manifest"}[
                        behavior.removesuffix("-rebind")]
            path = getattr(args, role)
            path.rename(path.with_suffix(path.suffix + ".bound"))
            _write(path, b"replacement", 0o700 if role == "adapter_executable" else 0o600)
        if behavior == "output-rebind":
            args.observed_prover_input.rename(args.observed_prover_input.with_suffix(".bound"))
            _write(args.observed_prover_input, b"replacement")
        if behavior == "expected-mutation":
            os.chmod(args.expected_prover_input, 0o600)
            args.expected_prover_input.write_bytes(EXPECTED[:-1] + b"z")
        if behavior == "parent-symlink":
            _swap_parent(args.source_pie)
        status = 3 if behavior == "nonzero" else 0
        stdout = b"injected" if behavior == "stdout" else b""
        stderr = f"prover input dumped: {len(payload)} bytes -> {environment['STWO_DUMP_INPUT']}\n".encode()
        if behavior == "stderr":
            stderr = b"wrong stderr\n"
        if behavior == "timeout":
            raise ValueError("PIE adapter exceeded 300s wall timeout")
        return subprocess.CompletedProcess(command, status, stdout, stderr)
    return run


def _run(root: Path, behavior: str = "success") -> tuple[dict[str, Any], SimpleNamespace]:
    args, tools = _fixture(root)
    descriptors_before = len(os.listdir("/dev/fd"))
    record = execute(args, _runner=_runner(args, behavior), _tool_paths=tools)
    require(len(os.listdir("/dev/fd")) == descriptors_before,
            "PIE adapter replay leaked retained descriptors")
    return record, args


def _failure_case(parent: Path, label: str, behavior: str) -> None:
    root = parent / label
    args, tools = _fixture(root)
    _expect_rejection(label, lambda: execute(
        args, _runner=_runner(args, behavior), _tool_paths=tools
    ))
    require(not args.record.exists(), f"{label} left a PASS record")
    if not behavior.endswith("-rebind"):
        require(not args.observed_prover_input.exists(), f"{label} left its bound output")


def _input_boundary_cases(parent: Path) -> None:
    args, tools = _fixture(parent / "input-parent-replacement-before-open")
    real_open = execution_module.RetainedLeaf.open
    moved: Path | None = None
    sentinel: Path | None = None
    def replace_before_open(leaf, flags, mode=0o777):
        nonlocal moved, sentinel
        if moved is None and leaf.label == "provenance_manifest":
            moved, sentinel = _replace_parent(leaf.path)
        return real_open(leaf, flags, mode)
    with patch.object(execution_module.RetainedLeaf, "open", new=replace_before_open):
        _expect_rejection("input parent replacement before open", lambda: execute(
            args, _runner=_runner(args, "success"), _tool_paths=tools
        ))
    require(moved is not None and sentinel is not None and sentinel.read_bytes() == b"untouched"
            and (moved / "provenance.json").exists() and not args.record.exists(),
            "input parent replacement escaped anchoring or touched attacker state")

    args, tools = _fixture(parent / "invocation-mutation")
    hostile = json.loads(args.adapter_invocation.read_text())
    hostile["engine"] = "hostile"
    args.adapter_invocation.write_bytes(canonical_bytes(hostile) + b"\n")
    args.adapter_invocation_sha256 = sha256_file(args.adapter_invocation)
    _expect_rejection("invocation mutation", lambda: execute(
        args, _runner=_runner(args, "success"), _tool_paths=tools
    ))

    args, tools = _fixture(parent / "manifest-mutation")
    args.provenance_manifest.write_text('{"schema_version":"wrong"}\n')
    args.provenance_manifest_sha256 = sha256_file(args.provenance_manifest)
    _expect_rejection("manifest mutation", lambda: execute(
        args, _runner=_runner(args, "success"), _tool_paths=tools
    ))

    for label, mutate in (
        ("manifest float size", lambda value: value["source_pie"].__setitem__(
            "byte_length", float(value["source_pie"]["byte_length"]))),
        ("manifest backslash path", lambda value: value["source_pie"].__setitem__(
            "path", "hostile\\source.zip")),
    ):
        args, tools = _fixture(parent / label.replace(" ", "-"))
        manifest = json.loads(args.provenance_manifest.read_text())
        mutate(manifest)
        args.provenance_manifest.write_bytes(canonical_bytes(manifest) + b"\n")
        args.provenance_manifest_sha256 = sha256_file(args.provenance_manifest)
        def must_not_run(*_):
            raise AssertionError(f"{label} reached the adapter child")
        _expect_rejection(label, lambda args=args, tools=tools: execute(
            args, _runner=must_not_run, _tool_paths=tools
        ))

    args, tools = _fixture(parent / "leaf-symlink")
    real = args.source_pie.rename(args.source_pie.with_name("real-source.zip"))
    args.source_pie.symlink_to(real)
    args.source_pie_sha256 = sha256_file(real)
    _expect_rejection("leaf symlink", lambda: execute(
        args, _runner=_runner(args, "success"), _tool_paths=tools
    ))

    real = parent / "component-real"
    args, tools = _fixture(real)
    linked = parent / "component-link"
    linked.symlink_to(real, target_is_directory=True)
    args.source_pie = linked / args.source_pie.name
    _expect_rejection("component symlink", lambda: execute(
        args, _runner=_runner(args, "success"), _tool_paths=tools
    ))

    args, tools = _fixture(parent / "hardlink-alias")
    args.bootloader_program.unlink()
    os.link(args.source_pie, args.bootloader_program)
    args.bootloader_program_sha256 = sha256_file(args.bootloader_program)
    _expect_rejection("hardlink alias", lambda: execute(
        args, _runner=_runner(args, "success"), _tool_paths=tools
    ))

    args, tools = _fixture(parent / "output-record-alias")
    args.observed_prover_input = args.record
    _expect_rejection("output record alias", lambda: execute(
        args, _runner=_runner(args, "success"), _tool_paths=tools
    ))

    args, tools = _fixture(parent / "tool-input-alias")
    _expect_rejection("tool input alias", lambda: execute(
        args, _runner=_runner(args, "success"), _tool_paths=(args.source_pie, *tools)
    ))

    args, tools = _fixture(parent / "oversized-executable")
    with args.adapter_executable.open("r+b") as handle:
        handle.truncate(ARTIFACT_SPECS["adapter_executable"][1] + 1)
    args.adapter_executable_sha256 = "0" * 64
    _expect_rejection("oversized executable", lambda: execute(
        args, _runner=_runner(args, "success"), _tool_paths=tools
    ))


def _freshness_and_publication_cases(parent: Path) -> None:
    for role in ("observed_prover_input", "record"):
        args, tools = _fixture(parent / f"stale-{role}")
        getattr(args, role).write_bytes(b"stale")
        _expect_rejection(f"stale {role}", lambda args=args, tools=tools: execute(
            args, _runner=_runner(args, "success"), _tool_paths=tools
        ))
    args, tools = _fixture(parent / "publication-failure")
    with patch("gpu_lab.pie_adapter_execution._link_record",
               side_effect=OSError("injected publication failure")):
        _expect_rejection("publication failure", lambda: execute(
            args, _runner=_runner(args, "success"), _tool_paths=tools
        ))
    require(not args.record.exists() and not args.observed_prover_input.exists(),
            "failed publication left an attestation or observed output")

    args, tools = _fixture(parent / "output-parent-swap-before-open")
    real_open = execution_module.RetainedLeaf.open
    swapped = False
    def swap_before_open(leaf, flags, mode=0o777):
        nonlocal swapped
        if not swapped and leaf.label == "observed ProverInput":
            _swap_parent(leaf.path)
            swapped = True
        return real_open(leaf, flags, mode)
    with patch.object(execution_module.RetainedLeaf, "open", new=swap_before_open):
        _expect_rejection("output parent swap before open", lambda: execute(
            args, _runner=_runner(args, "success"), _tool_paths=tools
        ))
    require(swapped and not args.record.exists() and not args.observed_prover_input.exists(),
            "output parent swap before open was accepted or leaked artifacts")

    args, tools = _fixture(parent / "output-parent-swap-after-create")
    real_verify = RetainedDirectory.verify
    verify_calls = 0
    def swap_after_create(directory, label):
        nonlocal verify_calls
        if label == "observed ProverInput":
            verify_calls += 1
            if verify_calls == 4:
                _swap_parent(args.observed_prover_input)
        return real_verify(directory, label)
    with patch.object(RetainedDirectory, "verify", new=swap_after_create):
        _expect_rejection("output parent swap after create", lambda: execute(
            args, _runner=_runner(args, "success"), _tool_paths=tools
        ))
    require(verify_calls == 4 and not args.record.exists()
            and not args.observed_prover_input.exists(),
            "output parent swap after create was accepted or leaked artifacts")


def _publisher_early_failure_cases(parent: Path) -> None:
    real_fchmod, real_write, real_fsync = os.fchmod, os.write, os.fsync
    def fail_record_fchmod(descriptor, mode):
        if os.fstat(descriptor).st_size == 0 and mode == 0o400:
            raise OSError("injected record fchmod failure")
        return real_fchmod(descriptor, mode)
    def fail_record_write(descriptor, payload):
        raise OSError("injected record write failure")
    def fail_record_fsync(descriptor):
        metadata = os.fstat(descriptor)
        if (stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o400
                and metadata.st_size > len(EXPECTED)):
            raise OSError("injected record fsync failure")
        return real_fsync(descriptor)
    for name, replacement in (("fchmod", fail_record_fchmod), ("write", fail_record_write),
                              ("fsync", fail_record_fsync)):
        args, tools = _fixture(parent / f"record-{name}-failure")
        with patch(f"gpu_lab.pie_adapter_execution.os.{name}", side_effect=replacement):
            _expect_rejection(f"record {name} failure", lambda: execute(
                args, _runner=_runner(args, "success"), _tool_paths=tools
            ))
        require(not args.record.exists() and not args.observed_prover_input.exists()
                and not list(args.record.parent.glob(f".{args.record.name}.*")),
                f"record {name} failure leaked temporary or authenticated artifacts")


def _retained_fs_api_cases(parent: Path) -> None:
    root = parent / "retained-fs-api"
    root.mkdir()
    directory = RetainedDirectory.bind(root, "retained filesystem test")
    descriptors = directory.descriptors
    try:
        require(all(descriptor >= 3 and not os.get_inheritable(descriptor)
                    for descriptor in descriptors),
                "retained directory descriptor is standard I/O or inheritable")
        hostile = (
            lambda: directory.open("../escape", os.O_RDONLY, "hostile open"),
            lambda: directory.stat("/etc/passwd"),
            lambda: directory.link("source", "../escape"),
            lambda: directory.unlink_if_same("../escape", root.stat()),
        )
        for index, operation in enumerate(hostile):
            _expect_rejection(f"retained filesystem leaf escape {index}", operation)
    finally:
        directory.close()
    for descriptor in descriptors:
        _expect_rejection("closed retained directory descriptor",
                          lambda descriptor=descriptor: os.fstat(descriptor))


def _tool_mutation_case(parent: Path) -> None:
    args, tools = _fixture(parent / "tool-mutation")
    real_runner = _runner(args, "success")
    def mutate(*runner_args):
        tools[0].write_text("mutated tool\n")
        return real_runner(*runner_args)
    _expect_rejection("tool mutation", lambda: execute(args, _runner=mutate, _tool_paths=tools))
    require(not args.record.exists(), "tool mutation left a PASS record")


def _record_rebinding_cases(parent: Path) -> None:
    real_link = execution_module._link_record
    for behavior in ("same-payload", "symlink"):
        args, tools = _fixture(parent / f"record-{behavior}-replacement")
        published = args.record.with_suffix(".published")
        def replace(destination, temporary):
            real_link(destination, temporary)
            path = destination.path
            path.rename(published)
            if behavior == "same-payload":
                _write(path, published.read_bytes(), 0o400)
            else:
                path.symlink_to(published)
        with patch("gpu_lab.pie_adapter_execution._link_record", side_effect=replace):
            _expect_rejection(f"record {behavior} replacement", lambda: execute(
                args, _runner=_runner(args, "success"), _tool_paths=tools
            ))
        require(published.exists() and not args.observed_prover_input.exists(),
                f"record {behavior} replacement was accepted or leaked output")

    args, tools = _fixture(parent / "record-replacement-during-hash")
    published = args.record.with_suffix(".published")
    real_hash = execution_module._sha256_descriptor
    swapped = False
    def replace_during_hash(descriptor: int, length: int) -> str:
        nonlocal swapped
        digest = real_hash(descriptor, length)
        metadata = os.fstat(descriptor)
        if (not swapped and stat.S_IMODE(metadata.st_mode) == 0o400
                and args.record.exists()):
            path_state = args.record.lstat()
            if (path_state.st_dev, path_state.st_ino) == (metadata.st_dev, metadata.st_ino):
                args.record.rename(published)
                _write(args.record, published.read_bytes(), 0o400)
                swapped = True
        return digest
    with patch("gpu_lab.pie_adapter_execution._sha256_descriptor",
               side_effect=replace_during_hash):
        _expect_rejection("record replacement during hash", lambda: execute(
            args, _runner=_runner(args, "success"), _tool_paths=tools
        ))
    require(swapped and published.exists() and args.record.exists()
            and not published.samefile(args.record)
            and not args.observed_prover_input.exists(),
            "during-hash record replacement was accepted or leaked output")

    args, tools = _fixture(parent / "record-parent-swap-after-link")
    real_link = execution_module._link_record
    swapped = False
    def swap_after_link(destination, temporary):
        nonlocal swapped
        real_link(destination, temporary)
        _swap_parent(destination.path)
        swapped = True
    with patch("gpu_lab.pie_adapter_execution._link_record", side_effect=swap_after_link):
        _expect_rejection("record parent swap after link", lambda: execute(
            args, _runner=_runner(args, "success"), _tool_paths=tools
        ))
    require(swapped and not args.record.exists() and not args.observed_prover_input.exists(),
            "record parent swap after link was accepted or leaked artifacts")

    args, tools = _fixture(parent / "record-parent-replacement-before-temp")
    real_open = RetainedDirectory.open
    moved = sentinel = None
    def replace_before_temp(directory, name, flags, label, mode=0o777):
        nonlocal moved, sentinel
        if moved is None and label == "PIE adapter record temporary":
            moved, sentinel = _replace_parent(args.record)
        return real_open(directory, name, flags, label, mode)
    with patch.object(RetainedDirectory, "open", new=replace_before_temp):
        _expect_rejection("record parent replacement before temp", lambda: execute(
            args, _runner=_runner(args, "success"), _tool_paths=tools
        ))
    require(moved is not None and sentinel is not None and sentinel.read_bytes() == b"untouched"
            and not (moved / "observed.bin").exists() and not args.record.exists(),
            "record parent replacement escaped cleanup anchoring or touched attacker state")


def _post_publication_mutation_cases(parent: Path) -> None:
    real_link = execution_module._link_record
    for behavior in ("source", "output"):
        args, tools = _fixture(parent / f"post-publication-{behavior}-mutation")
        def mutate(destination, temporary):
            real_link(destination, temporary)
            target = (args.source_pie if behavior == "source"
                      else args.observed_prover_input)
            if behavior == "output":
                os.chmod(target, 0o600)
                target.write_bytes(b"x" * len(EXPECTED))
            else:
                target.write_bytes(b"hostilePIE")
        with patch("gpu_lab.pie_adapter_execution._link_record", side_effect=mutate):
            _expect_rejection(f"post-publication {behavior} mutation", lambda: execute(
                args, _runner=_runner(args, "success"), _tool_paths=tools
            ))
        require(not args.record.exists() and not args.observed_prover_input.exists(),
                f"post-publication {behavior} mutation left authenticated artifacts")

    args, tools = _fixture(parent / "output-mutation-after-content-sweep")
    real_verify = execution_module._verify_bound
    content_checks = 0
    def mutate_after_sweep(bound, *, content: bool):
        nonlocal content_checks
        real_verify(bound, content=content)
        if content:
            content_checks += 1
            if content_checks == 2 * (len(ARTIFACT_SPECS) + len(tools)):
                os.chmod(args.observed_prover_input, 0o600)
                args.observed_prover_input.write_bytes(b"x" * len(EXPECTED))
    with patch("gpu_lab.pie_adapter_execution._verify_bound", side_effect=mutate_after_sweep):
        _expect_rejection("output mutation after content sweep", lambda: execute(
            args, _runner=_runner(args, "success"), _tool_paths=tools
        ))
    require(content_checks == 2 * (len(ARTIFACT_SPECS) + len(tools))
            and not args.record.exists() and not args.observed_prover_input.exists(),
            "output tail mutation was accepted or left authenticated artifacts")


def _record_mutations(record: dict[str, Any], args: SimpleNamespace) -> None:
    for label, mutate in (
        ("admission", lambda value: value.__setitem__("production_admissible", True)),
        ("bool inode", lambda value: value["execution_contract"]["bindings"]
         ["source_pie"].__setitem__("inode", True)),
        ("output lifecycle", lambda value: value["output_lifecycle"]
         ["pre_launch"].__setitem__("byte_length", 1)),
        ("provenance trust", lambda value: value["provenance_binding"]
         .__setitem__("trust_root", "self-declared")),
    ):
        hostile = copy.deepcopy(record)
        mutate(hostile)
        _expect_rejection(label, lambda hostile=hostile: validate_execution_record(
            hostile, provenance_manifest_bytes=args.provenance_manifest.read_bytes(),
            expected_provenance_manifest_sha256=args.provenance_manifest_sha256,
        ))
    manifest = json.loads(args.provenance_manifest.read_text())
    manifest["extended_cairo_proof_bincode"]["sha256"] = "f" * 64
    substituted = canonical_bytes(manifest) + b"\n"
    hostile = copy.deepcopy(record)
    hostile["provenance_manifest"]["byte_length"] = len(substituted)
    hostile["provenance_manifest"]["sha256"] = sha256_bytes(substituted)
    _expect_rejection("self-consistent manifest substitution", lambda: validate_execution_record(
        hostile, provenance_manifest_bytes=substituted,
        expected_provenance_manifest_sha256=args.provenance_manifest_sha256,
    ))


def _sealed_process_extension_test(root: Path) -> None:
    marker = _write(root / "child-setup-marker", b"")
    descriptor = os.open(marker, os.O_RDWR)
    try:
        result = run_bounded_child(
            [sys.executable, "-c", "import os;os.write(2,b'accepted')"], root,
            (descriptor,), {}, timeout_seconds=2, max_stdout_bytes=0,
            process_name="adapter test", stdout_bound_name="zero bytes",
            max_stderr_bytes=8, stderr_bound_name="8 bytes",
            child_setup=lambda: os.write(descriptor, b"setup"),
        )
    finally:
        os.close(descriptor)
    require(result.stderr == b"accepted" and marker.read_bytes() == b"setup",
            "sealed process did not return bounded stderr or run child setup")
    _expect_rejection("bounded stderr overflow", lambda: run_bounded_child(
        [sys.executable, "-c", "import os;os.write(2,b'123456789')"], root, (), {},
        timeout_seconds=2, max_stdout_bytes=0, process_name="adapter test",
        stdout_bound_name="zero bytes", max_stderr_bytes=8, stderr_bound_name="8 bytes",
    ))
    configured: list[tuple[int, tuple[int, int]]] = []
    with patch("gpu_lab.pie_adapter_execution.resource.getrlimit",
               return_value=(resource.RLIM_INFINITY, resource.RLIM_INFINITY)), patch(
                   "gpu_lab.pie_adapter_execution.resource.setrlimit",
                   side_effect=lambda key, value: configured.append((key, value))):
        _install_limits(len(EXPECTED))()
    require(len(configured) == 4 and configured[-1][1] == (len(EXPECTED), len(EXPECTED)),
            "adapter resource-limit callback differs")


def pie_adapter_execution_self_test(lab_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix=".pie-adapter-execution-test-", dir=lab_root) as name:
        parent = Path(name).resolve()
        record, args = _run(parent / "success")
        require(record["exact_byte_equal"] is True and args.observed_prover_input.read_bytes()
                == EXPECTED and stat.S_IMODE(args.observed_prover_input.stat().st_mode) == 0o400,
                "valid fake adapter did not publish exact immutable output")
        require(all(record["execution_contract"].get(key) == value
                    for key, value in EXECUTION_HONESTY.items()),
                "execution honesty fields did not survive exact record parsing")
        _record_mutations(record, args)
        with patch.dict(os.environ, {"LD_PRELOAD": "/tmp/hostile.so", "STWO_ADAPT_ONLY": "1"}):
            inherited_record, _ = _run(parent / "hostile-parent-environment")
        require(inherited_record["passed"] is True, "parent environment reached the adapter")
        for behavior in (
            "wrong-byte", "truncated", "oversized", "nonzero", "stdout", "stderr", "timeout",
            "source-rebind", "bootloader-rebind", "executable-rebind", "invocation-rebind",
            "closure-rebind", "manifest-rebind", "output-rebind", "expected-mutation",
            "parent-symlink",
        ):
            _failure_case(parent, behavior, behavior)
        _input_boundary_cases(parent)
        _freshness_and_publication_cases(parent)
        _publisher_early_failure_cases(parent)
        _retained_fs_api_cases(parent)
        _tool_mutation_case(parent)
        _record_rebinding_cases(parent)
        _post_publication_mutation_cases(parent)
        _sealed_process_extension_test(parent)
        if sys.platform == "linux" and Path("/proc/self/fd").is_dir():
            args, tools = _fixture(parent / "linux-real-process")
            require(execute(args, _tool_paths=tools)["passed"] is True,
                    "real Linux proc-fd fake adapter failed")


if __name__ == "__main__":
    pie_adapter_execution_self_test(LAB_ROOT)
    print("pie_adapter_execution_tests: PASS")
