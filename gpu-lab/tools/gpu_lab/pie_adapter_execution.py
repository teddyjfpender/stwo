"""Authenticated, GPU-free execution of the sealed PIE adapter boundary."""
# gpu-lab-cohesion-review: one retained transaction keeps cleanup and attestation auditable.

from __future__ import annotations

import fcntl
import hashlib
import os
import resource
import secrets
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .common import LAB_ROOT, canonical_bytes, require
from .common import require_sha256, sha256_bytes
from .retained_fs import RetainedLeaf
from .sealed_process import run_bounded_child
from .pie_adapter_contract import (
    ADDRESS_SPACE_LIMIT_BYTES, ARTIFACT_SPECS, CPU_LIMIT_SECONDS, DATA_LIMIT_BYTES,
    EMPTY_SHA256, EXECUTION_HONESTY, INVOCATION_SCHEMA, MAX_EXECUTION_RECORD_BYTES,
    MAX_STREAM_BYTES,
    OUTPUT_ENCODING, PROTOCOL, WALL_TIMEOUT_SECONDS, invocation_bytes, parse_execution_record_bytes,
    parse_invocation_bytes, validate_adapter_provenance_projection, validate_execution_record,
)

EXECUTABLE_POLICY = "release-or-stripped-provenance-compatible-max-256mib-v1"
TOOL_RELATIVE_PATHS = tuple(f"tools/{path}" for path in (
    "pie-adapter-replay", "gpu_lab/__init__.py", "gpu_lab/common.py",
    "gpu_lab/pie_adapter_contract.py", "gpu_lab/pie_adapter_execution.py",
    "gpu_lab/retained_fs.py", "gpu_lab/sealed_process.py",
))
INHERITED_ROLES = {
    "adapter_executable", "source_pie", "bootloader_program", "observed_prover_input",
}
Fingerprint = tuple[int, int, int, int, int, int, int]

@dataclass(frozen=True)
class BoundFile:
    role: str; kind: str; leaf: RetainedLeaf; sha256: str
    byte_length: int; fingerprint: Fingerprint; descriptor: int
    @property
    def path(self) -> Path: return self.leaf.path
    @property
    def proc_path(self) -> str: return f"/proc/self/fd/{self.descriptor}"
    def artifact(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": str(self.path), "byte_length": self.byte_length,
                "sha256": self.sha256}
@dataclass(frozen=True)
class OutputFile:
    leaf: RetainedLeaf; descriptor: int; initial: os.stat_result
    expected_sha256: str; expected_bytes: int
    @property
    def path(self) -> Path: return self.leaf.path
    @property
    def proc_path(self) -> str: return f"/proc/self/fd/{self.descriptor}"
    def artifact(self) -> dict[str, Any]:
        return {"kind": ARTIFACT_SPECS["expected_prover_input"][0],
                "path": str(self.path), "byte_length": self.expected_bytes,
                "sha256": self.expected_sha256}
def _fingerprint(metadata: os.stat_result) -> Fingerprint:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns,
            metadata.st_ctime_ns, stat.S_IMODE(metadata.st_mode), metadata.st_nlink)
def _descriptor_at_least_three(descriptor: int) -> int:
    if descriptor >= 3:
        return descriptor
    duplicate = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 3)
    try:
        os.close(descriptor)
    except BaseException as primary:
        try: os.close(duplicate)
        except BaseException as cleanup: raise primary from cleanup
        raise
    return duplicate
def _attempt(first_error: BaseException | None, operation: Callable[[], Any]) -> BaseException | None:
    try: operation()
    except BaseException as error: return first_error or error
    return first_error
def _sha256_descriptor(descriptor: int, length: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < length:
        try:
            block = os.pread(descriptor, min(1024 * 1024, length - offset), offset)
        except InterruptedError:
            continue
        require(block, "sealed PIE adapter input ended early")
        digest.update(block)
        offset += len(block)
    require(os.pread(descriptor, 1, length) == b"", "sealed PIE adapter input grew")
    return digest.hexdigest()
def _read_descriptor(bound: BoundFile, maximum: int) -> bytes:
    require(bound.byte_length <= maximum, f"{bound.role} exceeds its read bound")
    result = bytearray()
    offset = 0
    while offset < bound.byte_length:
        block = os.pread(bound.descriptor, min(1024 * 1024, bound.byte_length - offset), offset)
        require(block, f"{bound.role} ended early")
        result.extend(block)
        offset += len(block)
    return bytes(result)
def _close_bound(bound: BoundFile) -> None:
    error = _attempt(None, lambda: os.close(bound.descriptor))
    error = _attempt(error, bound.leaf.close)
    if error is not None: raise error
def _close_bounds(bounds: list[BoundFile]) -> None:
    error: BaseException | None = None
    for bound in bounds:
        error = _attempt(error, lambda bound=bound: _close_bound(bound))
    if error is not None: raise error
def _bind(path: Path, digest: str | None, role: str, kind: str, maximum: int) -> BoundFile:
    if digest is not None: require_sha256(digest, f"{role} sha256")
    leaf = RetainedLeaf.bind(path, role)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = _descriptor_at_least_three(leaf.open(flags))
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
                f"{role} is not a single-link regular file")
        require(0 < metadata.st_size <= maximum, f"{role} byte length is out of bounds")
        if role == "adapter_executable":
            require(metadata.st_size <= ARTIFACT_SPECS[role][1], "adapter executable exceeds "
                    "the provenance-compatible 256 MiB policy")
            require(metadata.st_mode & 0o111, "adapter executable is not executable")
        fingerprint = _fingerprint(metadata)
        observed_digest = _sha256_descriptor(descriptor, metadata.st_size)
        digest = digest or observed_digest
        require(observed_digest == digest,
                f"{role} sha256 differs")
        require(_fingerprint(leaf.stat()) == fingerprint,
                f"{role} path changed while it was bound")
        leaf.verify()
        return BoundFile(role, kind, leaf, digest, metadata.st_size, fingerprint, descriptor)
    except BaseException as primary:
        cleanup: BaseException | None = None
        if descriptor is not None:
            cleanup = _attempt(cleanup, lambda: os.close(descriptor))
        cleanup = _attempt(cleanup, leaf.close)
        if cleanup is not None: raise primary from cleanup
        raise
def _verify_bound(bound: BoundFile, *, content: bool) -> None:
    bound.leaf.verify()
    require(_fingerprint(os.fstat(bound.descriptor)) == bound.fingerprint,
            f"{bound.role} descriptor changed")
    require(_fingerprint(bound.leaf.stat()) == bound.fingerprint,
            f"{bound.role} path was rebound")
    if content:
        require(_sha256_descriptor(bound.descriptor, bound.byte_length) == bound.sha256,
                f"{bound.role} content changed")
        require(_fingerprint(os.fstat(bound.descriptor)) == bound.fingerprint,
                f"{bound.role} descriptor changed while it was rehashed")
        require(_fingerprint(bound.leaf.stat()) == bound.fingerprint,
                f"{bound.role} path changed while it was rehashed")
    bound.leaf.verify()
def _tool_sources(paths: tuple[Path, ...] | None) -> tuple[list[BoundFile], dict[str, Any]]:
    selected = paths or tuple(LAB_ROOT / relative for relative in TOOL_RELATIVE_PATHS)
    bounds: list[BoundFile] = []
    try:
        for index, path in enumerate(selected):
            canonical = Path(os.path.abspath(path))
            relative = canonical.relative_to(LAB_ROOT).as_posix()
            bounds.append(_bind(canonical, None, f"replay tool source {relative}",
                                "python-source", 2 * 1024 * 1024))
        sources = [{"path": bound.path.relative_to(LAB_ROOT).as_posix(),
                    "sha256": bound.sha256} for bound in bounds]
        require(len({item["path"] for item in sources}) == len(sources),
                "replay tool source paths are not distinct")
        return bounds, {"sources": sources,
                        "closure_sha256": sha256_bytes(canonical_bytes(sources))}
    except BaseException as primary:
        cleanup = _attempt(None, lambda: _close_bounds(bounds))
        if cleanup is not None: raise primary from cleanup
        raise
def _rollback_fresh_leaf(leaf: RetainedLeaf,
                         identity: os.stat_result | None) -> None:
    error: BaseException | None = None
    def remove() -> None:
        if identity is not None:
            leaf.unlink_if_same(identity)
            return
        try: os.unlink(leaf.name, dir_fd=leaf.directory.descriptor)
        except FileNotFoundError: pass
    # A second identity-checked attempt handles a one-shot unlink failure safely.
    error = _attempt(error, remove)
    error = _attempt(error, remove)
    error = _attempt(error, leaf.directory.sync)
    if error is not None: raise error
def _create_output(path: Path, expected: BoundFile,
                   aliases: set[tuple[int, int, str]]) -> OutputFile:
    leaf = RetainedLeaf.bind(path, "observed ProverInput", fresh=True)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor: int | None = None
    initial: os.stat_result | None = None
    try:
        require(leaf.key not in aliases, "observed ProverInput aliases an input")
        descriptor = _descriptor_at_least_three(leaf.open(flags, 0o600))
        os.fchmod(descriptor, 0o600)
        initial = os.fstat(descriptor)
        require(stat.S_ISREG(initial.st_mode) and initial.st_size == 0
                and initial.st_nlink == 1 and stat.S_IMODE(initial.st_mode) == 0o600,
                "observed ProverInput initial inode is not empty single-link mode 0600")
        path_state = leaf.stat()
        require((path_state.st_dev, path_state.st_ino) == (initial.st_dev, initial.st_ino),
                "observed ProverInput path differs from its retained inode")
        leaf.verify()
        return OutputFile(leaf, descriptor, initial, expected.sha256, expected.byte_length)
    except BaseException as primary:
        cleanup: BaseException | None = None
        cleanup_identity = initial
        if cleanup_identity is None and descriptor is not None:
            try: cleanup_identity = os.stat(descriptor)
            except BaseException as error: cleanup = cleanup or error
        cleanup = _attempt(
            cleanup, lambda: _rollback_fresh_leaf(leaf, cleanup_identity)
        )
        if descriptor is not None:
            cleanup = _attempt(cleanup, lambda: os.close(descriptor))
        cleanup = _attempt(cleanup, leaf.close)
        if cleanup is not None: raise primary from cleanup
        raise
def _remove_output(output: OutputFile) -> None:
    _rollback_fresh_leaf(output.leaf, output.initial)
def _compare_output(output: OutputFile, expected: BoundFile, *, sealed: bool = False) -> os.stat_result:
    output.leaf.verify()
    require(output.leaf.same(output.initial), "observed ProverInput path was rebound")
    actual = os.fstat(output.descriptor)
    require(stat.S_ISREG(actual.st_mode) and actual.st_nlink == 1
            and (actual.st_dev, actual.st_ino) == (output.initial.st_dev, output.initial.st_ino)
            and actual.st_size == expected.byte_length
            and (not sealed or stat.S_IMODE(actual.st_mode) == 0o400),
            "observed ProverInput final inode or byte length differs")
    offset = 0
    digest = hashlib.sha256()
    while offset < expected.byte_length:
        length = min(1024 * 1024, expected.byte_length - offset)
        observed = os.pread(output.descriptor, length, offset)
        golden = os.pread(expected.descriptor, length, offset)
        require(observed and observed == golden, "observed ProverInput bytes differ")
        digest.update(observed)
        offset += len(observed)
    require(os.pread(output.descriptor, 1, expected.byte_length) == b""
            and digest.hexdigest() == expected.sha256,
            "observed ProverInput digest or extent differs")
    if not sealed:
        os.fchmod(output.descriptor, 0o400)
        os.fsync(output.descriptor)
    final = os.fstat(output.descriptor)
    require(stat.S_IMODE(final.st_mode) == 0o400 and final.st_nlink == 1
            and final.st_size == expected.byte_length
            and (not sealed or _fingerprint(final) == _fingerprint(actual))
            and _fingerprint(output.leaf.stat()) == _fingerprint(final),
            "observed ProverInput immutable final state differs")
    output.leaf.directory.sync(); output.leaf.verify()
    return final
def _install_limits(file_size: int) -> Callable[[], None]:
    limits = (
        (resource.RLIMIT_AS, ADDRESS_SPACE_LIMIT_BYTES),
        (resource.RLIMIT_DATA, DATA_LIMIT_BYTES),
        (resource.RLIMIT_CPU, CPU_LIMIT_SECONDS),
        (resource.RLIMIT_FSIZE, file_size),
    )
    for identifier, value in limits:
        hard = resource.getrlimit(identifier)[1]
        require(hard == resource.RLIM_INFINITY or hard >= value,
                "host hard resource limit is below the adapter contract")
    def apply() -> None:
        for identifier, value in limits:
            resource.setrlimit(identifier, (value, value))
    return apply
def _run_adapter(command: list[str], cwd: Path, descriptors: tuple[int, ...],
                 environment: dict[str, str], file_size: int) -> subprocess.CompletedProcess[bytes]:
    return run_bounded_child(
        command, cwd, descriptors, environment, timeout_seconds=WALL_TIMEOUT_SECONDS,
        max_stdout_bytes=0, process_name="PIE adapter",
        stdout_bound_name="the zero-byte bound", max_stderr_bytes=MAX_STREAM_BYTES,
        stderr_bound_name="4096 bytes", child_setup=_install_limits(file_size),
    )
def _binding(bound: BoundFile, *, inherited: bool) -> dict[str, Any]:
    return {"fd": bound.descriptor, "proc_path": bound.proc_path,
        "artifact_path": str(bound.path), "byte_length": bound.byte_length,
        "sha256": bound.sha256, "access": "retained-read-only",
        "child_inherited": inherited,
        "device": bound.fingerprint[0], "inode": bound.fingerprint[1],
        "link_count": bound.fingerprint[6]}
def _output_binding(output: OutputFile) -> dict[str, Any]:
    return {"fd": output.descriptor, "proc_path": output.proc_path,
        "artifact_path": str(output.path), "byte_length": output.expected_bytes,
        "sha256": output.expected_sha256, "access": "retained-read-write",
        "child_inherited": True, "device": output.initial.st_dev,
        "inode": output.initial.st_ino, "link_count": output.initial.st_nlink}
def _verify_record(descriptor: int, destination: RetainedLeaf, published: os.stat_result,
                   payload: bytes) -> None:
    before = os.fstat(descriptor)
    expected_id = (published.st_dev, published.st_ino)
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
            and stat.S_IMODE(before.st_mode) == 0o400 and before.st_size == len(payload)
            and (before.st_dev, before.st_ino) == expected_id,
            "published PIE adapter record inode state differs")
    digest = _sha256_descriptor(descriptor, len(payload))
    after = os.fstat(descriptor)
    require(_fingerprint(after) == _fingerprint(before) and digest == sha256_bytes(payload)
            and _fingerprint(destination.stat()) == _fingerprint(after),
            "published PIE adapter record path or bytes changed while rehashed")
    destination.verify()
def _link_record(destination: RetainedLeaf, temporary: str) -> None:
    try:
        destination.directory.link(temporary, destination.name)
    except FileExistsError as error:
        raise ValueError("PIE adapter record appeared during publication") from error
def _rollback_record(destination: RetainedLeaf, temporary: str,
                     identity: os.stat_result | None) -> None:
    error: BaseException | None = None
    def remove_destination() -> None:
        if identity is not None: destination.unlink_if_same(identity)
    def remove_temporary() -> None:
        if identity is not None:
            destination.directory.unlink_if_same(temporary, identity)
            return
        try: os.unlink(temporary, dir_fd=destination.directory.descriptor)
        except FileNotFoundError: pass
    for remove in (remove_destination, remove_temporary):
        error = _attempt(error, remove)
        error = _attempt(error, remove)
    error = _attempt(error, destination.directory.sync)
    if error is not None: raise error
def _publish_record(destination: RetainedLeaf, payload: bytes, before: Callable[[], None],
                    after: Callable[[], None]) -> None:
    descriptor: int | None = None
    created: os.stat_result | None = None
    published: os.stat_result | None = None
    temporary = f".pie-adapter-record.{secrets.token_hex(16)}"
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        descriptor = _descriptor_at_least_three(destination.directory.open(
            temporary, flags, "PIE adapter record temporary", 0o600
        ))
        created = os.fstat(descriptor)
        os.fchmod(descriptor, 0o400)
        remaining = memoryview(payload)
        while remaining:
            try: count = os.write(descriptor, remaining)
            except InterruptedError: continue
            require(count > 0, "cannot write complete PIE adapter record")
            remaining = remaining[count:]
        os.fsync(descriptor); published = os.fstat(descriptor)
        require(stat.S_ISREG(published.st_mode) and stat.S_IMODE(published.st_mode) == 0o400
                and published.st_nlink == 1 and published.st_size == len(payload)
                and (published.st_dev, published.st_ino) == (created.st_dev, created.st_ino)
                and _sha256_descriptor(descriptor, len(payload)) == sha256_bytes(payload),
                "PIE adapter record temporary differs")
        before(); destination.verify(); _link_record(destination, temporary)
        require(destination.directory.unlink_if_same(temporary, published),
                "PIE adapter record temporary was rebound")
        destination.directory.sync()
        _verify_record(descriptor, destination, published, payload)
        after(); _verify_record(descriptor, destination, published, payload)
        owned_descriptor, descriptor = descriptor, None
        os.close(owned_descriptor)
    except BaseException as primary:
        cleanup: BaseException | None = None
        cleanup_identity = created
        if cleanup_identity is None and descriptor is not None:
            try: cleanup_identity = os.stat(descriptor)
            except BaseException as error: cleanup = cleanup or error
        cleanup = _attempt(cleanup, lambda: _rollback_record(
            destination, temporary, cleanup_identity
        ))
        if descriptor is not None:
            cleanup = _attempt(cleanup, lambda: os.close(descriptor))
        if cleanup is not None: raise primary from cleanup
        raise
Runner = Callable[[list[str], Path, tuple[int, ...], dict[str, str], int],
                  subprocess.CompletedProcess[bytes]]
def execute(args: Any) -> dict[str, Any]:
    return _execute(args)
def _execute(args: Any, *, _runner: Runner | None = None,
             _tool_paths: tuple[Path, ...] | None = None) -> dict[str, Any]:
    require(sys.platform == "linux" or _runner is not None,
            "PIE adapter replay requires Linux /proc descriptor execution")
    require(_runner is not None or Path("/proc/self/fd").is_dir(),
            "Linux /proc/self/fd is unavailable")
    inputs: dict[str, BoundFile] = {}
    tools: list[BoundFile] = []
    output: OutputFile | None = None
    record_destination: RetainedLeaf | None = None
    record_installed = False
    try:
        for role, (kind, maximum) in ARTIFACT_SPECS.items():
            path = getattr(args, role)
            digest = getattr(args, f"{role}_sha256")
            inputs[role] = _bind(path, digest, role, kind, maximum)
        manifest_bytes = _read_descriptor(
            inputs["provenance_manifest"], ARTIFACT_SPECS["provenance_manifest"][1]
        )
        artifacts = {role: bound.artifact() for role, bound in inputs.items()}
        validate_adapter_provenance_projection(
            manifest_bytes, args.provenance_manifest_sha256, artifacts
        )
        invocation = parse_invocation_bytes(_read_descriptor(
            inputs["adapter_invocation"], ARTIFACT_SPECS["adapter_invocation"][1]
        ))
        expected_invocation = {
            "schema_version": INVOCATION_SCHEMA, "protocol": PROTOCOL,
            "bootloader_program": inputs["bootloader_program"].artifact(), "pie_copies": 1,
            "backend": "simd", "engine": "legacy", "output_encoding": OUTPUT_ENCODING,
        }
        require(invocation == expected_invocation
                and _read_descriptor(inputs["adapter_invocation"],
                                     ARTIFACT_SPECS["adapter_invocation"][1])
                == invocation_bytes(expected_invocation),
                "adapter invocation bytes differ from the executed bootloader contract")
        tools, tool_closure = _tool_sources(_tool_paths)
        all_bounds = [*inputs.values(), *tools]
        identities = [(bound.fingerprint[0], bound.fingerprint[1]) for bound in all_bounds]
        require(len(set(identities)) == len(identities), "PIE adapter inputs/tools reuse an inode")
        require(len({bound.path for bound in all_bounds}) == len(all_bounds),
                "PIE adapter inputs/tools reuse a path")
        record_destination = RetainedLeaf.bind(
            args.record, "PIE adapter execution record", fresh=True
        )
        output = _create_output(args.observed_prover_input, inputs["expected_prover_input"],
                                {record_destination.key,
                                 *(bound.leaf.key for bound in all_bounds)})
        require(output.leaf.key != record_destination.key,
                "observed ProverInput aliases the execution record")
        for bound in [*inputs.values(), *tools]:
            _verify_bound(bound, content=False)
        bindings = {role: _binding(bound, inherited=role in INHERITED_ROLES)
                    for role, bound in inputs.items()}
        bindings["observed_prover_input"] = _output_binding(output)
        command = [
            inputs["adapter_executable"].proc_path,
            "--pie", inputs["source_pie"].proc_path,
            "--backend", "simd", "--engine", "legacy", "--adapt-only",
        ]
        environment = {
            "LANG": "C", "LC_ALL": "C", "RAYON_NUM_THREADS": "8", "RUST_BACKTRACE": "0",
            "STWO_BOOTLOADER_JSON": inputs["bootloader_program"].proc_path,
            "STWO_DUMP_INPUT": output.proc_path,
        }
        inherited = tuple(bindings[role]["fd"] for role in bindings
                          if bindings[role]["child_inherited"])
        run = (_runner or _run_adapter)(command, Path("/"), inherited, environment,
                                        output.expected_bytes)
        require(run.returncode == 0, f"PIE adapter exit status differs: {run.returncode}")
        require(run.stdout == b"", "PIE adapter emitted unexpected stdout")
        expected_stderr = (
            f"prover input dumped: {output.expected_bytes} bytes -> {output.proc_path}\n"
        ).encode()
        require(run.stderr == expected_stderr, "PIE adapter stderr differs")
        for bound in [*inputs.values(), *tools]:
            _verify_bound(bound, content=True)
        final = _compare_output(output, inputs["expected_prover_input"])
        observed = output.artifact()
        # The contract validator is the sole authority for whether this checked state is a PASS.
        record = {
            "schema_version": "stwo.gpu-lab.pie-adapter-execution-record.v1",
            "evidence_mode": "authenticated-pie-adapter-replay",
            "passed": True, "adapter_execution_attested": True,
            "source_closure_status": "identity-only",
            "adapter_executable_policy": EXECUTABLE_POLICY,
            "production_admissible": False, "correctness_admissible": False,
            "performance_admissible": False,
            **{role: bound.artifact() for role, bound in inputs.items()},
            "replay_tool_source_closure": tool_closure,
            "invocation_contract": invocation,
            "provenance_binding": {
                "contract": "fri-round6-provenance-v1-external-raw-binding-v1",
                "trust_root": "required-out-of-band-sha256-v1",
                "output_relation": "manifest-expected-equals-observed-byte-for-byte-v1",
                "bootloader_coverage": "separate-required-semantic-live-in-v1",
            },
            "execution_contract": {
                **EXECUTION_HONESTY,
                "contract": "linux-retained-descriptor-exec-v1",
                "shell": False, "stdin": "devnull", "working_directory": "/",
                "argv": command, "environment": environment, "bindings": bindings,
                "input_recheck": "descriptor-and-path-before-and-after-v1",
                "resource_limits": {
                    "address_space_bytes": ADDRESS_SPACE_LIMIT_BYTES,
                    "data_bytes": DATA_LIMIT_BYTES, "cpu_seconds": CPU_LIMIT_SECONDS,
                    "file_size_bytes": output.expected_bytes,
                },
                "wall_timeout_seconds": WALL_TIMEOUT_SECONDS, "exit_status": 0,
                "stdout": {"byte_length": 0, "sha256": EMPTY_SHA256},
                "stderr": {"byte_length": len(expected_stderr),
                           "sha256": sha256_bytes(expected_stderr)},
            },
            "output_lifecycle": {
                "contract": "fresh-bound-output-inode-transition-v1",
                "creation": "o_creat-o_excl-o_nofollow-0600-v1",
                "failure_cleanup": "unlink-only-if-path-still-bound-inode-v1",
                "finalization": "fchmod-0400-fsync-before-attestation-v1",
                "pre_launch": {
                    "path": str(output.path), "file_type": "regular",
                    "device": output.initial.st_dev, "inode": output.initial.st_ino,
                    "link_count": 1, "mode": "0600", "byte_length": 0,
                },
                "retained_descriptor": {
                    "fd": output.descriptor, "device": output.initial.st_dev,
                    "inode": output.initial.st_ino, "access": "retained-read-write",
                },
                "post_launch": {
                    "path": str(output.path), "file_type": "regular",
                    "device": final.st_dev, "inode": final.st_ino,
                    "link_count": final.st_nlink, "mode": "0400",
                    "byte_length": final.st_size, "sha256": observed["sha256"],
                },
                "same_inode_transition": True,
            },
            "observed_prover_input": observed,
            "exact_byte_equal": True,
        }
        validated = validate_execution_record(
            record, provenance_manifest_bytes=manifest_bytes,
            expected_provenance_manifest_sha256=args.provenance_manifest_sha256,
        )
        payload = canonical_bytes(record) + b"\n"
        require(len(payload) <= MAX_EXECUTION_RECORD_BYTES, "PIE adapter record is too large")
        installed = parse_execution_record_bytes(
            payload, provenance_manifest_bytes=manifest_bytes,
            expected_provenance_manifest_sha256=args.provenance_manifest_sha256,
        )
        require(installed == validated, "serialized PIE adapter execution record differs")
        def before_publish() -> None:
            for bound in all_bounds:
                _verify_bound(bound, content=False)
            output.leaf.verify()
            require(output.leaf.same(output.initial),
                    "observed ProverInput changed before publication")
        def after_publish() -> None:
            for bound in all_bounds:
                _verify_bound(bound, content=True)
            sealed_output = _compare_output(
                output, inputs["expected_prover_input"], sealed=True
            )
            for bound in all_bounds:
                _verify_bound(bound, content=False)
            require(_fingerprint(os.fstat(output.descriptor)) == _fingerprint(sealed_output)
                    and _fingerprint(output.leaf.stat()) == _fingerprint(sealed_output),
                    "observed ProverInput changed after post-publication validation")
            output.leaf.verify()
        _publish_record(record_destination, payload, before_publish, after_publish)
        record_installed = True
        return installed
    finally:
        active_error = sys.exc_info()[1]
        cleanup: BaseException | None = None
        if output is not None:
            if not record_installed:
                cleanup = _attempt(cleanup, lambda: _remove_output(output))
            cleanup = _attempt(cleanup, lambda: os.close(output.descriptor))
            cleanup = _attempt(cleanup, output.leaf.close)
        cleanup = _attempt(cleanup, lambda: _close_bounds([*inputs.values(), *tools]))
        if record_destination is not None:
            cleanup = _attempt(cleanup, record_destination.close)
        # A verified PASS record is the commit point. Every post-commit close is still
        # attempted, but close(2)'s ambiguous release state cannot revoke that durable
        # record/output pair or turn a successful CLI invocation into FAIL.
        committed_cleanup = record_installed and active_error is None
        if cleanup is not None and not committed_cleanup:
            if active_error is not None: raise active_error from cleanup
            raise cleanup
