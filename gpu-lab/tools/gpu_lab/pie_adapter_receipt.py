"""Non-admitting live source closure and declared-build receipt for the PIE adapter."""

# gpu-lab-cohesion-review: one module keeps one source/receipt contract and hash model auditable.

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .common import (WORKSPACE_ROOT, canonical_bytes, require, require_exact_keys,
                     require_int, require_sha256, sha256_bytes)
from .immutable_output import write_immutable_bytes
from .sealed_process import run_bounded_child

INVENTORY_SCHEMA = "stwo.gpu-lab.pie-adapter-source-inventory.v2"
SOURCE_CLOSURE_SCHEMA = "stwo.gpu-lab.pie-adapter-source-closure.v2"
BUILD_RECEIPT_SCHEMA = "stwo.gpu-lab.pie-adapter-build-receipt.v2"
REPOSITORIES = ("stwo", "stwo-cairo")
ADAPTER_REPOSITORY = "stwo-cairo"
ADAPTER_ROOT = "gpu_benchmarks/lab/pie-adapter"
ADAPTER_BINARY = "stwo-gpu-lab-pie-adapter"
ADAPTER_PROFILE = "release"
TARGET_NAMESPACE = PurePosixPath(f"{ADAPTER_ROOT}/receipt-targets")
LINUX_TARGETS = frozenset({"aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"})
MAX_RUN_ID_CHARS = 128
SOURCE_STATUS = "live-explicit-inventory-identity-only-v2"
RECEIPT_STATUS = "declared-build-inputs-live-outputs-no-build-attestation-v2"
UNATTESTED_BUILD_INPUTS = (
    "cargo-config-compiler-and-user-environment-not-execution-attested-v1"
)
MAX_JSON_BYTES = 4 << 20
MAX_SOURCES = 1024
MAX_SOURCE_BYTES = 32 << 20
MAX_TOTAL_SOURCE_BYTES = 256 << 20
MAX_OUTPUTS = 64
MAX_OUTPUT_BYTES = 512 << 20
MAX_TOTAL_OUTPUT_BYTES = 1 << 30
MAX_EXECUTABLE_BYTES = 256 << 20
MAX_PATH_CHARS = 512
MAX_ARGUMENTS = 64
MAX_ARGUMENT_CHARS = 1024
WRITER_PRECONDITION = "caller-guaranteed-no-concurrent-source-or-output-writers-v1"
ATTESTATION_SCOPE = "point-in-time-live-identity-observations-v1"

# Cargo.lock binds registry/git inputs; every file below the seven local package
# roots plus both workspace manifests must appear in the reviewed inventory.
DISCOVERY_FILES = (
    ("stwo", "Cargo.toml"),
    ("stwo", "crates/constraint-framework/Cargo.toml"),
    ("stwo", "crates/constraint-framework/src/lib.rs"),
    ("stwo", "crates/stwo/Cargo.toml"),
    ("stwo", "crates/stwo/src/lib.rs"),
    ("stwo-cairo", "gpu_benchmarks/lab/pie-adapter/Cargo.lock"),
    ("stwo-cairo", "gpu_benchmarks/lab/pie-adapter/Cargo.toml"),
    ("stwo-cairo", "gpu_benchmarks/lab/pie-adapter/rust-toolchain.toml"),
    ("stwo-cairo", "gpu_benchmarks/lab/pie-adapter/src/main.rs"),
    ("stwo-cairo", "stwo_cairo_prover/Cargo.toml"),
    ("stwo-cairo", "stwo_cairo_prover/crates/adapter/Cargo.toml"),
    ("stwo-cairo", "stwo_cairo_prover/crates/adapter/src/lib.rs"),
    ("stwo-cairo", "stwo_cairo_prover/crates/cairo-serialize-derive/Cargo.toml"),
    ("stwo-cairo", "stwo_cairo_prover/crates/cairo-serialize-derive/src/lib.rs"),
    ("stwo-cairo", "stwo_cairo_prover/crates/cairo-serialize/Cargo.toml"),
    ("stwo-cairo", "stwo_cairo_prover/crates/cairo-serialize/src/lib.rs"),
    ("stwo-cairo", "stwo_cairo_prover/crates/common/Cargo.toml"),
    ("stwo-cairo", "stwo_cairo_prover/crates/common/src/lib.rs"),
)
DISCOVERY_PACKAGE_ROOTS = (
    ("stwo", "crates/constraint-framework"),
    ("stwo", "crates/stwo"),
    ("stwo-cairo", ADAPTER_ROOT),
    ("stwo-cairo", "stwo_cairo_prover/crates/adapter"),
    ("stwo-cairo", "stwo_cairo_prover/crates/cairo-serialize-derive"),
    ("stwo-cairo", "stwo_cairo_prover/crates/cairo-serialize"),
    ("stwo-cairo", "stwo_cairo_prover/crates/common"),
)
Locator = dict[str, str]
CommitReader = Callable[[Path], str]


def _parse_json(payload: bytes, label: str) -> dict[str, Any]:
    require(type(payload) is bytes and 0 < len(payload) <= MAX_JSON_BYTES,
            f"{label} byte length is out of bounds")

    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            require(key not in value, f"{label} contains duplicate key {key!r}")
            value[key] = item
        return value

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{label} contains non-finite number {value}")

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=object_without_duplicates,
                           parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {error}") from error
    require(isinstance(value, dict), f"{label} top level must be an object")
    return value


def _read_bounded(path: Path, label: str) -> bytes:
    initial = path.lstat()
    require(stat.S_ISREG(initial.st_mode) and not stat.S_ISLNK(initial.st_mode)
            and 0 < initial.st_size <= MAX_JSON_BYTES,
            f"{label} is not a bounded regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size)
                == (initial.st_dev, initial.st_ino, initial.st_size),
                f"{label} changed before it was read")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(descriptor, min(1 << 20, before.st_size - offset), offset)
            require(chunk, f"{label} ended while it was read")
            chunks.append(chunk)
            offset += len(chunk)
        require(not os.pread(descriptor, 1, before.st_size), f"{label} grew while it was read")
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                 before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size,
                                          after.st_mtime_ns, after.st_ctime_ns),
                f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    final = path.lstat()
    require((final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns,
             final.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size,
                                     after.st_mtime_ns, after.st_ctime_ns),
            f"{label} path changed while it was read")
    return b"".join(chunks)


def _bounded_text(value: Any, label: str, maximum: int = MAX_ARGUMENT_CHARS) -> str:
    require(isinstance(value, str) and 0 < len(value) <= maximum,
            f"{label} is not bounded text")
    require(all(0x20 <= ord(char) <= 0x7E for char in value),
            f"{label} contains non-printable text")
    return value


def _relative_path(value: Any, label: str) -> str:
    path = _bounded_text(value, label, MAX_PATH_CHARS)
    require(path.isascii() and "\\" not in path, f"{label} is not portable ASCII")
    parsed = PurePosixPath(path)
    require(not parsed.is_absolute() and str(parsed) == path,
            f"{label} must be a canonical relative path")
    require(all(part not in {"", ".", ".."} for part in parsed.parts),
            f"{label} contains an ambiguous component")
    return path


def _locator(value: Any, label: str) -> Locator:
    require(isinstance(value, dict), f"{label} must be an object")
    require_exact_keys(value, {"repository", "path"}, label)
    require(value["repository"] in REPOSITORIES, f"{label} repository differs")
    return {"repository": value["repository"], "path": _relative_path(value["path"], f"{label} path")}


def _locator_key(value: Locator) -> tuple[str, str]:
    return value["repository"], value["path"]


def _identity_locator(value: Any, label: str, *, output: bool) -> Locator:
    require(isinstance(value, dict), f"{label} must be an object")
    keys = {"repository", "path", "byte_length", "sha256", "mode"} if output else {
        "repository", "path", "byte_length", "sha256"
    }
    require_exact_keys(value, keys, label)
    locator = _locator({key: value[key] for key in ("repository", "path")}, label)
    maximum = MAX_OUTPUT_BYTES if output else MAX_SOURCE_BYTES
    require(require_int(value["byte_length"], f"{label} byte length", 1) <= maximum,
            f"{label} exceeds its byte bound")
    require_sha256(value["sha256"], f"{label} sha256")
    if output:
        mode = value["mode"]
        require(isinstance(mode, str) and len(mode) == 4 and mode[0] == "0"
                and all(char in "01234567" for char in mode), f"{label} mode differs")
    return locator


def _canonical_build_command(build: dict[str, Any]) -> list[str]:
    command = ["cargo", "build", "--manifest-path", "Cargo.toml"]
    command.extend(["--release"] if build["profile"] == "release"
                   else ["--profile", build["profile"]])
    command.extend(["--locked", "--offline", "--target", build["target"]])
    if not build["default_features"]:
        command.append("--no-default-features")
    if build["features"]:
        command.extend(["--features", ",".join(build["features"])])
    return command


def _require_adapter_build_layout(build: dict[str, Any], executable: Locator) -> None:
    target_directory = build["target_directory"]
    target = build["target"]
    relative = PurePosixPath(target_directory["path"])
    run_id = relative.name
    require(build["manifest"] == {
        "repository": ADAPTER_REPOSITORY, "path": f"{ADAPTER_ROOT}/Cargo.toml",
    }, "PIE adapter build manifest differs from the fixed host adapter")
    require(build["profile"] == ADAPTER_PROFILE and build["features"] == []
            and build["default_features"] is True,
            "PIE adapter release feature profile differs")
    require(target in LINUX_TARGETS,
            "PIE adapter target is not an allowlisted Linux target triple")
    require(target_directory["repository"] == ADAPTER_REPOSITORY
            and relative.parent == TARGET_NAMESPACE,
            "PIE adapter target directory must be a direct child of the receipt-targets namespace")
    require(0 < len(run_id) <= MAX_RUN_ID_CHARS and run_id[0].isalnum()
            and all(char.isalnum() or char in "_.-" for char in run_id),
            "PIE adapter target directory has an invalid bounded run id")
    expected = PurePosixPath(target_directory["path"]) / target / ADAPTER_PROFILE / ADAPTER_BINARY
    require(executable == {"repository": ADAPTER_REPOSITORY, "path": str(expected)},
            "PIE adapter executable locator is not the exact Cargo release output")


def validate_inventory(value: Any, workspace_root: Path = WORKSPACE_ROOT) -> dict[str, Any]:
    require(isinstance(value, dict), "PIE adapter source inventory must be an object")
    require_exact_keys(value, {"schema_version", "repository_roots", "expected_sources",
                       "build", "generated_outputs", "adapter_executable"},
                       "PIE adapter source inventory")
    require(value["schema_version"] == INVENTORY_SCHEMA,
            "PIE adapter source inventory schema differs")
    require(value["repository_roots"] == {name: name for name in REPOSITORIES},
            "PIE adapter repository roots differ")
    sources = value["expected_sources"]
    require(isinstance(sources, list) and 2 <= len(sources) <= MAX_SOURCES,
            "PIE adapter expected source count is out of bounds")
    checked_sources = [_locator(source, f"PIE adapter expected source {index}")
                       for index, source in enumerate(sources)]
    source_keys = [_locator_key(source) for source in checked_sources]
    require(source_keys == sorted(source_keys) and len(source_keys) == len(set(source_keys)),
            "PIE adapter expected sources are not sorted and distinct")
    require({repository for repository, _ in source_keys} == set(REPOSITORIES),
            "PIE adapter expected sources do not cover both repositories")
    require(checked_sources == discover_expected_sources(workspace_root),
            "PIE adapter expected sources differ from fixed live local-input discovery")

    build = value["build"]
    require(isinstance(build, dict), "PIE adapter build declaration must be an object")
    require_exact_keys(build, {"manifest", "cargo_lock", "rust_toolchain", "profile",
                       "features", "default_features", "target", "target_directory",
                       "working_directory", "environment", "command"},
                       "PIE adapter build declaration")
    for name in ("manifest", "cargo_lock", "rust_toolchain", "target_directory"):
        _locator(build[name], f"PIE adapter build {name}")
    manifest = build["manifest"]
    require(manifest["repository"] == build["cargo_lock"]["repository"]
            == build["rust_toolchain"]["repository"],
            "PIE adapter Cargo inputs span repositories")
    manifest_parent = str(PurePosixPath(manifest["path"]).parent)
    require(build["cargo_lock"]["path"] == f"{manifest_parent}/Cargo.lock"
            and build["rust_toolchain"]["path"] == f"{manifest_parent}/rust-toolchain.toml",
            "PIE adapter Cargo lock/toolchain do not accompany the manifest")
    for name in ("manifest", "cargo_lock", "rust_toolchain"):
        require(_locator_key(build[name]) in set(source_keys),
                f"PIE adapter expected inventory omits build {name}")
    _bounded_text(build["profile"], "PIE adapter build profile", 64)
    _bounded_text(build["target"], "PIE adapter build target", 128)
    require(all(char.isalnum() or char in "_-" for char in build["profile"]),
            "PIE adapter build profile is not canonical")
    require(all(char.isalnum() or char in "_.-" for char in build["target"]),
            "PIE adapter build target is not canonical")
    require(build["default_features"] is True or build["default_features"] is False,
            "PIE adapter default-features declaration is not boolean")
    features = build["features"]
    require(isinstance(features, list) and len(features) <= 64,
            "PIE adapter feature count is out of bounds")
    for index, feature in enumerate(features):
        _bounded_text(feature, f"PIE adapter feature {index}", 128)
        require(all(char.isalnum() or char in "_-" for char in feature),
                f"PIE adapter feature {index} is not canonical")
    require(features == sorted(features) and len(features) == len(set(features)),
            "PIE adapter features are not sorted and distinct")
    expected_working_directory = (
        f"<workspace>/{manifest['repository']}/{manifest_parent}"
    )
    require(build["working_directory"] == expected_working_directory,
            "PIE adapter build working directory differs")
    target_dir = build["target_directory"]
    expected_environment = {
        "CARGO_TARGET_DIR": f"<workspace>/{target_dir['repository']}/{target_dir['path']}"
    }
    require(build["environment"] == expected_environment,
            "PIE adapter build environment differs")
    command = build["command"]
    require(isinstance(command, list) and 1 <= len(command) <= MAX_ARGUMENTS,
            "PIE adapter build command length is out of bounds")
    for index, argument in enumerate(command):
        _bounded_text(argument, f"PIE adapter build argument {index}")
    require(command == _canonical_build_command(build),
            "PIE adapter build command is not the canonical declared command")

    outputs = value["generated_outputs"]
    require(isinstance(outputs, list) and 1 <= len(outputs) <= MAX_OUTPUTS,
            "PIE adapter generated-output count is out of bounds")
    checked_outputs = [_locator(output, f"PIE adapter generated output {index}")
                       for index, output in enumerate(outputs)]
    output_keys = [_locator_key(output) for output in checked_outputs]
    require(output_keys == sorted(output_keys) and len(output_keys) == len(set(output_keys)),
            "PIE adapter generated outputs are not sorted and distinct")
    target_prefix = PurePosixPath(target_dir["path"])
    require(all(output["repository"] == target_dir["repository"]
                and PurePosixPath(output["path"]).is_relative_to(target_prefix)
                and PurePosixPath(output["path"]) != target_prefix for output in checked_outputs),
            "PIE adapter generated output escapes CARGO_TARGET_DIR")
    executable = _locator(value["adapter_executable"], "PIE adapter executable")
    _require_adapter_build_layout(build, executable)
    require(_locator_key(executable) in set(output_keys),
            "PIE adapter executable is not a declared generated output")
    require(not set(source_keys) & set(output_keys),
            "PIE adapter generated output aliases an expected source path")
    return value


def parse_inventory_bytes(payload: bytes, expected_sha256: str,
                          workspace_root: Path = WORKSPACE_ROOT) -> dict[str, Any]:
    """Authenticate exact file bytes; closure records hash the canonical parsed value."""
    require_sha256(expected_sha256, "expected PIE adapter inventory sha256")
    require(sha256_bytes(payload) == expected_sha256,
            "PIE adapter inventory differs from its out-of-band sha256")
    return validate_inventory(_parse_json(payload, "PIE adapter source inventory"),
                              workspace_root)


def _safe_directory(repository: str, relative: str, workspace_root: Path) -> Path:
    repository_root = workspace_root / repository
    root_status = repository_root.lstat()
    require(stat.S_ISDIR(root_status.st_mode) and not stat.S_ISLNK(root_status.st_mode),
            f"PIE adapter repository root is not a real directory: {repository_root}")
    current = repository_root
    for part in PurePosixPath(relative).parts:
        current = current / part
        status = current.lstat()
        require(stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode),
                f"PIE adapter discovery path is not a real directory: {current}")
    return current


def discover_expected_sources(workspace_root: Path = WORKSPACE_ROOT) -> list[Locator]:
    """Discover all files in the seven local package roots, excluding only build outputs."""
    discovered = {(repository, path): {"repository": repository, "path": path}
                  for repository, path in DISCOVERY_FILES}
    excluded = (ADAPTER_REPOSITORY, str(TARGET_NAMESPACE))
    for repository, relative_root in DISCOVERY_PACKAGE_ROOTS:
        root = _safe_directory(repository, relative_root, workspace_root)
        for directory, names, files in os.walk(root, topdown=True, followlinks=False):
            names.sort()
            files.sort()
            base = Path(directory)
            retained_names: list[str] = []
            for name in names:
                path = base / name
                relative = str(path.relative_to(workspace_root / repository))
                status = path.lstat()
                require(stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode),
                        f"PIE adapter discovery tree contains a non-directory or symlink: {path}")
                if (repository, relative) == excluded:
                    continue
                retained_names.append(name)
            names[:] = retained_names
            for name in files:
                path = base / name
                status = path.lstat()
                require(stat.S_ISREG(status.st_mode) and not stat.S_ISLNK(status.st_mode),
                        f"PIE adapter discovery tree contains a non-regular file: {path}")
                relative = str(path.relative_to(workspace_root / repository))
                discovered[(repository, relative)] = {"repository": repository, "path": relative}
    ordered = [discovered[key] for key in sorted(discovered)]
    require(2 <= len(ordered) <= MAX_SOURCES,
            "PIE adapter fixed source discovery is out of bounds")
    for locator in ordered:
        _safe_file(locator, workspace_root)
    return ordered


def _safe_file(locator: Locator, workspace_root: Path) -> Path:
    repository_root = workspace_root / locator["repository"]
    root_status = repository_root.lstat()
    require(stat.S_ISDIR(root_status.st_mode) and not stat.S_ISLNK(root_status.st_mode),
            f"PIE adapter repository root is not a real directory: {repository_root}")
    current = repository_root
    parts = PurePosixPath(locator["path"]).parts
    for index, part in enumerate(parts):
        current = current / part
        status = current.lstat()
        require(not stat.S_ISLNK(status.st_mode),
                f"PIE adapter inventory path crosses a symlink: {current}")
        require(stat.S_ISREG(status.st_mode) if index == len(parts) - 1
                else stat.S_ISDIR(status.st_mode),
                f"PIE adapter inventory path has the wrong file type: {current}")
    return current


def _capture_file(locator: Locator, workspace_root: Path, *, output: bool) -> tuple[dict[str, Any], tuple[int, int]]:
    path = _safe_file(locator, workspace_root)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        maximum = MAX_OUTPUT_BYTES if output else MAX_SOURCE_BYTES
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and 0 < before.st_size <= maximum,
                f"PIE adapter file identity is unsafe or out of bounds: {path}")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(descriptor, min(1 << 20, before.st_size - offset), offset)
            require(chunk, f"PIE adapter file ended while hashing: {path}")
            digest.update(chunk)
            offset += len(chunk)
        require(not os.pread(descriptor, 1, before.st_size),
                f"PIE adapter file grew while hashing: {path}")
        after = os.fstat(descriptor)
        fingerprint = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                                     value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        require(fingerprint(before) == fingerprint(after),
                f"PIE adapter file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    final = path.lstat()
    require(fingerprint(final) == fingerprint(after),
            f"PIE adapter file path changed while hashing: {path}")
    identity: dict[str, Any] = {**locator, "byte_length": after.st_size,
                                "sha256": digest.hexdigest()}
    if output:
        identity["mode"] = f"0{stat.S_IMODE(after.st_mode):03o}"
    return identity, (after.st_dev, after.st_ino)


def _git_commit(repository_root: Path) -> str:
    executable = shutil.which("git")
    require(executable is not None, "git is unavailable for repository identity")
    result = run_bounded_child(
        [executable, "-C", str(repository_root), "rev-parse", "HEAD"],
        repository_root, (), {"PATH": str(Path(executable).parent), "LANG": "C", "LC_ALL": "C"},
        timeout_seconds=10, max_stdout_bytes=128, max_stderr_bytes=1024,
        process_name="PIE adapter git identity", stdout_bound_name="128-byte bound",
        stderr_bound_name="1024-byte bound",
    )
    require(result.returncode == 0, "git failed while reading PIE adapter repository identity")
    try:
        return result.stdout.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("PIE adapter git identity is not ASCII") from error


def _commits(workspace_root: Path, commit_reader: CommitReader) -> dict[str, str]:
    commits = {repository: commit_reader(workspace_root / repository)
               for repository in REPOSITORIES}
    for repository, commit in commits.items():
        require(isinstance(commit, str) and len(commit) == 40
                and all(char in "0123456789abcdef" for char in commit),
                f"PIE adapter {repository} commit is not a lowercase SHA-1")
    return commits


def _source_body(inventory: dict[str, Any], inventory_sha256: str, workspace_root: Path,
                 commit_reader: CommitReader) -> tuple[dict[str, Any], list[Path]]:
    commits = _commits(workspace_root, commit_reader)
    sources: list[dict[str, Any]] = []
    paths: list[Path] = []
    file_ids: list[tuple[int, int]] = []
    total = 0
    for locator in inventory["expected_sources"]:
        identity, file_id = _capture_file(locator, workspace_root, output=False)
        sources.append(identity)
        paths.append(_safe_file(locator, workspace_root))
        file_ids.append(file_id)
        total += identity["byte_length"]
    require(total <= MAX_TOTAL_SOURCE_BYTES, "PIE adapter source closure is too large")
    require(len(file_ids) == len(set(file_ids)), "PIE adapter source inventory aliases an inode")
    require(commits == _commits(workspace_root, commit_reader),
            "PIE adapter repository commit changed while sources were captured")
    return {
        "evidence_status": SOURCE_STATUS,
        "writer_precondition": WRITER_PRECONDITION,
        "attestation_scope": ATTESTATION_SCOPE,
        "production_admissible": False,
        "correctness_admissible": False,
        "performance_admissible": False,
        "inventory_sha256": inventory_sha256,
        "repository_commits": commits,
        "sources": sources,
    }, paths


def _closure_document(body: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": SOURCE_CLOSURE_SCHEMA, "source_closure": body,
            "source_closure_sha256": sha256_bytes(canonical_bytes(body))}


def _output_identities(inventory: dict[str, Any], workspace_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    identities: list[dict[str, Any]] = []
    paths: list[Path] = []
    file_ids: list[tuple[int, int]] = []
    total = 0
    for locator in inventory["generated_outputs"]:
        identity, file_id = _capture_file(locator, workspace_root, output=True)
        identities.append(identity)
        paths.append(_safe_file(locator, workspace_root))
        file_ids.append(file_id)
        total += identity["byte_length"]
    require(total <= MAX_TOTAL_OUTPUT_BYTES, "PIE adapter generated outputs are too large")
    require(len(file_ids) == len(set(file_ids)), "PIE adapter generated outputs alias an inode")
    executable_key = _locator_key(inventory["adapter_executable"])
    executable = next(identity for identity in identities
                      if _locator_key(identity) == executable_key)
    require(executable["byte_length"] <= MAX_EXECUTABLE_BYTES,
            "PIE adapter executable exceeds 256 MiB")
    require(int(executable["mode"], 8) & 0o111,
            "PIE adapter executable has no execute permission")
    return identities, paths


def _require_no_cross_alias(source_paths: list[Path], output_paths: list[Path]) -> None:
    require(all(not source.samefile(output) for source in source_paths for output in output_paths),
            "PIE adapter generated output aliases a source inode")


def _source_lookup(body: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {_locator_key(source): source for source in body["sources"]}


def _receipt_document(inventory: dict[str, Any], inventory_sha256: str,
                      closure: dict[str, Any], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    source_body = closure["source_closure"]
    sources = _source_lookup(source_body)
    build = inventory["build"]
    executable = next(output for output in outputs
                      if _locator_key(output) == _locator_key(inventory["adapter_executable"]))
    body = {
        "evidence_status": RECEIPT_STATUS,
        "build_execution_attested": False,
        "unattested_build_inputs": UNATTESTED_BUILD_INPUTS,
        "writer_precondition": WRITER_PRECONDITION,
        "attestation_scope": ATTESTATION_SCOPE,
        "production_admissible": False,
        "correctness_admissible": False,
        "performance_admissible": False,
        "inventory_sha256": inventory_sha256,
        "source_closure_sha256": closure["source_closure_sha256"],
        "repository_commits": source_body["repository_commits"],
        "build_inputs": {
            "manifest": sources[_locator_key(build["manifest"])],
            "cargo_lock": sources[_locator_key(build["cargo_lock"])],
            "rust_toolchain": sources[_locator_key(build["rust_toolchain"])],
            "profile": build["profile"], "features": build["features"],
            "default_features": build["default_features"], "target": build["target"],
            "target_directory": build["target_directory"],
            "working_directory": build["working_directory"],
            "environment": build["environment"], "command": build["command"],
        },
        "generated_outputs": outputs,
        "adapter_executable": executable,
    }
    return {"schema_version": BUILD_RECEIPT_SCHEMA, "build_receipt": body,
            "build_receipt_sha256": sha256_bytes(canonical_bytes(body))}


def _admission_false(value: dict[str, Any], label: str) -> None:
    require(value["production_admissible"] is False
            and value["correctness_admissible"] is False
            and value["performance_admissible"] is False,
            f"{label} overclaims admission")


def validate_source_closure(value: Any, inventory: dict[str, Any],
                            workspace_root: Path = WORKSPACE_ROOT,
                            _commit_reader: CommitReader = _git_commit) -> dict[str, Any]:
    inventory = validate_inventory(inventory, workspace_root)
    inventory_sha256 = sha256_bytes(canonical_bytes(inventory))
    require(isinstance(value, dict), "PIE adapter source closure must be an object")
    require_exact_keys(value, {"schema_version", "source_closure", "source_closure_sha256"},
                       "PIE adapter source closure")
    require(value["schema_version"] == SOURCE_CLOSURE_SCHEMA,
            "PIE adapter source closure schema differs")
    body = value["source_closure"]
    require(isinstance(body, dict), "PIE adapter source closure body must be an object")
    require_exact_keys(body, {"evidence_status", "production_admissible",
                       "correctness_admissible", "performance_admissible",
                       "writer_precondition", "attestation_scope", "inventory_sha256",
                       "repository_commits", "sources"},
                       "PIE adapter source closure body")
    require(body["evidence_status"] == SOURCE_STATUS,
            "PIE adapter source closure status differs")
    require(body["writer_precondition"] == WRITER_PRECONDITION
            and body["attestation_scope"] == ATTESTATION_SCOPE,
            "PIE adapter source closure observation boundary differs")
    _admission_false(body, "PIE adapter source closure")
    require(body["inventory_sha256"] == inventory_sha256,
            "PIE adapter source closure inventory hash differs")
    require_sha256(value["source_closure_sha256"], "PIE adapter source closure sha256")
    require(value["source_closure_sha256"] == sha256_bytes(canonical_bytes(body)),
            "PIE adapter source closure canonical hash differs")
    expected, _ = _source_body(inventory, inventory_sha256, workspace_root, _commit_reader)
    require(body == expected, "PIE adapter source closure differs from live expected inventory")
    return value


def validate_build_receipt(value: Any, inventory: dict[str, Any],
                           closure: dict[str, Any], workspace_root: Path = WORKSPACE_ROOT,
                           _commit_reader: CommitReader = _git_commit) -> dict[str, Any]:
    inventory = validate_inventory(inventory, workspace_root)
    inventory_sha256 = sha256_bytes(canonical_bytes(inventory))
    validate_source_closure(closure, inventory, workspace_root, _commit_reader)
    require(isinstance(value, dict), "PIE adapter build receipt must be an object")
    require_exact_keys(value, {"schema_version", "build_receipt", "build_receipt_sha256"},
                       "PIE adapter build receipt")
    require(value["schema_version"] == BUILD_RECEIPT_SCHEMA,
            "PIE adapter build receipt schema differs")
    body = value["build_receipt"]
    require(isinstance(body, dict), "PIE adapter build receipt body must be an object")
    require_exact_keys(body, {"evidence_status", "build_execution_attested",
                       "unattested_build_inputs",
                       "writer_precondition", "attestation_scope",
                       "production_admissible", "correctness_admissible",
                       "performance_admissible", "inventory_sha256",
                       "source_closure_sha256", "repository_commits", "build_inputs",
                       "generated_outputs", "adapter_executable"},
                       "PIE adapter build receipt body")
    require(body["evidence_status"] == RECEIPT_STATUS
            and body["build_execution_attested"] is False
            and body["unattested_build_inputs"] == UNATTESTED_BUILD_INPUTS
            and body["writer_precondition"] == WRITER_PRECONDITION
            and body["attestation_scope"] == ATTESTATION_SCOPE,
            "PIE adapter build receipt status overclaims execution")
    _admission_false(body, "PIE adapter build receipt")
    require(body["inventory_sha256"] == inventory_sha256
            and body["source_closure_sha256"] == closure["source_closure_sha256"]
            and body["repository_commits"] == closure["source_closure"]["repository_commits"],
            "PIE adapter build receipt does not bind its source closure")
    require_sha256(value["build_receipt_sha256"], "PIE adapter build receipt sha256")
    require(value["build_receipt_sha256"] == sha256_bytes(canonical_bytes(body)),
            "PIE adapter build receipt canonical hash differs")
    outputs = body["generated_outputs"]
    require(isinstance(outputs, list) and len(outputs) == len(inventory["generated_outputs"]),
            "PIE adapter build receipt generated-output count differs")
    for index, output in enumerate(outputs):
        _identity_locator(output, f"PIE adapter receipt output {index}", output=True)
    output_keys = [_locator_key(output) for output in outputs]
    require(output_keys == sorted(output_keys) and output_keys
            == [_locator_key(item) for item in inventory["generated_outputs"]],
            "PIE adapter build receipt output inventory differs")
    live_outputs, output_paths = _output_identities(inventory, workspace_root)
    _require_no_cross_alias(
        [_safe_file(locator, workspace_root) for locator in inventory["expected_sources"]],
        output_paths,
    )
    require(outputs == live_outputs, "PIE adapter build receipt outputs differ from live files")
    executable = body["adapter_executable"]
    _identity_locator(executable, "PIE adapter receipt executable", output=True)
    expected_executable = next(output for output in outputs
                               if _locator_key(output) == _locator_key(inventory["adapter_executable"]))
    require(executable == expected_executable,
            "PIE adapter receipt executable differs from its exact output identity")
    build = body["build_inputs"]
    require(isinstance(build, dict), "PIE adapter receipt build inputs must be an object")
    require_exact_keys(build, {"manifest", "cargo_lock", "rust_toolchain", "profile",
                       "features", "default_features", "target", "target_directory",
                       "working_directory", "environment", "command"},
                       "PIE adapter receipt build inputs")
    source_lookup = _source_lookup(closure["source_closure"])
    declared = inventory["build"]
    expected_build = {
        "manifest": source_lookup[_locator_key(declared["manifest"])],
        "cargo_lock": source_lookup[_locator_key(declared["cargo_lock"])],
        "rust_toolchain": source_lookup[_locator_key(declared["rust_toolchain"])],
        **{key: declared[key] for key in ("profile", "features", "default_features", "target",
                                           "target_directory", "working_directory", "environment",
                                           "command")},
    }
    require(build == expected_build, "PIE adapter receipt build inputs differ")
    validate_source_closure(closure, inventory, workspace_root, _commit_reader)
    return value


def parse_source_closure_bytes(payload: bytes, inventory: dict[str, Any],
                               workspace_root: Path = WORKSPACE_ROOT,
                               _commit_reader: CommitReader = _git_commit) -> dict[str, Any]:
    return validate_source_closure(_parse_json(payload, "PIE adapter source closure"), inventory,
                                   workspace_root, _commit_reader)


def parse_build_receipt_bytes(payload: bytes, inventory: dict[str, Any],
                              closure: dict[str, Any], workspace_root: Path = WORKSPACE_ROOT,
                              _commit_reader: CommitReader = _git_commit) -> dict[str, Any]:
    return validate_build_receipt(_parse_json(payload, "PIE adapter build receipt"), inventory,
                                  closure, workspace_root, _commit_reader)


def validate_documents(inventory_payload: bytes, raw_inventory_sha256: str,
                       closure_payload: bytes, receipt_payload: bytes,
                       workspace_root: Path = WORKSPACE_ROOT,
                       _commit_reader: CommitReader = _git_commit) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = parse_inventory_bytes(inventory_payload, raw_inventory_sha256, workspace_root)
    closure = parse_source_closure_bytes(
        closure_payload, inventory, workspace_root, _commit_reader)
    receipt = parse_build_receipt_bytes(
        receipt_payload, inventory, closure, workspace_root, _commit_reader)
    return closure, receipt


def generate(inventory_path: Path, raw_inventory_sha256: str, closure_path: Path,
             receipt_path: Path, workspace_root: Path = WORKSPACE_ROOT,
             _commit_reader: CommitReader = _git_commit) -> tuple[dict[str, Any], dict[str, Any]]:
    require(closure_path.resolve(strict=False) != receipt_path.resolve(strict=False),
            "PIE adapter closure and receipt paths alias")
    inventory_bytes = _read_bounded(inventory_path, "PIE adapter source inventory")
    inventory = parse_inventory_bytes(inventory_bytes, raw_inventory_sha256, workspace_root)
    canonical_inventory_sha256 = sha256_bytes(canonical_bytes(inventory))
    source_body, source_paths = _source_body(inventory, canonical_inventory_sha256,
                                             workspace_root, _commit_reader)
    closure = _closure_document(source_body)
    outputs, output_paths = _output_identities(inventory, workspace_root)
    _require_no_cross_alias(source_paths, output_paths)
    receipt = _receipt_document(inventory, canonical_inventory_sha256, closure, outputs)
    validate_build_receipt(receipt, inventory, closure, workspace_root, _commit_reader)
    closure_bytes = json.dumps(closure, indent=2, sort_keys=True).encode() + b"\n"
    receipt_bytes = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    immutable_inputs = [inventory_path, *source_paths, *output_paths]
    write_immutable_bytes(closure_path, closure_bytes, immutable_inputs)
    write_immutable_bytes(receipt_path, receipt_bytes, [*immutable_inputs, closure_path])
    parse_build_receipt_bytes(_read_bounded(receipt_path, "PIE adapter build receipt"),
                              inventory,
                              parse_source_closure_bytes(
                                  _read_bounded(closure_path, "PIE adapter source closure"), inventory,
                                  workspace_root, _commit_reader),
                              workspace_root, _commit_reader)
    return closure, receipt


def validate_files(inventory_path: Path, raw_inventory_sha256: str, closure_path: Path,
                   receipt_path: Path, workspace_root: Path = WORKSPACE_ROOT,
                   _commit_reader: CommitReader = _git_commit) -> tuple[dict[str, Any], dict[str, Any]]:
    return validate_documents(
        _read_bounded(inventory_path, "PIE adapter source inventory"),
        raw_inventory_sha256,
        _read_bounded(closure_path, "PIE adapter source closure"),
        _read_bounded(receipt_path, "PIE adapter build receipt"),
        workspace_root, _commit_reader,
    )
