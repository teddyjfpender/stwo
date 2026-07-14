"""Sealed, non-admitting execution record for one fresh host-adapter build."""

# gpu-lab-cohesion-review: one module keeps build execution, validation, and rollback atomic.

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .common import (WORKSPACE_ROOT, canonical_bytes, require, require_exact_keys,
                     require_int, require_sha256, sha256_bytes)
from .immutable_output import write_immutable_bytes
from .pie_adapter_receipt import (CommitReader, _git_commit, _read_bounded, generate,
                                  parse_inventory_bytes, parse_source_closure_bytes,
                                  validate_documents)
from .sealed_process import run_bounded_child


SCHEMA = "stwo.gpu-lab.pie-adapter-build-execution-record.v1"
CACHE_SCHEMA = "stwo.gpu-lab.pie-adapter-builder-cache-identity.v1"
EVIDENCE_MODE = "authenticated-fresh-pie-adapter-build"
CAUSAL_SCOPE = "declared-image-cache-trust-roots-and-live-source-direct-build-v1"
BUILDER_STATUS = "caller-declared-image-and-image-baked-cache-not-runtime-observed-v1"
WRITER_PRECONDITION = "caller-guaranteed-no-concurrent-source-target-or-artifact-writers-v1"
MAX_JSON_BYTES = 4 << 20
MAX_EXECUTABLE_BYTES = 256 << 20
MAX_STDOUT_BYTES = 1 << 20
MAX_STDERR_BYTES = 8 << 20
MAX_TARGET_BYTES = 16 << 30
MAX_TARGET_ENTRIES = 200_000
WALL_TIMEOUT_SECONDS = 3600
CPU_LIMIT_SECONDS = 1800
ADDRESS_SPACE_LIMIT_BYTES = 16 << 30
DATA_LIMIT_BYTES = 12 << 30
FILE_SIZE_LIMIT_BYTES = 1 << 30
FORBIDDEN_ENV = {
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD",
    "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
}

Runner = Callable[[list[str], Path, dict[str, str]], subprocess.CompletedProcess[bytes]]


def _parse_json(payload: bytes, label: str) -> dict[str, Any]:
    require(type(payload) is bytes and 0 < len(payload) <= MAX_JSON_BYTES,
            f"{label} byte length is out of bounds")

    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=object_without_duplicates,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               ValueError(f"{label} contains non-finite number {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {error}") from error
    require(isinstance(value, dict), f"{label} top level must be an object")
    return value


def _image_sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.startswith("sha256:"),
            f"{label} is not a sha256 image identity")
    require_sha256(value.removeprefix("sha256:"), label)
    return value


def _absolute_path(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.isascii() and 1 <= len(value) <= 2048
            and "\\" not in value and all(0x20 <= ord(char) <= 0x7E for char in value),
            f"{label} is not bounded portable text")
    path = PurePosixPath(value)
    require(path.is_absolute() and str(path) == value and value != "/"
            and not value.startswith("//")
            and all(part not in {"", ".", ".."} for part in path.parts[1:]),
            f"{label} is not a canonical absolute path")
    return value


def _artifact(path: Path, label: str, expected_sha256: str | None = None,
              maximum: int = MAX_JSON_BYTES, executable: bool = False,
              single_link: bool = True) -> dict[str, Any]:
    require_sha256(expected_sha256, f"expected {label} sha256") if expected_sha256 else None
    status = path.lstat()
    require(stat.S_ISREG(status.st_mode) and not stat.S_ISLNK(status.st_mode)
            and (not single_link or status.st_nlink == 1)
            and status.st_nlink >= 1 and 0 < status.st_size <= maximum,
            f"{label} is not a bounded safe regular file")
    mode = stat.S_IMODE(status.st_mode)
    require(mode == 0o400 or executable and mode & 0o111,
            f"{label} mode is not immutable or executable")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(descriptor, min(1 << 20, before.st_size - offset), offset)
            require(chunk, f"{label} ended while hashing")
            digest.update(chunk)
            offset += len(chunk)
        require(not os.pread(descriptor, 1, before.st_size), f"{label} grew while hashing")
        after = os.fstat(descriptor)
        fingerprint = lambda item: (item.st_dev, item.st_ino, item.st_mode, item.st_nlink,
                                    item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        require(fingerprint(before) == fingerprint(after) == fingerprint(path.lstat()),
                f"{label} changed while hashing")
    finally:
        os.close(descriptor)
    identity = {"path": str(path), "byte_length": after.st_size,
                "sha256": digest.hexdigest(), "mode": f"0{mode:03o}",
                "link_count": after.st_nlink}
    require(expected_sha256 is None or identity["sha256"] == expected_sha256,
            f"{label} differs from its out-of-band sha256")
    return identity


def _cache_identity(payload: bytes, image_id: str, image_digest: str) -> dict[str, Any]:
    value = _parse_json(payload, "PIE adapter builder cache identity")
    require_exact_keys(value, {"schema_version", "kind", "path", "builder_image_id",
                       "builder_image_digest", "storage", "external_mount", "read_only",
                       "runtime_observed"},
                       "PIE adapter builder cache identity")
    require(value["schema_version"] == CACHE_SCHEMA
            and value["kind"] == "builder-image-baked-cargo-home-declared-v1"
            and value["builder_image_id"] == image_id
            and value["builder_image_digest"] == image_digest
            and value["storage"] == "builder-image-layer"
            and value["external_mount"] is False
            and value["read_only"] is True
            and value["runtime_observed"] is False,
            "PIE adapter builder cache identity status differs")
    _absolute_path(value["path"], "PIE adapter builder cache path")
    return value


def _linker_environment_name(target_triple: str) -> str:
    require(isinstance(target_triple, str) and target_triple,
            "PIE adapter target triple is absent")
    suffix = "".join(char.upper() if char.isalnum() else "_" for char in target_triple)
    return f"CARGO_TARGET_{suffix}_LINKER"


def _environment(payload: str, target: Path, target_triple: str, cache: dict[str, Any],
                 rustc: dict[str, Any], compiler_linker: dict[str, Any],
                 archiver: dict[str, Any]) -> dict[str, str]:
    value = _parse_json(payload.encode(), "PIE adapter builder environment")
    linker_key = _linker_environment_name(target_triple)
    required = {"CARGO_HOME", "CARGO_TARGET_DIR", "CARGO_NET_OFFLINE", "LANG", "LC_ALL",
                "RUSTC", "CC", "AR", linker_key}
    require(set(value) == required and not set(value) & FORBIDDEN_ENV,
            "PIE adapter builder environment is not the exact sealed environment")
    for key, item in value.items():
        require(isinstance(key, str) and key.isascii() and 1 <= len(key) <= 128
                and all(char.isalnum() or char == "_" for char in key)
                and isinstance(item, str) and item.isascii() and len(item) <= 4096
                and "\x00" not in item,
                "PIE adapter builder environment contains invalid text")
    require(value.get("CARGO_HOME") == cache["path"]
            and value.get("CARGO_TARGET_DIR") == str(target)
            and value.get("CARGO_NET_OFFLINE") == "true"
            and value.get("LANG") == value.get("LC_ALL") == "C"
            and value.get("RUSTC") == rustc["path"]
            and value.get("CC") == value.get(linker_key) == compiler_linker["path"]
            and value.get("AR") == archiver["path"],
            "PIE adapter environment does not bind cache, target, locale, rustc, CC/linker, or AR")
    return dict(sorted(value.items()))


def _source_context(inventory_path: Path, inventory_sha256: str, closure_path: Path,
                    closure_sha256: str, workspace_root: Path,
                    commit_reader: CommitReader) -> tuple[bytes, bytes, dict[str, Any], dict[str, Any]]:
    inventory_bytes = _read_bounded(inventory_path, "PIE adapter source inventory")
    closure_bytes = _read_bounded(closure_path, "PIE adapter source closure")
    require_sha256(closure_sha256, "expected PIE adapter source closure sha256")
    require(sha256_bytes(closure_bytes) == closure_sha256,
            "PIE adapter source closure differs from its out-of-band sha256")
    inventory = parse_inventory_bytes(inventory_bytes, inventory_sha256, workspace_root)
    closure = parse_source_closure_bytes(closure_bytes, inventory, workspace_root, commit_reader)
    return inventory_bytes, closure_bytes, inventory, closure


def _target_paths(inventory: dict[str, Any], workspace_root: Path) -> tuple[Path, Path, Path]:
    target_locator = inventory["build"]["target_directory"]
    executable_locator = inventory["adapter_executable"]
    target = workspace_root / target_locator["repository"] / target_locator["path"]
    executable = workspace_root / executable_locator["repository"] / executable_locator["path"]
    cwd = workspace_root / inventory["build"]["manifest"]["repository"]
    cwd /= PurePosixPath(inventory["build"]["manifest"]["path"]).parent
    require(target.is_absolute() and executable.is_relative_to(target) and cwd.is_dir(),
            "PIE adapter build paths differ from the inventory")
    return target, executable, cwd


def _require_fresh_target(target: Path) -> None:
    current = Path(target.anchor)
    for part in target.parts[1:]:
        current /= part
        try:
            status = current.lstat()
        except FileNotFoundError:
            break
        require(stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode),
                f"PIE adapter target path crosses a non-directory or symlink: {current}")
    require(not target.exists() and not target.is_symlink(),
            "PIE adapter target directory is not fresh and absent")


def _remove_fresh_output(path: Path) -> None:
    """Remove only a path proven absent before this transaction."""
    try:
        status = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def _target_state(target: Path) -> dict[str, Any]:
    status = target.lstat()
    require(stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode),
            "PIE adapter builder did not create a real target directory")
    entries = 0
    total = 0
    for directory, names, files in os.walk(target, topdown=True, followlinks=False):
        entries += len(names) + len(files)
        require(entries <= MAX_TARGET_ENTRIES, "PIE adapter target entry count is too large")
        base = Path(directory)
        for name in [*names, *files]:
            item = (base / name).lstat()
            require(not stat.S_ISLNK(item.st_mode), "PIE adapter target contains a symlink")
            if stat.S_ISREG(item.st_mode):
                total += item.st_size
                require(total <= MAX_TARGET_BYTES, "PIE adapter target is too large")
    return {"path": str(target), "file_type": "directory", "device": status.st_dev,
            "inode": status.st_ino, "mode": f"0{stat.S_IMODE(status.st_mode):03o}",
            "entries": entries, "total_regular_bytes": total}


def _stream(payload: bytes, maximum: int, label: str) -> dict[str, Any]:
    require(type(payload) is bytes and len(payload) <= maximum, f"{label} exceeds its bound")
    return {"byte_length": len(payload), "sha256": sha256_bytes(payload)}


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _detach_executable(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _artifact(path, "Cargo adapter executable", maximum=MAX_EXECUTABLE_BYTES,
                       executable=True, single_link=False)
    if source["link_count"] > 1:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.detach.", dir=path.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(path, temporary)
            os.chmod(temporary, int(source["mode"], 8))
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
            detached = _artifact(temporary, "detached adapter executable",
                                 maximum=MAX_EXECUTABLE_BYTES, executable=True)
            require(detached["sha256"] == source["sha256"],
                    "PIE adapter detached executable bytes differ")
            os.replace(temporary, path)
            _sync_directory(path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
    final = _artifact(path, "adapter executable", maximum=MAX_EXECUTABLE_BYTES, executable=True)
    publication = {"contract": "byte-checked-single-link-detach-v1",
        "source_link_count": source["link_count"], "detached": source["link_count"] > 1,
        "source_sha256": source["sha256"], "final_sha256": final["sha256"],
        "byte_equal": True}
    require(source["sha256"] == final["sha256"],
            "PIE adapter executable changed during detach")
    return publication, final


def _limits() -> None:
    for kind, value in ((resource.RLIMIT_AS, ADDRESS_SPACE_LIMIT_BYTES),
                        (resource.RLIMIT_DATA, DATA_LIMIT_BYTES),
                        (resource.RLIMIT_CPU, CPU_LIMIT_SECONDS),
                        (resource.RLIMIT_FSIZE, FILE_SIZE_LIMIT_BYTES)):
        resource.setrlimit(kind, (value, value))


def _run_builder(argv: list[str], cwd: Path,
                 environment: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    return run_bounded_child(
        argv, cwd, (), environment, timeout_seconds=WALL_TIMEOUT_SECONDS,
        max_stdout_bytes=MAX_STDOUT_BYTES, max_stderr_bytes=MAX_STDERR_BYTES,
        process_name="PIE adapter builder", stdout_bound_name="1 MiB",
        stderr_bound_name="8 MiB", child_setup=_limits,
    )


def _checked_artifact(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} identity must be an object")
    require_exact_keys(value, {"path", "byte_length", "sha256", "mode", "link_count"}, label)
    _absolute_path(value["path"], f"{label} path")
    require_int(value["byte_length"], f"{label} byte length", 1)
    require_sha256(value["sha256"], f"{label} sha256")
    require(isinstance(value["mode"], str) and len(value["mode"]) == 4
            and value["mode"][0] == "0" and all(char in "01234567" for char in value["mode"])
            and value["link_count"] == 1, f"{label} mode or link count differs")
    return value


def validate_record(value: Any, *, inventory_sha256: str, closure_sha256: str,
                    cache_sha256: str, cargo_sha256: str, rustc_sha256: str,
                    native_compiler_linker_sha256: str, native_archiver_sha256: str,
                    builder_image_id: str, builder_image_digest: str,
                    workspace_root: Path = WORKSPACE_ROOT,
                    commit_reader: CommitReader = _git_commit) -> dict[str, Any]:
    require(isinstance(value, dict), "PIE adapter build execution record must be an object")
    keys = {"schema_version", "evidence_mode", "passed", "build_execution_attested",
            "causal_scope", "builder_status", "builder_runtime_attested", "writer_precondition",
            "production_admissible",
            "correctness_admissible", "performance_admissible", "source_inventory",
            "source_closure", "builder", "invocation", "source_revalidation",
            "target_lifecycle", "child_result", "executable_publication", "adapter_executable",
            "build_receipt", "receipt_binding"}
    require_exact_keys(value, keys, "PIE adapter build execution record")
    require(value["schema_version"] == SCHEMA and value["evidence_mode"] == EVIDENCE_MODE
            and value["passed"] is True and value["build_execution_attested"] is True
            and value["causal_scope"] == CAUSAL_SCOPE and value["builder_status"] == BUILDER_STATUS
            and value["builder_runtime_attested"] is False
            and value["writer_precondition"] == WRITER_PRECONDITION,
            "PIE adapter build execution status differs")
    require(all(value[key] is False for key in
                ("production_admissible", "correctness_admissible", "performance_admissible")),
            "PIE adapter build execution record overclaims admission")

    builder = value["builder"]
    require(isinstance(builder, dict), "PIE adapter builder identity must be an object")
    require_exact_keys(builder, {"image_id", "image_digest", "tools", "cache_manifest",
                       "cache_identity"}, "PIE adapter builder identity")
    tools = builder["tools"]
    require(isinstance(tools, dict), "PIE adapter builder tools must be an object")
    require_exact_keys(tools, {"cargo", "rustc", "native_compiler_linker", "native_archiver"},
                       "PIE adapter builder tools")
    inventory_artifact = _checked_artifact(value["source_inventory"], "source inventory")
    closure_artifact = _checked_artifact(value["source_closure"], "source closure")
    cache_artifact = _checked_artifact(builder["cache_manifest"], "cache manifest")
    cargo_artifact = _checked_artifact(tools["cargo"], "Cargo executable")
    rustc_artifact = _checked_artifact(tools["rustc"], "rustc executable")
    compiler_linker_artifact = _checked_artifact(
        tools["native_compiler_linker"], "native compiler/linker executable")
    archiver_artifact = _checked_artifact(tools["native_archiver"],
                                          "native archiver executable")
    receipt_artifact = _checked_artifact(value["build_receipt"], "build receipt")
    output_artifact = _checked_artifact(value["adapter_executable"], "adapter executable")
    live_inventory = _artifact(Path(inventory_artifact["path"]), "source inventory",
                               inventory_sha256)
    live_closure = _artifact(Path(closure_artifact["path"]), "source closure", closure_sha256)
    live_cache = _artifact(Path(cache_artifact["path"]), "cache manifest", cache_sha256)
    live_cargo = _artifact(Path(cargo_artifact["path"]), "Cargo executable", cargo_sha256,
                           MAX_EXECUTABLE_BYTES, True)
    live_rustc = _artifact(Path(rustc_artifact["path"]), "rustc executable", rustc_sha256,
                           MAX_EXECUTABLE_BYTES, True)
    live_compiler_linker = _artifact(
        Path(compiler_linker_artifact["path"]), "native compiler/linker executable",
        native_compiler_linker_sha256, MAX_EXECUTABLE_BYTES, True)
    live_archiver = _artifact(Path(archiver_artifact["path"]), "native archiver executable",
                              native_archiver_sha256, MAX_EXECUTABLE_BYTES, True)
    require((inventory_artifact, closure_artifact, cache_artifact,
             cargo_artifact, rustc_artifact, compiler_linker_artifact, archiver_artifact)
            == (live_inventory, live_closure, live_cache, live_cargo, live_rustc,
                live_compiler_linker, live_archiver),
            "PIE adapter build input identity differs from live files")
    cargo_path = Path(cargo_artifact["path"])
    rustc_path = Path(rustc_artifact["path"])
    require(cargo_path.name == "cargo" and rustc_path.name == "rustc"
            and cargo_path.parent == rustc_path.parent
            and len({item["path"] for item in
                     (cargo_artifact, rustc_artifact, compiler_linker_artifact,
                      archiver_artifact)}) == 4,
            "PIE adapter record does not bind distinct sibling cargo and rustc tools")
    inventory_bytes, closure_bytes, inventory, closure = _source_context(
        Path(inventory_artifact["path"]), inventory_sha256, Path(closure_artifact["path"]),
        closure_sha256, workspace_root, commit_reader)
    image_id = _image_sha(builder_image_id, "builder image id")
    image_digest = _image_sha(builder_image_digest, "builder image digest")
    cache_bytes = _read_bounded(Path(cache_artifact["path"]), "builder cache identity")
    cache = _cache_identity(cache_bytes, image_id, image_digest)
    require(builder["image_id"] == image_id and builder["image_digest"] == image_digest
            and builder["cache_identity"] == cache,
            "PIE adapter builder trust root differs")
    target, executable, cwd = _target_paths(inventory, workspace_root)
    invocation = value["invocation"]
    require(isinstance(invocation, dict), "PIE adapter build invocation must be an object")
    require_exact_keys(invocation, {"contract", "shell", "argv", "environment",
                       "working_directory", "resource_limits", "wall_timeout_seconds"},
                       "PIE adapter build invocation")
    expected_argv = [cargo_artifact["path"], *inventory["build"]["command"][1:]]
    environment = _environment(canonical_bytes(invocation["environment"]).decode(), target,
                               inventory["build"]["target"], cache, rustc_artifact,
                               compiler_linker_artifact, archiver_artifact)
    limits = {"address_space_bytes": ADDRESS_SPACE_LIMIT_BYTES, "data_bytes": DATA_LIMIT_BYTES,
              "cpu_seconds": CPU_LIMIT_SECONDS, "file_size_bytes": FILE_SIZE_LIMIT_BYTES}
    require(invocation == {"contract": "direct-cargo-explicit-rustc-cc-linker-ar-v1",
            "shell": False,
            "argv": expected_argv, "environment": environment, "working_directory": str(cwd),
            "resource_limits": limits, "wall_timeout_seconds": WALL_TIMEOUT_SECONDS},
            "PIE adapter build invocation differs")
    revalidation = value["source_revalidation"]
    expected_revalidation = {"contract": "same-live-source-inventory-and-closure-before-after-v1",
        "before_inventory_raw_sha256": inventory_sha256,
        "after_inventory_raw_sha256": inventory_sha256,
        "before_source_closure_raw_sha256": closure_sha256,
        "after_source_closure_raw_sha256": closure_sha256,
        "canonical_inventory_sha256": sha256_bytes(canonical_bytes(inventory)),
        "canonical_source_closure_sha256": closure["source_closure_sha256"]}
    require(revalidation == expected_revalidation,
            "PIE adapter source revalidation record differs")
    target_state = _target_state(target)
    require(value["target_lifecycle"] == {"contract": "fresh-absent-target-to-bounded-tree-v1",
            "pre_launch": "absent", "post_launch": target_state, "fresh_transition": True},
            "PIE adapter target lifecycle differs")
    child = value["child_result"]
    require(isinstance(child, dict), "PIE adapter child result must be an object")
    require_exact_keys(child, {"exit_status", "stdout", "stderr"}, "PIE adapter child result")
    require(child["exit_status"] == 0, "PIE adapter builder exit status differs")
    for name, maximum in (("stdout", MAX_STDOUT_BYTES), ("stderr", MAX_STDERR_BYTES)):
        stream = child[name]
        require(isinstance(stream, dict) and set(stream) == {"byte_length", "sha256"}
                and require_int(stream["byte_length"], f"builder {name} bytes") <= maximum,
                f"PIE adapter builder {name} identity differs")
        require_sha256(stream["sha256"], f"builder {name} sha256")
    live_output = _artifact(executable, "adapter executable", maximum=MAX_EXECUTABLE_BYTES,
                            executable=True)
    live_receipt = _artifact(Path(receipt_artifact["path"]), "build receipt")
    require(output_artifact == live_output and receipt_artifact == live_receipt,
            "PIE adapter build output identity differs from live files")
    publication = value["executable_publication"]
    require(isinstance(publication, dict), "PIE adapter executable publication must be an object")
    require_exact_keys(publication, {"contract", "source_link_count", "detached", "source_sha256",
                       "final_sha256", "byte_equal"}, "PIE adapter executable publication")
    links = require_int(publication["source_link_count"], "Cargo executable source links", 1)
    require(publication == {"contract": "byte-checked-single-link-detach-v1",
            "source_link_count": links, "detached": links > 1,
            "source_sha256": output_artifact["sha256"],
            "final_sha256": output_artifact["sha256"], "byte_equal": True},
            "PIE adapter executable publication differs")
    receipt_bytes = _read_bounded(Path(receipt_artifact["path"]), "PIE adapter build receipt")
    _, receipt = validate_documents(inventory_bytes, inventory_sha256, closure_bytes,
                                    receipt_bytes, workspace_root, commit_reader)
    receipt_output = receipt["build_receipt"]["adapter_executable"]
    require(output_artifact == {"path": str(executable), "byte_length": receipt_output["byte_length"],
            "sha256": receipt_output["sha256"], "mode": receipt_output["mode"], "link_count": 1},
            "PIE adapter execution output does not bind its v2 receipt")
    require(value["receipt_binding"] == {
        "raw_sha256": receipt_artifact["sha256"],
        "canonical_build_receipt_sha256": receipt["build_receipt_sha256"],
        "v2_build_execution_attested": False,
    }, "PIE adapter v2 receipt binding differs")
    return value


def parse_record_bytes(payload: bytes, **context: Any) -> dict[str, Any]:
    return validate_record(_parse_json(payload, "PIE adapter build execution record"), **context)


def execute(args: Any, *, runner: Runner | None = None,
            workspace_root: Path = WORKSPACE_ROOT,
            commit_reader: CommitReader = _git_commit) -> dict[str, Any]:
    inventory_path = args.inventory.absolute()
    closure_path = args.source_closure.absolute()
    cache_path = args.cache_identity.absolute()
    cargo_path = args.cargo.absolute()
    rustc_path = args.rustc.absolute()
    compiler_linker_path = args.native_compiler_linker.absolute()
    archiver_path = args.native_archiver.absolute()
    receipt_path = args.build_receipt.absolute()
    record_path = args.record.absolute()
    paths = [inventory_path, closure_path, cache_path, cargo_path, rustc_path,
             compiler_linker_path, archiver_path, receipt_path, record_path]
    require(len(set(paths)) == len(paths), "PIE adapter build transaction paths are not distinct")
    require(cargo_path.name == "cargo" and rustc_path.name == "rustc"
            and cargo_path.parent == rustc_path.parent,
            "PIE adapter cargo and rustc must be exact sibling toolchain executables")
    require(not receipt_path.exists() and not receipt_path.is_symlink()
            and not record_path.exists() and not record_path.is_symlink(),
            "PIE adapter build receipt and execution record paths must be fresh")
    image_id = _image_sha(args.builder_image_id, "builder image id")
    image_digest = _image_sha(args.builder_image_digest, "builder image digest")
    inventory_bytes, closure_bytes, inventory, closure = _source_context(
        inventory_path, args.inventory_sha256, closure_path, args.source_closure_sha256,
        workspace_root, commit_reader)
    inventory_artifact = _artifact(inventory_path, "source inventory", args.inventory_sha256)
    closure_artifact = _artifact(closure_path, "source closure", args.source_closure_sha256)
    cache_artifact = _artifact(cache_path, "cache identity", args.cache_identity_sha256)
    cargo_artifact = _artifact(cargo_path, "Cargo executable", args.cargo_sha256,
                               MAX_EXECUTABLE_BYTES, True)
    rustc_artifact = _artifact(rustc_path, "rustc executable", args.rustc_sha256,
                               MAX_EXECUTABLE_BYTES, True)
    compiler_linker_artifact = _artifact(
        compiler_linker_path, "native compiler/linker executable",
        args.native_compiler_linker_sha256, MAX_EXECUTABLE_BYTES, True)
    archiver_artifact = _artifact(archiver_path, "native archiver executable",
                                  args.native_archiver_sha256, MAX_EXECUTABLE_BYTES, True)
    cache_bytes = _read_bounded(cache_path, "PIE adapter builder cache identity")
    cache = _cache_identity(cache_bytes, image_id, image_digest)
    target, executable, cwd = _target_paths(inventory, workspace_root)
    require(not receipt_path.is_relative_to(target) and not record_path.is_relative_to(target),
            "PIE adapter build records must live outside the fresh target")
    _require_fresh_target(target)
    environment = _environment(args.environment_json, target, inventory["build"]["target"],
                               cache, rustc_artifact, compiler_linker_artifact,
                               archiver_artifact)
    argv = [str(cargo_path), *inventory["build"]["command"][1:]]
    committed = False
    try:
        run = (runner or _run_builder)(argv, cwd, environment)
        require(isinstance(run, subprocess.CompletedProcess),
                "PIE adapter builder did not return a completed process")
        require(run.returncode == 0, f"PIE adapter builder exit status differs: {run.returncode}")
        stdout = _stream(run.stdout, MAX_STDOUT_BYTES, "PIE adapter builder stdout")
        stderr = _stream(run.stderr, MAX_STDERR_BYTES, "PIE adapter builder stderr")
        target_state = _target_state(target)
        publication, output_artifact = _detach_executable(executable)
        after = _source_context(inventory_path, args.inventory_sha256, closure_path,
                                args.source_closure_sha256, workspace_root, commit_reader)
        require(after[:2] == (inventory_bytes, closure_bytes) and after[2:] == (inventory, closure),
                "PIE adapter live source context changed during build execution")
        require(_artifact(cache_path, "cache identity", args.cache_identity_sha256)
                == cache_artifact
                and _artifact(cargo_path, "Cargo executable", args.cargo_sha256,
                              MAX_EXECUTABLE_BYTES, True) == cargo_artifact
                and _artifact(rustc_path, "rustc executable", args.rustc_sha256,
                              MAX_EXECUTABLE_BYTES, True) == rustc_artifact
                and _artifact(compiler_linker_path, "native compiler/linker executable",
                              args.native_compiler_linker_sha256, MAX_EXECUTABLE_BYTES,
                              True) == compiler_linker_artifact
                and _artifact(archiver_path, "native archiver executable",
                              args.native_archiver_sha256, MAX_EXECUTABLE_BYTES,
                              True) == archiver_artifact,
                "PIE adapter toolchain or cache declaration changed during execution")
        generate(inventory_path, args.inventory_sha256, closure_path, receipt_path,
                 workspace_root, commit_reader)
        receipt_artifact = _artifact(receipt_path, "build receipt")
        receipt_bytes = _read_bounded(receipt_path, "PIE adapter build receipt")
        _, receipt = validate_documents(inventory_bytes, args.inventory_sha256, closure_bytes,
                                        receipt_bytes, workspace_root, commit_reader)
        record = {
            "schema_version": SCHEMA, "evidence_mode": EVIDENCE_MODE, "passed": True,
            "build_execution_attested": True, "causal_scope": CAUSAL_SCOPE,
            "builder_status": BUILDER_STATUS, "builder_runtime_attested": False,
            "writer_precondition": WRITER_PRECONDITION,
            "production_admissible": False, "correctness_admissible": False,
            "performance_admissible": False, "source_inventory": inventory_artifact,
            "source_closure": closure_artifact,
            "builder": {"image_id": image_id, "image_digest": image_digest,
                        "tools": {"cargo": cargo_artifact, "rustc": rustc_artifact,
                                  "native_compiler_linker": compiler_linker_artifact,
                                  "native_archiver": archiver_artifact},
                        "cache_manifest": cache_artifact,
                        "cache_identity": cache},
            "invocation": {"contract": "direct-cargo-explicit-rustc-cc-linker-ar-v1",
                "shell": False,
                "argv": argv, "environment": environment, "working_directory": str(cwd),
                "resource_limits": {"address_space_bytes": ADDRESS_SPACE_LIMIT_BYTES,
                    "data_bytes": DATA_LIMIT_BYTES, "cpu_seconds": CPU_LIMIT_SECONDS,
                    "file_size_bytes": FILE_SIZE_LIMIT_BYTES},
                "wall_timeout_seconds": WALL_TIMEOUT_SECONDS},
            "source_revalidation": {
                "contract": "same-live-source-inventory-and-closure-before-after-v1",
                "before_inventory_raw_sha256": args.inventory_sha256,
                "after_inventory_raw_sha256": args.inventory_sha256,
                "before_source_closure_raw_sha256": args.source_closure_sha256,
                "after_source_closure_raw_sha256": args.source_closure_sha256,
                "canonical_inventory_sha256": sha256_bytes(canonical_bytes(inventory)),
                "canonical_source_closure_sha256": closure["source_closure_sha256"]},
            "target_lifecycle": {"contract": "fresh-absent-target-to-bounded-tree-v1",
                "pre_launch": "absent", "post_launch": target_state, "fresh_transition": True},
            "child_result": {"exit_status": 0, "stdout": stdout, "stderr": stderr},
            "executable_publication": publication, "adapter_executable": output_artifact,
            "build_receipt": receipt_artifact,
            "receipt_binding": {"raw_sha256": receipt_artifact["sha256"],
                "canonical_build_receipt_sha256": receipt["build_receipt_sha256"],
                "v2_build_execution_attested": False},
        }
        context = {"inventory_sha256": args.inventory_sha256,
                   "closure_sha256": args.source_closure_sha256,
                   "cache_sha256": args.cache_identity_sha256,
                   "cargo_sha256": args.cargo_sha256, "rustc_sha256": args.rustc_sha256,
                   "native_compiler_linker_sha256": args.native_compiler_linker_sha256,
                   "native_archiver_sha256": args.native_archiver_sha256,
                   "builder_image_id": image_id, "builder_image_digest": image_digest,
                   "workspace_root": workspace_root, "commit_reader": commit_reader}
        validate_record(record, **context)
        payload = canonical_bytes(record) + b"\n"
        require(len(payload) <= MAX_JSON_BYTES, "PIE adapter build record is too large")
        write_immutable_bytes(record_path, payload,
                              [inventory_path, closure_path, cache_path, cargo_path, rustc_path,
                               compiler_linker_path, archiver_path, receipt_path, executable],
                              fresh_only=True,
                              _before_publish=lambda _: validate_record(record, **context))
        installed = parse_record_bytes(record_path.read_bytes(), **context)
        committed = True
        return installed
    finally:
        if not committed:
            _remove_fresh_output(record_path)
            _remove_fresh_output(receipt_path)
            _remove_fresh_output(target)
