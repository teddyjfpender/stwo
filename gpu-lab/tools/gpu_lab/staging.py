"""Content-addressed pod-local inputs and authenticated execution paths."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

from .common import canonical_bytes, load_json, require, sha256_bytes, sha256_file


LOCAL_MARKER_SCHEMA = "stwo.gpu-lab.local-root.v1"
STAGING_SCHEMA = "stwo.gpu-lab.staging.v1"
EXECUTION_ROLES = {
    "build_commands", "execution", "harness", "module", "module_index", "plan", "replay",
}


def _is_workspace(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == Path("/workspace") or Path("/workspace") in resolved.parents


def _regular(path: Path, label: str) -> Path:
    require(path.is_absolute(), f"{label} path must be absolute")
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
            f"{label} must be a regular non-symlink file: {path}")
    return path


def _artifact(path: Path, role: str, origin: str, source: Path | None = None) -> dict[str, Any]:
    path = _regular(path.absolute(), role).resolve()
    source = _regular((source or path).absolute(), f"{role} source").resolve()
    source_hash, destination_hash = sha256_file(source), sha256_file(path)
    require(source_hash == destination_hash and source.stat().st_size == path.stat().st_size,
            f"{role} changed while staging")
    return {
        "bytes": path.stat().st_size,
        "destination_path": str(path),
        "destination_sha256": destination_hash,
        "origin": origin,
        "role": role,
        "source_path": str(source),
        "source_sha256": source_hash,
    }


def _with_hash(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["record_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def _validate_marker(local_root: Path, pod_id: str) -> dict[str, Any]:
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", pod_id) is not None,
            "pod-local staging requires a safe RUNPOD_POD_ID")
    require(local_root.is_absolute() and not _is_workspace(local_root),
            "pod-local root must be absolute and outside /workspace")
    resolved = local_root.resolve()
    require(resolved == local_root, "pod-local root contains a symlink or noncanonical component")
    marker = load_json(resolved / "LOCAL_ROOT.json")
    require(set(marker) == {
        "schema_version", "pod_id", "volume_id", "local_root", "local_mount", "local_device",
        "workspace_mount", "workspace_device",
    }, "pod-local marker keys differ")
    require(marker["schema_version"] == LOCAL_MARKER_SCHEMA and marker["pod_id"] == pod_id,
            "pod-local marker belongs to another lease")
    require(Path(marker["local_root"]).resolve() == resolved,
            "pod-local marker names another root")
    require(isinstance(marker["volume_id"], str) and marker["volume_id"],
            "pod-local marker has no volume identity")
    require(isinstance(marker["local_mount"], str) and marker["local_mount"],
            "pod-local marker has no local mount identity")
    require(isinstance(marker["workspace_mount"], str) and marker["workspace_mount"],
            "pod-local marker has no workspace mount identity")
    require(marker["local_mount"] != marker["workspace_mount"],
            "declared local root resolves to the network-volume mount")
    require(isinstance(marker["local_device"], str) and marker["local_device"]
            and isinstance(marker["workspace_device"], str) and marker["workspace_device"]
            and marker["local_device"] != marker["workspace_device"],
            "declared local root is not a distinct device")
    require(not (resolved / "SEALED").exists(), "pod-local lease is sealed against new work")
    return marker


def _chunk_descriptors(fixture: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if fixture.get("schema_version") != "stwo.gpu-lab.semantic-fixture-index.v2":
        return []
    payload = fixture.get("semantic_payload")
    require(isinstance(payload, dict), "indexed fixture semantic payload is missing")
    table = payload.get("address_to_id")
    inputs, expected = payload.get("input_chunks"), payload.get("expected_chunks")
    require(isinstance(table, dict) and isinstance(inputs, list) and isinstance(expected, list),
            "indexed fixture chunk descriptors are missing")
    descriptors = [("chunk.address_to_id", table)]
    descriptors += [(f"chunk.input.{index}", item) for index, item in enumerate(inputs)]
    descriptors += [(f"chunk.expected.{index}", item) for index, item in enumerate(expected)]
    return descriptors


def _validated_chunk(index: Path, role: str, descriptor: dict[str, Any]) -> tuple[Path, Path]:
    relative = descriptor.get("path")
    digest, size = descriptor.get("sha256"), descriptor.get("byte_len")
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, f"unsafe {role} path")
    address = re.fullmatch(
        r"chunks/sha256/([0-9a-f]{64})\.[a-z0-9][a-z0-9.-]{0,31}", relative
    )
    require(address is not None, f"noncanonical {role} content address")
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
            and address.group(1) == digest, f"invalid {role} digest")
    require(isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            f"invalid {role} byte length")
    raw_source = (index.parent / relative).absolute()
    _regular(raw_source, role)
    source = raw_source.resolve()
    require(source.is_relative_to(index.parent.resolve()), f"{role} escapes fixture store")
    require(source.stat().st_size == size and sha256_file(source) == digest,
            f"{role} content differs from its descriptor")
    return source, Path(relative)


def _copy_verified(source: Path, destination: Path, root: Path, digest: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(destination.parent.resolve().is_relative_to(root), "staging destination escapes root")
    if destination.exists():
        _regular(destination, "existing staged object")
        require(sha256_file(destination) == digest, "immutable staged object was mutated")
        return
    descriptor, name = tempfile.mkstemp(prefix=".stage-", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o400)
        require(sha256_file(temporary) == digest and sha256_file(source) == digest,
                "source changed while copying to local storage")
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    _regular(destination, "staged object")
    require(sha256_file(destination) == digest, "post-copy staging verification failed")


def stage_fixture(source: Path, local_root: Path | None, pod_id: str,
                  build_root: Path) -> dict[str, Any]:
    source = _regular(source.absolute(), "fixture").resolve()
    fixture, fixture_hash = load_json(source), sha256_file(source)
    pod_mode = bool(pod_id) or _is_workspace(source)
    if not pod_mode:
        entries = [_artifact(source, "fixture", "local_developer")]
        for role, descriptor in _chunk_descriptors(fixture):
            chunk, _ = _validated_chunk(source, role, descriptor)
            entries.append(_artifact(chunk, role, "local_developer"))
        return _with_hash({
            "schema_version": STAGING_SCHEMA,
            "mode": "local_developer",
            "pod_id": None,
            "local_root": None,
            "build_root": str(build_root.resolve()),
            "entries": sorted(entries, key=lambda item: item["role"]),
        })

    require(local_root is not None, "pod execution requires a declared GPU_LAB_LOCAL_ROOT")
    declared_root = local_root.absolute()
    _validate_marker(declared_root, pod_id)
    root = declared_root.resolve()
    build = build_root.resolve()
    require(build.is_relative_to(root) and not _is_workspace(build),
            "pod build output must be beneath the declared local root")
    fixture_root = root / "fixtures" / fixture_hash
    destination = fixture_root / "index.json"
    _copy_verified(source, destination, root, fixture_hash)
    entries = [
        _artifact(destination, "fixture", "verified_copy", source),
        _artifact(root / "LOCAL_ROOT.json", "local_root_marker", "generated_local"),
    ]
    for role, descriptor in _chunk_descriptors(fixture):
        chunk, relative = _validated_chunk(source, role, descriptor)
        staged = fixture_root / relative
        _copy_verified(chunk, staged, root, descriptor["sha256"])
        entries.append(_artifact(staged, role, "verified_copy", chunk))
    return _with_hash({
        "schema_version": STAGING_SCHEMA,
        "mode": "pod_local_nvme",
        "pod_id": pod_id,
        "local_root": str(root),
        "build_root": str(build),
        "entries": sorted(entries, key=lambda item: item["role"]),
    })


def validate_staging(value: Any, *, sealed: bool) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {
        "schema_version", "mode", "pod_id", "local_root", "build_root", "entries",
        "record_sha256",
    }, "staging record keys differ")
    require(value["schema_version"] == STAGING_SCHEMA,
            "unsupported staging record schema")
    expected_hash = sha256_bytes(canonical_bytes({
        name: item for name, item in value.items() if name != "record_sha256"
    }))
    require(value["record_sha256"] == expected_hash, "staging record self-hash mismatch")
    require(value["mode"] in {"local_developer", "pod_local_nvme"},
            "invalid staging mode")
    entries = value["entries"]
    require(isinstance(entries, list) and entries, "staging entries are empty")
    roles = []
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == {
            "bytes", "destination_path", "destination_sha256", "origin", "role",
            "source_path", "source_sha256",
        }, "staging entry keys differ")
        require(entry["origin"] in {"generated_local", "local_developer", "verified_copy"},
                "invalid staging entry origin")
        require(isinstance(entry["role"], str) and entry["role"], "empty staging role")
        require(all(isinstance(entry[name], str) and Path(entry[name]).is_absolute()
                    for name in ("source_path", "destination_path")),
                f"staging paths are not absolute for {entry['role']}")
        if entry["origin"] != "verified_copy":
            require(entry["source_path"] == entry["destination_path"],
                    f"non-copy staging paths differ for {entry['role']}")
        require(re.fullmatch(r"[0-9a-f]{64}", str(entry["source_sha256"])) is not None
                and entry["source_sha256"] == entry["destination_sha256"],
                f"staging hash mismatch for {entry['role']}")
        require(isinstance(entry["bytes"], int) and not isinstance(entry["bytes"], bool)
                and entry["bytes"] >= 0, f"invalid staged byte count for {entry['role']}")
        roles.append(entry["role"])
    require(len(roles) == len(set(roles)), "duplicate staging role")
    require(roles == sorted(roles), "staging entries are not canonically ordered")
    require("fixture" in roles, "staging record has no fixture")
    if sealed:
        require(EXECUTION_ROLES.issubset(roles), "sealed staging record lacks execution artifacts")
    if value["mode"] == "pod_local_nvme":
        require(isinstance(value["pod_id"], str) and isinstance(value["local_root"], str),
                "pod staging identity is missing")
        root = Path(value["local_root"]).resolve()
        require(not _is_workspace(root), "pod staging root is on /workspace")
        require("local_root_marker" in roles, "pod staging record lacks its mount declaration")
        build = Path(str(value["build_root"])).resolve()
        require(build.is_relative_to(root) and not _is_workspace(build),
                "pod build output is not local")
        for entry in entries:
            destination = Path(entry["destination_path"]).resolve()
            require(destination.is_relative_to(root) and not _is_workspace(destination),
                    f"active {entry['role']} is not pod-local")
            expected_origin = "generated_local" if (entry["role"] in EXECUTION_ROLES
                                                      or entry["role"] == "local_root_marker") \
                else "verified_copy"
            require(entry["origin"] == expected_origin,
                    f"pod staging origin is wrong for {entry['role']}")
    else:
        require(value["pod_id"] is None and value["local_root"] is None,
                "local developer staging unexpectedly names a pod root")
        require(all(entry["origin"] == "local_developer" for entry in entries),
                "local developer staging contains a copied/generated origin")
    return value


def seal_staging(seed: dict[str, Any], build_root: Path,
                 artifacts: dict[str, Path]) -> dict[str, Any]:
    validate_staging(seed, sealed=False)
    if seed["mode"] == "pod_local_nvme":
        _validate_marker(Path(seed["local_root"]), seed["pod_id"])
    require(Path(str(seed["build_root"])).resolve() == build_root.resolve(),
            "build root changed after fixture staging")
    require(set(artifacts) == EXECUTION_ROLES, "execution staging roles differ")
    entries = list(seed["entries"])
    for role, path in sorted(artifacts.items()):
        entries.append(_artifact(path.resolve(), role,
                                 "generated_local" if seed["mode"] == "pod_local_nvme"
                                 else "local_developer"))
    result = _with_hash({
        **{name: item for name, item in seed.items() if name not in {"entries", "record_sha256"}},
        "build_root": str(build_root.resolve()),
        "entries": sorted(entries, key=lambda item: item["role"]),
    })
    validate_staging(result, sealed=True)
    return result


def verify_staging(value: dict[str, Any]) -> None:
    validate_staging(value, sealed=True)
    if value["mode"] == "pod_local_nvme":
        _validate_marker(Path(value["local_root"]), value["pod_id"])
    for entry in value["entries"]:
        path = _regular(Path(entry["destination_path"]).absolute(), entry["role"]).resolve()
        require(path.stat().st_size == entry["bytes"]
                and sha256_file(path) == entry["destination_sha256"],
                f"active staged artifact changed: {entry['role']}")


def _rejected(call, label: str) -> None:
    try:
        call()
    except (OSError, ValueError, KeyError, TypeError):
        return
    raise AssertionError(f"accepted invalid staging case: {label}")


def staging_self_test() -> None:
    with tempfile.TemporaryDirectory() as name:
        temporary = Path(name)
        source = temporary / "source"
        source.mkdir()
        chunk_bytes = b"\x01\x00\x00\x00" * 3
        chunk_hash = sha256_bytes(chunk_bytes)
        chunk_relative = Path("chunks/sha256") / f"{chunk_hash}.m31le"
        chunk = source / chunk_relative
        chunk.parent.mkdir(parents=True)
        chunk.write_bytes(chunk_bytes)
        descriptor = {
            "encoding": "m31-le-u32-row-major-v1", "path": str(chunk_relative),
            "sha256": chunk_hash, "byte_len": len(chunk_bytes),
        }
        fixture = source / "fixture.json"
        fixture.write_text(json.dumps({
            "schema_version": "stwo.gpu-lab.semantic-fixture-index.v2",
            "semantic_payload": {
                "address_to_id": {**descriptor, "element_count": 3},
                "input_chunks": [{**descriptor, "row_start": 0, "row_count": 1,
                                  "words_per_row": 3}],
                "expected_chunks": [{**descriptor, "row_start": 0, "row_count": 1,
                                     "words_per_row": 23}],
            },
        }))
        root = (temporary / "local").resolve()
        root.mkdir()
        marker = {
            "schema_version": LOCAL_MARKER_SCHEMA, "pod_id": "pod-test",
            "volume_id": "volume-test", "local_root": str(root),
            "local_mount": "overlay overlay /", "workspace_mount": "nfs nfs /workspace",
            "local_device": "0:1 overlay overlay", "workspace_device": "0:2 nfs nfs",
        }
        (root / "LOCAL_ROOT.json").write_text(json.dumps(marker))
        build = root / "build" / "sm86"
        seed = stage_fixture(fixture, root, "pod-test", build)
        require(seed == stage_fixture(fixture, root, "pod-test", build),
                "identical fixture staging was not deterministic")
        _rejected(lambda: stage_fixture(fixture, root, "pod-test", Path("/workspace/build")),
                  "network-volume build root")
        staged_fixture = next(item for item in seed["entries"] if item["role"] == "fixture")
        require(Path(staged_fixture["destination_path"]).is_relative_to(root),
                "fixture was not staged locally")
        files = {}
        for role in EXECUTION_ROLES:
            path = build / role
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(role)
            files[role] = path
        sealed = seal_staging(seed, build, files)
        require(sealed == seal_staging(seed, build, files),
                "identical execution staging was not deterministic")
        verify_staging(sealed)
        hostile = json.loads(json.dumps(sealed))
        hostile["entries"][0]["destination_sha256"] = "0" * 64
        _rejected(lambda: validate_staging(hostile, sealed=True), "tampered record")
        _rejected(lambda: seal_staging(seed, root / "other", files), "changed build root")
        incomplete = dict(files)
        incomplete.pop("module")
        _rejected(lambda: seal_staging(seed, build, incomplete), "missing module")
        destination = Path(staged_fixture["destination_path"])
        os.chmod(destination, 0o600)
        destination.write_text("changed")
        _rejected(lambda: verify_staging(sealed), "mutated staged fixture")
        _rejected(lambda: stage_fixture(fixture, Path("/workspace/not-local"),
                                        "pod-test", Path("/workspace/build")),
                  "network-volume local root")
        link = temporary / "linked-root"
        link.symlink_to(root, target_is_directory=True)
        _rejected(lambda: stage_fixture(fixture, link, "pod-test", root / "build"),
                  "symlink local root")
        local_fixture = temporary / "local.json"
        local_fixture.write_text("{}")
        local = stage_fixture(local_fixture, None, "", temporary / "build")
        require(local["mode"] == "local_developer", "local developer path was not preserved")


def _assignments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        role, separator, path = value.partition("=")
        require(separator == "=" and role in EXECUTION_ROLES and role not in result and path,
                f"invalid staging artifact: {value}")
        result[role] = Path(path)
    return result


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f"staging record path is a symlink: {path}")
    payload = (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=".staging-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("fixture")
    stage.add_argument("--source", required=True, type=Path)
    stage.add_argument("--local-root", type=Path)
    stage.add_argument("--pod-id", default=os.environ.get("RUNPOD_POD_ID", ""))
    stage.add_argument("--build-root", required=True, type=Path)
    stage.add_argument("--output", required=True, type=Path)
    seal = sub.add_parser("seal")
    seal.add_argument("--seed", required=True, type=Path)
    seal.add_argument("--build-root", required=True, type=Path)
    seal.add_argument("--artifact", action="append", default=[])
    seal.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--record", required=True, type=Path)
    sub.add_parser("self-test")
    args = parser.parse_args(argv)
    try:
        if args.command == "fixture":
            value = stage_fixture(args.source, args.local_root, args.pod_id, args.build_root)
            if value["mode"] == "pod_local_nvme":
                require(args.output.absolute().resolve().parent.is_relative_to(
                    Path(value["local_root"])), "staging record output is not pod-local")
            _write(args.output, value)
            fixture = next(item for item in value["entries"] if item["role"] == "fixture")
            print(fixture["destination_path"])
        elif args.command == "seal":
            value = seal_staging(load_json(args.seed), args.build_root,
                                 _assignments(args.artifact))
            if value["mode"] == "pod_local_nvme":
                require(args.output.absolute().resolve().parent.is_relative_to(
                    Path(value["local_root"])), "sealed staging record output is not pod-local")
            _write(args.output, value)
            print(value["record_sha256"])
        elif args.command == "verify":
            verify_staging(load_json(args.record))
            print(f"staging valid: {args.record}")
        else:
            staging_self_test()
            print("stage-run self-test: PASS")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"stage-run: {error}", file=os.sys.stderr)
        return 1
    return 0
