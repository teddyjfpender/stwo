"""Standalone hostile checks for the PIE adapter data contracts."""

# gpu-lab-cohesion-review: one mutation suite keeps runtime and schema parity auditable.

from __future__ import annotations

import copy
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .common import LAB_ROOT, canonical_bytes, require, sha256_bytes
from .pie_adapter_contract import (
    ADDRESS_SPACE_LIMIT_BYTES,
    ARTIFACT_SPECS,
    CPU_LIMIT_SECONDS,
    DATA_LIMIT_BYTES,
    BUILD_RECEIPT_STATUS,
    EMPTY_SHA256,
    EVIDENCE_MODE,
    EXECUTION_HONESTY,
    EXECUTABLE_POLICY,
    EXECUTION_RECORD_SCHEMA,
    INVOCATION_SCHEMA,
    MAX_EXECUTION_RECORD_BYTES,
    MAX_INVOCATION_BYTES,
    OUTPUT_ENCODING,
    PROTOCOL,
    RECEIPT_TRUST,
    SOURCE_CLOSURE_STATUS,
    WALL_TIMEOUT_SECONDS,
    invocation_bytes,
    parse_execution_record_bytes,
    parse_invocation_bytes,
    validate_adapter_receipt_projection,
    validate_execution_record,
    validate_invocation,
)
from .pie_adapter_receipt import generate
from .pie_adapter_receipt_tests import _commit, _inventory_bytes, _workspace


def _digest(label: str) -> str:
    return sha256_bytes(label.encode())


def _artifact(role: str, filename: str, size: int = 17,
              sha256: str | None = None) -> dict[str, Any]:
    kind, _ = ARTIFACT_SPECS[role]
    return {"kind": kind, "path": f"/sealed/{filename}",
            "byte_length": size, "sha256": sha256 or _digest(role)}


def _invocation() -> dict[str, Any]:
    return {
        "schema_version": INVOCATION_SCHEMA, "protocol": PROTOCOL,
        "bootloader_program": _artifact("bootloader_program", "bootloader.json", 31),
        "pie_copies": 1, "backend": "simd", "engine": "legacy",
        "output_encoding": OUTPUT_ENCODING,
    }


def _retained_artifact(role: str, path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "kind": ARTIFACT_SPECS[role][0], "path": str(path),
        "byte_length": len(payload), "sha256": sha256_bytes(payload),
    }


def _record(executable: Path, inventory_path: Path, inventory_bytes: bytes,
            closure_path: Path, closure_bytes: bytes, receipt_path: Path,
            receipt_bytes: bytes) -> dict[str, Any]:
    invocation = _invocation()
    invocation_bytes = canonical_bytes(invocation) + b"\n"
    artifacts = {
        "source_pie": _artifact("source_pie", "source.zip", 103),
        "bootloader_program": copy.deepcopy(invocation["bootloader_program"]),
        "expected_prover_input": _artifact("expected_prover_input", "expected.bin", 107),
        "adapter_executable": _retained_artifact(
            "adapter_executable", executable, executable.read_bytes()),
        "adapter_invocation": {
            "kind": ARTIFACT_SPECS["adapter_invocation"][0], "path": "/sealed/invocation.json",
            "byte_length": len(invocation_bytes), "sha256": sha256_bytes(invocation_bytes),
        },
        "adapter_source_inventory": _retained_artifact(
            "adapter_source_inventory", inventory_path, inventory_bytes),
        "adapter_source_closure": _retained_artifact(
            "adapter_source_closure", closure_path, closure_bytes),
        "adapter_build_receipt": _retained_artifact(
            "adapter_build_receipt", receipt_path, receipt_bytes),
    }
    sources = [
        {"path": "tools/pie-adapter-replay", "sha256": _digest("launcher")},
        {"path": "tools/gpu_lab/pie_adapter_contract.py", "sha256": _digest("contract")},
    ]
    observed = {
        "kind": ARTIFACT_SPECS["expected_prover_input"][0],
        "path": "/sealed/observed.bin",
        "byte_length": artifacts["expected_prover_input"]["byte_length"],
        "sha256": artifacts["expected_prover_input"]["sha256"],
    }
    bound_artifacts = {**artifacts, "observed_prover_input": observed}
    bindings = {}
    for descriptor, role in enumerate(bound_artifacts, 10):
        artifact = bound_artifacts[role]
        bindings[role] = {
            "fd": descriptor, "proc_path": f"/proc/self/fd/{descriptor}",
            "artifact_path": artifact["path"], "byte_length": artifact["byte_length"],
            "sha256": artifact["sha256"],
            "access": (
                "retained-read-write" if role == "observed_prover_input"
                else "retained-read-only"
            ),
            "child_inherited": role in {
                "adapter_executable", "source_pie", "bootloader_program",
                "observed_prover_input",
            },
            "device": 1, "inode": 100 + descriptor, "link_count": 1,
        }
    stderr = (
        "prover input dumped: "
        f"{observed['byte_length']} bytes -> {bindings['observed_prover_input']['proc_path']}\n"
    ).encode()
    return {
        "schema_version": EXECUTION_RECORD_SCHEMA, "evidence_mode": EVIDENCE_MODE,
        "passed": True, "adapter_execution_attested": True,
        "source_closure_status": SOURCE_CLOSURE_STATUS,
        "build_receipt_status": BUILD_RECEIPT_STATUS,
        "build_execution_attested": False, "production_admissible": False,
        "correctness_admissible": False, "performance_admissible": False,
        "adapter_executable_policy": EXECUTABLE_POLICY,
        **artifacts,
        "replay_tool_source_closure": {
            "sources": sources,
            "closure_sha256": sha256_bytes(canonical_bytes(sources)),
        },
        "invocation_contract": invocation,
        "receipt_binding": {**RECEIPT_TRUST, "adapter_executable_mode": "0755"},
        "execution_contract": {
            "contract": "linux-retained-descriptor-exec-v1",
            "shell": False,
            "stdin": "devnull",
            "working_directory": "/",
            **EXECUTION_HONESTY,
            "argv": [
                bindings["adapter_executable"]["proc_path"], "--pie",
                bindings["source_pie"]["proc_path"], "--backend", "simd",
                "--engine", "legacy", "--adapt-only",
            ],
            "environment": {
                "LANG": "C", "LC_ALL": "C", "RAYON_NUM_THREADS": "8",
                "RUST_BACKTRACE": "0",
                "STWO_BOOTLOADER_JSON": bindings["bootloader_program"]["proc_path"],
                "STWO_DUMP_INPUT": bindings["observed_prover_input"]["proc_path"],
            },
            "bindings": bindings,
            "input_recheck": "descriptor-and-path-before-and-after-v1",
            "resource_limits": {
                "address_space_bytes": ADDRESS_SPACE_LIMIT_BYTES, "data_bytes": DATA_LIMIT_BYTES,
                "cpu_seconds": CPU_LIMIT_SECONDS,
                "file_size_bytes": observed["byte_length"],
            },
            "wall_timeout_seconds": WALL_TIMEOUT_SECONDS,
            "exit_status": 0,
            "stdout": {"byte_length": 0, "sha256": EMPTY_SHA256},
            "stderr": {"byte_length": len(stderr), "sha256": sha256_bytes(stderr)},
        },
        "output_lifecycle": {
            "contract": "fresh-bound-output-inode-transition-v1",
            "creation": "o_creat-o_excl-o_nofollow-0600-v1",
            "failure_cleanup": "unlink-only-if-path-still-bound-inode-v1",
            "finalization": "fchmod-0400-fsync-before-attestation-v1",
            "pre_launch": {
                "path": observed["path"], "file_type": "regular",
                "device": bindings["observed_prover_input"]["device"],
                "inode": bindings["observed_prover_input"]["inode"],
                "link_count": 1, "mode": "0600", "byte_length": 0,
            },
            "retained_descriptor": {
                "fd": bindings["observed_prover_input"]["fd"],
                "device": bindings["observed_prover_input"]["device"],
                "inode": bindings["observed_prover_input"]["inode"],
                "access": "retained-read-write",
            },
            "post_launch": {
                "path": observed["path"], "file_type": "regular",
                "device": bindings["observed_prover_input"]["device"],
                "inode": bindings["observed_prover_input"]["inode"],
                "link_count": 1, "mode": "0400",
                "byte_length": observed["byte_length"], "sha256": observed["sha256"],
            },
            "same_inode_transition": True,
        },
        "observed_prover_input": observed,
        "exact_byte_equal": True,
    }


def _reject(label: str, callback: Callable[[], Any]) -> None:
    try:
        callback()
    except (TypeError, ValueError):
        return
    raise AssertionError(f"hostile case was accepted: {label}")


Case = tuple[dict[str, Any], dict[str, Any]]


@contextmanager
def _case() -> Iterator[Case]:
    with tempfile.TemporaryDirectory(prefix="gpu-lab-adapter-contract-") as temporary:
        base = Path(temporary)
        workspace = base / "workspace"
        inventory, executable = _workspace(workspace)
        inventory_path = base / "inventory.json"
        inventory_path.write_bytes(_inventory_bytes(inventory))
        inventory_bytes = inventory_path.read_bytes()
        closure_path, receipt_path = base / "closure.json", base / "receipt.json"
        generate(inventory_path, sha256_bytes(inventory_bytes), closure_path,
                 receipt_path, workspace, _commit)
        closure_bytes, receipt_bytes = closure_path.read_bytes(), receipt_path.read_bytes()
        record = _record(executable, inventory_path, inventory_bytes, closure_path,
                         closure_bytes, receipt_path, receipt_bytes)
        arguments = {
            "adapter_source_inventory_bytes": inventory_bytes,
            "expected_adapter_source_inventory_sha256": sha256_bytes(inventory_bytes),
            "adapter_source_closure_bytes": closure_bytes,
            "expected_adapter_source_closure_sha256": sha256_bytes(closure_bytes),
            "adapter_build_receipt_bytes": receipt_bytes,
            "expected_adapter_build_receipt_sha256": sha256_bytes(receipt_bytes),
            "workspace_root": workspace,
            "_commit_reader": _commit,
        }
        yield record, arguments


def _validate(record: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    return validate_execution_record(record, **arguments)


def _rebind_raw(record: dict[str, Any], arguments: dict[str, Any], role: str,
                payload: bytes) -> None:
    digest = sha256_bytes(payload)
    for identity in (record[role], _binding(record, role)):
        identity.update({"byte_length": len(payload), "sha256": digest})
    stem = role.removeprefix("adapter_")
    arguments[f"adapter_{stem}_bytes"] = payload
    arguments[f"expected_adapter_{stem}_sha256"] = digest


def _mutate_record(case: Case, label: str,
                   mutate: Callable[[dict[str, Any]], None]) -> None:
    hostile = copy.deepcopy(case[0])
    mutate(hostile)
    _reject(label, lambda: _validate(hostile, case[1]))


def _contract(record: dict[str, Any]) -> dict[str, Any]:
    return record["execution_contract"]


def _binding(record: dict[str, Any], role: str) -> dict[str, Any]:
    return _contract(record)["bindings"][role]


def _lifecycle(record: dict[str, Any]) -> dict[str, Any]:
    return record["output_lifecycle"]


def test_invocation_contract() -> None:
    invocation = _invocation()
    require(validate_invocation(invocation) is invocation, "valid invocation was not retained")
    require(invocation_bytes(invocation) == canonical_bytes(invocation) + b"\n",
            "canonical invocation bytes differ")
    payload = canonical_bytes(invocation)
    require(parse_invocation_bytes(payload) == invocation, "invocation byte parser differs")

    for key in sorted(invocation):
        hostile = copy.deepcopy(invocation)
        del hostile[key]
        _reject(f"missing invocation {key}", lambda hostile=hostile: validate_invocation(hostile))
    hostile = copy.deepcopy(invocation)
    hostile["extra"] = False
    _reject("unknown invocation key", lambda: validate_invocation(hostile))
    for value in (True, 1.0, 2, -1):
        hostile = copy.deepcopy(invocation)
        hostile["pie_copies"] = value
        _reject(f"non-exact PIE copy count {value!r}", lambda hostile=hostile: validate_invocation(hostile))
    for value in (True, 31.0, 0, ARTIFACT_SPECS["bootloader_program"][1] + 1):
        hostile = copy.deepcopy(invocation)
        hostile["bootloader_program"]["byte_length"] = value
        _reject(f"invalid bootloader size {value!r}", lambda hostile=hostile: validate_invocation(hostile))
    hostile = copy.deepcopy(invocation)
    hostile["bootloader_program"]["sha256"] = "A" * 64
    _reject("uppercase bootloader sha256", lambda: validate_invocation(hostile))
    hostile = copy.deepcopy(invocation)
    hostile["bootloader_program"]["path"] = "/sealed/../bootloader.json"
    _reject("ambiguous bootloader path", lambda: validate_invocation(hostile))
    for path in ("/", "/sealed/", "/sealed\\bootloader.json", "/sealed/" + "é" * 1600):
        hostile = copy.deepcopy(invocation); hostile["bootloader_program"]["path"] = path
        _reject(f"noncanonical bootloader path {path!r}", lambda hostile=hostile: validate_invocation(hostile))


def test_execution_record_contract() -> None:
    with _case() as case:
        record, arguments = case
        require(_validate(record, arguments) is record,
                "valid execution record was not retained")
        require(parse_execution_record_bytes(canonical_bytes(record), **arguments) == record,
                "execution record byte parser differs")
        require(not ({"provenance_manifest", "proof_shape", "fri_capture"} & set(record)),
                "adapter-only record retained proof or FRI metadata")
        _reject("run-record hash alone", lambda: validate_execution_record({
            "schema_version": EXECUTION_RECORD_SCHEMA, "sha256": _digest("run")
        }, **arguments))
        for key in (
            "adapter_execution_attested", "build_execution_attested",
            "execution_contract", "invocation_contract", "receipt_binding",
            "adapter_source_inventory", "adapter_source_closure",
            "adapter_build_receipt", "output_lifecycle", "bootloader_program",
            "observed_prover_input", "exact_byte_equal",
        ):
            _mutate_record(case, f"missing authenticated field {key}",
                           lambda value, key=key: value.pop(key))
        _mutate_record(case, "unknown record field",
                       lambda value: value.__setitem__("extra", False))
        for key in ("production_admissible", "correctness_admissible",
                    "performance_admissible"):
            _mutate_record(case, f"overclaimed {key}",
                           lambda value, key=key: value.__setitem__(key, True))
        _mutate_record(case, "overclaimed build execution", lambda value: value.__setitem__(
            "build_execution_attested", True))
        _mutate_record(case, "overclaimed source closure", lambda value: value.__setitem__(
            "source_closure_status", "verified-build"))
        _mutate_record(case, "wrong receipt status", lambda value: value.__setitem__(
            "build_receipt_status", "build-attested"))
        _mutate_record(case, "wrong executable policy", lambda value: value.__setitem__(
            "adapter_executable_policy", "debug-allowed"))
        _mutate_record(case, "false execution attestation", lambda value: value.__setitem__(
            "adapter_execution_attested", False))
        _mutate_record(case, "false exact equality", lambda value: value.__setitem__(
            "exact_byte_equal", False))
        _mutate_record(case, "bootloader invocation mismatch", lambda value: value[
            "invocation_contract"]["bootloader_program"].__setitem__("sha256", _digest("other")))
        _mutate_record(case, "invocation artifact mismatch", lambda value: value[
            "adapter_invocation"].__setitem__("sha256", _digest("other")))
        _mutate_record(case, "observed hash mismatch", lambda value: value[
            "observed_prover_input"].__setitem__("sha256", _digest("other")))
        _mutate_record(case, "observed input alias", lambda value: value[
            "observed_prover_input"].__setitem__("path", value["expected_prover_input"]["path"]))
        _mutate_record(case, "input artifact alias", lambda value: value[
            "adapter_source_closure"].__setitem__("path", value["adapter_source_inventory"]["path"]))
        _mutate_record(case, "wrong receipt executable mode", lambda value: value[
            "receipt_binding"].__setitem__("adapter_executable_mode", "0750"))


def test_raw_receipt_trust() -> None:
    with _case() as case:
        record, arguments = case
        artifacts = {role: record[role] for role in ARTIFACT_SPECS}
        receipt = validate_adapter_receipt_projection(
            arguments["adapter_source_inventory_bytes"],
            arguments["expected_adapter_source_inventory_sha256"],
            arguments["adapter_source_closure_bytes"],
            arguments["expected_adapter_source_closure_sha256"],
            arguments["adapter_build_receipt_bytes"],
            arguments["expected_adapter_build_receipt_sha256"],
            artifacts, arguments["workspace_root"], arguments["_commit_reader"])
        require(receipt["build_receipt"]["build_execution_attested"] is False,
                "public receipt validator overclaimed build execution")
        for role in ("adapter_source_inventory", "adapter_source_closure",
                     "adapter_build_receipt"):
            hostile = dict(arguments)
            stem = role.removeprefix("adapter_")
            hostile[f"expected_adapter_{stem}_sha256"] = "0" * 64
            _reject(f"wrong raw {role} pin", lambda hostile=hostile:
                    _validate(record, hostile))
            rebound = copy.deepcopy(record)
            for identity in (rebound[role], _binding(rebound, role)):
                identity["sha256"] = _digest(f"substituted-{role}")
            _reject(f"self-consistent {role} record substitution",
                    lambda rebound=rebound: _validate(rebound, arguments))

        reordered = copy.deepcopy(record)
        reordered_arguments = dict(arguments)
        document = json.loads(arguments["adapter_source_inventory_bytes"])
        pretty = (json.dumps(dict(reversed(list(document.items()))), indent=1) + "\n").encode()
        _rebind_raw(reordered, reordered_arguments, "adapter_source_inventory", pretty)
        require(_validate(reordered, reordered_arguments) is reordered,
                "caller-pinned key-reordered inventory bytes were rejected")
        _reject("original inventory pin after exact-byte reserialization",
                lambda: _validate(reordered, arguments))

        mutated = copy.deepcopy(record)
        mutated_arguments = dict(arguments)
        receipt_document = json.loads(arguments["adapter_build_receipt_bytes"])
        receipt_document["build_receipt"]["build_execution_attested"] = True
        receipt_document["build_receipt_sha256"] = sha256_bytes(
            canonical_bytes(receipt_document["build_receipt"]))
        payload = canonical_bytes(receipt_document)
        _rebind_raw(mutated, mutated_arguments, "adapter_build_receipt", payload)
        _reject("self-consistent build-attestation overclaim",
                lambda: _validate(mutated, mutated_arguments))


def test_authenticated_execution_fields() -> None:
    with _case() as case:
        mutate = lambda label, callback: _mutate_record(case, label, callback)
        mutate("shell execution", lambda value: _contract(value).__setitem__("shell", True))
        mutate("non-root cwd", lambda value: _contract(value).__setitem__(
            "working_directory", "/sealed"))
        for field in EXECUTION_HONESTY:
            mutate(f"missing honesty field {field}", lambda value, field=field:
                   _contract(value).pop(field))
            mutate(f"wrong honesty field {field}", lambda value, field=field:
                   _contract(value).__setitem__(field, "wrong"))
        mutate("argv injection", lambda value: _contract(value)["argv"].append("--prove"))
        mutate("environment injection", lambda value: _contract(value)[
            "environment"].__setitem__("LD_PRELOAD", "/tmp/inject.so"))
        mutate("descriptor as bool", lambda value: _binding(
            value, "source_pie").__setitem__("fd", True))

        def reuse_descriptor(value: dict[str, Any]) -> None:
            descriptor = _binding(value, "adapter_executable")["fd"]
            _binding(value, "source_pie").update({
                "fd": descriptor, "proc_path": f"/proc/self/fd/{descriptor}",
            })

        mutate("descriptor reuse", reuse_descriptor)
        mutate("golden inode used as output", lambda value: _binding(
            value, "observed_prover_input").update({key: _binding(
                value, "expected_prover_input")[key] for key in ("device", "inode")}))
        mutate("golden inheritance as int", lambda value: _binding(
            value, "expected_prover_input").__setitem__("child_inherited", 1))
        for size in (True, 103.0):
            mutate(f"binding size {size!r}", lambda value, size=size: _binding(
                value, "source_pie").__setitem__("byte_length", size))
        mutate("stdout injection", lambda value: _contract(value)["stdout"].update({
            "byte_length": 1, "sha256": _digest("stdout")}))
        mutate("preexisting output", lambda value: _lifecycle(value)[
            "pre_launch"].__setitem__("byte_length", 1))
        mutate("hardlinked fresh output", lambda value: _lifecycle(value)[
            "pre_launch"].__setitem__("link_count", 2))
        mutate("output inode substitution", lambda value: _lifecycle(value)[
            "post_launch"].__setitem__("inode", 999))


def test_tool_closure_and_strict_json() -> None:
    with _case() as case:
        mutate = lambda label, callback: _mutate_record(case, label, callback)
        mutate("tool closure digest mismatch", lambda value: value[
            "replay_tool_source_closure"].__setitem__("closure_sha256", _digest("other")))
        mutate("tool source path alias", lambda value: value[
            "replay_tool_source_closure"]["sources"][1].__setitem__(
                "path", value["replay_tool_source_closure"]["sources"][0]["path"]))
        mutate("tool source traversal", lambda value: value[
            "replay_tool_source_closure"]["sources"][0].__setitem__("path", "../escape.py"))

        _reject("duplicate invocation JSON key", lambda: parse_invocation_bytes(
            b'{"schema_version":"a","schema_version":"b"}'))
        _reject("nonfinite invocation JSON",
                lambda: parse_invocation_bytes(b'{"pie_copies":NaN}'))
        _reject("oversized invocation JSON", lambda: parse_invocation_bytes(
            b"{" + b" " * MAX_INVOCATION_BYTES))
        _reject("duplicate execution JSON key", lambda: parse_execution_record_bytes(
            b'{"schema_version":"a","schema_version":"b"}', **case[1]))
        _reject("oversized execution JSON", lambda: parse_execution_record_bytes(
            b"{" + b" " * MAX_EXECUTION_RECORD_BYTES, **case[1]))


def test_schema_documents() -> None:
    invocation_path = LAB_ROOT / "schemas/pie-adapter-invocation.schema.json"
    execution_path = LAB_ROOT / "schemas/pie-adapter-execution-record.schema.json"
    for path in (invocation_path, execution_path):
        value = json.loads(Path(path).read_text())
        require(value["type"] == "object" and value["additionalProperties"] is False,
                f"{path.name} is not fail-closed")
    invocation = json.loads(invocation_path.read_text())
    require(invocation["properties"]["bootloader_program"],
            "invocation schema omits bootloader semantic live-in")
    execution = json.loads(execution_path.read_text())
    required = set(execution["required"])
    require({"adapter_execution_attested", "build_execution_attested", "receipt_binding",
             "adapter_source_inventory", "adapter_source_closure", "adapter_build_receipt",
             "execution_contract", "output_lifecycle", "exact_byte_equal"} <= required,
            "execution schema permits a hash-only success record")
    contract_schema = execution["$defs"]["execution_contract"]
    require(set(EXECUTION_HONESTY) <= set(contract_schema["required"]),
            "execution schema omits an execution honesty field")
    try:
        import jsonschema
    except ImportError as error:
        raise AssertionError("jsonschema unavailable: full offline schema gate is HOLD") from error
    jsonschema.Draft202012Validator.check_schema(execution)
    jsonschema.Draft202012Validator(invocation).validate(_invocation())
    validator = jsonschema.Draft202012Validator(execution)
    with _case() as case:
        validator.validate(case[0])
        for field in EXECUTION_HONESTY:
            hostile = copy.deepcopy(case[0]); _contract(hostile).pop(field)
            require(not validator.is_valid(hostile), f"full schema accepted missing {field}")
            hostile = copy.deepcopy(case[0]); _contract(hostile)[field] = "wrong"
            require(not validator.is_valid(hostile), f"full schema accepted wrong {field}")
        for label, mutate in (
            ("nonempty stdout", lambda value: _contract(value)["stdout"].update(
                {"byte_length": 1, "sha256": _digest("stdout")})),
            ("non-root cwd", lambda value: _contract(value).__setitem__(
                "working_directory", "/sealed")),
            ("relative source trailing slash", lambda value: value[
                "replay_tool_source_closure"]["sources"][0].__setitem__("path", "tools/")),
            ("over-3072-byte path", lambda value: value[
                "source_pie"].__setitem__("path", "/sealed/" + "é" * 1600)),
            ("receipt executable mode", lambda value: value[
                "receipt_binding"].__setitem__("adapter_executable_mode", "755")),
            ("legacy provenance manifest", lambda value: value.__setitem__(
                "provenance_manifest", _artifact("source_pie", "legacy.json"))),
            ("legacy provenance binding", lambda value: value.__setitem__(
                "provenance_binding", {"contract": "legacy"})),
            ("legacy source-closure kind", lambda value: value[
                "adapter_source_closure"].__setitem__(
                    "kind", "adapter-source-closure-json-v1")),
            ("overclaimed production admission", lambda value: value.__setitem__(
                "production_admissible", True)),
            ("overclaimed correctness admission", lambda value: value.__setitem__(
                "correctness_admissible", True)),
            ("overclaimed performance admission", lambda value: value.__setitem__(
                "performance_admissible", True)),
        ):
            hostile = copy.deepcopy(case[0]); mutate(hostile)
            require(not validator.is_valid(hostile), f"full schema accepted {label}")


def pie_adapter_contract_self_test(_: Path | None = None) -> None:
    test_invocation_contract()
    test_execution_record_contract()
    test_raw_receipt_trust()
    test_authenticated_execution_fields()
    test_tool_closure_and_strict_json()
    test_schema_documents()


def main() -> None:
    pie_adapter_contract_self_test()
    print("pie_adapter_contract_tests: PASS")


if __name__ == "__main__":
    main()
