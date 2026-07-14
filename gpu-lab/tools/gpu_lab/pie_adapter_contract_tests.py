"""Standalone hostile checks for the PIE adapter data contracts."""

# gpu-lab-cohesion-review: one mutation suite keeps runtime and schema parity auditable.

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

from .common import LAB_ROOT, canonical_bytes, require, sha256_bytes
from .pie_adapter_contract import (
    ADDRESS_SPACE_LIMIT_BYTES,
    ARTIFACT_SPECS,
    CPU_LIMIT_SECONDS,
    DATA_LIMIT_BYTES,
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
    SOURCE_CLOSURE_STATUS,
    WALL_TIMEOUT_SECONDS,
    invocation_bytes,
    parse_execution_record_bytes,
    parse_invocation_bytes,
    validate_execution_record,
    validate_invocation,
    validate_adapter_provenance_projection,
)


def _digest(label: str) -> str:
    return sha256_bytes(label.encode())


def _artifact(role: str, filename: str, size: int = 17) -> dict[str, Any]:
    kind, _ = ARTIFACT_SPECS[role]
    return {"kind": kind, "path": f"/sealed/{filename}",
            "byte_length": size, "sha256": _digest(role)}


def _invocation() -> dict[str, Any]:
    return {
        "schema_version": INVOCATION_SCHEMA, "protocol": PROTOCOL,
        "bootloader_program": _artifact("bootloader_program", "bootloader.json", 31),
        "pie_copies": 1, "backend": "simd", "engine": "legacy",
        "output_encoding": OUTPUT_ENCODING,
    }


def _seal(kind: str, path: str, size: int, label: str) -> dict[str, Any]:
    return {"kind": kind, "path": path, "byte_length": size, "sha256": _digest(label)}


def _manifest_bytes(artifacts: dict[str, dict[str, Any]]) -> bytes:
    def bound(role: str) -> dict[str, Any]:
        value = artifacts[role]
        return {**value, "path": Path(value["path"]).name}

    shape = {
        "schema_version": "stwo.gpu-lab.cairo-proof-shape.v1",
        "pcs": {"pow_bits": 0, "log_blowup_factor": 1, "log_last_layer_degree_bound": 2,
                "n_queries": 3, "fold_step": 1, "lifting_log_size": None},
        "channel_salt": 0, "preprocessed_trace_variant_sha256": _digest("variant"),
        "component_slots": 0, "component_enable_bits_sha256": _digest("enable"),
        "component_log_sizes": [], "trace_column_log_sizes": [],
        "public_data_word_counts": [0, 0, 0], "interaction_claim_felts": 0,
        "commitment_trees": 0, "sampled_value_counts": [], "decommitment_trees": 0,
        "queried_value_counts": [], "fri_inner_layers": 0, "fri_witness_counts": [],
        "fri_last_layer_coefficients": 0, "unsorted_query_locations": 0,
    }
    manifest = {
        "schema_version": "stwo.gpu-lab.fri-round6-provenance.v1",
        "status": "captured-unsealed", "production_admissible": False,
        "source_pie": bound("source_pie"),
        "adapted_prover_input": bound("expected_prover_input"),
        "adapter": {
            "run_record": _seal("adapter-run-json-v1", "legacy-run.json", 17, "run"),
            "invocation": bound("adapter_invocation"),
            "executable": bound("adapter_executable"),
            "source_closure": bound("adapter_source_closure"),
        },
        "extended_cairo_proof_bincode": _seal(
            "extended-cairo-proof-bincode-v1", "proof.bin", 19, "proof"),
        "canonical_cairo_transport": _seal(
            "canonical-cairo-proof-felts-be32-v1", "transport.bin", 23, "transport"),
        "verifier_source_closure": _seal(
            "verifier-source-closure-json-v1", "verifier.json", 29, "verifier"),
        "proof_shape": {"sha256": _digest("shape"), "shape": shape},
    }
    return canonical_bytes(manifest) + b"\n"


def _record() -> dict[str, Any]:
    invocation = _invocation()
    invocation_bytes = canonical_bytes(invocation) + b"\n"
    artifacts = {
        "source_pie": _artifact("source_pie", "source.zip", 103),
        "bootloader_program": copy.deepcopy(invocation["bootloader_program"]),
        "expected_prover_input": _artifact("expected_prover_input", "expected.bin", 107),
        "adapter_executable": _artifact("adapter_executable", "adapter", 109),
        "adapter_invocation": {
            "kind": ARTIFACT_SPECS["adapter_invocation"][0], "path": "/sealed/invocation.json",
            "byte_length": len(invocation_bytes), "sha256": sha256_bytes(invocation_bytes),
        },
        "adapter_source_closure": _artifact("adapter_source_closure", "adapter-sources.json", 113),
    }
    manifest = _manifest_bytes(artifacts)
    artifacts = {"provenance_manifest": {
        "kind": ARTIFACT_SPECS["provenance_manifest"][0], "path": "/sealed/provenance.json",
        "byte_length": len(manifest), "sha256": sha256_bytes(manifest),
    }, **artifacts}
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
        "source_closure_status": SOURCE_CLOSURE_STATUS, "production_admissible": False,
        "correctness_admissible": False, "performance_admissible": False,
        "adapter_executable_policy": EXECUTABLE_POLICY,
        **artifacts,
        "replay_tool_source_closure": {
            "sources": sources,
            "closure_sha256": sha256_bytes(canonical_bytes(sources)),
        },
        "invocation_contract": invocation,
        "provenance_binding": {
            "contract": "fri-round6-provenance-v1-external-raw-binding-v1",
            "trust_root": "required-out-of-band-sha256-v1",
            "output_relation": "manifest-expected-equals-observed-byte-for-byte-v1",
            "bootloader_coverage": "separate-required-semantic-live-in-v1",
        },
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


def _validate(record: dict[str, Any], manifest: bytes, expected: str) -> dict[str, Any]:
    return validate_execution_record(
        record, provenance_manifest_bytes=manifest,
        expected_provenance_manifest_sha256=expected,
    )


def _rebind_manifest(record: dict[str, Any], manifest: bytes) -> str:
    digest = sha256_bytes(manifest)
    for identity in (record["provenance_manifest"], _binding(record, "provenance_manifest")):
        identity.update({"byte_length": len(manifest), "sha256": digest})
    return digest


def _mutate_record(label: str, mutate: Callable[[dict[str, Any]], None]) -> None:
    hostile = _record()
    manifest = _manifest_bytes(hostile)
    expected = sha256_bytes(manifest)
    mutate(hostile)
    _reject(label, lambda: _validate(hostile, manifest, expected))


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
    record = _record()
    manifest = _manifest_bytes(record)
    expected = sha256_bytes(manifest)
    require(_validate(record, manifest, expected) is record, "valid execution record was not retained")
    require(parse_execution_record_bytes(
        canonical_bytes(record), provenance_manifest_bytes=manifest,
        expected_provenance_manifest_sha256=expected) == record,
            "execution record byte parser differs")

    _reject("run-record hash alone", lambda: validate_execution_record({
        "schema_version": EXECUTION_RECORD_SCHEMA, "sha256": _digest("run")
    }, provenance_manifest_bytes=manifest, expected_provenance_manifest_sha256=expected))
    for key in (
        "adapter_execution_attested", "execution_contract", "invocation_contract",
        "provenance_binding", "output_lifecycle", "bootloader_program",
        "observed_prover_input", "exact_byte_equal",
    ):
        _mutate_record(f"missing authenticated field {key}", lambda value, key=key: value.pop(key))
    _mutate_record("unknown record field", lambda value: value.__setitem__("extra", False))
    for key in ("production_admissible", "correctness_admissible", "performance_admissible"):
        _mutate_record(f"overclaimed {key}", lambda value, key=key: value.__setitem__(key, True))
    _mutate_record("overclaimed source closure", lambda value: value.__setitem__(
        "source_closure_status", "verified-build"
    ))
    _mutate_record("wrong executable policy", lambda value: value.__setitem__(
        "adapter_executable_policy", "debug-allowed"))
    _mutate_record("oversized debug executable", lambda value: value[
        "adapter_executable"].__setitem__("byte_length", 268435457))
    _mutate_record("false execution attestation", lambda value: value.__setitem__(
        "adapter_execution_attested", False))
    _mutate_record("false exact equality", lambda value: value.__setitem__("exact_byte_equal", False))
    _mutate_record("bootloader invocation mismatch", lambda value: value[
        "invocation_contract"]["bootloader_program"].__setitem__("sha256", _digest("other")))
    _mutate_record("invocation artifact mismatch", lambda value: value[
        "adapter_invocation"].__setitem__("sha256", _digest("other")))
    _mutate_record("observed hash mismatch", lambda value: value[
        "observed_prover_input"].__setitem__("sha256", _digest("other")))
    _mutate_record("observed input alias", lambda value: value[
        "observed_prover_input"].__setitem__("path", value["expected_prover_input"]["path"]))
    _mutate_record("input artifact alias", lambda value: value[
        "adapter_source_closure"].__setitem__("path", value["adapter_invocation"]["path"]))
    _mutate_record("bootloader smuggled into manifest", lambda value: value[
        "provenance_binding"].__setitem__("bootloader_coverage", "manifest-covered"))


def test_raw_manifest_trust() -> None:
    record = _record(); manifest = _manifest_bytes(record); expected = sha256_bytes(manifest)
    artifacts = {role: record[role] for role in ARTIFACT_SPECS}
    require(validate_adapter_provenance_projection(manifest, expected, artifacts)["status"]
            == "captured-unsealed", "public manifest validator differs")
    document = json.loads(manifest)
    pretty = (json.dumps(dict(reversed(list(document.items()))), indent=2) + "\n").encode()
    record = _record(); pretty_digest = _rebind_manifest(record, pretty)
    require(_validate(record, pretty, pretty_digest) is record,
            "exact pretty/key-reordered manifest bytes were not accepted")
    canonical = canonical_bytes(document) + b"\n"
    _reject("canonical reserialization against original raw pin",
            lambda: _validate(record, canonical, pretty_digest))
    unrelated = manifest.replace(b"proof.bin", b"other.bin")
    _rebind_manifest(record, unrelated)
    _reject("consistent unrelated manifest substitution",
            lambda: _validate(record, unrelated, expected))
    for label, hostile in (
        ("duplicate manifest key", b'{"schema_version":"a","schema_version":"b"}'),
        ("unknown manifest key", canonical_bytes({**json.loads(manifest), "extra": False})),
    ):
        record = _record(); digest = _rebind_manifest(record, hostile)
        _reject(label, lambda record=record, hostile=hostile, digest=digest:
                _validate(record, hostile, digest))
    record = _record(); document = json.loads(manifest)
    document["source_pie"]["sha256"] = _digest("substituted")
    hostile = canonical_bytes(document); digest = _rebind_manifest(record, hostile)
    _reject("manifest-to-executed-PIE substitution", lambda: _validate(record, hostile, digest))
    for label, key, bad in (("seal bool size", "byte_length", True),
                            ("seal float size", "byte_length", 103.0),
                            ("seal backslash path", "path", "bad\\path"),
                            ("seal trailing path", "path", "bad/")):
        document = json.loads(manifest); document["source_pie"][key] = bad
        hostile = canonical_bytes(document); record = _record(); digest = _rebind_manifest(record, hostile)
        _reject(label, lambda record=record, hostile=hostile, digest=digest:
                _validate(record, hostile, digest))
    for label, hostile in (("non-UTF8 manifest", b"\xff"),
                           ("oversized manifest", b"{" + b" " * (1 << 20))):
        record = _record(); digest = _rebind_manifest(record, hostile)
        _reject(label, lambda record=record, hostile=hostile, digest=digest:
                _validate(record, hostile, digest))


def test_authenticated_execution_fields() -> None:
    _mutate_record("shell execution", lambda value: _contract(value).__setitem__("shell", True))
    _mutate_record("non-root cwd", lambda value: _contract(value).__setitem__("working_directory", "/sealed"))
    for field in EXECUTION_HONESTY:
        _mutate_record(f"missing honesty field {field}", lambda value, field=field:
                       _contract(value).pop(field))
        _mutate_record(f"wrong honesty field {field}", lambda value, field=field:
                       _contract(value).__setitem__(field, "wrong"))
    _mutate_record("argv injection", lambda value: _contract(value)["argv"].append("--prove"))
    _mutate_record("environment injection", lambda value: _contract(value)[
        "environment"].__setitem__("LD_PRELOAD", "/tmp/inject.so"))
    _mutate_record("descriptor as bool", lambda value: _binding(
        value, "source_pie").__setitem__("fd", True))
    def reuse_descriptor(value: dict[str, Any]) -> None:
        descriptor = _binding(value, "adapter_executable")["fd"]
        _binding(value, "source_pie").update({
            "fd": descriptor, "proc_path": f"/proc/self/fd/{descriptor}",
        })

    _mutate_record("descriptor reuse", reuse_descriptor)
    _mutate_record("golden inode used as output", lambda value: _binding(
        value, "observed_prover_input").update({key: _binding(
            value, "expected_prover_input")[key] for key in ("device", "inode")}))
    _mutate_record("golden inheritance as int", lambda value: _binding(
        value, "expected_prover_input").__setitem__("child_inherited", 1))
    for size in (True, 103.0):
        _mutate_record(f"binding size {size!r}", lambda value, size=size: _binding(
            value, "source_pie").__setitem__("byte_length", size))
    _mutate_record("stdout injection", lambda value: _contract(value)["stdout"].update({
        "byte_length": 1, "sha256": _digest("stdout")
    }))
    _mutate_record("preexisting output", lambda value: _lifecycle(value)[
        "pre_launch"].__setitem__("byte_length", 1))
    _mutate_record("hardlinked fresh output", lambda value: _lifecycle(value)[
        "pre_launch"].__setitem__("link_count", 2))
    _mutate_record("output inode substitution", lambda value: _lifecycle(value)[
        "post_launch"].__setitem__("inode", 999))


def test_tool_closure_and_strict_json() -> None:
    record = _record(); manifest = _manifest_bytes(record); expected = sha256_bytes(manifest)
    _mutate_record("tool closure digest mismatch", lambda value: value[
        "replay_tool_source_closure"].__setitem__("closure_sha256", _digest("other")))
    _mutate_record("tool source path alias", lambda value: value[
        "replay_tool_source_closure"]["sources"][1].__setitem__(
            "path", value["replay_tool_source_closure"]["sources"][0]["path"]
        ))
    _mutate_record("tool source traversal", lambda value: value[
        "replay_tool_source_closure"]["sources"][0].__setitem__("path", "../escape.py"))

    _reject("duplicate invocation JSON key", lambda: parse_invocation_bytes(
        b'{"schema_version":"a","schema_version":"b"}'
    ))
    _reject("nonfinite invocation JSON", lambda: parse_invocation_bytes(b'{"pie_copies":NaN}'))
    _reject("oversized invocation JSON", lambda: parse_invocation_bytes(
        b"{" + b" " * MAX_INVOCATION_BYTES
    ))
    _reject("duplicate execution JSON key", lambda: parse_execution_record_bytes(
        b'{"schema_version":"a","schema_version":"b"}',
        provenance_manifest_bytes=manifest, expected_provenance_manifest_sha256=expected,
    ))
    _reject("oversized execution JSON", lambda: parse_execution_record_bytes(
        b"{" + b" " * MAX_EXECUTION_RECORD_BYTES,
        provenance_manifest_bytes=manifest, expected_provenance_manifest_sha256=expected,
    ))


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
    require({"adapter_execution_attested", "provenance_binding", "execution_contract",
             "output_lifecycle", "exact_byte_equal"} <= required,
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
    validator.validate(_record())
    for field in EXECUTION_HONESTY:
        hostile = _record(); _contract(hostile).pop(field)
        require(not validator.is_valid(hostile), f"full schema accepted missing {field}")
        hostile = _record(); _contract(hostile)[field] = "wrong"
        require(not validator.is_valid(hostile), f"full schema accepted wrong {field}")
    for label, mutate in (
        ("nonempty stdout", lambda value: _contract(value)["stdout"].update(
            {"byte_length": 1, "sha256": _digest("stdout")})),
        ("non-root cwd", lambda value: _contract(value).__setitem__("working_directory", "/sealed")),
        ("relative source trailing slash", lambda value: value[
            "replay_tool_source_closure"]["sources"][0].__setitem__("path", "tools/")),
        ("over-3072-byte path", lambda value: value[
            "source_pie"].__setitem__("path", "/sealed/" + "é" * 1600)),
    ):
        hostile = _record(); mutate(hostile)
        require(not validator.is_valid(hostile), f"full schema accepted {label}")


def main() -> None:
    test_invocation_contract()
    test_execution_record_contract()
    test_raw_manifest_trust()
    test_authenticated_execution_fields()
    test_tool_closure_and_strict_json()
    test_schema_documents()
    print("pie_adapter_contract_tests: PASS")


if __name__ == "__main__":
    main()
