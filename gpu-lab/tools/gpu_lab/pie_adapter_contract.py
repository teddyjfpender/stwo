"""Strict data contracts for authenticated, GPU-free PIE adapter replay."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

from .common import (canonical_bytes, require, require_exact_keys, require_int,
                     require_sha256, sha256_bytes)


INVOCATION_SCHEMA = "stwo.gpu-lab.pie-adapter-invocation.v1"
EXECUTION_RECORD_SCHEMA = "stwo.gpu-lab.pie-adapter-execution-record.v1"
PROTOCOL = "stwo-cairo.gpu-bench.pie-adapt-only.v1"
EVIDENCE_MODE = "authenticated-pie-adapter-replay"
SOURCE_CLOSURE_STATUS = "identity-only"
OUTPUT_ENCODING = "bincode-v1-fixed-int"
EXECUTABLE_POLICY = "release-or-stripped-provenance-compatible-max-256mib-v1"
EMPTY_SHA256 = sha256_bytes(b"")

MIB, GIB = 1 << 20, 1 << 30
MAX_INVOCATION_BYTES = MIB
MAX_EXECUTION_RECORD_BYTES = 4 * MIB
MAX_PATH_CHARS = 2048
MAX_REPLAY_SOURCES = 64
MAX_STREAM_BYTES = 4096
WALL_TIMEOUT_SECONDS = 300
ADDRESS_SPACE_LIMIT_BYTES = 16 * GIB
DATA_LIMIT_BYTES = 12 * GIB
CPU_LIMIT_SECONDS = 240
ARTIFACT_SPECS = {
    "provenance_manifest": ("fri-round6-provenance-json-v1", MIB),
    "source_pie": ("cairo-pie-zip", GIB),
    "bootloader_program": ("cairo-compiled-program-json-v1", 256 * MIB),
    "expected_prover_input": ("stwo-prover-input-bincode-v1", 256 * MIB),
    "adapter_executable": ("adapter-executable", 256 * MIB),
    "adapter_invocation": ("adapter-invocation-json-v1", MIB),
    "adapter_source_closure": ("adapter-source-closure-json-v1", 4 * MIB),
}
BINDING_ROLES = (*ARTIFACT_SPECS, "observed_prover_input")
INVOCATION_KEYS = {
    "schema_version", "protocol", "bootloader_program", "pie_copies",
    "backend", "engine", "output_encoding",
}
RECORD_KEYS = {
    "schema_version", "evidence_mode", "passed", "adapter_execution_attested",
    "source_closure_status", "production_admissible", "correctness_admissible",
    "performance_admissible", "adapter_executable_policy",
    *ARTIFACT_SPECS,
    "replay_tool_source_closure", "invocation_contract", "provenance_binding",
    "execution_contract", "output_lifecycle", "observed_prover_input", "exact_byte_equal",
}


def _parse_json(payload: bytes, label: str, maximum: int) -> dict[str, Any]:
    require(type(payload) is bytes, f"{label} must be bytes")
    require(0 < len(payload) <= maximum, f"{label} byte length is out of bounds")

    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{label} contains non-finite number {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {error}") from error
    require(isinstance(value, dict), f"{label} top level must be an object")
    return value


def _bounded_text(value: Any, label: str, maximum: int) -> str:
    require(isinstance(value, str) and 0 < len(value) <= maximum, f"{label} is not bounded text")
    require(all(0x20 <= ord(char) <= 0x7E for char in value),
            f"{label} contains a control character")
    return value


def _absolute_path(value: Any, label: str) -> str:
    path = _bounded_text(value, label, MAX_PATH_CHARS)
    require(path.isascii() and "\\" not in path, f"{label} must be printable ASCII without backslashes")
    parsed = PurePosixPath(path)
    require(parsed.is_absolute() and path != "/", f"{label} must be absolute")
    require(str(parsed) == path and not path.startswith("//"), f"{label} is not canonical")
    require(all(part not in {"", ".", ".."} for part in parsed.parts[1:]),
            f"{label} contains an ambiguous component")
    return path


def _relative_source_path(value: Any, label: str) -> str:
    path = _bounded_text(value, label, 512)
    require(path.isascii() and "\\" not in path, f"{label} must be printable ASCII without backslashes")
    parsed = PurePosixPath(path)
    require(not parsed.is_absolute() and str(parsed) == path,
            f"{label} must be a canonical relative path")
    require(all(part not in {"", ".", ".."} for part in parsed.parts),
            f"{label} contains an ambiguous component")
    return path


def _artifact(value: Any, role: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"PIE adapter {role} identity must be an object")
    require_exact_keys(value, {"kind", "path", "byte_length", "sha256"},
                       f"PIE adapter {role} identity")
    expected_kind, maximum = ARTIFACT_SPECS[role]
    require(value["kind"] == expected_kind, f"PIE adapter {role} kind differs")
    _absolute_path(value["path"], f"PIE adapter {role} path")
    size = require_int(value["byte_length"], f"PIE adapter {role} byte length", 1)
    diagnostic = (
        "adapter executable exceeds the provenance-compatible 256 MiB policy; "
        "use an exact release/stripped or host-only adapter"
        if role == "adapter_executable" else f"PIE adapter {role} is too large"
    )
    require(size <= maximum, diagnostic)
    require_sha256(value["sha256"], f"PIE adapter {role} sha256")
    return value


def _observed_artifact(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "observed ProverInput identity must be an object")
    require_exact_keys(value, {"kind", "path", "byte_length", "sha256"},
                       "observed ProverInput identity")
    require(value["kind"] == ARTIFACT_SPECS["expected_prover_input"][0],
            "observed ProverInput kind differs")
    _absolute_path(value["path"], "observed ProverInput path")
    size = require_int(value["byte_length"], "observed ProverInput byte length", 1)
    require(size <= ARTIFACT_SPECS["expected_prover_input"][1],
            "observed ProverInput is too large")
    require_sha256(value["sha256"], "observed ProverInput sha256")
    return value


def _stream_identity(value: Any, label: str, maximum: int = MAX_STREAM_BYTES) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} identity must be an object")
    require_exact_keys(value, {"byte_length", "sha256"}, f"{label} identity")
    size = require_int(value["byte_length"], f"{label} byte length")
    require(size <= maximum, f"{label} exceeds its bound")
    require_sha256(value["sha256"], f"{label} sha256")
    return value


def _replay_tool_closure(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "PIE replay-tool source closure must be an object")
    require_exact_keys(value, {"sources", "closure_sha256"},
                       "PIE replay-tool source closure")
    sources = value["sources"]
    require(isinstance(sources, list) and 0 < len(sources) <= MAX_REPLAY_SOURCES,
            "PIE replay-tool source count is out of bounds")
    paths: list[str] = []
    for index, source in enumerate(sources):
        label = f"PIE replay-tool source {index}"
        require(isinstance(source, dict), f"{label} must be an object")
        require_exact_keys(source, {"path", "sha256"}, label)
        paths.append(_relative_source_path(source["path"], f"{label} path"))
        require_sha256(source["sha256"], f"{label} sha256")
    require(len(paths) == len(set(paths)), "PIE replay-tool source paths are not distinct")
    require_sha256(value["closure_sha256"], "PIE replay-tool closure sha256")
    require(value["closure_sha256"] == sha256_bytes(canonical_bytes(sources)),
            "PIE replay-tool closure sha256 does not bind its sources")
    return value


def validate_invocation(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "PIE adapter invocation must be an object")
    require_exact_keys(value, INVOCATION_KEYS, "PIE adapter invocation")
    require(value["schema_version"] == INVOCATION_SCHEMA, "PIE adapter invocation schema differs")
    require(value["protocol"] == PROTOCOL, "PIE adapter invocation protocol differs")
    _artifact(value["bootloader_program"], "bootloader_program")
    require(require_int(value["pie_copies"], "PIE adapter copy count", 1) == 1,
            "PIE adapter copy count differs")
    require(value["backend"] == "simd" and value["engine"] == "legacy",
            "PIE adapter backend or engine differs")
    require(value["output_encoding"] == OUTPUT_ENCODING,
            "PIE adapter output encoding differs")
    return value


def invocation_bytes(value: Any) -> bytes:
    return canonical_bytes(validate_invocation(value)) + b"\n"


def parse_invocation_bytes(payload: bytes) -> dict[str, Any]:
    return validate_invocation(_parse_json(payload, "PIE adapter invocation", MAX_INVOCATION_BYTES))


def _bindings(value: Any, artifacts: dict[str, dict[str, Any]],
              output: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(value, dict), "PIE adapter descriptor bindings must be an object")
    require(set(value) == set(BINDING_ROLES), "PIE adapter descriptor binding roles differ")
    identities = {**artifacts, "observed_prover_input": output}
    descriptors: list[int] = []
    for role in BINDING_ROLES:
        binding = value[role]
        require(isinstance(binding, dict), f"PIE adapter {role} binding must be an object")
        require_exact_keys(binding, {
            "fd", "proc_path", "artifact_path", "byte_length", "sha256", "access",
            "child_inherited", "device", "inode", "link_count",
        }, f"PIE adapter {role} binding")
        descriptor = require_int(binding["fd"], f"PIE adapter {role} descriptor", 3)
        require(descriptor <= 1_048_575, f"PIE adapter {role} descriptor is too large")
        size = require_int(binding["byte_length"], f"PIE adapter {role} binding byte length", 1)
        device = require_int(binding["device"], f"PIE adapter {role} binding device")
        inode = require_int(binding["inode"], f"PIE adapter {role} binding inode", 1)
        links = require_int(binding["link_count"], f"PIE adapter {role} binding link count", 1)
        require(device <= (1 << 64) - 1 and inode <= (1 << 64) - 1 and links == 1,
                f"PIE adapter {role} binding filesystem identity differs")
        expected = identities[role]
        access = "retained-read-write" if role == "observed_prover_input" else "retained-read-only"
        inherited = role in {
            "adapter_executable", "source_pie", "bootloader_program", "observed_prover_input",
        }
        require(binding == {
            "fd": descriptor,
            "proc_path": f"/proc/self/fd/{descriptor}",
            "artifact_path": expected["path"],
            "byte_length": size,
            "sha256": expected["sha256"],
            "access": access,
            "child_inherited": binding["child_inherited"],
            "device": device,
            "inode": inode,
            "link_count": links,
        }, f"PIE adapter {role} descriptor binding differs")
        require(size == expected["byte_length"], f"PIE adapter {role} binding size differs")
        require(binding["child_inherited"] is inherited,
                f"PIE adapter {role} child inheritance differs")
        descriptors.append(descriptor)
    require(len(descriptors) == len(set(descriptors)),
            "PIE adapter descriptor bindings reuse a descriptor")
    file_ids = [(value[role]["device"], value[role]["inode"]) for role in BINDING_ROLES]
    require(len(file_ids) == len(set(file_ids)), "PIE adapter bindings reuse an inode")
    return value


def _expected_argv(bindings: dict[str, Any]) -> list[str]:
    return [
        bindings["adapter_executable"]["proc_path"],
        "--pie", bindings["source_pie"]["proc_path"],
        "--backend", "simd",
        "--engine", "legacy",
        "--adapt-only",
    ]


def _expected_environment(bindings: dict[str, Any]) -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "RAYON_NUM_THREADS": "8",
        "RUST_BACKTRACE": "0",
        "STWO_BOOTLOADER_JSON": bindings["bootloader_program"]["proc_path"],
        "STWO_DUMP_INPUT": bindings["observed_prover_input"]["proc_path"],
    }


def _expected_stderr(output: dict[str, Any], bindings: dict[str, Any]) -> dict[str, Any]:
    payload = (
        "prover input dumped: "
        f"{output['byte_length']} bytes -> {bindings['observed_prover_input']['proc_path']}\n"
    ).encode("utf-8")
    require(len(payload) <= MAX_STREAM_BYTES, "PIE adapter expected stderr exceeds its bound")
    return {"byte_length": len(payload), "sha256": sha256_bytes(payload)}


def _validate_execution_contract(
    value: Any,
    artifacts: dict[str, dict[str, Any]],
    output: dict[str, Any],
) -> dict[str, Any]:
    require(isinstance(value, dict), "PIE adapter execution contract must be an object")
    require_exact_keys(value, {
        "contract", "shell", "stdin", "working_directory", "argv", "environment",
        "bindings", "input_recheck", "resource_limits", "wall_timeout_seconds",
        "exit_status", "stdout", "stderr",
    }, "PIE adapter execution contract")
    require(value["contract"] == "linux-retained-descriptor-exec-v1"
            and value["shell"] is False and value["stdin"] == "devnull",
            "PIE adapter direct-execution contract differs")
    bindings = _bindings(value["bindings"], artifacts, output)
    require(value["working_directory"] == "/",
            "PIE adapter working directory is not the fixed filesystem root")
    require(value["argv"] == _expected_argv(bindings), "PIE adapter argv differs")
    require(value["environment"] == _expected_environment(bindings),
            "PIE adapter environment differs")
    require(value["input_recheck"] == "descriptor-and-path-before-and-after-v1",
            "PIE adapter input recheck contract differs")
    limits = value["resource_limits"]
    require(isinstance(limits, dict), "PIE adapter resource limits must be an object")
    require_exact_keys(limits, {
        "address_space_bytes", "data_bytes", "cpu_seconds", "file_size_bytes",
    }, "PIE adapter resource limits")
    for name in limits:
        require_int(limits[name], f"PIE adapter resource limit {name}", 1)
    require(limits == {
        "address_space_bytes": ADDRESS_SPACE_LIMIT_BYTES,
        "data_bytes": DATA_LIMIT_BYTES,
        "cpu_seconds": CPU_LIMIT_SECONDS,
        "file_size_bytes": artifacts["expected_prover_input"]["byte_length"],
    }, "PIE adapter resource limits differ")
    require(require_int(value["wall_timeout_seconds"], "PIE adapter wall timeout", 1)
            == WALL_TIMEOUT_SECONDS, "PIE adapter wall timeout differs")
    require(require_int(value["exit_status"], "PIE adapter exit status") == 0,
            "PIE adapter exit status differs")
    stdout = _stream_identity(value["stdout"], "PIE adapter stdout")
    stderr = _stream_identity(value["stderr"], "PIE adapter stderr")
    require(stdout == {"byte_length": 0, "sha256": EMPTY_SHA256},
            "PIE adapter stdout is not empty")
    require(stderr == _expected_stderr(output, bindings), "PIE adapter stderr differs")
    return bindings


def validate_adapter_provenance_projection(payload: bytes, expected_sha256: str,
    artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    require(isinstance(artifacts, dict) and set(artifacts) == set(ARTIFACT_SPECS),
            "PIE adapter provenance artifacts differ")
    artifacts = {role: _artifact(artifacts[role], role) for role in ARTIFACT_SPECS}
    require(type(payload) is bytes and 0 < len(payload) <= MIB,
            "PIE adapter raw provenance manifest bytes are out of bounds")
    require_sha256(expected_sha256, "expected PIE adapter provenance manifest sha256")
    identity = artifacts["provenance_manifest"]
    require(len(payload) == identity["byte_length"]
            and sha256_bytes(payload) == expected_sha256 == identity["sha256"],
            "PIE adapter raw manifest identity differs")
    manifest = _parse_json(payload, "PIE adapter raw provenance manifest", MIB)
    require_exact_keys(manifest, {
        "schema_version", "status", "production_admissible", "source_pie",
        "adapted_prover_input", "adapter", "extended_cairo_proof_bincode",
        "canonical_cairo_transport", "verifier_source_closure", "proof_shape",
    }, "PIE adapter raw provenance manifest")
    require(manifest["schema_version"] == "stwo.gpu-lab.fri-round6-provenance.v1"
            and manifest["status"] == "captured-unsealed"
            and manifest["production_admissible"] is False,
            "PIE adapter raw provenance manifest status differs")
    root = PurePosixPath(artifacts["provenance_manifest"]["path"]).parent
    def seal(value: Any, label: str, kind: str, maximum: int,
             artifact: dict[str, Any] | None = None) -> None:
        require(isinstance(value, dict), f"PIE adapter manifest {label} must be an object")
        require_exact_keys(value, {"kind", "path", "byte_length", "sha256"}, f"PIE adapter manifest {label}")
        relative = _relative_source_path(value["path"], f"PIE adapter manifest {label} path")
        size = require_int(value["byte_length"], f"PIE adapter manifest {label} bytes", 1)
        require(value["kind"] == kind and size <= maximum,
                f"PIE adapter manifest {label} kind or size differs")
        require_sha256(value["sha256"], f"PIE adapter manifest {label} sha256")
        if artifact is not None:
            require(str(root / relative) == artifact["path"]
                    and value["kind"] == artifact["kind"]
                    and size == artifact["byte_length"] and value["sha256"] == artifact["sha256"],
                    f"PIE adapter manifest {label} does not bind the executed artifact")
    adapter = manifest["adapter"]
    require(isinstance(adapter, dict), "PIE adapter raw-manifest adapter binding must be an object")
    require_exact_keys(adapter, {"run_record", "invocation", "executable", "source_closure"}, "PIE adapter raw-manifest adapter binding")
    for item, label, kind, maximum, artifact in (
        (manifest["source_pie"], "source PIE", "cairo-pie-zip", GIB, artifacts["source_pie"]),
        (manifest["adapted_prover_input"], "adapted input", "stwo-prover-input-bincode-v1", 256 * MIB, artifacts["expected_prover_input"]),
        (adapter["invocation"], "invocation", "adapter-invocation-json-v1", MIB, artifacts["adapter_invocation"]),
        (adapter["executable"], "executable", "adapter-executable", 256 * MIB, artifacts["adapter_executable"]),
        (adapter["source_closure"], "source closure", "adapter-source-closure-json-v1", 4 * MIB, artifacts["adapter_source_closure"]),
        (adapter["run_record"], "legacy run record", "adapter-run-json-v1", 4 * MIB, None),
        (manifest["extended_cairo_proof_bincode"], "proof", "extended-cairo-proof-bincode-v1", 256 * MIB, None),
        (manifest["canonical_cairo_transport"], "transport", "canonical-cairo-proof-felts-be32-v1", 256 * MIB, None),
        (manifest["verifier_source_closure"], "verifier closure", "verifier-source-closure-json-v1", 4 * MIB, None),
    ):
        seal(item, label, kind, maximum, artifact)
    proof_shape = manifest["proof_shape"]
    require(isinstance(proof_shape, dict), "PIE adapter manifest proof shape must be an object")
    require_exact_keys(proof_shape, {"sha256", "shape"}, "PIE adapter manifest proof shape")
    require_sha256(proof_shape["sha256"], "PIE adapter manifest proof-shape sha256")
    shape = proof_shape["shape"]
    require(isinstance(shape, dict), "PIE adapter manifest proof shape body must be an object")
    require_exact_keys(shape, {
        "schema_version", "pcs", "channel_salt", "preprocessed_trace_variant_sha256",
        "component_slots", "component_enable_bits_sha256", "component_log_sizes",
        "trace_column_log_sizes", "public_data_word_counts", "interaction_claim_felts",
        "commitment_trees", "sampled_value_counts", "decommitment_trees",
        "queried_value_counts", "fri_inner_layers", "fri_witness_counts",
        "fri_last_layer_coefficients", "unsorted_query_locations",
    }, "PIE adapter manifest proof shape body")
    require(shape["schema_version"] == "stwo.gpu-lab.cairo-proof-shape.v1"
            and isinstance(shape["pcs"], dict), "PIE adapter manifest proof shape differs")
    require_exact_keys(shape["pcs"], {"pow_bits", "log_blowup_factor", "log_last_layer_degree_bound",
        "n_queries", "fold_step", "lifting_log_size"}, "PIE adapter manifest proof-shape PCS")
    return manifest


def _output_lifecycle(value: Any, output: dict[str, Any], bindings: dict[str, Any]) -> None:
    require(isinstance(value, dict), "PIE adapter output lifecycle must be an object")
    require_exact_keys(value, {
        "contract", "creation", "failure_cleanup", "finalization", "pre_launch",
        "retained_descriptor", "post_launch", "same_inode_transition",
    }, "PIE adapter output lifecycle")
    require(value["contract"] == "fresh-bound-output-inode-transition-v1"
            and value["creation"] == "o_creat-o_excl-o_nofollow-0600-v1"
            and value["failure_cleanup"] == "unlink-only-if-path-still-bound-inode-v1"
            and value["finalization"] == "fchmod-0400-fsync-before-attestation-v1"
            and value["same_inode_transition"] is True,
            "PIE adapter output lifecycle policy differs")
    binding = bindings["observed_prover_input"]
    pre, descriptor, post = value["pre_launch"], value["retained_descriptor"], value["post_launch"]
    require(isinstance(pre, dict) and isinstance(descriptor, dict) and isinstance(post, dict),
            "PIE adapter output lifecycle states must be objects")
    require_exact_keys(pre, {"path", "file_type", "device", "inode", "link_count", "mode", "byte_length"}, "PIE adapter pre-launch output")
    require_exact_keys(descriptor, {"fd", "device", "inode", "access"}, "PIE adapter retained output descriptor")
    require_exact_keys(post, {"path", "file_type", "device", "inode", "link_count", "mode", "byte_length", "sha256"}, "PIE adapter post-launch output")
    for label, state in (("pre-launch", pre), ("retained", descriptor), ("post-launch", post)):
        require_int(state["device"], f"PIE adapter {label} output device")
        require_int(state["inode"], f"PIE adapter {label} output inode", 1)
    require_int(pre["link_count"], "PIE adapter pre-launch output links", 1)
    require_int(pre["byte_length"], "PIE adapter pre-launch output bytes")
    require_int(descriptor["fd"], "PIE adapter retained output fd", 3)
    require_int(post["link_count"], "PIE adapter post-launch output links", 1)
    require_int(post["byte_length"], "PIE adapter post-launch output bytes", 1)
    require_sha256(post["sha256"], "PIE adapter post-launch output sha256")
    file_id = {"device": binding["device"], "inode": binding["inode"]}
    require(pre == {"path": output["path"], "file_type": "regular", **file_id,
            "link_count": 1, "mode": "0600", "byte_length": 0},
            "PIE adapter pre-launch output was not fresh and empty")
    require(descriptor == {"fd": binding["fd"], **file_id, "access": "retained-read-write"},
            "PIE adapter retained output descriptor differs")
    require(post == {"path": output["path"], "file_type": "regular", **file_id,
            "link_count": 1, "mode": "0400", "byte_length": output["byte_length"],
            "sha256": output["sha256"]}, "PIE adapter final output inode differs")


def validate_execution_record(value: Any, *, provenance_manifest_bytes: bytes,
                              expected_provenance_manifest_sha256: str) -> dict[str, Any]:
    require(isinstance(value, dict), "PIE adapter execution record must be an object")
    require_exact_keys(value, RECORD_KEYS, "PIE adapter execution record")
    require(value["schema_version"] == EXECUTION_RECORD_SCHEMA,
            "PIE adapter execution record schema differs")
    require(value["evidence_mode"] == EVIDENCE_MODE
            and value["passed"] is True
            and value["adapter_execution_attested"] is True,
            "PIE adapter execution is not structurally attested")
    require(value["source_closure_status"] == SOURCE_CLOSURE_STATUS,
            "PIE adapter source-closure status overclaims its evidence")
    require(value["production_admissible"] is False
            and value["correctness_admissible"] is False
            and value["performance_admissible"] is False,
            "PIE adapter execution record overclaims admission")
    require(value["adapter_executable_policy"] == EXECUTABLE_POLICY,
            "PIE adapter executable policy differs")

    artifacts = {role: _artifact(value[role], role) for role in ARTIFACT_SPECS}
    paths = [artifact["path"] for artifact in artifacts.values()]
    require(len(paths) == len(set(paths)), "PIE adapter input artifact paths are not distinct")
    _replay_tool_closure(value["replay_tool_source_closure"])
    invocation = validate_invocation(value["invocation_contract"])
    require(invocation["bootloader_program"] == artifacts["bootloader_program"],
            "PIE adapter invocation does not bind the executed bootloader")
    serialized_invocation = invocation_bytes(invocation)
    require(artifacts["adapter_invocation"]["byte_length"] == len(serialized_invocation)
            and artifacts["adapter_invocation"]["sha256"]
            == sha256_bytes(serialized_invocation),
            "PIE adapter invocation identity does not bind its canonical contract")

    output = _observed_artifact(value["observed_prover_input"])
    require(output["path"] not in set(paths), "observed ProverInput aliases an input artifact")
    expected = artifacts["expected_prover_input"]
    require(output["kind"] == expected["kind"]
            and output["byte_length"] == expected["byte_length"]
            and output["sha256"] == expected["sha256"]
            and value["exact_byte_equal"] is True,
            "observed ProverInput is not attested byte-for-byte equal")
    provenance = value["provenance_binding"]
    require(isinstance(provenance, dict), "PIE adapter provenance binding must be an object")
    require_exact_keys(provenance, {"contract", "trust_root", "output_relation",
        "bootloader_coverage"}, "PIE adapter provenance binding")
    require(provenance == {
        "contract": "fri-round6-provenance-v1-external-raw-binding-v1",
        "trust_root": "required-out-of-band-sha256-v1",
        "output_relation": "manifest-expected-equals-observed-byte-for-byte-v1",
        "bootloader_coverage": "separate-required-semantic-live-in-v1",
    }, "PIE adapter provenance trust-root contract differs")
    validate_adapter_provenance_projection(provenance_manifest_bytes,
                                           expected_provenance_manifest_sha256, artifacts)
    bindings = _validate_execution_contract(value["execution_contract"], artifacts, output)
    _output_lifecycle(value["output_lifecycle"], output, bindings)
    return value


def parse_execution_record_bytes(payload: bytes, *, provenance_manifest_bytes: bytes,
    expected_provenance_manifest_sha256: str) -> dict[str, Any]:
    return validate_execution_record(
        _parse_json(payload, "PIE adapter execution record", MAX_EXECUTION_RECORD_BYTES),
        provenance_manifest_bytes=provenance_manifest_bytes,
        expected_provenance_manifest_sha256=expected_provenance_manifest_sha256,
    )
