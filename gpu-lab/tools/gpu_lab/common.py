"""Shared constants and strict, dependency-free validation primitives."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_FIXTURE = "stwo.gpu-lab.semantic-fixture.v1"
SCHEMA_EXECUTION = "stwo.gpu-lab.execution-manifest.v1"
KERNEL_ENTRY_SCHEMA = "stwo.gpu-lab.kernel-entry.v1"
REPLAY_MAGIC = b"STWOLAB1"
REPLAY_VERSION = 2
PLAN_MAGIC = b"STWOPLN1"
PLAN_VERSION = 2
PLAN_BINDING_VERSION = 1
M31_P = 2_147_483_647
FIRST_SLICE_ABI_SHA256 = "0c0302965abd6c325519b73823fe9351c174c89bbf9cb809be02214833581dce"
FIRST_SLICE_ORACLE_INDEX_SHA256 = (
    "8f85f4a879a246b8efc88dd20615d998431456eed85835ac7a030875c5d03050"
)
# Fail-closed integration hook. Set only to the SHA256 of the reviewed wrapper
# after the production exporter source closure and its v2 artifacts are frozen.
FIRST_SLICE_INDEXED_ORACLE_WRAPPER_SHA256 = (
    "209b3022c874125699d81c6e3df3c9bfa5ffc5507a57302dd9c05fa8c9ceee0b"
)
FIRST_SLICE_ORACLE_EXPORTER_PATHS = [
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/Cargo.lock",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/Cargo.toml",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/rust-toolchain.toml",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/artifact_io.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/legacy.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/main.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/model.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/oracle.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/streaming.rs",
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/streaming_tests.rs",
]
FIRST_SLICE_GENERATORS = [
    {"repository": "stwo", "path": "crates/backend-cuda/src/backend/jit_witness/codegen.rs"},
    {"repository": "stwo", "path": "crates/backend-cuda/src/backend/jit_witness/isa.rs"},
    {"repository": "stwo", "path": "crates/backend-cuda/src/backend/jit_witness/recording.rs"},
    {"repository": "stwo", "path": "crates/backend-cuda/src/backend/jit_witness/programs.rs"},
    {
        "repository": "stwo-cairo",
        "path": "stwo_cairo_prover/crates/gpu-prover/src/bin/kernel_emit.rs",
    },
    {
        "repository": "stwo-cairo",
        "path": "stwo_cairo_prover/crates/prover/src/witness/jit_prove_backend.rs",
    },
    {
        "repository": "stwo-cairo",
        "path": "stwo_cairo_prover/crates/prover/src/witness/components/pedersen_builtin.rs",
    },
    {
        "repository": "stwo-cairo",
        "path": "stwo_cairo_prover/crates/prover/src/witness/witness_eval/mod.rs",
    },
    {
        "repository": "stwo-cairo",
        "path": "stwo_cairo_prover/crates/prover/src/witness/witness_eval/recording.rs",
    },
    {
        "repository": "stwo",
        "path": "crates/backend-cuda-kernels/cuda/generated/aot_manifest.json",
    },
]
HOST_REFERENCE_RELATIVE = (
    "stwo-cairo/stwo_cairo_prover/crates/prover/src/witness/components/pedersen_builtin.rs"
)

PACKAGE_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = PACKAGE_ROOT.parent
LAB_ROOT = TOOLS_ROOT.parent
REPO_ROOT = LAB_ROOT.parent
WORKSPACE_ROOT = REPO_ROOT.parent


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lab_tool_source_paths() -> list[Path]:
    """Return the complete handwritten Python closure bound into every cubin recipe."""
    paths = [TOOLS_ROOT / "lab.py", TOOLS_ROOT / "stage-run",
             *sorted(PACKAGE_ROOT.glob("*.py"))]
    require(all(path.is_file() for path in paths), "lab tool source closure is incomplete")
    return paths


def lab_tool_sha256() -> str:
    identities = [
        {"path": str(path.relative_to(LAB_ROOT)), "sha256": sha256_file(path)}
        for path in lab_tool_source_paths()
    ]
    return sha256_bytes(canonical_bytes(identities))


def load_json(path: Path) -> dict[str, Any]:
    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top level must be an object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_exact_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    require(
        actual == keys,
        f"{label} keys differ: missing={sorted(keys - actual)} extra={sorted(actual - keys)}",
    )


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64, f"{label} must be 64 lowercase hex digits")
    require(
        all(character in "0123456789abcdef" for character in value),
        f"{label} must be 64 lowercase hex digits",
    )
    return value


def require_words(values: Any, count: int, label: str) -> list[int]:
    require(isinstance(values, list) and len(values) == count, f"{label} length mismatch")
    require(
        all(
            isinstance(value, int) and not isinstance(value, bool) and 0 <= value < M31_P
            for value in values
        ),
        f"{label} contains a non-canonical M31 word",
    )
    return values


def require_int(value: Any, label: str, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return value


def require_number(
    value: Any, label: str, minimum: float = 0.0, exclusive: bool = False
) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        f"{label} must be a finite number",
    )
    require(
        value > minimum if exclusive else value >= minimum,
        f"{label} is below its accepted minimum",
    )
    return float(value)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
