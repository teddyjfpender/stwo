"""Small local mutation suite for the lab's trust boundaries."""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

from .artifacts import reverse_rows, write_plan, write_replay
from .baseline_tests import baseline_self_test
from .common import (
    PLAN_MAGIC,
    PLAN_BINDING_VERSION,
    PLAN_VERSION,
    FIRST_SLICE_ABI_SHA256,
    FIRST_SLICE_GENERATORS,
    REPLAY_MAGIC,
    REPLAY_VERSION,
    WORKSPACE_ROOT,
    lab_tool_sha256,
    load_json,
    require,
    sha256_file,
)
from .identity import validate_abi, validate_generator_sources
from .indexed_fixture_tests import indexed_fixture_self_test
from .fri_module_tests import fri_module_self_test
from .fri_round6_execution_tests import fri_round6_execution_self_test
from .fri_round6_loop_tests import fri_round6_loop_self_test
from .loop import loop_self_test
from .oracle import validate_oracle_index
from .pie_adapter_cleanup_tests import pie_adapter_cleanup_self_test
from .pie_adapter_contract_tests import pie_adapter_contract_self_test
from .pie_adapter_execution_tests import pie_adapter_execution_self_test
from .pie_adapter_receipt_tests import pie_adapter_receipt_self_test
from .result_tests import result_self_test
from .results import _validate_correctness, _validate_timing
from .semantics import make_tiny_fixture, validate_fixture


def _has_cohesion_review(path: Path) -> bool:
    marker = "gpu-lab-cohesion-review:"
    for candidate in (path, path.with_name(path.name + ".cohesion.md")):
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            if marker in line and line.split(marker, 1)[1].strip():
                return True
    return False


def _expect_fixture_rejection(path: Path, fixture: dict, label: str, mutate) -> None:
    mutated = json.loads(json.dumps(fixture))
    mutate(mutated)
    path.write_text(json.dumps(mutated))
    try:
        validate_fixture(path)
    except ValueError:
        return
    raise ValueError(f"{label} mutation was accepted")


def _source_shape_test(root: Path) -> tuple[int, Path, int]:
    companion = WORKSPACE_ROOT / "stwo-cairo/gpu_benchmarks/lab"
    suffixes = {
        ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp", ".hxx",
        ".inc", ".inl", ".ipp", ".py", ".pyi", ".rs", ".sh",
    }
    exact_names = {
        "CMakeLists.txt", "Cargo.toml", "rust-toolchain.toml", "Dockerfile.cuda11.8",
        "quick-loop", "sanitize-slab", "profile-slab", "stage-run", "accept-baseline",
        "check-loop-slo", "cpu-self-test", "labctl",
    }
    sources = []
    for code_root in (root, companion):
        for path in code_root.rglob("*"):
            if (not path.is_file() or "__pycache__" in path.parts or "target" in path.parts
                    or "cases" in path.parts or "schemas" in path.parts):
                continue
            if path.suffix in suffixes or path.name in exact_names:
                sources.append(path)
    require(sources, "gpu-lab source-shape gate found no handwritten code")
    counts = [(len(path.read_text().splitlines()), path) for path in sources]
    oversized = [(lines, path) for lines, path in counts if lines > 750]
    require(
        not oversized,
        "gpu-lab handwritten file exceeds 750 lines: "
        + ", ".join(f"{path}:{lines}" for lines, path in oversized),
    )
    unreviewed = [
        (lines, path) for lines, path in counts
        if 500 < lines <= 750 and not _has_cohesion_review(path)
    ]
    require(
        not unreviewed,
        "gpu-lab handwritten file over 500 lines lacks an explicit cohesion review: "
        + ", ".join(f"{path}:{lines}" for lines, path in unreviewed),
    )
    maximum, path = max(counts)
    return len(sources), path, maximum


def _fail_soft_result_test() -> None:
    correctness: dict[str, object] = {}
    for prefix in ("eager", "graph", "mutated_eager", "mutated_graph"):
        correctness.update({
            f"{prefix}_passed": True,
            f"{prefix}_checked_words": 23,
            f"{prefix}_error": "",
        })
    for prefix in ("post_eager", "post_graph"):
        correctness.update({
            f"{prefix}_passed": False,
            f"{prefix}_checked_words": 0,
            f"{prefix}_error": "not completed after performance failure",
        })
    failed_timing = {
        "post_benchmark_passed": False,
        "performance_admissible": False,
        "error": "shared benchmark budget cannot fit minimum samples",
    }
    _validate_correctness(
        {"correctness": correctness}, "benchmark", 23, performance_failed=True,
    )
    _validate_timing({"timing": failed_timing}, "benchmark", performance_failed=True)
    try:
        _validate_timing({"timing": failed_timing}, "benchmark")
    except ValueError:
        pass
    else:
        raise ValueError("non-admissible benchmark timing was accepted as performance evidence")


def self_test(root: Path) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        cohesion_candidate = Path(temporary) / "large.py"
        cohesion_candidate.write_text("pass\n" * 501)
        require(not _has_cohesion_review(cohesion_candidate),
                "missing cohesion-review marker was accepted")
        cohesion_candidate.with_name("large.py.cohesion.md").write_text(
            "gpu-lab-cohesion-review: one invariant is clearer in one module\n"
        )
        require(_has_cohesion_review(cohesion_candidate),
                "explicit cohesion-review marker was rejected")
        fixture_path = Path(temporary) / "tiny.json"
        first = make_tiny_fixture(fixture_path)
        validate_fixture(fixture_path)
        first_bytes = fixture_path.read_bytes()
        second = make_tiny_fixture(fixture_path)
        require(first == second and first_bytes == fixture_path.read_bytes(),
                "fixture generation is not deterministic")
        indexed_fixture_self_test(Path(temporary) / "indexed", second)
        output = second["semantic_payload"]["expected"]["output_columns"]
        inputs = second["semantic_payload"]["inputs"]
        primary_inputs = [inputs["segment_start"], inputs["enabler"], inputs["iota"]]
        mutated_inputs = reverse_rows(primary_inputs)
        require(mutated_inputs != primary_inputs and
                all(mutated[0] == primary[-1]
                    for primary, mutated in zip(primary_inputs, mutated_inputs)),
                "proof-varying reverse-row mutation is missing or malformed")
        table = second["semantic_payload"]["tables"]["address_to_id"]
        require([column[-2] for column in output] == table[-3:],
                "canonical high table edge semantics changed")
        require([column[-1] for column in output] == table[1:4],
                "canonical low table edge semantics changed")
        replay = Path(temporary) / "case.replay"
        write_replay(second, replay)
        magic, version, rows = struct.unpack("<8sII", replay.read_bytes()[:16])
        require((magic, version, rows) == (REPLAY_MAGIC, REPLAY_VERSION, 32),
                "replay header round trip failed")
        plan = Path(temporary) / "case.plan"
        write_plan({
            "target": {"sm": 86},
            "semantic_fixture_sha256": "22" * 32,
            "program_image": {
                "path": "/tmp/module.cubin",
                "module_content_sha256": "11" * 32,
                "build_recipe_hash": "66" * 32,
            },
            "kernel": {"abi_sha256": FIRST_SLICE_ABI_SHA256,
                       "symbol": "kernel", "launch": {
                "block": [256, 1, 1], "dynamic_shared_bytes": 0,
            }},
            "host_oracle": {
                "index_sha256": "33" * 32,
                "artifact_sha256": "44" * 32,
                "exporter_closure_sha256": "55" * 32,
            },
            "physical_layout": {
                "input_columns": 3, "table_pointer_slots": 37, "table_stride_words": 3,
                "output_columns": 3, "lookup_words_per_row": 14, "sub_words_per_row": 6,
            },
            "artifacts": {"replay_image_sha256": sha256_file(replay)},
        }, plan)
        plan_magic, plan_version, binding_version, plan_sm = struct.unpack(
            "<8sIII", plan.read_bytes()[:20]
        )
        require((plan_magic, plan_version, binding_version, plan_sm) ==
                (PLAN_MAGIC, PLAN_VERSION, PLAN_BINDING_VERSION, 86),
                "launch-plan header round trip failed")

        corrupted = json.loads(json.dumps(second))
        corrupted["semantic_payload"]["expected"]["output_columns"][0][0] ^= 1
        fixture_path.write_text(json.dumps(corrupted))
        try:
            validate_fixture(fixture_path)
        except ValueError as error:
            require("oracle" in str(error), "corrupt golden failed for the wrong reason")
        else:
            raise ValueError("corrupt golden was accepted")
        _expect_fixture_rejection(
            fixture_path, second, "unknown field",
            lambda value: value.update({"candidate_hash": "0" * 64}),
        )
        _expect_fixture_rejection(
            fixture_path, second, "bad boundary",
            lambda value: value["boundary"].update({"entry": 7}),
        )
        _expect_fixture_rejection(
            fixture_path, second, "bad enabler",
            lambda value: value["semantic_payload"]["inputs"]["enabler"].__setitem__(0, 2),
        )
        for name, content in (("duplicate.json", '{"x":1,"x":2}'),
                              ("nonfinite.json", '{"x":NaN}')):
            hostile = Path(temporary) / name
            hostile.write_text(content)
            try:
                load_json(hostile)
            except ValueError:
                pass
            else:
                raise ValueError(f"hostile JSON was accepted: {name}")

    checked = root / "cases/tiny/witness_pedersen_builtin.semantic.json"
    if checked.exists():
        checked_fixture = validate_fixture(checked)
        oracle_index = root / "cases/tiny/witness_pedersen_builtin.oracle-index.json"
        validate_oracle_index(oracle_index, checked, checked_fixture, WORKSPACE_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            mutated_index = Path(temporary) / "oracle-index.json"
            mutated = load_json(oracle_index)
            mutated["oracle_artifact"]["sha256"] = "0" * 64
            mutated_index.write_text(json.dumps(mutated, indent=2) + "\n")
            try:
                validate_oracle_index(mutated_index, checked, checked_fixture, WORKSPACE_ROOT)
            except ValueError:
                pass
            else:
                raise ValueError("mutated host-oracle index was accepted")
    abi_path = root / "manifests/witness_pedersen_builtin.abi.json"
    validate_abi(load_json(abi_path), abi_path, root.parent)
    generator_sources = [
        {
            **identity,
            "sha256": sha256_file(WORKSPACE_ROOT / identity["repository"] / identity["path"]),
        }
        for identity in FIRST_SLICE_GENERATORS
    ]
    validate_generator_sources(generator_sources, WORKSPACE_ROOT)
    generator_sources[0]["sha256"] = "0" * 64
    try:
        validate_generator_sources(generator_sources, WORKSPACE_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("mutated generator closure was accepted")
    fri_module_self_test(root)
    fri_round6_loop_self_test(root)
    fri_round6_execution_self_test(root)
    pie_adapter_contract_self_test(root)
    pie_adapter_execution_self_test(root)
    pie_adapter_cleanup_self_test(root)
    pie_adapter_receipt_self_test(root)
    for schema in ("semantic-fixture.schema.json", "semantic-fixture-index.schema.json",
                   "execution-manifest.schema.json", "host-oracle-index.schema.json",
                   "host-oracle-artifact-v2.schema.json",
                   "result.schema.json", "kernel-entry.schema.json", "build-recipe.schema.json",
                   "module-index.schema.json", "environment.schema.json",
                   "loop-result.schema.json", "baseline-envelope.schema.json",
                   "baseline-comparison.schema.json", "pie-adapter-invocation.schema.json",
                   "pie-adapter-execution-record.schema.json",
                   "pie-adapter-source-inventory-v2.schema.json",
                   "pie-adapter-source-closure-v2.schema.json",
                   "pie-adapter-build-receipt-v2.schema.json"):
        require(load_json(root / "schemas" / schema).get("$schema") is not None,
                f"invalid schema document: {schema}")
    require(len(lab_tool_sha256()) == 64, "lab tool closure hash is invalid")
    _fail_soft_result_test()
    loop_self_test()
    baseline_self_test()
    result_self_test()
    source_count, largest_source, largest_lines = _source_shape_test(root)
    print(
        f"gpu-lab source-shape gate: {source_count} files, "
        f"max={largest_source.relative_to(WORKSPACE_ROOT)}:{largest_lines}"
    )
    print("gpu-lab local self-test: PASS")
