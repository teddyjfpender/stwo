"""Hostile local checks for fail-soft benchmark diagnostics."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from .common import sha256_file
from .results import _validate_correctness, _validate_harness_identity, _validate_timing


def _expect_rejection(call, label: str) -> None:
    try:
        call()
    except ValueError:
        return
    raise ValueError(f"result hostile mutation was accepted: {label}")


def result_self_test() -> None:
    words = 736
    correctness = {
        "eager_passed": True, "graph_passed": True,
        "mutated_eager_passed": True, "mutated_graph_passed": True,
        "eager_checked_words": words, "graph_checked_words": words,
        "mutated_eager_checked_words": words, "mutated_graph_checked_words": words,
        "eager_error": "", "graph_error": "",
        "mutated_eager_error": "", "mutated_graph_error": "",
        "post_eager_passed": False, "post_graph_passed": False,
        "post_eager_checked_words": 0, "post_graph_checked_words": 0,
        "post_eager_error": "not completed after performance failure",
        "post_graph_error": "not completed after performance failure",
    }
    result = {
        "correctness": correctness,
        "timing": {
            "post_benchmark_passed": False,
            "performance_admissible": False,
            "error": "benchmark budget exhausted",
        },
    }
    _validate_correctness(result, "benchmark", words, True)
    _validate_timing(result, "benchmark", True)

    broken_core = copy.deepcopy(result)
    broken_core["correctness"]["mutated_graph_passed"] = False
    broken_core["correctness"]["mutated_graph_error"] = "wrong output"
    _expect_rejection(
        lambda: _validate_correctness(broken_core, "benchmark", words, True),
        "failed mutated correctness",
    )
    fake_metrics = copy.deepcopy(result)
    fake_metrics["timing"]["p50_gpu_ms"] = 0.1
    _expect_rejection(
        lambda: _validate_timing(fake_metrics, "benchmark", True),
        "metrics on failed benchmark",
    )
    _expect_rejection(
        lambda: _validate_timing(result, "benchmark", False),
        "diagnostic accepted as successful timing",
    )

    with tempfile.TemporaryDirectory(prefix="gpu-lab-harness-test-") as directory:
        harness = Path(directory) / "harness"
        harness.write_bytes(b"reviewed harness bytes")
        bound = {"harness_executable_sha256": sha256_file(harness)}
        _validate_harness_identity(bound, harness)
        harness.write_bytes(b"substituted harness bytes")
        _expect_rejection(
            lambda: _validate_harness_identity(bound, harness),
            "substituted harness executable",
        )
