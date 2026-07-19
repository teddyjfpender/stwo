//! Machine-readable result and promotion envelope for the retained SN3 A/B.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use blake3::{Hash, Hasher};
use serde_json::json;

use super::sn3_quotient_numerator_bench::{artifact_identity, percentile};
use super::sn3_quotient_retained_fixture::staged_lde_kernel_nodes;
use super::sn3_quotient_retained_run_sum_ab::{
    CanonicalFriInput, RetainedMutation, EXPECTED_DIRECT_NODES, EXPECTED_RUN_SUM_NODES, SAMPLES,
    SCHEMA, WARMUPS,
};
use super::*;

const SEALED_A40_RETAINED_P50_MS: f64 = 335.177_948;
const SEALED_A40_RETAINED_P95_MS: f64 = 336.626_038;
const THREE_X_RETAINED_MAX_MS: f64 = 144.268;
const FIVE_X_RETAINED_MAX_MS: f64 = 106.086;
const ARCHIVE_LTO_REQUESTED: bool = matches!(option_env!("STWO_CUDA_ARCHIVE_LTO"), Some("1"));
const STATIC_SASS_SHA256_ENV: &str = "STWO_SN3_RETAINED_RUN_SUM_STATIC_SASS_SHA256";
const STATIC_SASS_GATE_ENV: &str = "STWO_SN3_RETAINED_RUN_SUM_STATIC_SASS_GATE";
const EXACT_NUMERATOR_AND_AUXILIARY_BYTES: u64 = 402_645_136;
const REQUIRED_EXACT_DIGESTS: usize = 13;

#[allow(clippy::too_many_arguments)]
pub(crate) fn publish_result(
    sn3: &sn3_quotient_topology_fixture::LoadedTopologyFixture,
    shape: &sn3_quotient_retained_fixture::RetainedShape,
    fixture: &BenchmarkArena,
    receipt: &stwo_backend_cuda::QuotientNumeratorRunSumReceipt,
    mutation: RetainedMutation,
    canonical: &sn3_quotient_numerator_bench::CanonicalOutput,
    canonical_fri: &CanonicalFriInput,
    digests: &[(&str, (Hash, Hash))],
    direct_ms: Vec<f64>,
    candidate_ms: Vec<f64>,
    numerator_recipe: Hash,
    boundary_recipe: Hash,
) {
    let direct_p50 = percentile(&direct_ms, 50);
    let direct_p95 = percentile(&direct_ms, 95);
    let candidate_p50 = percentile(&candidate_ms, 50);
    let candidate_p95 = percentile(&candidate_ms, 95);
    let artifact = artifact_identity();
    let measurement_source = retained_measurement_source_digest();
    let artifact_eligible =
        ARCHIVE_LTO_REQUESTED && artifact.cuda_build_mode == "cuda" && artifact.is_complete();
    let exact_sample_count = direct_ms.len() == SAMPLES && candidate_ms.len() == SAMPLES;
    let paired_candidate_wins = direct_ms
        .iter()
        .zip(&candidate_ms)
        .filter(|(direct, candidate)| candidate < direct)
        .count();
    let all_pairs_won = exact_sample_count && paired_candidate_wins == SAMPLES;
    let p95_non_regression = candidate_p95 <= direct_p95;
    let canonical_pair = (*canonical.digest(), canonical_fri.digest);
    let source_direct = required_digest(digests, "source_mutation_direct");
    let source_candidate = required_digest(digests, "source_mutation_candidate");
    let coefficient_direct = required_digest(digests, "coefficient_mutation_direct");
    let coefficient_candidate = required_digest(digests, "coefficient_mutation_candidate");
    let canonical_labels = [
        "eager_candidate",
        "captured_direct",
        "captured_candidate",
        "source_restored_candidate",
        "source_restored_direct",
        "coefficient_restored_candidate",
        "coefficient_restored_direct",
        "post_timing_direct",
        "post_timing_candidate",
    ];
    let exact_checks_passed = digests.len() == REQUIRED_EXACT_DIGESTS
        && canonical.len_bytes() == EXACT_NUMERATOR_AND_AUXILIARY_BYTES
        && canonical_fri.len_bytes() == FRI_INPUT_OUTPUT_BYTES
        && canonical_labels
            .iter()
            .all(|label| required_digest(digests, label) == &canonical_pair)
        && source_direct == source_candidate
        && source_direct.0 != canonical_pair.0
        && source_direct.1 != canonical_pair.1
        && coefficient_direct == coefficient_candidate
        && coefficient_direct.0 != canonical_pair.0
        && coefficient_direct.1 != canonical_pair.1;
    assert!(exact_checks_passed, "formal exact-check gate failed");
    let static_sass_sha256 = static_sass_sha256();
    let static_sass_gate_declaration = std::env::var(STATIC_SASS_GATE_ENV).ok();
    let static_sass_gate_passed =
        static_sass_sha256.is_some() && static_sass_gate_declaration.as_deref() == Some("pass");
    let formal_promotion_eligible = artifact_eligible
        && all_pairs_won
        && p95_non_regression
        && exact_checks_passed
        && static_sass_gate_passed;
    let result = json!({
        "schema": SCHEMA,
        "passed": true,
        "result_class": "observed retained-production-shape same-prepared-object A/B; not an end-to-end proof MHz claim",
        "timing_scope": "retained FixedImage numerator plus unchanged ordinary quotient-to-FRI-input, CUDA-event device elapsed",
        "baseline": "retained_group_direct",
        "candidate": "retained_native_domain_run_sum_group0_victim_group12",
        "prepared_ownership": {
            "timed_retained_numerator_objects": 1,
            "same_prepared_object": true,
            "oracle_dropped_before_retained_prepare": true,
            "cold_fixed_image_materialization_in_timing": false,
        },
        "fixed_image": {
            "retained_evaluation_count": RETAINED_COLUMN_COUNT,
            "retained_words": RETAINED_IMAGE_WORDS,
            "retained_bytes": RETAINED_IMAGE_WORDS as u64 * 4,
            "manifest_blake3": fixture.retained_manifest_blake3.to_string(),
            "staged_coefficient_sources": 0,
            "staged_lde_kernel_nodes": staged_lde_kernel_nodes(&shape.retained_requirements),
        },
        "bytes": {
            "validated_numerator_and_auxiliary": canonical.len_bytes(),
            "validated_fri_input": canonical_fri.len_bytes(),
            "exact_word_comparison": true,
            "poisoned_complete_writes": true,
        },
        "plan": {
            "identity_blake3": Hash::from_bytes(receipt.identity).to_string(),
            "target_group": receipt.target_group,
            "victim_group": receipt.victim_group,
            "run_count": receipt.manifest.run_count,
            "precomputed_terms": receipt.precomputed_term_count,
            "direct_terms": receipt.direct_term_count,
            "scratch_words_per_coordinate": receipt.scratch_words_per_coordinate,
            "margin_words_per_coordinate": receipt.margin_words_per_coordinate,
            "incremental_arena_bytes": 0,
            "baseline_row_terms": receipt.baseline_row_terms,
            "candidate_add_units": receipt.candidate_add_units,
        },
        "checks": {
            "eager_candidate_exact": true,
            "captured_direct_exact": true,
            "captured_candidate_exact": true,
            "retained_source_only_direct_diff": true,
            "retained_source_only_candidate_full_byte_exact": true,
            "retained_source_restoration_candidate_exact": true,
            "retained_source_restoration_direct_exact": true,
            "random_coefficient_only_direct_diff": true,
            "random_coefficient_only_candidate_full_byte_exact": true,
            "random_coefficient_restoration_candidate_exact": true,
            "random_coefficient_restoration_direct_exact": true,
            "post_timing_direct_exact": true,
            "post_timing_candidate_exact": true,
            "formal_exact_check_gate": exact_checks_passed,
        },
        "mutation": {
            "retained_column": mutation.column,
            "source_index": mutation.source_index,
            "source_log_size": mutation.source_log_size,
            "image_words": mutation.words,
            "run_sum_consumed": true,
            "phases": ["retained_source_only", "random_coefficient_only"],
            "each_phase_restored_before_next_phase": true,
        },
        "digests": {
            "canonical_numerator_blake3": canonical.digest().to_string(),
            "canonical_fri_blake3": canonical_fri.digest.to_string(),
            "validated": digests.iter().map(|(label, (numerator, fri))| {
                ((*label).to_owned(), json!({
                    "numerator_blake3": numerator.to_string(),
                    "fri_blake3": fri.to_string(),
                }))
            }).collect::<serde_json::Map<_, _>>(),
        },
        "graph": {
            "direct_kernel_nodes": EXPECTED_DIRECT_NODES,
            "candidate_kernel_nodes": EXPECTED_RUN_SUM_NODES,
            "candidate_minus_direct": EXPECTED_RUN_SUM_NODES - EXPECTED_DIRECT_NODES,
            "expected_delta_from_run_count": receipt.manifest.run_count,
        },
        "timing": {
            "ordering": "AB,BA alternating",
            "warmups_each": WARMUPS,
            "samples_each": SAMPLES,
            "percentile_method": "nearest-rank",
            "direct_samples_ms": direct_ms,
            "candidate_samples_ms": candidate_ms,
            "direct": {"p50_ms": direct_p50, "p95_ms": direct_p95},
            "candidate": {"p50_ms": candidate_p50, "p95_ms": candidate_p95},
            "speedup": {"p50": direct_p50 / candidate_p50, "p95": direct_p95 / candidate_p95},
        },
        "objective_reward": {
            "reference": {
                "sealed_a40_retained_p50_ms": SEALED_A40_RETAINED_P50_MS,
                "sealed_a40_retained_p95_ms": SEALED_A40_RETAINED_P95_MS,
            },
            "observed": {
                "p50_saved_ms": direct_p50 - candidate_p50,
                "p95_saved_ms": direct_p95 - candidate_p95,
                "positive_p50": candidate_p50 < direct_p50,
                "positive_p95": candidate_p95 < direct_p95,
            },
            "diagnostic_thresholds": {
                "three_x_boundary_max_ms": THREE_X_RETAINED_MAX_MS,
                "five_x_boundary_max_ms": FIVE_X_RETAINED_MAX_MS,
                "diagnostic_three_x": candidate_p50 <= THREE_X_RETAINED_MAX_MS,
                "diagnostic_five_x": candidate_p50 <= FIVE_X_RETAINED_MAX_MS,
            },
        },
        "artifact_eligibility": {
            "eligible": artifact_eligible,
            "requires": "CUDA archive-LTO request and complete linked-module artifact identity",
            "archive_lto_requested_at_compile": ARCHIVE_LTO_REQUESTED,
            "cuda_build_mode": artifact.cuda_build_mode,
            "identity_complete": artifact.is_complete(),
        },
        "static_sass_evidence": {
            "sha256": static_sass_sha256,
            "sha256_env": STATIC_SASS_SHA256_ENV,
            "gate_declaration": static_sass_gate_declaration,
            "gate_env": STATIC_SASS_GATE_ENV,
            "gate_passed": static_sass_gate_passed,
            "required_for_formal_promotion": true,
        },
        "formal_promotion": {
            "eligible": formal_promotion_eligible,
            "requires": "artifact eligibility, 20/20 paired candidate wins, p95 non-regression, exact-output checks, and passing static-SASS evidence",
            "required_paired_candidate_wins": SAMPLES,
            "observed_paired_candidate_wins": paired_candidate_wins,
            "exact_sample_count": exact_sample_count,
            "all_pairs_won": all_pairs_won,
            "p95_non_regression": p95_non_regression,
            "exact_checks_passed": exact_checks_passed,
            "static_sass_gate_passed": static_sass_gate_passed,
            "three_x_boundary_max_ms": THREE_X_RETAINED_MAX_MS,
            "five_x_boundary_max_ms": FIVE_X_RETAINED_MAX_MS,
            "passes_three_x": formal_promotion_eligible && candidate_p50 <= THREE_X_RETAINED_MAX_MS,
            "passes_five_x": formal_promotion_eligible && candidate_p50 <= FIVE_X_RETAINED_MAX_MS,
        },
        "identity": {
            "topology_fixture_blake3": sn3.digest.to_string(),
            "numerator_input_recipe_blake3": numerator_recipe.to_string(),
            "boundary_input_recipe_blake3": boundary_recipe.to_string(),
            "retained_measurement_source_blake3": measurement_source.to_string(),
            "base_boundary_seal_blake3": artifact.boundary_seal_blake3.to_string(),
            "ordinary_cuda_source_blake3": artifact.ordinary_cuda_source_blake3.to_string(),
            "test_binary_blake3": artifact.test_binary_blake3.to_string(),
            "cuda_build_mode": artifact.cuda_build_mode,
            "archive_lto_requested_at_compile": ARCHIVE_LTO_REQUESTED,
            "expected_cuda_module_build_identity": artifact.expected_cuda_module_build_identity.to_string(),
            "linked_cuda_module_build_identity": artifact.linked_cuda_module_build_identity.to_string(),
            "cuda_module_target_sms": artifact.cuda_module_target_sms,
            "identity_complete": artifact.is_complete(),
        },
    });
    publish(&result);
}

fn required_digest<'a>(
    digests: &'a [(&str, (Hash, Hash))],
    required_label: &str,
) -> &'a (Hash, Hash) {
    &digests
        .iter()
        .find(|(label, _)| *label == required_label)
        .unwrap_or_else(|| panic!("missing exact-check digest {required_label}"))
        .1
}

fn static_sass_sha256() -> Option<String> {
    let value = std::env::var(STATIC_SASS_SHA256_ENV).ok()?;
    assert!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "{STATIC_SASS_SHA256_ENV} must be a lowercase 64-character SHA-256"
    );
    Some(value)
}

fn retained_measurement_source_digest() -> Hash {
    let sources: &[(&str, &[u8])] = &[
        (
            "tests/prepared_quotient_numerator_sn3_retained_run_sum_native.rs",
            include_bytes!("../prepared_quotient_numerator_sn3_retained_run_sum_native.rs"),
        ),
        (
            "tests/support/sn3_quotient_retained_fixture.rs",
            include_bytes!("sn3_quotient_retained_fixture.rs"),
        ),
        (
            "tests/support/sn3_quotient_retained_run_sum_ab.rs",
            include_bytes!("sn3_quotient_retained_run_sum_ab.rs"),
        ),
        (
            "tests/support/sn3_quotient_retained_result.rs",
            include_bytes!("sn3_quotient_retained_result.rs"),
        ),
        (
            "src/backend/quotient_numerator_run_sum.rs",
            include_bytes!("../../src/backend/quotient_numerator_run_sum.rs"),
        ),
        (
            "src/backend/prepared_quotient_numerator/run_sum.rs",
            include_bytes!("../../src/backend/prepared_quotient_numerator/run_sum.rs"),
        ),
        (
            "cuda/quotient_numerator_native_run_sum.cu",
            include_bytes!(
                "../../../backend-cuda-kernels/cuda/quotient_numerator_native_run_sum.cu"
            ),
        ),
    ];
    let mut hasher = Hasher::new();
    hasher.update(b"stwo.sn3.retained-run-sum.measurement-sources.v1\0");
    for (name, bytes) in sources {
        hasher.update(&u64::try_from(name.len()).unwrap().to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&u64::try_from(bytes.len()).unwrap().to_le_bytes());
        hasher.update(bytes);
    }
    hasher.finalize()
}

fn publish(receipt: &serde_json::Value) {
    let bytes = serde_json::to_vec_pretty(receipt).unwrap();
    if let Some(path) = std::env::var_os("STWO_SN3_RETAINED_RUN_SUM_AB_RECEIPT").map(PathBuf::from)
    {
        assert!(!path.exists(), "refusing to replace {}", path.display());
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .unwrap();
        file.write_all(&bytes).unwrap();
        file.write_all(b"\n").unwrap();
        file.sync_all().unwrap();
        fs::rename(temporary, path).unwrap();
    }
    println!(
        "STWO_SN3_RETAINED_RUN_SUM_AB_RECEIPT_JSON={}",
        serde_json::to_string(receipt).unwrap()
    );
}
