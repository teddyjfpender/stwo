//! Process-level default-off admission gate for domain-progressive commitment.

use stwo_backend_cuda::{
    progressive_leaf_workspace_requirements, progressive_leaf_workspace_requirements_for_mode,
    progressive_prepare_mode_admission, PreparedProgressiveCommitError, ProgressiveCommitGeometry,
    ProgressiveCommitGroupGeometry, ProgressiveCommitMode,
};

fn geometry() -> ProgressiveCommitGeometry {
    ProgressiveCommitGeometry {
        lifting_log_size: 5,
        log_blowup_factor: 1,
        groups: vec![ProgressiveCommitGroupGeometry {
            coefficient_log_sizes: vec![4],
            retain_evaluations: false,
        }],
    }
}

#[test]
fn absent_flag_rejects_requirements_and_prepared_admission_without_dispatch() {
    unsafe { std::env::remove_var("STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE") };
    assert_eq!(
        ProgressiveCommitMode::from_env(),
        ProgressiveCommitMode::FullLifting
    );
    assert_eq!(
        progressive_leaf_workspace_requirements(geometry()).unwrap_err(),
        PreparedProgressiveCommitError::Disabled
    );

    // An explicitly constructed topology (the native differential seam) still
    // cannot enter `PreparedProgressiveLeaves::prepare`: this is the exact
    // admission function called before arena binding. No fallback or production
    // dispatcher is invoked by either rejection.
    let explicit = progressive_leaf_workspace_requirements_for_mode(
        ProgressiveCommitMode::DomainProgressive,
        geometry(),
    )
    .unwrap();
    assert_eq!(
        progressive_prepare_mode_admission(&explicit).unwrap_err(),
        PreparedProgressiveCommitError::Disabled
    );
}
