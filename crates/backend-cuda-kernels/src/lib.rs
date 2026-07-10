//! Compile-gated CUDA kernel library — the staging crate for a future `stwo-backend-cuda`.
//!
//! See the crate README for the kernel-to-trait mapping, the API delta against the
//! prototype these kernels came from, and the known issues to address while wiring up.
//!
//! On a machine with `nvcc`, building this crate compiles every kernel under `cuda/`
//! into a static archive (the first validation gate for the staged sources). Without
//! `nvcc` it builds as a stub, so the workspace never requires a CUDA toolkit.
//!
//! No FFI bindings are exposed yet: bindings should be written together with the
//! `stwo-backend-cuda` trait implementations and validated against
//! `stwo-backend-testkit` (proof byte-equality on both Blake2s channels), mirroring
//! `stwo-backend-metal`.

pub mod aot_pack;
pub mod raw;
#[cfg(not(stwo_cuda_link))]
mod stubs;

/// True when the CUDA kernels were compiled and linked into this build.
pub const CUDA_KERNELS_BUILT: bool = cfg!(stwo_cuda_link);

/// `"cuda"` when the kernels were compiled, `"no-cuda"` for the stub build.
pub const BUILD_MODE: &str = env!("STWO_CUDA_BUILD_MODE");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_mode_is_consistent() {
        assert_eq!(BUILD_MODE == "cuda", CUDA_KERNELS_BUILT);
    }

    #[test]
    fn prepared_quotient_symbols_are_linked_in_cuda_and_stub_builds() {
        assert_ne!(raw::stwo_combine_quotients_from_numerators_on as usize, 0);
        assert_ne!(raw::stwo_prepare_quotient_numerator_terms_on as usize, 0);
        assert_ne!(raw::stwo_finalize_quotient_numerator_groups_on as usize, 0);
        assert_ne!(raw::stwo_zero_quotient_numerator_outputs_on as usize, 0);
        assert_ne!(raw::stwo_accumulate_quotient_numerator_batch_on as usize, 0);
        assert_ne!(raw::stwo_ntt_b2n_columns_on as usize, 0);
        assert_ne!(raw::stwo_lde_n2b_columns_on as usize, 0);
    }

    #[test]
    fn hash_from_tile_symbols_are_linked_in_cuda_and_stub_builds() {
        assert_ne!(raw::stwo_lde_n2b_columns_before_circle_on as usize, 0);
        assert_ne!(raw::stwo_blake2s_leaf_group_from_lde_on as usize, 0);
        assert_ne!(raw::stwo_lde_n2b_hash16_configure as usize, 0);
        assert_ne!(raw::stwo_lde_n2b_hash16_on as usize, 0);
    }

    #[test]
    fn ntt_leaf_fused_symbols_are_linked_in_cuda_and_stub_builds() {
        assert_ne!(raw::stwo_ntt_leaf_fused_configure as usize, 0);
        assert_ne!(raw::stwo_ntt_leaf_fused_on as usize, 0);
    }

    #[test]
    fn prepared_oods_symbols_are_linked_in_cuda_and_stub_builds() {
        assert_ne!(raw::stwo_oods_derive_points_on as usize, 0);
        assert_ne!(raw::stwo_oods_eval_first_on as usize, 0);
        assert_ne!(raw::stwo_oods_eval_reduce_on as usize, 0);
        assert_ne!(raw::stwo_oods_store_results_on as usize, 0);
        assert_ne!(raw::stwo_oods_barycentric_weights_on as usize, 0);
        assert_ne!(raw::stwo_oods_barycentric_eval_many_on as usize, 0);
    }

    #[test]
    fn prepared_composition_symbols_are_linked_in_cuda_and_stub_builds() {
        assert_ne!(
            raw::stwo_composition_generate_descending_powers_on as usize,
            0
        );
        assert_ne!(raw::stwo_composition_lift_accumulate_on as usize, 0);
        assert_ne!(raw::stwo_composition_materialize_ext_params_on as usize, 0);
        assert_ne!(raw::stwo_cuda_jit_eval_fused_on as usize, 0);
    }

    #[test]
    fn prepared_final_fri_and_pow_symbols_are_linked_in_cuda_and_stub_builds() {
        assert_ne!(raw::stwo_fri_last_layer_on as usize, 0);
        assert_ne!(raw::stwo_blake2s_pow_persistent_on as usize, 0);
    }

    #[test]
    fn prepared_decommit_symbols_are_linked_in_cuda_and_stub_builds() {
        assert_ne!(raw::stwo_blake2s_sparse_leaf_group_on as usize, 0);
        assert_ne!(raw::stwo_decommit_normalize_queries_on as usize, 0);
        assert_ne!(raw::stwo_decommit_prepare_trace_queries_on as usize, 0);
        assert_ne!(raw::stwo_decommit_gather_trace_values_on as usize, 0);
        assert_ne!(raw::stwo_decommit_sparse_parent_on as usize, 0);
        assert_ne!(raw::stwo_decommit_assemble_trace_on as usize, 0);
        assert_ne!(raw::stwo_decommit_prepare_fri_queries_on as usize, 0);
        assert_ne!(raw::stwo_decommit_gather_fri_values_on as usize, 0);
        assert_ne!(raw::stwo_decommit_assemble_fri_on as usize, 0);
    }
}
