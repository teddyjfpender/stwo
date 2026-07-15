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
    fn recent_checked_abi_symbols_are_linked_in_cuda_and_stub_builds() {
        assert_ne!(raw::cuda_default_pool_alloc_checked as usize, 0);
        assert_ne!(raw::cuda_default_pool_copy_h2d_checked as usize, 0);
        assert_ne!(raw::cuda_default_pool_free_checked as usize, 0);
        assert_ne!(raw::cuda_default_pool_stream_sync_checked as usize, 0);
        assert_ne!(raw::stwo_pedersen_table_init_borrowed_checked as usize, 0);
        assert_ne!(raw::stwo_cuda_device_snapshot as usize, 0);
        assert_ne!(raw::stwo_preprocessed_alloc_u32_checked as usize, 0);
        assert_ne!(raw::stwo_preprocessed_copy_h2d_checked as usize, 0);
        assert_ne!(raw::stwo_preprocessed_gen_seq_checked as usize, 0);
        assert_ne!(raw::stwo_preprocessed_gen_range_checked as usize, 0);
        assert_ne!(raw::stwo_preprocessed_gen_xor_checked as usize, 0);
        assert_ne!(raw::stwo_preprocessed_stream_sync_checked as usize, 0);
        assert_ne!(raw::stwo_cuda_jit_witness_phase_pair_launch as usize, 0);
        assert_ne!(raw::stwo_witness_casm_input_scatter_on as usize, 0);
    }

    #[test]
    fn witness_phase_pair_abi_is_scratch_explicit() {
        type PhasePairFn = unsafe extern "C" fn(
            *const *const core::ffi::c_char,
            *const u64,
            *const *const u32,
            *const *const u32,
            *const u32,
            *const *mut u32,
            *const *mut u32,
            *mut u32,
            *mut u32,
            *mut u32,
            u32,
            *mut core::ffi::c_void,
        ) -> bool;
        let _: PhasePairFn = raw::stwo_cuda_jit_witness_phase_pair_launch;
    }

    #[test]
    fn prepared_quotient_symbols_are_linked_in_cuda_and_stub_builds() {
        assert_ne!(raw::stwo_combine_quotients_from_numerators_on as usize, 0);
        assert_ne!(raw::stwo_prepare_quotient_numerator_terms_on as usize, 0);
        assert_ne!(raw::stwo_finalize_quotient_numerator_groups_on as usize, 0);
        assert_ne!(raw::stwo_zero_quotient_numerator_outputs_on as usize, 0);
        assert_ne!(raw::stwo_accumulate_quotient_numerator_batch_on as usize, 0);
        assert_ne!(
            raw::stwo_accumulate_quotient_numerator_single_write_on as usize,
            0
        );
        assert_ne!(raw::stwo_ntt_b2n_columns_on as usize, 0);
        assert_ne!(raw::stwo_lde_n2b_columns_on as usize, 0);
    }

    #[test]
    fn prepared_quotient_abi_has_no_denominator_scratch_arguments() {
        type CombineFn = unsafe extern "C" fn(
            u32,
            u32,
            u32,
            u32,
            *const u32,
            u32,
            *const raw::CudaSecureField,
            *const u32,
            *const *const u32,
            *const *const u32,
            *const *const u32,
            *const *const u32,
            *mut u32,
            *mut u32,
            *mut u32,
            *mut u32,
            *mut core::ffi::c_void,
        ) -> i32;
        let _: CombineFn = raw::stwo_combine_quotients_from_numerators_on;

        let source = include_str!("../cuda/quotients.cu");
        assert_eq!(source.matches("quotient_inverse_chunk").count(), 3);
        assert_eq!(source.matches("denominator_for_sample(").count(), 3);
        assert!(source.contains("constexpr uint32_t ACCUMULATE_QUOTIENT_INVERSE_CHUNK = 4;"));
        assert!(source.contains("constexpr uint32_t COMBINE_QUOTIENT_INVERSE_CHUNK = 8;"));
        assert!(source.contains("constexpr int QUOTIENT_COMBINE_BLOCK_DIM = 512;"));
        assert_eq!(
            source
                .matches("__launch_bounds__(QUOTIENT_COMBINE_BLOCK_DIM, 1)")
                .count(),
            2
        );
        assert!(source.contains("inverses[CHUNK_SIZE - 1]"));
        assert!(source.contains("inverse_product = mul(inverse_product, denominator)"));
        assert!(source.contains("zero_mask |= static_cast<uint32_t>(is_zero) << offset"));
        assert!(!source.contains("__shfl_sync"));
        assert!(!source.contains("__shfl_down_sync"));
        assert!(!source.contains("cm31 *denominator_inverses"));
        assert!(!source.contains("cuda_proving_malloc<cm31>(sample_size * domain_size)"));
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
        assert_ne!(raw::stwo_ntt_progressive_leaf_fused_configure as usize, 0);
        assert_ne!(raw::stwo_ntt_progressive_leaf_fused_on as usize, 0);
    }

    #[test]
    fn progressive_ntt_leaf_sink_preserves_lazy_block_and_retained_write_contract() {
        let source = include_str!("../cuda/ntt_leaf_fused.cu");
        assert!(source.contains("if constexpr (PROGRESSIVE)"));
        assert!(source.contains("4u * cols_done, 0u"));
        assert!(source.contains("state->pending[word] = message[word]"));
        assert_eq!(
            source
                .matches("writes_completed_evaluation<PROGRESSIVE>")
                .count(),
            2
        );
        assert!(source.contains("(retained_write_mask & ~0xffffu) != 0"));
        assert!(source.contains("uint32_t retained_write_mask"));
    }

    #[test]
    fn nofinal_ntt_shapes_pin_their_distinct_occupancy_bounds() {
        let source = include_str!("../cuda/rfft.cu");
        assert_eq!(
            source.matches("LOG_VALS_PER_THREAD == 3 ? 6 : 2").count(),
            1
        );
        assert!(source.contains("1u << (LOG_WARP + LOG_VALS_PER_THREAD)"));
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
        assert_ne!(raw::stwo_decommit_pack_trace_group_on as usize, 0);
        assert_ne!(raw::stwo_decommit_sparse_parent_on as usize, 0);
        assert_ne!(raw::stwo_decommit_assemble_trace_on as usize, 0);
        assert_ne!(raw::stwo_decommit_prepare_fri_queries_on as usize, 0);
        assert_ne!(raw::stwo_decommit_assemble_fri_on as usize, 0);
    }

    #[test]
    fn gpu_lab_fri_entry_names_are_stable() {
        let sources = [
            include_str!("../cuda/fold_line.cu"),
            include_str!("../cuda/blake2s.cu"),
            include_str!("../cuda/device_transcript.cu"),
        ]
        .join("\n");
        for (name, declaration) in [
            (
                "stwo_gpu_lab_fold_line_device_alpha",
                "extern \"C\" __global__ void stwo_gpu_lab_fold_line_device_alpha",
            ),
            (
                "stwo_gpu_lab_blake2s_fri_leaf",
                "extern \"C\" __global__ void __launch_bounds__(BLOCK_SIZE) stwo_gpu_lab_blake2s_fri_leaf",
            ),
            (
                "stwo_gpu_lab_blake2s_layer",
                "extern \"C\" __global__ void __launch_bounds__(BLOCK_SIZE, STWO_LEAF_MIN_BLOCKS) stwo_gpu_lab_blake2s_layer",
            ),
            (
                "stwo_gpu_lab_blake2s_transcript_mix_words",
                "}  // namespace\n\nextern \"C\" __global__ void stwo_gpu_lab_blake2s_transcript_mix_words",
            ),
            (
                "stwo_gpu_lab_blake2s_transcript_draw_secure",
                "}  // namespace\n\nextern \"C\" __global__ void stwo_gpu_lab_blake2s_transcript_draw_secure",
            ),
        ] {
            assert!(sources.contains(declaration));
            assert!(sources.contains(&format!("{name}<<<")));
        }
    }
}
