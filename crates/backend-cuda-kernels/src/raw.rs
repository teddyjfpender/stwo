//! Raw FFI declarations for the staged CUDA kernel entry points.
//!
//! Link-gated: with `stwo_cuda_archive` (set only by this crate's build script when
//! nvcc compiled the kernels) these resolve to the static archive; otherwise
//! `stubs.rs` provides panicking `no_mangle` definitions so the crate links
//! everywhere — including when `stwo_cuda_link` is forced via `RUSTFLAGS` for
//! compile-only validation of the downstream link-gated tests without a CUDA
//! toolkit.

use core::ffi::c_void;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct CudaSecureField {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
}

impl CudaSecureField {
    pub fn zero() -> Self {
        Self {
            a: 0,
            b: 0,
            c: 0,
            d: 0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct CirclePointBaseField {
    pub x: u32,
    pub y: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct LayerIndexPair {
    pub layer_idx: u32,
    pub hash_idx: u32,
}

#[repr(C, align(32))]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Blake2sHash(pub [u8; 32]);

#[cfg_attr(stwo_cuda_archive, link(name = "stwo_cuda_kernels", kind = "static"))]
extern "C" {
    /// Returns a CUDA error code (0 = success). Sets the default mem pool's release
    /// threshold to never-release so warm proves reuse allocations.
    pub fn cuda_mem_pool_init() -> i32;
    // Upload lane (async H2D on a dedicated copy stream; see cuda_mem_pool.cuh).
    pub fn stwo_upload_alloc_uint32(count: usize) -> *mut u32;
    pub fn stwo_upload_h2d_async(pinned_src: *const u32, device_dst: *mut u32, n_words: u64);
    pub fn stwo_upload_record_half(half: i32);
    pub fn stwo_upload_half_sync(half: i32);
    pub fn stwo_legacy_wait_uploads();
    // Generic witness logup-input kernels (witness_logup.cu).
    pub fn tuple_pair_logup(
        base0: CudaSecureField,
        cols0: *const *const u32,
        alphas0: *const u32,
        n0: u32,
        base1: CudaSecureField,
        cols1: *const *const u32,
        alphas1: *const u32,
        n1: u32,
        mult0_col: *const u32,
        enabler0: u32,
        mult1_col: *const u32,
        enabler1: u32,
        negate: u32,
        column_length: u32,
        denoms: *const u32,
        num0: *const u32,
        num1: *const u32,
        num2: *const u32,
        num3: *const u32,
    );
    pub fn tuple_single_logup(
        base: CudaSecureField,
        cols: *const *const u32,
        alphas: *const u32,
        n: u32,
        mult_col: *const u32,
        enabler: u32,
        negate: u32,
        column_length: u32,
        denoms: *const u32,
        num0: *const u32,
        num1: *const u32,
        num2: *const u32,
        num3: *const u32,
    );

    pub fn ret_opcode_trace(
        pc: *const u32,
        ap: *const u32,
        fp: *const u32,
        addr_table: *const u32,
        big_words: *const u32,
        small_words: *const u32,
        n_rows: u32,
        column_length: u32,
        trace: *const *const u32,
        addr0: *const u32,
        addr1: *const u32,
        next_pc: *const u32,
        next_fp: *const u32,
    );
    pub fn add_opcode_small_trace(
        pc: *const u32,
        ap: *const u32,
        fp: *const u32,
        addr_table: *const u32,
        big_words: *const u32,
        small_words: *const u32,
        n_rows: u32,
        column_length: u32,
        trace: *const *const u32,
        staged: *const *const u32,
    );
    pub fn tuple_count(
        cols: *const *const u32,
        n_tuples: u32,
        width: u32,
        slot_bits: *const u32,
        n_relations: u32,
        column_length: u32,
        input_to_row_lut: *const u32,
        table_size: u32,
        counts: *mut u32,
    );
    pub fn verify_instruction_trace(
        pc: *const u32,
        off0: *const u32,
        off1: *const u32,
        off2: *const u32,
        felt5_high: *const u32,
        felt6: *const u32,
        opcode_ext: *const u32,
        instruction_id: *const u32,
        mult: *const u32,
        column_length: u32,
        trace: *const *const u32,
        enc1: *const u32,
        enc3: *const u32,
        enc5b: *const u32,
    );
    /// Batched-OODS barycentric eval: result lands in a device slot (no D2H).
    pub fn barycentric_eval_base_field_into(
        eval_values: *const u32,
        weights: *const u32,
        size: u32,
        out_slot: *mut u32,
    );
    /// Chunked atomicMin nonce search; returns the LOWEST valid nonce, matching the
    /// SIMD grind's search order byte-exactly (non-M31 Blake2s channel only).
    pub fn grind_blake2s(host_prefixed_digest: *const u32, pow_bits: u32) -> u64;
    /// GPU generation of preprocessed columns (values identical to the CPU
    /// constructors; commitment roots must be byte-equal).
    pub fn gen_seq_column_on_gpu(output: *mut u32, log_size: u32);
    pub fn gen_range_check_columns_on_gpu(
        output_columns: *const *mut u32,
        n_columns: u32,
        bits_per_segment: *const u32,
        n_segments: u32,
    );
    pub fn gen_bitwise_xor_columns_on_gpu(output_columns: *const *mut u32, n_bits: u32);
    /// JIT-compile (NVRTC; cached by the CONTENT semantic hash, never pointers) and
    /// launch a generated fused constraint kernel. Returns false on any compile or
    /// launch failure; the caller falls back to the CPU lane.
    #[allow(clippy::too_many_arguments)]
    pub fn stwo_cuda_jit_eval_fused(
        source: *const core::ffi::c_char,
        kernel_name: *const core::ffi::c_char,
        semantic_hash: u64,
        trace_values: *const u32,
        interaction_offsets: *const u32,
        base_params: *const u32,
        ext_params: *const u32,
        random_coeff_powers: *const u32,
        denom_inv: *const u32,
        coord_0: *mut u32,
        coord_1: *mut u32,
        coord_2: *mut u32,
        coord_3: *mut u32,
        row_count: u32,
        log_n_rows: u32,
    ) -> bool;
    /// Per-component constraint-quotient kernel dispatch (NitrooZK lineage). The first
    /// 4 bytes behind `eval` are an FNV1a hash of the component name selecting the
    /// kernel; the rest is the raw `FrameworkEval` struct the kernel's generated code
    /// reads. Returns false when no kernel matches (caller falls back to CPU).
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_constraint_quotients_on_domain(
        quotients_0: *const u32,
        quotients_1: *const u32,
        quotients_2: *const u32,
        quotients_3: *const u32,
        trace0_evaluations: *const *const u32,
        trace0_evaluations_len: u32,
        trace1_evaluations: *const *const u32,
        trace1_evaluations_len: u32,
        trace2_evaluations: *const *const u32,
        trace2_evaluations_len: u32,
        random_coeff_powers: *const u32,
        denominator_inverses: *const u32,
        domain_log_size: u32,
        eval_domain_log_size: u32,
        number_of_columns: u32,
        logup_counts: u32,
        eval: *mut c_void,
        cumsum_shift: CudaSecureField,
        should_accumulate: bool,
        use_assert_evaluator: bool,
    ) -> bool;
    pub fn copy_uint32_t_vec_from_device_to_host(
        device_ptr: *const u32,
        host_ptr: *const u32,
        size: u32,
    );

    pub fn copy_uint32_t_vec_from_host_to_device(host_ptr: *const u32, size: u32) -> *const u32;

    pub fn copy_uint32_t_vec_from_device_to_device(
        from: *const u32,
        dst: *const u32,
        size: u32,
    ) -> *const u32;

    pub fn copy_uint32_t_vec_from_device_to_device_offset(
        from: *const u32,
        dst: *const u32,
        size: u32,
        offset: u32,
    );

    pub fn cuda_zero_device_region(ptr: *const u32, offset_words: u64, n_words: u64);

    pub fn cuda_alloc_pinned_host_u32(n_words: u64) -> *mut u32;

    pub fn cuda_free_pinned_host_u32(ptr: *mut u32);

    pub fn copy_uint32_t_vec_from_host_to_device_into(
        host_ptr: *const u32,
        device_ptr: *const u32,
        n_words: u64,
    );

    pub fn cuda_gather_uint32_t(
        device_src: *const u32,
        host_indices: *const u32,
        n_indices: u32,
        host_out: *mut u32,
    );

    pub fn cuda_malloc_uint32_t(size: u32) -> *const u32;

    pub fn cuda_set_uint32_t(device_ptr: *const c_void, index: usize, val: u32);

    pub fn cuda_get_uint32_t(device_ptr: *const c_void, index: usize) -> u32;

    pub fn cuda_increase_at(device_ptr: *const c_void, addr: u32);

    pub fn cuda_get_secure_field(device_ptr: *const c_void, index: usize) -> CudaSecureField;

    pub fn cuda_malloc_blake_2s_hash(size: usize) -> *const Blake2sHash;

    pub fn cuda_alloc_zeroes_uint32_t(size: u32) -> *const u32;

    pub fn cuda_alloc_zeroes_blake_2s_hash(size: usize) -> *const Blake2sHash;

    pub fn cuda_free_memory(device_ptr: *const c_void);

    pub fn cuda_get_memory_info(free_mem: *mut usize, total_mem: *mut usize);

    pub fn bit_reverse_base_field(array: *const u32, size: usize);

    pub fn bit_reverse_secure_field(array: *const u32, size: usize);

    pub fn batch_inverse_base_field(from: *const u32, dst: *const u32, size: usize);

    // pub fn batch_inverse_secure_field(from: *const u32, dst: *const u32, size: usize);

    pub fn sort_values_and_permute_with_bit_reverse_order(
        from: *const u32,
        size: usize,
    ) -> *const u32;

    pub fn precompute_twiddles(
        initial: CirclePointBaseField,
        step: CirclePointBaseField,
        total_size: usize,
    ) -> *const u32;

    pub fn evaluate_columns(
        eval_domain_sizes: *const u32,
        values: *const *const u32,
        twiddles_tree: *const u32,
        twiddle_tree_size: u32,
        number_of_columns: u32,
        column_sizes: *const u32,
    );

    pub fn eval_at_point(
        coeffs: *const u32,
        coeffs_size: u32,
        point_x: CudaSecureField,
        point_y: CudaSecureField,
    ) -> CudaSecureField;

    pub fn batch_eval_at_points(
        coeffs_ptrs: *const *const u32,
        coeffs_size: i32,
        num_polys: i32,
        point_x: CudaSecureField,
        point_y: CudaSecureField,
        results: *mut CudaSecureField,
    );

    pub fn barycentric_point_vanishings(
        half_coset_initial_index: u32,
        half_coset_step_size: u32,
        size: u32,
        log_size: u32,
        point_x: CudaSecureField,
        point_y: CudaSecureField,
        result: *const u32,
    );

    pub fn barycentric_weights_from_point_vanishings(
        point_vanishings: *const u32,
        size: u32,
        even_scale: CudaSecureField,
        odd_scale: CudaSecureField,
        result_weights: *const u32,
    );

    pub fn barycentric_eval_base_field(
        eval_values: *const u32,
        weights: *const u32,
        size: u32,
    ) -> CudaSecureField;

    pub fn fold_line(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values: *const *const u32,
        alpha: CudaSecureField,
        folded_values: *const *const u32,
    );

    pub fn fold_circle_into_line(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values: *const *const u32,
        alpha: CudaSecureField,
        folded_values: *const *const u32,
    );

    pub fn accumulate(size: u32, left_columns: *const *const u32, right_columns: *const *const u32);

    pub fn lift_accumulate_secure_columns(
        size: u32,
        log_ratio: u32,
        previous_columns: *const *const u32,
        current_columns: *const *const u32,
    );

    pub fn commit_on_first_layer(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        result: *mut Blake2sHash,
    );

    pub fn commit_on_first_layer_lifted(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        column_log_sizes: *const u32,
        lifting_log_size: u32,
        result: *mut Blake2sHash,
    );

    pub fn commit_on_first_layer_lifted_indexed(
        n_indices: u32,
        indices: *const u32,
        amount_of_columns: u32,
        columns: *const *const u32,
        column_log_sizes: *const u32,
        lifting_log_size: u32,
        result: *mut Blake2sHash,
    );

    pub fn commit_on_layer_with_previous(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        previous_layer: *const Blake2sHash,
        result: *mut Blake2sHash,
    );

    pub fn copy_blake_2s_hash_vec_from_host_to_device(
        from: *const Blake2sHash,
        size: usize,
    ) -> *mut Blake2sHash;

    pub fn copy_blake_2s_hash_vec_from_device_to_host(
        from: *const Blake2sHash,
        to: *mut Blake2sHash,
        size: usize,
    );

    pub fn copy_blake_2s_hash_vec_from_device_to_device(
        from: *const Blake2sHash,
        dst: *const Blake2sHash,
        size: usize,
    );

    pub fn cuda_get_blake_2s_hash(
        device_ptr: *const Blake2sHash,
        host_ptr: *mut Blake2sHash,
        index: usize,
    );

    pub fn cuda_set_blake_2s_hash(
        device_ptr: *mut Blake2sHash,
        index: usize,
        host_ptr: *const Blake2sHash,
    );

    pub fn cuda_batch_get_blake_2s_hash(
        device_ptr: *const Blake2sHash,
        host_ptr: *mut Blake2sHash,
        indices: *const u32,
        n_indices: u32,
    );

    pub fn cuda_multi_layer_batch_get_blake_2s_hash(
        layer_device_ptrs: *const *const Blake2sHash,
        host_ptr: *mut Blake2sHash,
        pairs: *const LayerIndexPair,
        n_pairs: u32,
    );

    pub fn copy_device_pointer_vec_from_host_to_device(
        from: *const *const u32,
        size: usize,
    ) -> *const *const u32;

    pub fn cuda_release_uploaded_pointer_vec(device_ptr: *const *const u32);

    pub fn accumulate_quotients(
        half_coset_initial_index: u32,
        half_coset_step_size: u32,
        domain_size: u32,
        columns: *const *const u32,
        number_of_columns: usize,
        random_coeff: CudaSecureField,
        sample_points: *const u32,
        sample_columns_indexes: *const u32,
        sample_columns_indexes_size: u32,
        sample_column_values: *const CudaSecureField,
        sample_column_and_values_sizes: *const u32,
        sample_size: u32,
        result_column_0: *const u32,
        result_column_1: *const u32,
        result_column_2: *const u32,
        result_column_3: *const u32,
        flattened_line_coeffs_size: u32,
    );

    pub fn accumulate_partial_quotient_numerators(
        domain_size: u32,
        columns: *const *const u32,
        sample_column_indexes: *const u32,
        sample_column_indexes_size: u32,
        line_coeffs_b: *const CudaSecureField,
        line_coeffs_c: *const CudaSecureField,
        result_column_0: *const u32,
        result_column_1: *const u32,
        result_column_2: *const u32,
        result_column_3: *const u32,
    );

    pub fn combine_quotients_from_numerators(
        half_coset_initial_index: u32,
        half_coset_step_size: u32,
        domain_size: u32,
        domain_log_size: u32,
        sample_points: *const CudaSecureField,
        sample_size: u32,
        first_linear_term_accs: *const CudaSecureField,
        partial_numerator_log_sizes: *const u32,
        partial_numerators_0: *const *const u32,
        partial_numerators_1: *const *const u32,
        partial_numerators_2: *const *const u32,
        partial_numerators_3: *const *const u32,
        result_column_0: *const u32,
        result_column_1: *const u32,
        result_column_2: *const u32,
        result_column_3: *const u32,
    );

    pub fn gen_eq_evals(
        v: CudaSecureField,
        y: *const CudaSecureField,
        y_size: u32,
        evals: *const CudaSecureField,
        evals_size: u32,
    );

    pub fn gkr_next_grand_product_layer(
        input_layer: *const CudaSecureField,
        input_size: u32,
        output_layer: *const CudaSecureField,
    );

    pub fn gkr_next_logup_generic_layer(
        numerators: *const CudaSecureField,
        denominators: *const CudaSecureField,
        input_size: u32,
        next_numerators: *const CudaSecureField,
        next_denominators: *const CudaSecureField,
    );

    pub fn gkr_next_logup_multiplicities_layer(
        numerators: *const u32,
        denominators: *const CudaSecureField,
        input_size: u32,
        next_numerators: *const CudaSecureField,
        next_denominators: *const CudaSecureField,
    );

    pub fn gkr_next_logup_singles_layer(
        denominators: *const CudaSecureField,
        input_size: u32,
        next_numerators: *const CudaSecureField,
        next_denominators: *const CudaSecureField,
    );

    pub fn gkr_sum_grand_product(
        eq_evals: *const CudaSecureField,
        input_layer: *const CudaSecureField,
        n_terms: u32,
        eval_at_0: *mut CudaSecureField,
        eval_at_2: *mut CudaSecureField,
    );

    pub fn gkr_sum_logup_generic(
        eq_evals: *const CudaSecureField,
        numerators: *const CudaSecureField,
        denominators: *const CudaSecureField,
        n_terms: u32,
        lambda: CudaSecureField,
        eval_at_0: *mut CudaSecureField,
        eval_at_2: *mut CudaSecureField,
    );

    pub fn gkr_sum_logup_multiplicities(
        eq_evals: *const CudaSecureField,
        numerators: *const u32,
        denominators: *const CudaSecureField,
        n_terms: u32,
        lambda: CudaSecureField,
        eval_at_0: *mut CudaSecureField,
        eval_at_2: *mut CudaSecureField,
    );

    pub fn gkr_sum_logup_singles(
        eq_evals: *const CudaSecureField,
        denominators: *const CudaSecureField,
        n_terms: u32,
        lambda: CudaSecureField,
        eval_at_0: *mut CudaSecureField,
        eval_at_2: *mut CudaSecureField,
    );

    pub fn fix_first_variable_base_field(
        evals: *const u32,
        evals_size: usize,
        assignment: CudaSecureField,
        output_evals: *const u32,
    );

    pub fn fix_first_variable_secure_field(
        evals: *const u32,
        evals_size: usize,
        assignment: CudaSecureField,
        output_evals: *const u32,
    );

    // Assert EQ FP IMM trace generation
    pub fn ntt_n2b_native_batch(
        value: *mut *mut u32,
        log_n: u32,
        num_poly: u32,
        start_stage: u32,
        end_stage: u32,
        g_twiddles: *const u32,
        twiddles_size: u32,
        eval_domain_size: u32,
    );

    pub fn ntt_b2n_column(
        values_columns: *mut *mut u32,
        log_n: u32,
        num_poly: u32,
        g_twiddles: *const u32,
        twiddles_size: u32,
        eval_domain_size: u32,
    );

    pub fn ntt_n2b_columns(
        values_columns: *mut *mut u32,
        log_n: u32,
        num_poly: u32,
        g_twiddles: *const u32,
        twiddles_size: u32,
        eval_domain_size: u32,
    );

    pub fn inclusive_prefix_sum(device_bit_rev_circle_domain_evals: *const u32, len: u32);

    pub fn inclusive_prefix_sum_x4(
        c0: *const u32,
        c1: *const u32,
        c2: *const u32,
        c3: *const u32,
        len: u32,
    );

    pub fn logup_fraction_chain(
        num0: *const u32,
        num1: *const u32,
        num2: *const u32,
        num3: *const u32,
        denom_packed: *const u32,
        prev0: *const u32,
        prev1: *const u32,
        prev2: *const u32,
        prev3: *const u32,
        size: u32,
    );

    pub fn logup_fraction_chain_dense(
        num0: *const u32,
        num1: *const u32,
        num2: *const u32,
        num3: *const u32,
        denoms_dense: *const u32,
        prev0: *const u32,
        prev1: *const u32,
        prev2: *const u32,
        prev3: *const u32,
        size: u32,
    );

    pub fn logup_sum_secure_coords(
        c0: *const u32,
        c1: *const u32,
        c2: *const u32,
        c3: *const u32,
        size: u32,
    ) -> CudaSecureField;

    pub fn memory_limb_split_big(
        values: *const u32,
        n_values: u32,
        column_length: u32,
        limb_cols: *const *const u32,
    );

    pub fn memory_limb_split_small(
        values: *const u32,
        n_values: u32,
        column_length: u32,
        limb_cols: *const *const u32,
    );

    pub fn memory_rc99_count(
        limb_cols: *const *const u32,
        n_pairs: u32,
        column_length: u32,
        input_to_row_lut: *const u32,
        rc_table_size: u32,
        counts: *mut u32,
    );

    pub fn memory_logup_inputs(
        limb_cols: *const *const u32,
        n_limbs: u32,
        mults: *const u32,
        relation_id: u32,
        id_offset: u32,
        id_tag: u32,
        column_length: u32,
        alpha_powers: *const u32,
        z: CudaSecureField,
        denoms: *const u32,
        num0: *const u32,
        num1: *const u32,
        num2: *const u32,
        num3: *const u32,
    );

    pub fn memory_rc_pair_logup(
        limb_a: *const u32,
        limb_b: *const u32,
        limb_c: *const u32,
        limb_d: *const u32,
        rel_id0: u32,
        rel_id1: u32,
        column_length: u32,
        alpha_powers: *const u32,
        z: CudaSecureField,
        denoms: *const u32,
        num0: *const u32,
        num1: *const u32,
        num2: *const u32,
        num3: *const u32,
    );

    pub fn addr_to_id_pair_logup(
        id0: *const u32,
        mult0: *const u32,
        id1: *const u32,
        mult1: *const u32,
        rel_id: u32,
        addr0_base: u32,
        addr1_base: u32,
        column_length: u32,
        alpha_powers: *const u32,
        z: CudaSecureField,
        denoms: *const u32,
        num0: *const u32,
        num1: *const u32,
        num2: *const u32,
        num3: *const u32,
    );

    pub fn logup_shift_secure_coords(
        c0: *const u32,
        c1: *const u32,
        c2: *const u32,
        c3: *const u32,
        shift: CudaSecureField,
        size: u32,
    );

    // Poseidon252 CUDA acceleration functions
    // Note: FieldElement252 is represented as 32 bytes (8 x u32)
    // Note: Poseidon252Hash is 32-byte struct, equivalent to [u8; 32]
    pub fn cuda_malloc_poseidon252_hash(size: usize) -> *mut [u8; 32];

    pub fn cuda_alloc_zeroes_poseidon252_hash(size: usize) -> *mut [u8; 32];

    pub fn copy_poseidon252_hash_vec_from_host_to_device(
        from: *const [u8; 32],
        size: usize,
    ) -> *mut [u8; 32];

    pub fn copy_poseidon252_hash_vec_from_device_to_host(
        from: *const [u8; 32],
        to: *mut [u8; 32],
        size: usize,
    );

    pub fn copy_poseidon252_hash_vec_from_device_to_device(
        from: *const [u8; 32],
        dst: *mut [u8; 32],
        size: usize,
    );

    pub fn cuda_get_poseidon252_hash(
        device_ptr: *const [u8; 32],
        host_ptr: *mut [u8; 32],
        index: usize,
    );

    pub fn cuda_set_poseidon252_hash(
        device_ptr: *mut [u8; 32],
        index: usize,
        value: *const [u8; 32],
    );

    // Hybrid Poseidon252 Merkle functions that use GPU for data processing
    // and CPU for verified hashing
    // GPU-only aliases matching Blake2s interface
    pub fn poseidon252_commit_on_first_layer(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        result: *mut [u8; 32],
    );

    pub fn poseidon252_commit_on_layer_with_previous(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        previous_layer: *const [u8; 32],
        result: *mut [u8; 32],
    );

    // Test function to compute offset_bit_reversed_circle_domain_index on GPU
    pub fn test_offset_bit_reversed_indices(
        result_host: *mut u32,
        domain_log_size: u32,
        eval_log_size: u32,
        offset: i32,
        n: u32,
    );
}
