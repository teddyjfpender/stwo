//! Raw FFI declarations for the staged CUDA kernel entry points.
//!
//! Link-gated: with `stwo_cuda_link` (set by the build script when nvcc compiled the
//! kernels) these resolve to the static archive; otherwise `stubs.rs` provides
//! panicking `no_mangle` definitions so the crate links everywhere.

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

#[cfg_attr(stwo_cuda_link, link(name = "stwo_cuda_kernels", kind = "static"))]
extern "C" {
    /// Returns a CUDA error code (0 = success). Sets the default mem pool's release
    /// threshold to never-release so warm proves reuse allocations.
    pub fn cuda_mem_pool_init() -> i32;
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
    /// launch a generated fused constraint kernel. `rc_base` is the kernel's first
    /// constraint's global index into `random_coeff_powers` (non-zero only for split
    /// kernels); `relax_opt` compiles with optimization disabled (nvrtc --dopt=off,
    /// ptxas -O0) for kernels too large to optimize in reasonable time. Returns false
    /// on any compile or launch failure; the caller falls back to the CPU lane.
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
        rc_base: u32,
        relax_opt: bool,
    ) -> bool;
    /// Compile a generated kernel into the JIT cache WITHOUT launching it. Used to
    /// compile every kernel of a split component before the first launch, so a
    /// compile failure can still fall back to the CPU lane with an untouched
    /// accumulator. Returns false on compile failure.
    pub fn stwo_cuda_jit_precompile(
        source: *const core::ffi::c_char,
        kernel_name: *const core::ffi::c_char,
        semantic_hash: u64,
        relax_opt: bool,
    ) -> bool;
    /// Precompile a batch of kernels into the JIT cache without launching, compiling
    /// across a worker pool when `STWO_JIT_PARALLEL_COMPILE` is enabled (default; set to
    /// `0` to force sequential, or to a positive integer to cap workers). The arrays are
    /// parallel (length `count`): `sources[i]`/`kernel_names[i]` are NUL-terminated,
    /// `cache_keys[i]` is the content semantic hash, `relax_opts[i]` the optimization
    /// relief flag. Returns false if ANY kernel fails to compile (caller falls back to
    /// the CPU lane). The populated cache is identical to a sequential precompile.
    pub fn stwo_cuda_jit_precompile_batch(
        sources: *const *const core::ffi::c_char,
        kernel_names: *const *const core::ffi::c_char,
        cache_keys: *const u64,
        relax_opts: *const bool,
        count: u32,
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

    pub fn copy_uint32_t_vec_from_host_to_device_into_async(
        host_ptr: *const u32,
        device_ptr: *const u32,
        n_words: u64,
    );

    pub fn stwo_legacy_stream_sync();

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
    pub fn cuda_pool_highwater(used_high: *mut usize, reserved_high: *mut usize);

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

    pub fn commit_on_layer_with_previous(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        previous_layer: *const Blake2sHash,
        result: *mut Blake2sHash,
    );

    /// Workstream D layer-pair fusion: hash two internal (column-free) tree levels
    /// per launch. `size` = grandparent hash count; `previous_layer` holds 4*size.
    pub fn commit_on_two_layers_with_previous(
        size: usize,
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

    /// Device logup pair generation from word-major witness-lane lookup flats
    /// (ENDGAME 6a). See cuda/logup_pairs.cu for the descriptor layout.
    pub fn stwo_logup_pairs_from_flats(
        flats: *const u32,
        n_rows: u32,
        descs_host: *const u32,
        n_cols: u32,
        alphas_host: *const u32,
        n_alphas: u32,
        z_host: *const u32,
        num_cols_device_table: *const *mut u32,
        den_dense_device_table: *const *mut u32,
    ) -> bool;

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

    /// Composed device `deduce_output` (ENDGAME §2 keystone): for each queried address,
    /// read the raw encoded id from `addr_to_id`, decode the tag, and gather the 28
    /// 9-bit value limbs from the device-resident big/small split tables — the device
    /// equivalent of the host `memory_address_to_id` + `memory_id_to_big` deduction.
    /// `big_limbs`/`small_limbs`/`out_limbs` are device-resident arrays of device
    /// column pointers (28 / 8 / 28). Addresses must be non-empty cells.
    pub fn exec_deduce_output(
        addr_to_id: *const u32,
        big_limbs: *const *const u32,
        small_limbs: *const *const u32,
        addresses: *const u32,
        n_queries: u32,
        out_ids: *mut u32,
        out_limbs: *const *mut u32,
    );

    pub fn logup_shift_secure_coords(
        c0: *const u32,
        c1: *const u32,
        c2: *const u32,
        c3: *const u32,
        shift: CudaSecureField,
        size: u32,
    );

    pub fn blake_g_write_trace(
        inputs: *const u32,
        n_rows: u32,
        column_length: u32,
        cols: *const *const u32,
    );

    pub fn blake_g_xor_count(
        a_cols: *const *const u32,
        b_cols: *const *const u32,
        rel_idx: *const u32,
        n_pairs: u32,
        column_length: u32,
        shift: u32,
        lut: *const u32,
        table_size: u32,
        counts: *mut u32,
    );

    pub fn blake_g_xor12_count(
        a_cols: *const *const u32,
        b_cols: *const *const u32,
        n_pairs: u32,
        column_length: u32,
        limb_bits: u32,
        expand_bits: u32,
        table_size: u32,
        counts: *mut u32,
    );

    pub fn blake_g_pair_logup(
        a0: *const u32,
        b0: *const u32,
        x0: *const u32,
        a1: *const u32,
        b1: *const u32,
        x1: *const u32,
        rel0: u32,
        rel1: u32,
        column_length: u32,
        alpha: *const u32,
        z: CudaSecureField,
        denoms: *const u32,
        num0: *const u32,
        num1: *const u32,
        num2: *const u32,
        num3: *const u32,
    );

    pub fn blake_g_final_logup(
        val_cols: *const *const u32,
        enabler: *const u32,
        rel: u32,
        column_length: u32,
        alpha: *const u32,
        z: CudaSecureField,
        denoms: *const u32,
        num0: *const u32,
        num1: *const u32,
        num2: *const u32,
        num3: *const u32,
    );

    // Pedersen family witness-on-GPU (partial_ec_mul / pedersen_aggregator).
    // See `cuda/pedersen_witness.cu` for the scope contract; these are the
    // reusable interaction/logup kernels (base-trace gadget kernels land per
    // component on hardware behind the STWO_CUDA_WITNESS_VERIFY differential).
    #[allow(clippy::too_many_arguments)]
    pub fn pedersen_pair_logup(
        vals0: *const *const u32,
        rel0: u32,
        vals1: *const *const u32,
        rel1: u32,
        n_vals: u32,
        m0: *const u32,
        m1: *const u32,
        sign0: i32,
        sign1: i32,
        column_length: u32,
        alpha: *const u32,
        z: CudaSecureField,
        denoms: *const u32,
        num0: *const u32,
        num1: *const u32,
        num2: *const u32,
        num3: *const u32,
    );

    #[allow(clippy::too_many_arguments)]
    pub fn pedersen_multi_logup(
        vals: *const *const u32,
        n_vals: u32,
        rel: u32,
        mult: *const u32,
        neg_num: i32,
        column_length: u32,
        alpha: *const u32,
        z: CudaSecureField,
        denoms: *const u32,
        num0: *const u32,
        num1: *const u32,
        num2: *const u32,
        num3: *const u32,
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

    /// Witness-JIT lane: NVRTC-compile (cached by `cache_key`, the CONTENT hash — never
    /// pointers) and launch a generated per-row witness kernel. The ABI matches
    /// `stwo-backend-cuda::backend::jit_witness::codegen`: one thread per row reads the
    /// packed input columns, replays the recorded decode in registers, writes committed
    /// trace columns, atomic-adds multiplicities, and stores lookup words.
    ///
    /// Pointer tables are device-resident arrays of device pointers (the pointer-table
    /// trace ABI — no flatten copies, no u32 length overflow at log >= 23). `relax_opt`
    /// compiles with optimization disabled for oversized kernels. Returns false on any
    /// compile or launch failure; the caller falls back to the host writer with an
    /// untouched output (this lane is default OFF and pod-gated — see
    /// `STWO_CUDA_WITNESS_JIT`).
    #[allow(clippy::too_many_arguments)]
    pub fn stwo_cuda_jit_witness_launch(
        source: *const core::ffi::c_char,
        kernel_name: *const core::ffi::c_char,
        cache_key: u64,
        input_cols: *const *const u32,
        table_bases: *const *const u32,
        table_strides: *const u32,
        out_cols: *const *mut u32,
        mult_counts: *const *mut u32,
        lookup_words: *mut u32,
        sub_words: *mut u32,
        row_count: u32,
        relax_opt: bool,
        // Stage B′: stream to launch on. Null = legacy default stream (pre-B′).
        stream: *mut core::ffi::c_void,
    ) -> bool;

    // Stage B′ fan-out primitives (see cuda_mem_pool.cu). `stwo_fanout_stream`
    // returns pool stream `i` (round-robin) as an opaque handle; `fork`/`join`
    // are the thread-safe (fresh-event) bridges around a lane's stream work.
    pub fn stwo_fanout_stream(i: i32) -> *mut core::ffi::c_void;
    pub fn stwo_fanout_fork(stream: *mut core::ffi::c_void);
    pub fn stwo_fanout_join(stream: *mut core::ffi::c_void);

    // Truth oracle for the witness-JIT computed EC deduces (ISA-V3 kinds 2/3):
    // runs the exact `stwo_wit_deduce_*` device functions the JIT kernels embed
    // from a precompiled kernel (see `stwo_wit_deduce_oracle.cu`), so the pod
    // ladder can compare against the host `fast_deduction` before trusting any
    // JIT kernel. Buffers are flat per the recorder shapes (kind 2: 72->72
    // words per item; kind 3: 1->56). Returns 0 on success; nonzero means "no
    // data" (unknown kind, table init failure, CUDA error) — never zeros.
    pub fn stwo_wit_deduce_oracle_run(
        kind: u32,
        h_in: *const u32,
        h_out: *mut u32,
        n_items: u32,
    ) -> i32;

    // Register caller-owned DEVICE columns as the pedersen points table
    // (borrowed mode; see pedersen_table_init.cu). 56 pointers, n_rows each.
    // Registration publishes the pointers to the precompiled module's device
    // globals; asserts if a different table was already registered.
    pub fn pedersen_table_init(columns: *const *mut u32, n_rows: u32);

    // Device DAG (B2): generalized multiplicity count feed over a witness
    // kernel's word-major sub buffer (see witness_feed_counts.cu). All pointer
    // args are DEVICE pointers; descs is the flat 11-u32-stride descriptor
    // array. Returns 0 on success.
    pub fn stwo_witness_feed_counts(
        sub_words_dev: *const u32,
        column_length: u32,
        descs_dev: *const u32,
        n_descs: u32,
        luts_dev: *const *const u32,
        counts_dev: *const *mut u32,
    ) -> i32;
}
