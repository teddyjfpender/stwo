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

/// Process-local provenance counters for generated CUDA kernels. Strict
/// GPU-native admission requires `aot_misses == runtime_loads ==
/// strict_rejections == 0`; cache hits retain their original provenance.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct CudaJitAotStats {
    pub aot_loads: u64,
    pub aot_cache_hits: u64,
    pub aot_misses: u64,
    pub runtime_loads: u64,
    pub runtime_cache_hits: u64,
    pub strict_rejections: u64,
}

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
    /// Allocation-free explicit-stream form used by resident composition graphs.
    #[allow(clippy::too_many_arguments)]
    pub fn stwo_cuda_jit_eval_fused_on(
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
        stream: *mut c_void,
    ) -> bool;
    /// Generate `[alpha^(count-1), ..., alpha, 1]` from a device-resident
    /// transcript parameter on the caller's stream.
    pub fn stwo_composition_generate_descending_powers_on(
        random_coefficient: *const CudaSecureField,
        powers: *mut CudaSecureField,
        count: u32,
        stream: *mut c_void,
    ) -> i32;
    /// Lift the smaller coordinate-major secure evaluation into the larger
    /// bit-reversed circle domain and add it in place, on `stream`.
    pub fn stwo_composition_lift_accumulate_on(
        previous_coordinates: *const u32,
        previous_log_size: u32,
        current_coordinates: *mut u32,
        current_log_size: u32,
        stream: *mut c_void,
    ) -> i32;
    /// Materialize statement-dependent extension parameters into stable
    /// per-component destinations. Source kinds are 0 = z, 1 = alpha power,
    /// and 2 = claimed sum; every result is multiplied by its M31 scale.
    /// `claimed_sums` may be null exactly when `claimed_sum_count` is zero.
    #[allow(clippy::too_many_arguments)]
    pub fn stwo_composition_materialize_ext_params_on(
        destinations: *const *mut CudaSecureField,
        source_kinds: *const u32,
        source_indices: *const u32,
        scales: *const u32,
        count: u32,
        z: *const CudaSecureField,
        alpha_powers: *const CudaSecureField,
        alpha_power_count: u32,
        claimed_sums: *const *const CudaSecureField,
        claimed_sum_count: u32,
        stream: *mut c_void,
    ) -> i32;
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
    /// Process-wide fail-closed policy for generated kernels. Set before any
    /// proof work; when true, an absent/unloadable embedded AOT entry is an
    /// error and NVRTC/disk-PTX paths are not entered.
    pub fn stwo_cuda_jit_set_require_aot(required: bool);
    pub fn stwo_cuda_jit_get_aot_stats(out: *mut CudaJitAotStats);
    /// Reset counters only. Cached functions and their AOT/runtime provenance
    /// remain intact, so subsequent cache-hit accounting stays truthful.
    pub fn stwo_cuda_jit_reset_aot_stats();
    /// Per-component constraint-quotient kernel dispatch (NitrooZK lineage). The first
    /// 4 bytes behind `eval` are an FNV1a hash of the component name selecting the
    /// kernel; the rest is the raw `FrameworkEval` struct the kernel's generated code
    /// reads. Returns false when no kernel matches (caller falls back to CPU).
    #[allow(clippy::too_many_arguments)]
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

    /// Allocation-free, explicit-stream multi-column gather. All descriptor
    /// arrays and the output are device-resident; returns a CUDA status.
    pub fn stwo_batch_gather_column_rows_launch(
        columns_device: *const *const u32,
        row_offsets_device: *const u32,
        row_indices_device: *const u32,
        n_columns: u32,
        total_rows: u32,
        output_device: *mut u32,
        stream: *mut c_void,
    ) -> i32;

    /// Host compatibility wrapper: uploads explicit descriptor arrays, performs
    /// one gather launch, and copies the flattened output D2H once.
    pub fn stwo_batch_gather_column_rows_host(
        columns_host: *const *const u32,
        column_lengths_host: *const u32,
        row_offsets_host: *const u32,
        row_indices_host: *const u32,
        n_columns: u32,
        total_rows: u32,
        output_host: *mut u32,
    ) -> i32;

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
    pub fn cuda_pool_highwater_reset();

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

    pub fn barycentric_eval_base_field_many(
        columns_dev: *const *const u32,
        n_cols: u32,
        weights: *const u32,
        size: u32,
        out_host: *mut CudaSecureField,
    );

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

    /// Allocation-free explicit-stream FRI folds. Pointer tables are device
    /// buffers and remain caller-owned for the full capture/replay lifetime.
    pub fn stwo_fold_line_on(
        gpu_domain: *const u32,
        twiddle_offset: u32,
        n: u32,
        eval_values: *const *mut u32,
        alpha: *const CudaSecureField,
        alpha_squarings: u32,
        folded_values: *const *mut u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn stwo_fold_circle_into_line_on(
        gpu_domain: *const u32,
        twiddle_offset: u32,
        n: u32,
        eval_values: *const *mut u32,
        alpha: *const CudaSecureField,
        alpha_squarings: u32,
        folded_values: *const *mut u32,
        stream: *mut c_void,
    ) -> i32;

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

    /// Streaming leaf commit (VRAM diet): the lifted first layer split so the
    /// caller LDEs base columns one group at a time. init -> update per group
    /// -> finalize; `state` holds h[8] per leaf. Byte-identical to
    /// `commit_on_first_layer_lifted`.
    pub fn stream_leaf_init(size: u32, state: *mut Blake2sHash);
    pub fn stream_leaf_update(
        size: u32,
        group_n_cols: u32,
        columns: *const *const u32,
        column_log_sizes: *const u32,
        lifting_log_size: u32,
        cols_done: u32,
        state: *mut Blake2sHash,
    );
    /// ILP2 leaf-update lane (Step 3.2, opt-in `STWO_CUDA_BLAKE2S_LEAF_ILP=1`):
    /// one thread hashes TWO adjacent rows with interleaved G-function streams,
    /// halving the grid. Byte-identical to `stream_leaf_update`; odd row counts
    /// hash the final unpaired row through the scalar stream.
    pub fn stream_leaf_update_ilp2(
        size: u32,
        group_n_cols: u32,
        columns: *const *const u32,
        column_log_sizes: *const u32,
        lifting_log_size: u32,
        cols_done: u32,
        state: *mut Blake2sHash,
    );
    pub fn stream_leaf_finalize(
        size: u32,
        rem_cols: u32,
        columns: *const *const u32,
        column_log_sizes: *const u32,
        lifting_log_size: u32,
        cols_done: u32,
        result: *mut Blake2sHash,
    );

    /// Allocation-free explicit-stream commit-island kernels. Pointer tables,
    /// state, scratch, and outputs are caller-owned device buffers.
    pub fn stwo_blake2s_leaf_init_on(
        size: u32,
        state: *mut Blake2sHash,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn stwo_blake2s_leaf_update_on(
        size: u32,
        group_n_cols: u32,
        columns: *const *mut u32,
        column_log_sizes: *const u32,
        lifting_log_size: u32,
        cols_done: u32,
        state: *mut Blake2sHash,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// ILP2 explicit-stream twin of `stwo_blake2s_leaf_update_on` (same
    /// contract; two rows per thread, halved grid, byte-identical digests).
    pub fn stwo_blake2s_leaf_update_ilp2_on(
        size: u32,
        group_n_cols: u32,
        columns: *const *mut u32,
        column_log_sizes: *const u32,
        lifting_log_size: u32,
        cols_done: u32,
        state: *mut Blake2sHash,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn stwo_blake2s_leaf_finalize_on(
        size: u32,
        rem_cols: u32,
        columns: *const *mut u32,
        column_log_sizes: *const u32,
        lifting_log_size: u32,
        cols_done: u32,
        result: *mut Blake2sHash,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// Complete each column's final circle butterfly inside the leaf-hash
    /// kernel instead of materializing and rereading the completed LDE.
    #[allow(clippy::too_many_arguments)]
    pub fn stwo_blake2s_leaf_group_from_lde_on(
        size: u32,
        group_n_cols: u32,
        prefinal_columns: *const *mut u32,
        column_log_sizes: *const u32,
        lifting_log_size: u32,
        cols_done: u32,
        is_final: u32,
        twiddles: *mut u32,
        twiddle_words: u32,
        state: *mut Blake2sHash,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn stwo_blake2s_layer_on(
        previous_layer: *const Blake2sHash,
        output_size: u32,
        result: *mut Blake2sHash,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// Four column-free interior Merkle levels in ONE launch (Step 3.2 fused
    /// interior lane, opt-in via `STWO_CUDA_BLAKE2S_INTERIOR_FUSED=1`).
    /// `previous_layer` holds `16 * output_size` child digests; `result[i]` is
    /// the level-4 ancestor of children `16i..16i+16`. Intermediate levels stay
    /// in shared memory and are never written to global. Byte-identical to four
    /// sequential `stwo_blake2s_layer_on` launches; buffers must not alias.
    pub fn stwo_blake2s_interior4_on(
        previous_layer: *const Blake2sHash,
        output_size: u32,
        result: *mut Blake2sHash,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// Hash four QM31 coordinate columns in the exact unpacked/packed FRI leaf
    /// byte order without materializing packed columns.
    pub fn stwo_blake2s_fri_leaf_on(
        evaluation_size: u32,
        coordinate_columns: *const *mut u32,
        log_rows_per_leaf: u32,
        result: *mut Blake2sHash,
        stream: *mut core::ffi::c_void,
    ) -> i32;

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

    /// Allocation-free quotient combination on an explicit proof stream.
    pub fn stwo_combine_quotients_from_numerators_on(
        half_coset_initial_index: u32,
        half_coset_step_size: u32,
        domain_size: u32,
        domain_log_size: u32,
        sample_points: *const u32,
        sample_size: u32,
        first_linear_term_accs: *const CudaSecureField,
        partial_numerator_log_sizes: *const u32,
        partial_numerators_0: *const *const u32,
        partial_numerators_1: *const *const u32,
        partial_numerators_2: *const *const u32,
        partial_numerators_3: *const *const u32,
        result_column_0: *mut u32,
        result_column_1: *mut u32,
        result_column_2: *mut u32,
        result_column_3: *mut u32,
        denominator_inverses: *mut u32,
        denominator_count: u64,
        stream: *mut c_void,
    ) -> i32;

    pub fn stwo_prepare_quotient_numerator_terms_on(
        term_descriptors: *const u32,
        term_count: u32,
        sample_points: *const u32,
        sample_values: *const CudaSecureField,
        random_coefficient: *const CudaSecureField,
        term_points: *mut u32,
        line_coefficients: *mut CudaSecureField,
        stream: *mut c_void,
    ) -> i32;

    pub fn stwo_finalize_quotient_numerator_groups_on(
        group_offsets: *const u32,
        group_term_indices: *const u32,
        group_count: u32,
        term_points: *const u32,
        line_coefficients: *const CudaSecureField,
        sample_points: *mut u32,
        first_linear_terms: *mut CudaSecureField,
        stream: *mut c_void,
    ) -> i32;

    pub fn stwo_zero_quotient_numerator_outputs_on(
        group_log_sizes: *const u32,
        group_count: u32,
        max_output_size: u32,
        outputs_0: *const *mut u32,
        outputs_1: *const *mut u32,
        outputs_2: *const *mut u32,
        outputs_3: *const *mut u32,
        stream: *mut c_void,
    ) -> i32;

    pub fn stwo_accumulate_quotient_numerator_batch_on(
        group_offsets: *const u32,
        term_descriptors: *const u32,
        group_count: u32,
        max_output_size: u32,
        source_evaluations: *const *const u32,
        line_coefficients: *const CudaSecureField,
        group_log_sizes: *const u32,
        outputs_0: *const *mut u32,
        outputs_1: *const *mut u32,
        outputs_2: *const *mut u32,
        outputs_3: *const *mut u32,
        stream: *mut c_void,
    ) -> i32;

    /// Allocation-free coefficient-form OODS evaluation on an explicit proof stream.
    pub fn stwo_oods_derive_points_on(
        oods_parameter: *const CudaSecureField,
        offset_points: *const CirclePointBaseField,
        fold_counts: *const u32,
        output_indices: *const u32,
        sample_count: u32,
        coefficient_log_size: u32,
        sample_points: *mut u32,
        evaluation_points: *mut u32,
        folding_factors: *mut CudaSecureField,
        stream: *mut c_void,
    ) -> i32;

    pub fn stwo_oods_eval_first_on(
        coefficients: *const *const u32,
        coefficient_size: u32,
        sample_count: u32,
        folding_factors: *const CudaSecureField,
        scratch: *mut CudaSecureField,
        stream: *mut c_void,
    ) -> i32;

    pub fn stwo_oods_eval_reduce_on(
        input: *const CudaSecureField,
        input_size: u32,
        input_stride: u32,
        factor_index: u32,
        coefficient_log_size: u32,
        sample_count: u32,
        folding_factors: *const CudaSecureField,
        output: *mut CudaSecureField,
        output_stride: u32,
        stream: *mut c_void,
    ) -> i32;

    pub fn stwo_oods_store_results_on(
        reduced: *const CudaSecureField,
        reduced_stride: u32,
        output_indices: *const u32,
        sample_count: u32,
        sampled_values: *mut CudaSecureField,
        stream: *mut c_void,
    ) -> i32;

    pub fn stwo_oods_barycentric_weights_on(
        half_coset_initial_index: u32,
        half_coset_step_size: u32,
        size: u32,
        log_size: u32,
        evaluation_point: *const u32,
        si0: CudaSecureField,
        vanishing_rotation: CirclePointBaseField,
        numerator_inverses: *mut CudaSecureField,
        weights: *mut CudaSecureField,
        scales: *mut CudaSecureField,
        stream: *mut c_void,
    ) -> i32;

    pub fn stwo_oods_barycentric_eval_many_on(
        columns: *const *const u32,
        column_count: u32,
        weights: *const CudaSecureField,
        size: u32,
        partial_sums: *mut CudaSecureField,
        reduction_blocks: u32,
        output_indices: *const u32,
        sampled_values: *mut CudaSecureField,
        stream: *mut c_void,
    ) -> i32;

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

    /// Allocation-free B2N transform using a device pointer table and explicit stream.
    pub fn stwo_ntt_b2n_columns_on(
        device_values: *const *mut u32,
        log_n: u32,
        num_poly: u32,
        g_twiddles: *mut u32,
        twiddles_size: u32,
        eval_domain_size: u32,
        stream: *mut c_void,
    ) -> i32;

    pub fn ntt_n2b_columns(
        values_columns: *mut *mut u32,
        log_n: u32,
        num_poly: u32,
        g_twiddles: *const u32,
        twiddles_size: u32,
        eval_domain_size: u32,
    );

    /// Allocation-free N2B transform. `device_values` is already a
    /// device-resident pointer table and every launch uses `stream`.
    pub fn stwo_ntt_n2b_columns_on(
        device_values: *const *mut u32,
        log_n: u32,
        num_poly: u32,
        g_twiddles: *mut u32,
        twiddles_size: u32,
        eval_domain_size: u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Allocation-free LDE. Pointer and exact coefficient-size tables are
    /// device-resident; staging and N2B use `stream`.
    pub fn stwo_lde_n2b_columns_on(
        coefficient_values: *const *const u32,
        coefficient_sizes: *const u32,
        device_values: *const *mut u32,
        log_n: u32,
        num_poly: u32,
        g_twiddles: *mut u32,
        twiddles_size: u32,
        eval_domain_size: u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Allocation-free LDE prefix ending immediately before the final circle
    /// butterfly; consumed by `stwo_blake2s_leaf_group_from_lde_on`.
    pub fn stwo_lde_n2b_columns_before_circle_on(
        coefficient_values: *const *const u32,
        coefficient_sizes: *const u32,
        device_values: *const *mut u32,
        log_n: u32,
        num_poly: u32,
        g_twiddles: *mut u32,
        twiddles_size: u32,
        eval_domain_size: u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Setup-time dynamic-shared-memory admission for the producer-fused
    /// 16-column N2B→Blake kernel selected by `log_n`.
    pub fn stwo_lde_n2b_hash16_configure(log_n: u32) -> i32;

    /// Stage and transform 16 same-log full-lifting columns, feeding final N2B
    /// values directly from registers/shared memory into leaf states.
    #[allow(clippy::too_many_arguments)]
    pub fn stwo_lde_n2b_hash16_on(
        coefficient_values: *const *const u32,
        coefficient_sizes: *const u32,
        device_values: *const *mut u32,
        log_n: u32,
        g_twiddles: *mut u32,
        twiddles_size: u32,
        eval_domain_size: u32,
        cols_done: u32,
        is_final: u32,
        states: *mut Blake2sHash,
        stream: *mut core::ffi::c_void,
    ) -> i32;

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
        n_real: u32,
        descs_host: *const u32,
        n_cols: u32,
        alphas_host: *const u32,
        n_alphas: u32,
        z_host: *const u32,
        num_cols_device_table: *const *mut u32,
        den_dense_device_table: *const *mut u32,
    ) -> bool;

    /// Exact CUB storage query used during prepared relation setup.
    pub fn stwo_relation_scan_temp_bytes(len: u32) -> usize;

    /// Expand the transcript draw `[z, alpha]` into z plus alpha powers on the
    /// explicit proof stream.
    pub fn stwo_relation_expand_challenges_on(
        drawn_z_alpha: *const u32,
        alpha_powers: *mut u32,
        n_alpha_powers: u32,
        z: *mut u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Allocation-free, stream-explicit generated relation pair engine.
    pub fn stwo_relation_pairs_on(
        sources: *const *const u32,
        n_sources: u32,
        n_rows: u32,
        n_real: u32,
        source_offset_rows: u32,
        descriptors: *const u32,
        n_columns: u32,
        alpha_powers: *const u32,
        n_alpha_powers: u32,
        z: *const u32,
        outputs: *const *mut u32,
        denominators: *mut u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn stwo_relation_pairs_global_on(
        source_tables: *const *const *const u32,
        descriptors: *const *const u32,
        output_tables: *const *const *mut u32,
        denominator_slabs: *const *mut u32,
        geometry: *const u32,
        n_instances: u32,
        total_pair_blocks: u32,
        alpha_powers: *const u32,
        n_alpha_powers: u32,
        z: *const u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Fused pairs -> single-inversion -> fraction-chain lane: one proof-wide
    /// launch over the shared 11-word geometry records. `eligible_mask` is a
    /// HOST pointer to 8 words (256 instance bits) copied into the by-value
    /// kernel parameter before launch; ineligible instances are skipped and
    /// must be executed by the caller on the 3-stage path.
    pub fn stwo_relation_fused_on(
        source_tables: *const *const *const u32,
        descriptors: *const *const u32,
        output_tables: *const *const *mut u32,
        geometry: *const u32,
        n_instances: u32,
        total_row_blocks: u32,
        alpha_powers: *const u32,
        n_alpha_powers: u32,
        z: *const u32,
        eligible_mask: *const u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn stwo_relation_fraction_chain_on(
        outputs: *const *mut u32,
        denominators: *mut u32,
        inverse_scratch: *mut u32,
        n_rows: u32,
        n_columns: u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn stwo_relation_reduce_shift_on(
        output_0: *mut u32,
        output_1: *mut u32,
        output_2: *mut u32,
        output_3: *mut u32,
        n_rows: u32,
        reduction_a: *mut u32,
        reduction_b: *mut u32,
        reduction_capacity: u32,
        claimed_sum: *mut u32,
        inverse_rows: u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn stwo_relation_prefix_scan_on(
        output: *mut u32,
        n_rows: u32,
        eval_scratch: *mut u32,
        scan_temp: *mut core::ffi::c_void,
        scan_temp_bytes: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn stwo_relation_tail_global_on(
        output_tables: *const *const *mut u32,
        claimed_sums: *const *mut u32,
        geometry: *const u32,
        n_instances: u32,
        total_row_blocks: u32,
        reduction_partials: *mut u32,
        reduction_capacity: u32,
        scan_block_sums: *mut u32,
        scan_capacity: u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

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

    pub fn memory_limb_split_big_into_on(
        values: *const u32,
        n_values: u32,
        column_length: u32,
        limb_cols_host: *const *mut u32,
        mults_host: *const u32,
        mults: *mut u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn stwo_relation_fraction_chain_global_on(
        output_tables: *const *const *mut u32,
        denominator_slabs: *const *mut u32,
        geometry: *const u32,
        n_instances: u32,
        total_inverse_blocks: u32,
        total_chain_blocks: u32,
        stream: *mut c_void,
    ) -> i32;

    pub fn memory_limb_split_small_into_on(
        values: *const u32,
        n_values: u32,
        column_length: u32,
        limb_cols_host: *const *mut u32,
        mults_host: *const u32,
        mults: *mut u32,
        stream: *mut c_void,
    ) -> i32;

    pub fn memory_limb_split_big_columns_on(
        values: *const u32,
        n_values: u32,
        column_length: u32,
        limb_cols_host: *const *mut u32,
        stream: *mut c_void,
    ) -> i32;

    pub fn memory_limb_split_small_columns_on(
        values: *const u32,
        n_values: u32,
        column_length: u32,
        limb_cols_host: *const *mut u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memory_address_base_trace_on(
        raw_addr_to_id: *const u32,
        n_addrs: u32,
        multiplicities: *const u32,
        count_words: u32,
        column_length: u32,
        outputs_host: *const *mut u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn memory_value_base_trace_on(
        sources_host: *const *const u32,
        n_limbs: u32,
        source_words: u32,
        source_offset: u32,
        multiplicities: *const u32,
        count_words: u32,
        column_length: u32,
        outputs_host: *const *mut u32,
        stream: *mut c_void,
    ) -> i32;

    /// Capture-safe native Cairo `ec_op_builtin` writer. The execution-table
    /// pointer table is device-resident; every destination points into the
    /// proof arena.
    #[allow(clippy::too_many_arguments)]
    pub fn ec_op_builtin_witness_on(
        execution_tables: *const *const u32,
        n_addresses: u32,
        n_big: u32,
        n_small: u32,
        segment_start_source: *const u32,
        row_count: u32,
        trace_columns_host: *const *mut u32,
        lookup_words: *mut u32,
        partial_input_columns_host: *const *mut u32,
        partial_row_count: u32,
        address_counts: *mut u32,
        address_count_words: u32,
        big_counts: *mut u32,
        big_count_words: u32,
        small_counts: *mut u32,
        small_count_words: u32,
        range_check_8_counts: *mut u32,
        range_check_8_count_words: u32,
        stream: *mut c_void,
    ) -> i32;

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

    /// Allocation-free arena-native blake_g writer. Exactly one of `inputs`
    /// (row-major) and `producer_sub` (blake_round word-major edge) is non-null.
    /// `trace_cols_host` is a host array of 53 device addresses copied into the
    /// kernel argument; lookup/sub are canonical word-major flat outputs.
    pub fn blake_g_write_trace_into_on(
        inputs: *const u32,
        producer_sub: *const u32,
        producer_rows: u32,
        producer_word_base: u32,
        producer_instances: u32,
        n_rows: u32,
        column_length: u32,
        trace_cols_host: *const *mut u32,
        lookup: *mut u32,
        sub: *mut u32,
        stream: *mut c_void,
    ) -> i32;

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

    // Resource-owning execution context for one resident proof (design §19,
    // cuda_exec_context.cu): an owned non-blocking stream + its own never-release
    // memory pool. Every function returns a CUDA status (0 = success); context
    // creation fails closed when an isolated pool cannot be created.
    pub fn stwo_exec_context_create(out_handle: *mut *mut core::ffi::c_void) -> i32;
    pub fn stwo_exec_context_destroy(handle: *mut core::ffi::c_void) -> i32;
    pub fn stwo_exec_context_sync(handle: *mut core::ffi::c_void) -> i32;
    pub fn stwo_exec_context_stream_sync(
        handle: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn stwo_exec_context_stream(
        handle: *mut core::ffi::c_void,
        out_stream: *mut *mut core::ffi::c_void,
    ) -> i32;
    pub fn stwo_exec_context_lane_count(handle: *mut core::ffi::c_void, out_count: *mut u32)
        -> i32;
    pub fn stwo_exec_context_lane_stream(
        handle: *mut core::ffi::c_void,
        lane: u32,
        out_stream: *mut *mut core::ffi::c_void,
    ) -> i32;
    pub fn stwo_exec_context_lane_fork(handle: *mut core::ffi::c_void, lane: u32) -> i32;
    pub fn stwo_exec_context_lane_join(handle: *mut core::ffi::c_void, lane: u32) -> i32;
    pub fn stwo_exec_context_alloc_u32(
        handle: *mut core::ffi::c_void,
        count: usize,
        out_ptr: *mut *mut u32,
    ) -> i32;
    pub fn stwo_exec_context_free_u32(handle: *mut core::ffi::c_void, ptr: *mut u32) -> i32;
    pub fn stwo_exec_context_memset_async(
        handle: *mut core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        value: i32,
        bytes: usize,
    ) -> i32;
    pub fn stwo_exec_context_fill_u32_async(
        handle: *mut core::ffi::c_void,
        dst: *mut u32,
        value: u32,
        count: usize,
    ) -> i32;
    pub fn stwo_exec_context_memcpy_d2d_async(
        handle: *mut core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        bytes: usize,
    ) -> i32;
    pub fn stwo_exec_context_memcpy_h2d_async(
        handle: *mut core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        bytes: usize,
    ) -> i32;
    pub fn stwo_exec_context_memcpy_d2h_async(
        handle: *mut core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        bytes: usize,
    ) -> i32;

    // Opaque CUDA graph lifecycle rooted on the context main stream. Explicit
    // proof-owned lane fork/join edges admit auxiliary streams into the same
    // captured segment; host transcript work remains outside.
    pub fn stwo_graph_capture_begin(handle: *mut core::ffi::c_void) -> i32;
    pub fn stwo_graph_capture_end(
        handle: *mut core::ffi::c_void,
        out_exec: *mut *mut core::ffi::c_void,
        out_kernel_nodes: *mut u64,
    ) -> i32;
    pub fn stwo_graph_capture_abort(handle: *mut core::ffi::c_void) -> i32;
    pub fn stwo_graph_launch(
        exec_handle: *mut core::ffi::c_void,
        context_handle: *mut core::ffi::c_void,
    ) -> i32;
    pub fn stwo_graph_destroy(exec_handle: *mut core::ffi::c_void) -> i32;

    // Ordinary-Blake2s Fiat-Shamir transcript kernels. The 16-word state,
    // mirror snapshots, all sources, and all outputs are caller-owned device
    // arena ranges. Every operation is enqueued on the explicit proof stream.
    pub fn stwo_blake2s_transcript_init_on(
        state: *mut u32,
        seed: *const u32,
        seed_snapshot: *mut u32,
        initial_chain: u64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn stwo_blake2s_transcript_mix_words_on(
        state: *mut u32,
        expected_step: u32,
        expected_chain: u64,
        next_chain: u64,
        source: *const u32,
        n_words: u32,
        validate_m31: u32,
        input_snapshot: *mut u32,
        boundary_snapshot: *mut u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn stwo_blake2s_transcript_absorb_pow_on(
        state: *mut u32,
        expected_step: u32,
        expected_chain: u64,
        next_chain: u64,
        nonce_words: *const u32,
        pow_bits: u32,
        input_snapshot: *mut u32,
        boundary_snapshot: *mut u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn stwo_blake2s_transcript_draw_u32s_on(
        state: *mut u32,
        expected_step: u32,
        expected_chain: u64,
        next_chain: u64,
        output: *mut u32,
        output_snapshot: *mut u32,
        boundary_snapshot: *mut u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn stwo_blake2s_transcript_draw_secure_on(
        state: *mut u32,
        expected_step: u32,
        expected_chain: u64,
        next_chain: u64,
        n_felts: u32,
        max_rejection_rounds: u32,
        output: *mut u32,
        output_snapshot: *mut u32,
        boundary_snapshot: *mut u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn stwo_blake2s_transcript_draw_queries_on(
        state: *mut u32,
        expected_step: u32,
        expected_chain: u64,
        next_chain: u64,
        log_domain_size: u32,
        n_queries: u32,
        output: *mut u32,
        output_snapshot: *mut u32,
        boundary_snapshot: *mut u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

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
    // args are DEVICE pointers; descs is the flat 14-u32-stride descriptor
    // array. Returns 0 on success.
    // Commit fusion (C2): the top K Merkle levels in one launch (see the tail
    // kernel in blake2s.cu). out_levels is a DEVICE array of per-level output
    // pointers; level l holds first_size >> (l+1) hashes.
    pub fn stwo_blake2s_tail(
        first_dev: *const Blake2sHash,
        first_size: u32,
        out_levels_dev: *const *mut Blake2sHash,
        n_levels: u32,
    ) -> i32;
    pub fn stwo_blake2s_tail_on(
        first_dev: *const Blake2sHash,
        first_size: u32,
        out_levels_dev: *const *mut Blake2sHash,
        n_levels: u32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    // Device edge (B3): blake_round's sub buffer -> blake_g's row-major input
    // buffer (the certified hand lane's ABI). See blake_witness.cu.
    pub fn stwo_blake_g_inputs_from_sub(
        producer_sub_dev: *const u32,
        producer_rows: u32,
        word_base: u32,
        n_instances: u32,
        consumer_rows: u32,
        out_row_major_dev: *mut u32,
    ) -> i32;

    // Device DAG (B3): gather a consumer's input columns from a producer's
    // word-major sub buffer (see witness_edge_gather.cu). Padding rows
    // replicate the first packed row, matching the host resize rule.
    pub fn stwo_witness_edge_gather(
        producer_sub_dev: *const u32,
        producer_rows: u32,
        word_base: u32,
        words_per_instance: u32,
        n_instances: u32,
        consumer_rows: u32,
        consumer_cols_dev: *const *mut u32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn stwo_witness_input_gather_on(
        producer_subs_dev: *const *const u32,
        edge_descs_dev: *const u32,
        n_edges: u32,
        input_width: u32,
        total_real_rows: u32,
        consumer_rows: u32,
        consumer_cols_dev: *const *mut u32,
        include_enabler: u32,
        include_iota: u32,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn stwo_witness_input_seed_on(
        scalars_dev: *const u32,
        n_scalars: u32,
        n_real_rows: u32,
        consumer_rows: u32,
        consumer_cols_dev: *const *mut u32,
        include_enabler: u32,
        include_iota: u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn stwo_witness_input_compact_sort_temp_bytes(rows: u32) -> usize;
    pub fn stwo_witness_input_compact_scan_temp_bytes(rows: u32) -> usize;
    #[allow(clippy::too_many_arguments)]
    pub fn stwo_witness_input_compact_on(
        producer_subs_dev: *const *const u32,
        edge_descs_dev: *const u32,
        n_edges: u32,
        tuple_words: u32,
        key_words: u32,
        total_rows: u32,
        sort_rows: u32,
        consumer_rows: u32,
        n_inputs: u32,
        consumer_cols_dev: *const *mut u32,
        enabler_slot: u32,
        iota_slot: u32,
        multiplicity_slot: u32,
        tuples_dev: *mut u32,
        keys_a_dev: *mut u32,
        keys_b_dev: *mut u32,
        indices_a_dev: *mut u32,
        indices_b_dev: *mut u32,
        heads_dev: *mut u32,
        positions_dev: *mut u32,
        n_unique_dev: *mut u32,
        sort_temp_dev: *mut c_void,
        sort_temp_bytes: usize,
        scan_temp_dev: *mut c_void,
        scan_temp_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Capture-safe fixed-table BaseTrace/LookupInputs materialization from
    /// stable device descriptor and pointer tables.
    #[allow(clippy::too_many_arguments)]
    pub fn stwo_fixed_table_materialize_on(
        source_columns_dev: *const *const u32,
        multiplicity_columns_dev: *const *const u32,
        trace_multiplicity_columns_dev: *const u32,
        trace_outputs_dev: *const *mut u32,
        n_trace_outputs: u32,
        lookup_descriptors_dev: *const u32,
        lookup_outputs_dev: *const *mut u32,
        n_lookup_outputs: u32,
        row_count: u32,
        stream: *mut c_void,
    ) -> i32;

    pub fn stwo_witness_feed_counts(
        sub_words_dev: *const u32,
        column_length: u32,
        descs_dev: *const u32,
        n_descs: u32,
        luts_dev: *const *const u32,
        counts_dev: *const *mut u32,
    ) -> i32;
    pub fn stwo_witness_feed_counts_on(
        sub_words_dev: *const u32,
        column_length: u32,
        descs_dev: *const u32,
        n_descs: u32,
        luts_dev: *const *const u32,
        counts_dev: *const *mut u32,
        stream: *mut c_void,
    ) -> i32;
    // Step 4.3 privatized variant of the count feed: per-block shared-memory
    // histograms for descriptors whose touched table footprint fits 48KB
    // static shared, block-local atomics, unconditional global merge;
    // oversized families keep the global-atomic path inside the same launch.
    // Byte-identical count slabs (wrapping u32 adds are commutative and
    // associative). Same ABI as stwo_witness_feed_counts[_on].
    pub fn stwo_witness_feed_counts_privatized(
        sub_words_dev: *const u32,
        column_length: u32,
        descs_dev: *const u32,
        n_descs: u32,
        luts_dev: *const *const u32,
        counts_dev: *const *mut u32,
    ) -> i32;
    pub fn stwo_witness_feed_counts_privatized_on(
        sub_words_dev: *const u32,
        column_length: u32,
        descs_dev: *const u32,
        n_descs: u32,
        luts_dev: *const *const u32,
        counts_dev: *const *mut u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn stwo_witness_feed_clear_on(
        destinations_dev: *const *mut u32,
        lengths_dev: *const u32,
        n_destinations: u32,
        max_words: u32,
        stream: *mut c_void,
    ) -> i32;

    /// Allocation-free final line interpolation, degree validation, and direct
    /// row-major transcript-input emission.
    pub fn stwo_fri_last_layer_on(
        evaluation: *const u32,
        evaluation_stride: u32,
        log_size: u32,
        inverse_twiddles: *const u32,
        inverse_twiddle_words: u32,
        log_degree_bound: u32,
        coefficients: *mut u32,
        degree_error: *mut u32,
        transcript_coefficients: *mut u32,
        stream: *mut c_void,
    ) -> i32;

    /// Persistent, globally minimal numeric-u64 Blake2s nonce search from the
    /// current device transcript state.
    pub fn stwo_blake2s_pow_persistent_on(
        transcript_state: *const u32,
        pow_bits: u32,
        best_nonce: *mut u64,
        completed_blocks: *mut u32,
        transcript_nonce: *mut u32,
        stream: *mut c_void,
    ) -> i32;

    pub fn stwo_blake2s_sparse_leaf_group_on(
        leaf_indices: *const u32,
        leaf_count: *const u32,
        max_leaf_count: u32,
        group_n_cols: u32,
        columns: *const *mut u32,
        column_log_sizes: *const u32,
        lifting_log_size: u32,
        cols_done: u32,
        is_final: u32,
        states: *mut Blake2sHash,
        stream: *mut c_void,
    ) -> i32;

    pub fn stwo_decommit_normalize_queries_on(
        raw_queries: *const u32,
        raw_query_count: u32,
        query_log_size: u32,
        tree_count: u32,
        unique_queries: *mut u32,
        unique_count: *mut u32,
        assembly: *mut u32,
        assembly_capacity_words: u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn stwo_decommit_prepare_trace_queries_on(
        unique_queries: *const u32,
        unique_count: *const u32,
        max_queries: u32,
        source_log_size: u32,
        tree_log_size: u32,
        leaf_log_size: u32,
        unretained_bottom_layers: u32,
        mapped_queries: *mut u32,
        mapped_count: *mut u32,
        walk_queries: *mut u32,
        walk_count: *mut u32,
        leaf_indices: *mut u32,
        leaf_count: *mut u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn stwo_decommit_gather_trace_values_on(
        columns: *const *const u32,
        column_log_sizes: *const u32,
        column_count: u32,
        lifting_log_size: u32,
        mapped_queries: *const u32,
        mapped_count: *const u32,
        max_queries: u32,
        destination_first_column: u32,
        destination_stride: u32,
        queried_values: *mut u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn stwo_decommit_sparse_parent_on(
        child_indices: *const u32,
        child_hashes: *const Blake2sHash,
        child_count: *const u32,
        max_child_count: u32,
        parent_indices: *mut u32,
        parent_hashes: *mut Blake2sHash,
        parent_count: *mut u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn stwo_decommit_assemble_trace_on(
        tree_index: u32,
        tree_role: u32,
        leaf_log_size: u32,
        first_retained_log_size: u32,
        column_count: u32,
        mapped_queries: *const u32,
        mapped_count: *const u32,
        max_queries: u32,
        walk_queries: *mut u32,
        walk_scratch: *mut u32,
        walk_count: *const u32,
        queried_values: *const u32,
        retained_layers_by_log: *const *const Blake2sHash,
        sparse_indices: *const u32,
        sparse_hashes: *const Blake2sHash,
        sparse_level_offsets: *const u32,
        sparse_level_counts: *const u32,
        sparse_level_count: u32,
        assembly: *mut u32,
        assembly_capacity_words: u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn stwo_decommit_prepare_fri_queries_on(
        unique_queries: *const u32,
        unique_count: *const u32,
        max_queries: u32,
        cumulative_fold: u32,
        fold_step: u32,
        log_rows_per_leaf: u32,
        tree_queries: *mut u32,
        tree_query_count: *mut u32,
        expanded_positions: *mut u32,
        expanded_count: *mut u32,
        walk_queries: *mut u32,
        walk_count: *mut u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn stwo_decommit_gather_fri_values_on(
        coordinate_columns: *const *const u32,
        expanded_positions: *const u32,
        expanded_count: *const u32,
        max_expanded_positions: u32,
        expanded_values: *mut u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn stwo_decommit_assemble_fri_on(
        tree_index: u32,
        leaf_log_size: u32,
        tree_queries: *const u32,
        tree_query_count: *const u32,
        expanded_positions: *const u32,
        expanded_count: *const u32,
        expanded_values: *const u32,
        walk_queries: *mut u32,
        walk_scratch: *mut u32,
        walk_count: *const u32,
        retained_layers_by_log: *const *const Blake2sHash,
        assembly: *mut u32,
        assembly_capacity_words: u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn stwo_relation_scan_tail_on(
        output_tables: *const *const *mut u32,
        claimed_sums: *const *mut u32,
        geometry: *const u32,
        n_instances: u32,
        total_row_blocks: u32,
        partition_descriptors: *mut u32,
        descriptor_capacity_words: u32,
        stream: *mut c_void,
    ) -> i32;
    // Fused FRI triple fold (Step 3.4, `fri_fold_fused.cu`): one 8-to-1 kernel
    // replacing the three per-fold launches of a full `fold_step == 3` round.
    // Byte-identical to that sequence; see the kernel-file comment.
    pub fn stwo_fri_fold_fused3_on(
        gpu_domain: *const u32,
        twiddle_offset_0: u32,
        twiddle_offset_1: u32,
        twiddle_offset_2: u32,
        n: u32,
        first_fold_is_circle: u32,
        eval_values: *const *mut u32,
        alpha: *const CudaSecureField,
        folded_values: *const *mut u32,
        stream: *mut c_void,
    ) -> i32;
}

// --- ntt_leaf_fused ---
#[cfg_attr(stwo_cuda_link, link(name = "stwo_cuda_kernels", kind = "static"))]
extern "C" {
    /// Setup-time dynamic-shared-memory admission for the retained
    /// write+hash final N2B kernel selected by `log_n` (`ntt_leaf_fused.cu`).
    pub fn stwo_ntt_leaf_fused_configure(log_n: u32) -> i32;

    /// Retained twin of [`stwo_lde_n2b_hash16_on`]: stage and transform 16
    /// same-log full-lifting columns, WRITE the completed evaluations into
    /// `device_values` (kept resident for decommitment) AND absorb the same
    /// final tile into the leaf states — zero evaluation re-read for hashing.
    #[allow(clippy::too_many_arguments)]
    pub fn stwo_ntt_leaf_fused_on(
        coefficient_values: *const *const u32,
        coefficient_sizes: *const u32,
        device_values: *const *mut u32,
        log_n: u32,
        g_twiddles: *mut u32,
        twiddles_size: u32,
        eval_domain_size: u32,
        cols_done: u32,
        is_final: u32,
        states: *mut Blake2sHash,
        stream: *mut c_void,
    ) -> i32;
}
