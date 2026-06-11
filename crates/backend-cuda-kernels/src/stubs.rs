//! Panicking `no_mangle` stand-ins for every kernel entry point, used when the build
//! script did not find nvcc (`stwo_cuda_link` unset). Generated mechanically from the
//! declarations in `raw.rs`; keep the two files in sync.
#![allow(unused_variables, clippy::missing_safety_doc)]

use core::ffi::c_void;

use crate::raw::{Blake2sHash, CirclePointBaseField, CudaSecureField, LayerIndexPair};

#[cold]
fn no_cuda_symbol(symbol: &str) -> ! {
    panic!(
        "CUDA kernel entry point `{symbol}` was called, but the kernels were not compiled \
         into this build (nvcc was not found). Build on a machine with the CUDA toolkit."
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn copy_uint32_t_vec_from_device_to_host(
    device_ptr: *const u32,
    host_ptr: *const u32,
    size: u32,
) {
    no_cuda_symbol("copy_uint32_t_vec_from_device_to_host")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn copy_uint32_t_vec_from_host_to_device(
    host_ptr: *const u32,
    size: u32,
) -> *const u32 {
    no_cuda_symbol("copy_uint32_t_vec_from_host_to_device")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn copy_uint32_t_vec_from_device_to_device(
    from: *const u32,
    dst: *const u32,
    size: u32,
) -> *const u32 {
    no_cuda_symbol("copy_uint32_t_vec_from_device_to_device")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn copy_uint32_t_vec_from_device_to_device_offset(
    from: *const u32,
    dst: *const u32,
    size: u32,
    offset: u32,
) {
    no_cuda_symbol("copy_uint32_t_vec_from_device_to_device_offset")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_zero_device_region(ptr: *const u32, offset_words: u64, n_words: u64) {
    let _ = (ptr, offset_words, n_words);
    no_cuda_symbol("cuda_zero_device_region")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_alloc_pinned_host_u32(n_words: u64) -> *mut u32 {
    let _ = n_words;
    no_cuda_symbol("cuda_alloc_pinned_host_u32")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_free_pinned_host_u32(ptr: *mut u32) {
    let _ = ptr;
    no_cuda_symbol("cuda_free_pinned_host_u32")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn copy_uint32_t_vec_from_host_to_device_into(
    host_ptr: *const u32,
    device_ptr: *const u32,
    n_words: u64,
) {
    let _ = (host_ptr, device_ptr, n_words);
    no_cuda_symbol("copy_uint32_t_vec_from_host_to_device_into")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_gather_uint32_t(
    device_src: *const u32,
    host_indices: *const u32,
    n_indices: u32,
    host_out: *mut u32,
) {
    let _ = (device_src, host_indices, n_indices, host_out);
    no_cuda_symbol("cuda_gather_uint32_t")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_malloc_uint32_t(size: u32) -> *const u32 {
    no_cuda_symbol("cuda_malloc_uint32_t")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_set_uint32_t(device_ptr: *const c_void, index: usize, val: u32) {
    no_cuda_symbol("cuda_set_uint32_t")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_get_uint32_t(device_ptr: *const c_void, index: usize) -> u32 {
    no_cuda_symbol("cuda_get_uint32_t")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_increase_at(device_ptr: *const c_void, addr: u32) {
    no_cuda_symbol("cuda_increase_at")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_get_secure_field(
    device_ptr: *const c_void,
    index: usize,
) -> CudaSecureField {
    no_cuda_symbol("cuda_get_secure_field")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_malloc_blake_2s_hash(size: usize) -> *const Blake2sHash {
    no_cuda_symbol("cuda_malloc_blake_2s_hash")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_alloc_zeroes_uint32_t(size: u32) -> *const u32 {
    no_cuda_symbol("cuda_alloc_zeroes_uint32_t")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_alloc_zeroes_blake_2s_hash(size: usize) -> *const Blake2sHash {
    no_cuda_symbol("cuda_alloc_zeroes_blake_2s_hash")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_free_memory(device_ptr: *const c_void) {
    no_cuda_symbol("cuda_free_memory")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_get_memory_info(free_mem: *mut usize, total_mem: *mut usize) {
    no_cuda_symbol("cuda_get_memory_info")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bit_reverse_base_field(array: *const u32, size: usize) {
    no_cuda_symbol("bit_reverse_base_field")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bit_reverse_secure_field(array: *const u32, size: usize) {
    no_cuda_symbol("bit_reverse_secure_field")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn batch_inverse_base_field(from: *const u32, dst: *const u32, size: usize) {
    no_cuda_symbol("batch_inverse_base_field")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sort_values_and_permute_with_bit_reverse_order(
    from: *const u32,
    size: usize,
) -> *const u32 {
    no_cuda_symbol("sort_values_and_permute_with_bit_reverse_order")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn precompute_twiddles(
    initial: CirclePointBaseField,
    step: CirclePointBaseField,
    total_size: usize,
) -> *const u32 {
    no_cuda_symbol("precompute_twiddles")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn evaluate_columns(
    eval_domain_sizes: *const u32,
    values: *const *const u32,
    twiddles_tree: *const u32,
    twiddle_tree_size: u32,
    number_of_columns: u32,
    column_sizes: *const u32,
) {
    no_cuda_symbol("evaluate_columns")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eval_at_point(
    coeffs: *const u32,
    coeffs_size: u32,
    point_x: CudaSecureField,
    point_y: CudaSecureField,
) -> CudaSecureField {
    no_cuda_symbol("eval_at_point")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn batch_eval_at_points(
    coeffs_ptrs: *const *const u32,
    coeffs_size: i32,
    num_polys: i32,
    point_x: CudaSecureField,
    point_y: CudaSecureField,
    results: *mut CudaSecureField,
) {
    no_cuda_symbol("batch_eval_at_points")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn barycentric_point_vanishings(
    half_coset_initial_index: u32,
    half_coset_step_size: u32,
    size: u32,
    log_size: u32,
    point_x: CudaSecureField,
    point_y: CudaSecureField,
    result: *const u32,
) {
    let _ = (
        half_coset_initial_index,
        half_coset_step_size,
        size,
        log_size,
        point_x,
        point_y,
        result,
    );
    no_cuda_symbol("barycentric_point_vanishings")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn barycentric_weights_from_point_vanishings(
    point_vanishings: *const u32,
    size: u32,
    even_scale: CudaSecureField,
    odd_scale: CudaSecureField,
    result_weights: *const u32,
) {
    no_cuda_symbol("barycentric_weights_from_point_vanishings")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn barycentric_eval_base_field(
    eval_values: *const u32,
    weights: *const u32,
    size: u32,
) -> CudaSecureField {
    no_cuda_symbol("barycentric_eval_base_field")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fold_line(
    gpu_domain: *const u32,
    twiddle_offset: usize,
    n: usize,
    eval_values: *const *const u32,
    alpha: CudaSecureField,
    folded_values: *const *const u32,
) {
    no_cuda_symbol("fold_line")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fold_circle_into_line(
    gpu_domain: *const u32,
    twiddle_offset: usize,
    n: usize,
    eval_values: *const *const u32,
    alpha: CudaSecureField,
    folded_values: *const *const u32,
) {
    no_cuda_symbol("fold_circle_into_line")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn accumulate(
    size: u32,
    left_columns: *const *const u32,
    right_columns: *const *const u32,
) {
    no_cuda_symbol("accumulate")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn lift_accumulate_secure_columns(
    size: u32,
    log_ratio: u32,
    previous_columns: *const *const u32,
    current_columns: *const *const u32,
) {
    no_cuda_symbol("lift_accumulate_secure_columns")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn commit_on_first_layer(
    size: usize,
    amount_of_columns: usize,
    columns: *const *const u32,
    result: *mut Blake2sHash,
) {
    no_cuda_symbol("commit_on_first_layer")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn commit_on_first_layer_lifted(
    size: usize,
    amount_of_columns: usize,
    columns: *const *const u32,
    column_log_sizes: *const u32,
    lifting_log_size: u32,
    result: *mut Blake2sHash,
) {
    no_cuda_symbol("commit_on_first_layer_lifted")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn commit_on_layer_with_previous(
    size: usize,
    amount_of_columns: usize,
    columns: *const *const u32,
    previous_layer: *const Blake2sHash,
    result: *mut Blake2sHash,
) {
    no_cuda_symbol("commit_on_layer_with_previous")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn copy_blake_2s_hash_vec_from_host_to_device(
    from: *const Blake2sHash,
    size: usize,
) -> *mut Blake2sHash {
    no_cuda_symbol("copy_blake_2s_hash_vec_from_host_to_device")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn copy_blake_2s_hash_vec_from_device_to_host(
    from: *const Blake2sHash,
    to: *mut Blake2sHash,
    size: usize,
) {
    no_cuda_symbol("copy_blake_2s_hash_vec_from_device_to_host")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn copy_blake_2s_hash_vec_from_device_to_device(
    from: *const Blake2sHash,
    dst: *const Blake2sHash,
    size: usize,
) {
    no_cuda_symbol("copy_blake_2s_hash_vec_from_device_to_device")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_get_blake_2s_hash(
    device_ptr: *const Blake2sHash,
    host_ptr: *mut Blake2sHash,
    index: usize,
) {
    no_cuda_symbol("cuda_get_blake_2s_hash")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_set_blake_2s_hash(
    device_ptr: *mut Blake2sHash,
    index: usize,
    host_ptr: *const Blake2sHash,
) {
    no_cuda_symbol("cuda_set_blake_2s_hash")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_batch_get_blake_2s_hash(
    device_ptr: *const Blake2sHash,
    host_ptr: *mut Blake2sHash,
    indices: *const u32,
    n_indices: u32,
) {
    no_cuda_symbol("cuda_batch_get_blake_2s_hash")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_multi_layer_batch_get_blake_2s_hash(
    layer_device_ptrs: *const *const Blake2sHash,
    host_ptr: *mut Blake2sHash,
    pairs: *const LayerIndexPair,
    n_pairs: u32,
) {
    no_cuda_symbol("cuda_multi_layer_batch_get_blake_2s_hash")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn copy_device_pointer_vec_from_host_to_device(
    from: *const *const u32,
    size: usize,
) -> *const *const u32 {
    no_cuda_symbol("copy_device_pointer_vec_from_host_to_device")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_release_uploaded_pointer_vec(device_ptr: *const *const u32) {
    no_cuda_symbol("cuda_release_uploaded_pointer_vec")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn accumulate_quotients(
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
) {
    no_cuda_symbol("accumulate_quotients")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn accumulate_partial_quotient_numerators(
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
) {
    no_cuda_symbol("accumulate_partial_quotient_numerators")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn combine_quotients_from_numerators(
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
) {
    no_cuda_symbol("combine_quotients_from_numerators")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gen_eq_evals(
    v: CudaSecureField,
    y: *const CudaSecureField,
    y_size: u32,
    evals: *const CudaSecureField,
    evals_size: u32,
) {
    no_cuda_symbol("gen_eq_evals")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gkr_next_grand_product_layer(
    input_layer: *const CudaSecureField,
    input_size: u32,
    output_layer: *const CudaSecureField,
) {
    no_cuda_symbol("gkr_next_grand_product_layer")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gkr_next_logup_generic_layer(
    numerators: *const CudaSecureField,
    denominators: *const CudaSecureField,
    input_size: u32,
    next_numerators: *const CudaSecureField,
    next_denominators: *const CudaSecureField,
) {
    no_cuda_symbol("gkr_next_logup_generic_layer")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gkr_next_logup_multiplicities_layer(
    numerators: *const u32,
    denominators: *const CudaSecureField,
    input_size: u32,
    next_numerators: *const CudaSecureField,
    next_denominators: *const CudaSecureField,
) {
    no_cuda_symbol("gkr_next_logup_multiplicities_layer")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gkr_next_logup_singles_layer(
    denominators: *const CudaSecureField,
    input_size: u32,
    next_numerators: *const CudaSecureField,
    next_denominators: *const CudaSecureField,
) {
    no_cuda_symbol("gkr_next_logup_singles_layer")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gkr_sum_grand_product(
    eq_evals: *const CudaSecureField,
    input_layer: *const CudaSecureField,
    n_terms: u32,
    eval_at_0: *mut CudaSecureField,
    eval_at_2: *mut CudaSecureField,
) {
    no_cuda_symbol("gkr_sum_grand_product")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gkr_sum_logup_generic(
    eq_evals: *const CudaSecureField,
    numerators: *const CudaSecureField,
    denominators: *const CudaSecureField,
    n_terms: u32,
    lambda: CudaSecureField,
    eval_at_0: *mut CudaSecureField,
    eval_at_2: *mut CudaSecureField,
) {
    no_cuda_symbol("gkr_sum_logup_generic")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gkr_sum_logup_multiplicities(
    eq_evals: *const CudaSecureField,
    numerators: *const u32,
    denominators: *const CudaSecureField,
    n_terms: u32,
    lambda: CudaSecureField,
    eval_at_0: *mut CudaSecureField,
    eval_at_2: *mut CudaSecureField,
) {
    no_cuda_symbol("gkr_sum_logup_multiplicities")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gkr_sum_logup_singles(
    eq_evals: *const CudaSecureField,
    denominators: *const CudaSecureField,
    n_terms: u32,
    lambda: CudaSecureField,
    eval_at_0: *mut CudaSecureField,
    eval_at_2: *mut CudaSecureField,
) {
    no_cuda_symbol("gkr_sum_logup_singles")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fix_first_variable_base_field(
    evals: *const u32,
    evals_size: usize,
    assignment: CudaSecureField,
    output_evals: *const u32,
) {
    no_cuda_symbol("fix_first_variable_base_field")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fix_first_variable_secure_field(
    evals: *const u32,
    evals_size: usize,
    assignment: CudaSecureField,
    output_evals: *const u32,
) {
    no_cuda_symbol("fix_first_variable_secure_field")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ntt_n2b_native_batch(
    value: *mut *mut u32,
    log_n: u32,
    num_poly: u32,
    start_stage: u32,
    end_stage: u32,
    g_twiddles: *const u32,
    twiddles_size: u32,
    eval_domain_size: u32,
) {
    no_cuda_symbol("ntt_n2b_native_batch")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ntt_b2n_column(
    values_columns: *mut *mut u32,
    log_n: u32,
    num_poly: u32,
    g_twiddles: *const u32,
    twiddles_size: u32,
    eval_domain_size: u32,
) {
    no_cuda_symbol("ntt_b2n_column")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ntt_n2b_columns(
    values_columns: *mut *mut u32,
    log_n: u32,
    num_poly: u32,
    g_twiddles: *const u32,
    twiddles_size: u32,
    eval_domain_size: u32,
) {
    no_cuda_symbol("ntt_n2b_columns")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_malloc_poseidon252_hash(size: usize) -> *mut [u8; 32] {
    no_cuda_symbol("cuda_malloc_poseidon252_hash")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_alloc_zeroes_poseidon252_hash(size: usize) -> *mut [u8; 32] {
    no_cuda_symbol("cuda_alloc_zeroes_poseidon252_hash")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn copy_poseidon252_hash_vec_from_host_to_device(
    from: *const [u8; 32],
    size: usize,
) -> *mut [u8; 32] {
    no_cuda_symbol("copy_poseidon252_hash_vec_from_host_to_device")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn copy_poseidon252_hash_vec_from_device_to_host(
    from: *const [u8; 32],
    to: *mut [u8; 32],
    size: usize,
) {
    no_cuda_symbol("copy_poseidon252_hash_vec_from_device_to_host")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn copy_poseidon252_hash_vec_from_device_to_device(
    from: *const [u8; 32],
    dst: *mut [u8; 32],
    size: usize,
) {
    no_cuda_symbol("copy_poseidon252_hash_vec_from_device_to_device")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_get_poseidon252_hash(
    device_ptr: *const [u8; 32],
    host_ptr: *mut [u8; 32],
    index: usize,
) {
    no_cuda_symbol("cuda_get_poseidon252_hash")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_set_poseidon252_hash(
    device_ptr: *mut [u8; 32],
    index: usize,
    value: *const [u8; 32],
) {
    no_cuda_symbol("cuda_set_poseidon252_hash")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn poseidon252_commit_on_first_layer(
    size: usize,
    amount_of_columns: usize,
    columns: *const *const u32,
    result: *mut [u8; 32],
) {
    no_cuda_symbol("poseidon252_commit_on_first_layer")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn poseidon252_commit_on_layer_with_previous(
    size: usize,
    amount_of_columns: usize,
    columns: *const *const u32,
    previous_layer: *const [u8; 32],
    result: *mut [u8; 32],
) {
    no_cuda_symbol("poseidon252_commit_on_layer_with_previous")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn test_offset_bit_reversed_indices(
    result_host: *mut u32,
    domain_log_size: u32,
    eval_log_size: u32,
    offset: i32,
    n: u32,
) {
    no_cuda_symbol("test_offset_bit_reversed_indices")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_mem_pool_init() -> i32 {
    no_cuda_symbol("cuda_mem_pool_init")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn grind_blake2s(_host_prefixed_digest: *const u32, _pow_bits: u32) -> u64 {
    no_cuda_symbol("grind_blake2s")
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn evaluate_constraint_quotients_on_domain(
    _q0: *const u32,
    _q1: *const u32,
    _q2: *const u32,
    _q3: *const u32,
    _t0: *const *const u32,
    _t0_len: u32,
    _t1: *const *const u32,
    _t1_len: u32,
    _t2: *const *const u32,
    _t2_len: u32,
    _rc: *const u32,
    _denom: *const u32,
    _domain_log_size: u32,
    _eval_domain_log_size: u32,
    _n_columns: u32,
    _logup_counts: u32,
    _eval: *mut c_void,
    _cumsum_shift: CudaSecureField,
    _should_accumulate: bool,
    _use_assert_evaluator: bool,
) -> bool {
    no_cuda_symbol("evaluate_constraint_quotients_on_domain")
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stwo_cuda_jit_eval_fused(
    _source: *const core::ffi::c_char,
    _kernel_name: *const core::ffi::c_char,
    _semantic_hash: u64,
    _trace_values: *const u32,
    _interaction_offsets: *const u32,
    _base_params: *const u32,
    _ext_params: *const u32,
    _random_coeff_powers: *const u32,
    _denom_inv: *const u32,
    _coord_0: *mut u32,
    _coord_1: *mut u32,
    _coord_2: *mut u32,
    _coord_3: *mut u32,
    _row_count: u32,
    _log_n_rows: u32,
) -> bool {
    no_cuda_symbol("stwo_cuda_jit_eval_fused")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gen_seq_column_on_gpu(_output: *mut u32, _log_size: u32) {
    no_cuda_symbol("gen_seq_column_on_gpu")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gen_range_check_columns_on_gpu(
    _output_columns: *const *mut u32,
    _n_columns: u32,
    _bits_per_segment: *const u32,
    _n_segments: u32,
) {
    no_cuda_symbol("gen_range_check_columns_on_gpu")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gen_bitwise_xor_columns_on_gpu(
    _output_columns: *const *mut u32,
    _n_bits: u32,
) {
    no_cuda_symbol("gen_bitwise_xor_columns_on_gpu")
}
