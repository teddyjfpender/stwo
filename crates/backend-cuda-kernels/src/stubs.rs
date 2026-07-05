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
pub unsafe extern "C" fn copy_uint32_t_vec_from_host_to_device_into_async(
    host_ptr: *const u32,
    device_ptr: *const u32,
    n_words: u64,
) {
    let _ = (host_ptr, device_ptr, n_words);
    no_cuda_symbol("copy_uint32_t_vec_from_host_to_device_into_async")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stwo_legacy_stream_sync() {
    no_cuda_symbol("stwo_legacy_stream_sync")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn inclusive_prefix_sum_x4(
    c0: *const u32,
    c1: *const u32,
    c2: *const u32,
    c3: *const u32,
    len: u32,
) {
    let _ = (c0, c1, c2, c3, len);
    no_cuda_symbol("inclusive_prefix_sum_x4")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn inclusive_prefix_sum(
    device_bit_rev_circle_domain_evals: *const u32,
    len: u32,
) {
    let _ = (device_bit_rev_circle_domain_evals, len);
    no_cuda_symbol("inclusive_prefix_sum")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn logup_fraction_chain(
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
) {
    let _ = (
        num0,
        num1,
        num2,
        num3,
        denom_packed,
        prev0,
        prev1,
        prev2,
        prev3,
        size,
    );
    no_cuda_symbol("logup_fraction_chain")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stwo_logup_pairs_from_flats(
    _flats: *const u32,
    _n_rows: u32,
    _descs_host: *const u32,
    _n_cols: u32,
    _alphas_host: *const u32,
    _n_alphas: u32,
    _z_host: *const u32,
    _num_cols_device_table: *const *mut u32,
    _den_dense_device_table: *const *mut u32,
) -> bool {
    let _ = (
        _flats,
        _n_rows,
        _descs_host,
        _n_cols,
        _alphas_host,
        _n_alphas,
        _z_host,
        _num_cols_device_table,
        _den_dense_device_table,
    );
    no_cuda_symbol("stwo_logup_pairs_from_flats")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn logup_fraction_chain_dense(
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
) {
    let _ = (
        num0,
        num1,
        num2,
        num3,
        denoms_dense,
        prev0,
        prev1,
        prev2,
        prev3,
        size,
    );
    no_cuda_symbol("logup_fraction_chain_dense")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn logup_sum_secure_coords(
    c0: *const u32,
    c1: *const u32,
    c2: *const u32,
    c3: *const u32,
    size: u32,
) -> CudaSecureField {
    let _ = (c0, c1, c2, c3, size);
    no_cuda_symbol("logup_sum_secure_coords")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memory_limb_split_big(
    values: *const u32,
    n_values: u32,
    column_length: u32,
    limb_cols: *const *const u32,
) {
    let _ = (values, n_values, column_length, limb_cols);
    no_cuda_symbol("memory_limb_split_big")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memory_limb_split_small(
    values: *const u32,
    n_values: u32,
    column_length: u32,
    limb_cols: *const *const u32,
) {
    let _ = (values, n_values, column_length, limb_cols);
    no_cuda_symbol("memory_limb_split_small")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memory_rc99_count(
    limb_cols: *const *const u32,
    n_pairs: u32,
    column_length: u32,
    input_to_row_lut: *const u32,
    rc_table_size: u32,
    counts: *mut u32,
) {
    let _ = (
        limb_cols,
        n_pairs,
        column_length,
        input_to_row_lut,
        rc_table_size,
        counts,
    );
    no_cuda_symbol("memory_rc99_count")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memory_logup_inputs(
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
) {
    let _ = (
        limb_cols,
        n_limbs,
        mults,
        relation_id,
        id_offset,
        id_tag,
        column_length,
    );
    let _ = (alpha_powers, z, denoms, num0, num1, num2, num3);
    no_cuda_symbol("memory_logup_inputs")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memory_rc_pair_logup(
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
) {
    let _ = (
        limb_a,
        limb_b,
        limb_c,
        limb_d,
        rel_id0,
        rel_id1,
        column_length,
    );
    let _ = (alpha_powers, z, denoms, num0, num1, num2, num3);
    no_cuda_symbol("memory_rc_pair_logup")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn exec_deduce_output(
    addr_to_id: *const u32,
    big_limbs: *const *const u32,
    small_limbs: *const *const u32,
    addresses: *const u32,
    n_queries: u32,
    out_ids: *mut u32,
    out_limbs: *const *mut u32,
) {
    let _ = (
        addr_to_id,
        big_limbs,
        small_limbs,
        addresses,
        n_queries,
        out_ids,
        out_limbs,
    );
    no_cuda_symbol("exec_deduce_output")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn logup_shift_secure_coords(
    c0: *const u32,
    c1: *const u32,
    c2: *const u32,
    c3: *const u32,
    shift: CudaSecureField,
    size: u32,
) {
    let _ = (c0, c1, c2, c3, shift, size);
    no_cuda_symbol("logup_shift_secure_coords")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn blake_g_write_trace(
    inputs: *const u32,
    n_rows: u32,
    column_length: u32,
    cols: *const *const u32,
) {
    let _ = (inputs, n_rows, column_length, cols);
    no_cuda_symbol("blake_g_write_trace")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn blake_g_xor_count(
    a_cols: *const *const u32,
    b_cols: *const *const u32,
    rel_idx: *const u32,
    n_pairs: u32,
    column_length: u32,
    shift: u32,
    lut: *const u32,
    table_size: u32,
    counts: *mut u32,
) {
    let _ = (
        a_cols,
        b_cols,
        rel_idx,
        n_pairs,
        column_length,
        shift,
        lut,
        table_size,
        counts,
    );
    no_cuda_symbol("blake_g_xor_count")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn blake_g_xor12_count(
    a_cols: *const *const u32,
    b_cols: *const *const u32,
    n_pairs: u32,
    column_length: u32,
    limb_bits: u32,
    expand_bits: u32,
    table_size: u32,
    counts: *mut u32,
) {
    let _ = (
        a_cols,
        b_cols,
        n_pairs,
        column_length,
        limb_bits,
        expand_bits,
        table_size,
        counts,
    );
    no_cuda_symbol("blake_g_xor12_count")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn blake_g_pair_logup(
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
) {
    let _ = (a0, b0, x0, a1, b1, x1, rel0, rel1, column_length);
    let _ = (alpha, z, denoms, num0, num1, num2, num3);
    no_cuda_symbol("blake_g_pair_logup")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn blake_g_final_logup(
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
) {
    let _ = (val_cols, enabler, rel, column_length);
    let _ = (alpha, z, denoms, num0, num1, num2, num3);
    no_cuda_symbol("blake_g_final_logup")
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn pedersen_pair_logup(
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
) {
    let _ = (
        vals0,
        rel0,
        vals1,
        rel1,
        n_vals,
        m0,
        m1,
        sign0,
        sign1,
        column_length,
    );
    let _ = (alpha, z, denoms, num0, num1, num2, num3);
    no_cuda_symbol("pedersen_pair_logup")
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn pedersen_multi_logup(
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
) {
    let _ = (vals, n_vals, rel, mult, neg_num, column_length);
    let _ = (alpha, z, denoms, num0, num1, num2, num3);
    no_cuda_symbol("pedersen_multi_logup")
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
pub unsafe extern "C" fn cuda_pool_highwater(used_high: *mut usize, reserved_high: *mut usize) {
    no_cuda_symbol("cuda_pool_highwater")
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
pub unsafe extern "C" fn commit_on_two_layers_with_previous(
    size: usize,
    previous_layer: *const Blake2sHash,
    result: *mut Blake2sHash,
) {
    no_cuda_symbol("commit_on_two_layers_with_previous")
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
    _rc_base: u32,
    _relax_opt: bool,
) -> bool {
    no_cuda_symbol("stwo_cuda_jit_eval_fused")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stwo_cuda_jit_precompile(
    _source: *const core::ffi::c_char,
    _kernel_name: *const core::ffi::c_char,
    _semantic_hash: u64,
    _relax_opt: bool,
) -> bool {
    no_cuda_symbol("stwo_cuda_jit_precompile")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stwo_cuda_jit_precompile_batch(
    _sources: *const *const core::ffi::c_char,
    _kernel_names: *const *const core::ffi::c_char,
    _cache_keys: *const u64,
    _relax_opts: *const bool,
    _count: u32,
) -> bool {
    no_cuda_symbol("stwo_cuda_jit_precompile_batch")
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

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stwo_cuda_jit_witness_launch(
    _source: *const core::ffi::c_char,
    _kernel_name: *const core::ffi::c_char,
    _cache_key: u64,
    _input_cols: *const *const u32,
    _table_bases: *const *const u32,
    _table_strides: *const u32,
    _out_cols: *const *mut u32,
    _mult_counts: *const *mut u32,
    _lookup_words: *mut u32,
    sub_words: *mut u32,
    _row_count: u32,
    _relax_opt: bool,
    _stream: *mut core::ffi::c_void,
) -> bool {
    no_cuda_symbol("stwo_cuda_jit_witness_launch")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stwo_fanout_stream(_i: i32) -> *mut core::ffi::c_void {
    no_cuda_symbol("stwo_fanout_stream")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stwo_fanout_fork(_stream: *mut core::ffi::c_void) {
    no_cuda_symbol("stwo_fanout_fork")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stwo_fanout_join(_stream: *mut core::ffi::c_void) {
    no_cuda_symbol("stwo_fanout_join")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stwo_wit_deduce_oracle_run(
    _kind: u32,
    _h_in: *const u32,
    _h_out: *mut u32,
    _n_items: u32,
) -> i32 {
    no_cuda_symbol("stwo_wit_deduce_oracle_run")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pedersen_table_init(_columns: *const *mut u32, _n_rows: u32) {
    no_cuda_symbol("pedersen_table_init")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn stwo_witness_feed_counts(
    _sub_words_dev: *const u32,
    _column_length: u32,
    _descs_dev: *const u32,
    _n_descs: u32,
    _luts_dev: *const *const u32,
    _counts_dev: *const *mut u32,
) -> i32 {
    no_cuda_symbol("stwo_witness_feed_counts")
}
