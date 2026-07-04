//! Device witness generation for the Cairo `blake_g` component (witness-on-GPU
//! W3 phase 2 — the BLAKE g-function family, the most GPU-natural port).
//!
//! Safe wrappers over the `blake_witness.cu` kernels: the stwo-cairo blake_g
//! component calls these to generate its 53 base-trace columns, feed the five
//! `verify_bitwise_xor_*` multiplicity families, and produce its 9 logup
//! interaction columns entirely on device — no host columns, no `lookup_data`.
//!
//! Every formula is a port of the generated SIMD writer (see the spec in the
//! stwo-cairo fork's `gpu_benchmarks/WITNESS_ON_GPU.md`); the authoritative gates
//! are the component differential (`STWO_CUDA_WITNESS_VERIFY`) and the Cairo e2e
//! proof byte-equality, both in stwo-cairo. The finalize lane and the
//! `DeviceRawLogupColumn` layout are shared with the `memory_witness` device lane.

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;

use crate::backend::memory_witness::DeviceRawLogupColumn;
use crate::backend::UploadedDevicePointerVec;
use crate::columns::bindings::{self, CudaSecureField};
use crate::columns::{BaseFieldVec, SecureFieldVec};

/// Number of committed base-trace columns (cairo-air blake_g N_TRACE_COLUMNS).
pub const BG_N_TRACE: usize = 53;
/// Number of auxiliary operand columns emitted alongside the trace (split low
/// parts + rot7/rot8 limbs the interaction and count feeds need).
pub const BG_N_AUX: usize = 20;
/// Total columns written by [`write_trace`] (0..53 trace, 53..73 aux).
pub const BG_N_COLS: usize = BG_N_TRACE + BG_N_AUX;

/// Generates the 73 device-resident blake_g columns from the raw input words.
///
/// `inputs` is `column_length * 6` raw u32 words, row-major (the 6 blake_g input
/// words per row, padding rows carrying the host's replicated first input).
/// `n_rows` is the number of real (non-padding) rows (the enabler cutoff).
pub fn write_trace(
    inputs: &BaseFieldVec,
    n_rows: usize,
    column_length: usize,
) -> Vec<BaseFieldVec> {
    bindings::ensure_mem_pool_init();
    let cols: Vec<BaseFieldVec> = (0..BG_N_COLS)
        .map(|_| BaseFieldVec::new_uninitialized(column_length))
        .collect();
    let ptrs: Vec<*const u32> = cols.iter().map(|c| c.device_ptr).collect();
    let table = UploadedDevicePointerVec::upload(&ptrs);
    unsafe {
        stwo_backend_cuda_kernels::raw::blake_g_write_trace(
            inputs.device_ptr,
            n_rows as u32,
            column_length as u32,
            table.as_ptr(),
        );
    }
    cols
}

/// Uploads a dense `(a << shift) | b -> row` LUT (row indices are `< P`, so the
/// M31 representation is exact) for the device xor multiplicity feed.
fn upload_lut(lut: &[u32]) -> BaseFieldVec {
    BaseFieldVec::from_vec(
        lut.iter()
            .map(|&r| BaseField::from_u32_unchecked(r))
            .collect(),
    )
}

/// Counts the xor multiplicities for one `verify_bitwise_xor_*` family, leaving
/// the count table **on device**.
///
/// For each `(a_cols[k], b_cols[k])` column pair, over every row, atomically adds
/// 1 into `counts[rel_idx[k] * table_size + lut[(a << shift) | b]]`. Returns the
/// device-resident `n_relations * table_size` count table.
///
/// The stored u32 counts have the exact wrap-around semantics of the host
/// `AtomicMultiplicityColumn` (`fetch_add`, no field reduction), so the count
/// buffer IS the component's multiplicity column bytes — [`xor_mult_columns`]
/// slices it into the base trace with no D2H at all (the device-resident xor
/// witness lane, kills the count-table D2H).
#[allow(clippy::too_many_arguments)]
pub fn xor_count_device(
    a_cols: &[&BaseFieldVec],
    b_cols: &[&BaseFieldVec],
    rel_idx: &[u32],
    column_length: usize,
    shift: u32,
    lut: &[u32],
    n_relations: usize,
    table_size: usize,
) -> BaseFieldVec {
    assert_eq!(a_cols.len(), b_cols.len());
    assert_eq!(a_cols.len(), rel_idx.len());
    let a_ptrs: Vec<*const u32> = a_cols.iter().map(|c| c.device_ptr).collect();
    let b_ptrs: Vec<*const u32> = b_cols.iter().map(|c| c.device_ptr).collect();
    let a_table = UploadedDevicePointerVec::upload(&a_ptrs);
    let b_table = UploadedDevicePointerVec::upload(&b_ptrs);
    let rel = BaseFieldVec::from_vec(
        rel_idx
            .iter()
            .map(|&r| BaseField::from_u32_unchecked(r))
            .collect(),
    );
    let lut_dev = upload_lut(lut);
    let counts = BaseFieldVec::new_zeroes(n_relations * table_size);
    unsafe {
        stwo_backend_cuda_kernels::raw::blake_g_xor_count(
            a_table.as_ptr(),
            b_table.as_ptr(),
            rel.device_ptr,
            a_cols.len() as u32,
            column_length as u32,
            shift,
            lut_dev.device_ptr,
            table_size as u32,
            counts.device_ptr.cast_mut(),
        );
    }
    counts
}

/// Host-merge variant of [`xor_count_device`]: downloads the count table for the
/// host to fold into the component's atomic multiplicities (order-independent —
/// byte-equal by construction). Used when the xor component still runs on the
/// host path (device-resident xor lane disabled or feeders not all on device).
#[allow(clippy::too_many_arguments)]
pub fn xor_count(
    a_cols: &[&BaseFieldVec],
    b_cols: &[&BaseFieldVec],
    rel_idx: &[u32],
    column_length: usize,
    shift: u32,
    lut: &[u32],
    n_relations: usize,
    table_size: usize,
) -> Vec<u32> {
    xor_count_device(
        a_cols,
        b_cols,
        rel_idx,
        column_length,
        shift,
        lut,
        n_relations,
        table_size,
    )
    .to_vec()
    .into_iter()
    .map(|f| f.0)
    .collect()
}

/// Counts the xor_12 multiplicities (expanded table, closed-form indexing — no
/// LUT), leaving the count table **on device**. Returns the device-resident
/// `n_mult_columns * table_size` count table.
#[allow(clippy::too_many_arguments)]
pub fn xor12_count_device(
    a_cols: &[&BaseFieldVec],
    b_cols: &[&BaseFieldVec],
    column_length: usize,
    limb_bits: u32,
    expand_bits: u32,
    n_mult_columns: usize,
    table_size: usize,
) -> BaseFieldVec {
    assert_eq!(a_cols.len(), b_cols.len());
    let a_ptrs: Vec<*const u32> = a_cols.iter().map(|c| c.device_ptr).collect();
    let b_ptrs: Vec<*const u32> = b_cols.iter().map(|c| c.device_ptr).collect();
    let a_table = UploadedDevicePointerVec::upload(&a_ptrs);
    let b_table = UploadedDevicePointerVec::upload(&b_ptrs);
    let counts = BaseFieldVec::new_zeroes(n_mult_columns * table_size);
    unsafe {
        stwo_backend_cuda_kernels::raw::blake_g_xor12_count(
            a_table.as_ptr(),
            b_table.as_ptr(),
            a_cols.len() as u32,
            column_length as u32,
            limb_bits,
            expand_bits,
            table_size as u32,
            counts.device_ptr.cast_mut(),
        );
    }
    counts
}

/// Host-merge variant of [`xor12_count_device`].
#[allow(clippy::too_many_arguments)]
pub fn xor12_count(
    a_cols: &[&BaseFieldVec],
    b_cols: &[&BaseFieldVec],
    column_length: usize,
    limb_bits: u32,
    expand_bits: u32,
    n_mult_columns: usize,
    table_size: usize,
) -> Vec<u32> {
    xor12_count_device(
        a_cols,
        b_cols,
        column_length,
        limb_bits,
        expand_bits,
        n_mult_columns,
        table_size,
    )
    .to_vec()
    .into_iter()
    .map(|f| f.0)
    .collect()
}

/// Splits a device-resident count table (layout `counts[rel * table_size + row]`,
/// as produced by [`xor_count_device`] / [`xor12_count_device`]) into the
/// `n_relations` per-relation multiplicity columns that form a
/// `verify_bitwise_xor_*` component's base trace — **entirely on device**.
///
/// This is the device-resident xor witness lane: the count table's u32 bytes are
/// exactly the host `AtomicMultiplicityColumn` bytes (identical `fetch_add`
/// wrap-around, no field reduction), so each relation's `table_size` slice is its
/// multiplicity column verbatim. Slicing is a borrowed-pointer view + a single
/// device-to-device copy per column (no host round trip). Correct only when every
/// feeder of this xor family accumulated into the same device count table.
pub fn xor_mult_columns(
    counts: &BaseFieldVec,
    n_relations: usize,
    table_size: usize,
) -> Vec<BaseFieldVec> {
    assert_eq!(
        counts.size,
        n_relations * table_size,
        "xor count table size does not match n_relations * table_size"
    );
    bindings::ensure_mem_pool_init();
    (0..n_relations)
        .map(|r| {
            // Borrowed view of relation r's slice (no allocation, no copy).
            let view = BaseFieldVec::from_borrowed_ptr(
                unsafe { counts.device_ptr.add(r * table_size) },
                table_size,
            );
            let mut col = BaseFieldVec::new_uninitialized(table_size);
            col.copy_from(&view); // D2D
            col
        })
        .collect()
}

fn new_device_raw_column(column_length: usize) -> DeviceRawLogupColumn {
    bindings::ensure_mem_pool_init();
    DeviceRawLogupColumn {
        numerator: std::array::from_fn(|_| BaseFieldVec::new_uninitialized(column_length)),
        denominator: BaseFieldVec::new_uninitialized(4 * column_length),
    }
}

/// One pair-batched xor logup column: `num = d0 + d1`, `den = d0 * d1` with
/// `d_i = combine([rel_i, a_i, b_i, xor_i])`.
#[allow(clippy::too_many_arguments)]
pub fn pair_logup(
    a0: &BaseFieldVec,
    b0: &BaseFieldVec,
    x0: &BaseFieldVec,
    a1: &BaseFieldVec,
    b1: &BaseFieldVec,
    x1: &BaseFieldVec,
    rel0: u32,
    rel1: u32,
    column_length: usize,
    alpha_powers: &[SecureField],
    z: SecureField,
) -> DeviceRawLogupColumn {
    assert!(alpha_powers.len() >= 4);
    let alphas = SecureFieldVec::from_vec(alpha_powers[..4].to_vec());
    let out = new_device_raw_column(column_length);
    unsafe {
        stwo_backend_cuda_kernels::raw::blake_g_pair_logup(
            a0.device_ptr,
            b0.device_ptr,
            x0.device_ptr,
            a1.device_ptr,
            b1.device_ptr,
            x1.device_ptr,
            rel0,
            rel1,
            column_length as u32,
            alphas.device_ptr,
            CudaSecureField::from(z).into_raw(),
            out.denominator.device_ptr,
            out.numerator[0].device_ptr,
            out.numerator[1].device_ptr,
            out.numerator[2].device_ptr,
            out.numerator[3].device_ptr,
        );
    }
    out
}

/// The final blake_g-relation logup column: `den = combine([rel, vals...])` over
/// 20 value columns (21 alpha powers), numerator `(-enabler, 0, 0, 0)`.
#[allow(clippy::too_many_arguments)]
pub fn final_logup(
    val_cols: &[&BaseFieldVec],
    enabler: &BaseFieldVec,
    rel: u32,
    column_length: usize,
    alpha_powers: &[SecureField],
    z: SecureField,
) -> DeviceRawLogupColumn {
    assert_eq!(val_cols.len(), 20);
    assert!(alpha_powers.len() >= 21);
    let ptrs: Vec<*const u32> = val_cols.iter().map(|c| c.device_ptr).collect();
    let table = UploadedDevicePointerVec::upload(&ptrs);
    let alphas = SecureFieldVec::from_vec(alpha_powers[..21].to_vec());
    let out = new_device_raw_column(column_length);
    unsafe {
        stwo_backend_cuda_kernels::raw::blake_g_final_logup(
            table.as_ptr(),
            enabler.device_ptr,
            rel,
            column_length as u32,
            alphas.device_ptr,
            CudaSecureField::from(z).into_raw(),
            out.denominator.device_ptr,
            out.numerator[0].device_ptr,
            out.numerator[1].device_ptr,
            out.numerator[2].device_ptr,
            out.numerator[3].device_ptr,
        );
    }
    out
}

pub use crate::backend::memory_witness::finalize_device_raw_logup;
