//! Device witness generation for the Cairo memory tables (witness-on-GPU P1).
//!
//! Safe wrappers over the `memory_witness.cu` kernels plus the device-resident
//! logup finalize: the stwo-cairo memory component calls these to generate its
//! base limb columns, feed the rc_9_9 multiplicity counts, and produce its logup
//! interaction columns entirely on device — no host columns, no `lookup_data`.
//!
//! Every formula is a port of the generated SIMD writer (see the spec in the
//! stwo-cairo fork's `gpu_benchmarks/WITNESS_ON_GPU.md`); the authoritative gates
//! are the component differential (`STWO_CUDA_WITNESS_VERIFY`) and the Cairo e2e
//! proof byte-equality, both in stwo-cairo.

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

use crate::backend::{CudaBackend, UploadedDevicePointerVec};
use crate::columns::base_field_vec::{BaseFieldVec, Uint32Vec};
use crate::columns::bindings::{self, CudaSecureField};

/// A logup column whose raw inputs already live on the device: numerator
/// coordinate columns plus element-major qm31 denominators. The device-native
/// counterpart of `stwo-constraint-framework`'s `RawLogupColumn`.
pub struct DeviceRawLogupColumn {
    pub numerator: [BaseFieldVec; 4],
    /// `4 * len` words: element-major qm31 denominators.
    pub denominator: BaseFieldVec,
}

/// Splits a device-resident f252 value table (8 u32 words per value, row-major)
/// into 28 9-bit limb columns of `column_length` (zero-padded past `n_values`).
pub fn limb_split_big(
    values: &BaseFieldVec,
    n_values: usize,
    column_length: usize,
) -> Vec<BaseFieldVec> {
    limb_split(values, n_values, column_length, 28, true)
}

/// Small-value variant: 4 words (u128 LE) per value into 8 limb columns.
pub fn limb_split_small(
    values: &BaseFieldVec,
    n_values: usize,
    column_length: usize,
) -> Vec<BaseFieldVec> {
    limb_split(values, n_values, column_length, 8, false)
}

fn limb_split(
    values: &BaseFieldVec,
    n_values: usize,
    column_length: usize,
    n_limbs: usize,
    big: bool,
) -> Vec<BaseFieldVec> {
    bindings::ensure_mem_pool_init();
    let cols: Vec<BaseFieldVec> = (0..n_limbs)
        .map(|_| BaseFieldVec::new_uninitialized(column_length))
        .collect();
    let ptrs: Vec<*const u32> = cols.iter().map(|c| c.device_ptr).collect();
    let table = UploadedDevicePointerVec::upload(&ptrs);
    unsafe {
        if big {
            stwo_backend_cuda_kernels::raw::memory_limb_split_big(
                values.device_ptr,
                n_values as u32,
                column_length as u32,
                table.as_ptr(),
            );
        } else {
            stwo_backend_cuda_kernels::raw::memory_limb_split_small(
                values.device_ptr,
                n_values as u32,
                column_length as u32,
                table.as_ptr(),
            );
        }
    }
    cols
}

/// Counts the rc_9_9 inputs fed by `limb_cols` (pairs (2j, 2j+1), relation index
/// j % 8 — padding rows included, matching the host loop) into 8 relation-indexed
/// count tables, and returns them to the host for merging into the rc state's
/// atomic multiplicity columns. `input_to_row_lut` is the dense
/// `(v0 << 9 | v1) -> rc row` map derived from the rc table's preprocessed layout.
pub fn rc99_count(
    limb_cols: &[BaseFieldVec],
    column_length: usize,
    input_to_row_lut: &[u32],
    rc_table_size: usize,
) -> Vec<u32> {
    assert_eq!(input_to_row_lut.len(), 1 << 18);
    assert!(limb_cols.len().is_multiple_of(2));
    let n_pairs = limb_cols.len() / 2;
    let ptrs: Vec<*const u32> = limb_cols.iter().map(|c| c.device_ptr).collect();
    let table = UploadedDevicePointerVec::upload(&ptrs);
    let lut_dev = unsafe {
        bindings::copy_uint32_t_vec_from_host_to_device(
            input_to_row_lut.as_ptr(),
            input_to_row_lut.len() as u32,
        )
    };
    let lut = BaseFieldVec::new(lut_dev, input_to_row_lut.len());
    let counts = BaseFieldVec::new_zeroes(8 * rc_table_size);
    unsafe {
        stwo_backend_cuda_kernels::raw::memory_rc99_count(
            table.as_ptr(),
            n_pairs as u32,
            column_length as u32,
            lut.device_ptr,
            rc_table_size as u32,
            counts.device_ptr.cast_mut(),
        );
    }
    // Synchronous D2H readback (the fence); the caller adds into the host atomics.
    counts.to_vec().into_iter().map(|f| f.0).collect()
}

/// The final memory-relation logup column for one segment:
/// `denom = combine([relation_id, (offset + row) | tag, limbs...])`,
/// numerator `(-mult, 0, 0, 0)`.
#[allow(clippy::too_many_arguments)]
pub fn memory_logup_inputs(
    limb_cols: &[BaseFieldVec],
    mults: &BaseFieldVec,
    relation_id: u32,
    id_offset: u32,
    id_tag: u32,
    column_length: usize,
    alpha_powers: &[SecureField],
    z: SecureField,
) -> DeviceRawLogupColumn {
    assert!(alpha_powers.len() >= limb_cols.len() + 2);
    let ptrs: Vec<*const u32> = limb_cols.iter().map(|c| c.device_ptr).collect();
    let table = UploadedDevicePointerVec::upload(&ptrs);
    let alphas = crate::columns::SecureFieldVec::from_vec(alpha_powers.to_vec());
    let out = new_device_raw_column(column_length);
    unsafe {
        stwo_backend_cuda_kernels::raw::memory_logup_inputs(
            table.as_ptr(),
            limb_cols.len() as u32,
            mults.device_ptr,
            relation_id,
            id_offset,
            id_tag,
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

/// One pair-batched rc_9_9 logup column: `num = d0 + d1`, `den = d0 * d1` with
/// `d_i = combine([rel_id_i, limb_a, limb_b])`.
#[allow(clippy::too_many_arguments)]
pub fn memory_rc_pair_logup(
    limbs: [&BaseFieldVec; 4],
    rel_id0: u32,
    rel_id1: u32,
    column_length: usize,
    alpha_powers: &[SecureField],
    z: SecureField,
) -> DeviceRawLogupColumn {
    assert!(alpha_powers.len() >= 3);
    let alphas = crate::columns::SecureFieldVec::from_vec(alpha_powers[..3].to_vec());
    let out = new_device_raw_column(column_length);
    unsafe {
        stwo_backend_cuda_kernels::raw::memory_rc_pair_logup(
            limbs[0].device_ptr,
            limbs[1].device_ptr,
            limbs[2].device_ptr,
            limbs[3].device_ptr,
            rel_id0,
            rel_id1,
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

/// One pair-batched `memory_address_to_id` logup column: split chunks `(2i, 2i+1)`
/// of the id table share a column with sequential addresses
/// `addr_k = addr_base_k + row`, numerator `d0·(-mult1) + d1·(-mult0)`,
/// denominator `d0·d1`.
#[allow(clippy::too_many_arguments)]
pub fn addr_to_id_pair_logup(
    ids: [&BaseFieldVec; 2],
    mults: [&BaseFieldVec; 2],
    rel_id: u32,
    addr0_base: u32,
    addr1_base: u32,
    column_length: usize,
    alpha_powers: &[SecureField],
    z: SecureField,
) -> DeviceRawLogupColumn {
    assert!(alpha_powers.len() >= 3);
    let alphas = crate::columns::SecureFieldVec::from_vec(alpha_powers[..3].to_vec());
    let out = new_device_raw_column(column_length);
    unsafe {
        stwo_backend_cuda_kernels::raw::addr_to_id_pair_logup(
            ids[0].device_ptr,
            mults[0].device_ptr,
            ids[1].device_ptr,
            mults[1].device_ptr,
            rel_id,
            addr0_base,
            addr1_base,
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

fn new_device_raw_column(column_length: usize) -> DeviceRawLogupColumn {
    bindings::ensure_mem_pool_init();
    DeviceRawLogupColumn {
        numerator: std::array::from_fn(|_| BaseFieldVec::new_uninitialized(column_length)),
        denominator: BaseFieldVec::new_uninitialized(4 * column_length),
    }
}

/// Finalizes device-resident raw logup columns: identical math to
/// [`super::logup::finalize_raw_logup`] (fraction chain, claimed sum, shift,
/// prefix sums) without any host-to-device transfer — the inputs were born here.
pub fn finalize_device_raw_logup(
    log_size: u32,
    columns: Vec<DeviceRawLogupColumn>,
) -> (
    Vec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();

    let mut finalized: Vec<[BaseFieldVec; 4]> = Vec::with_capacity(columns.len());
    for column in columns {
        let prev = finalized.last();
        unsafe {
            stwo_backend_cuda_kernels::raw::logup_fraction_chain_dense(
                column.numerator[0].device_ptr,
                column.numerator[1].device_ptr,
                column.numerator[2].device_ptr,
                column.numerator[3].device_ptr,
                column.denominator.device_ptr,
                prev.map_or(std::ptr::null(), |p| p[0].device_ptr),
                prev.map_or(std::ptr::null(), |p| p[1].device_ptr),
                prev.map_or(std::ptr::null(), |p| p[2].device_ptr),
                prev.map_or(std::ptr::null(), |p| p[3].device_ptr),
                size as u32,
            );
        }
        finalized.push(column.numerator);
    }

    let last = finalized.last().expect("device raw logup trace is empty");
    let claimed_sum: SecureField = unsafe {
        bindings::logup_sum_secure_coords(
            last[0].device_ptr,
            last[1].device_ptr,
            last[2].device_ptr,
            last[3].device_ptr,
            size as u32,
        )
    }
    .into();
    let cumsum_shift = claimed_sum / BaseField::from_u32_unchecked(1 << log_size);
    unsafe {
        bindings::logup_shift_secure_coords(
            last[0].device_ptr,
            last[1].device_ptr,
            last[2].device_ptr,
            last[3].device_ptr,
            cumsum_shift.into(),
            size as u32,
        );
        // P3: the four coordinate scans are independent — they run on the pool
        // streams with event bridges to the legacy stream on both sides.
        stwo_backend_cuda_kernels::raw::inclusive_prefix_sum_x4(
            last[0].device_ptr,
            last[1].device_ptr,
            last[2].device_ptr,
            last[3].device_ptr,
            size as u32,
        );
    }

    let trace = finalized
        .into_iter()
        .flat_map(|coords| coords.map(|col| CircleEvaluation::new(domain, col)))
        .collect();
    (trace, claimed_sum)
}

// ---------------------------------------------------------------------------
// Generic witness logup lane (witness_logup.cu): every generated interaction
// writer reduces to pair-of-tuples / single-tuple columns over staged device
// columns, so component ports only supply a base-trace kernel.
// ---------------------------------------------------------------------------

/// A logup multiplicity operand.
pub enum Mult<'a> {
    /// The constant 1 (most "use" tuples).
    One,
    /// A device column (mult columns, enabler-like data columns).
    Column(&'a BaseFieldVec),
    /// The Enabler pattern: 1 iff `row < offset` (opcode padding).
    Enabler(u32),
}

const NO_ENABLER: u32 = u32::MAX;

impl Mult<'_> {
    fn encode(&self) -> (*const u32, u32) {
        match self {
            Mult::One => (std::ptr::null(), NO_ENABLER),
            Mult::Column(col) => (col.device_ptr, NO_ENABLER),
            Mult::Enabler(offset) => (std::ptr::null(), *offset),
        }
    }
}

/// One slot of a lookup tuple (after the leading relation id): either a staged
/// device column or a row-invariant constant. Constants cost nothing on
/// device: the wrapper folds them (with the relation id and `-z`) into the
/// tuple's `base` and repacks the alpha powers to only the live columns.
pub enum TupleSlot<'a> {
    Col(&'a BaseFieldVec),
    Const(u32),
}

/// Folds a tuple's row-invariant parts: returns the combined
/// `base = alpha^0 * rel_id + sum_(const slots j) alpha^(1+j) * c_j - z`
/// and the (column pointer, alpha power) lists of the live slots.
fn fold_tuple_slots(
    rel_id: u32,
    slots: &[TupleSlot<'_>],
    alpha_powers: &[SecureField],
    z: SecureField,
) -> (SecureField, Vec<*const u32>, Vec<SecureField>) {
    assert!(alpha_powers.len() > slots.len());
    let mut base = alpha_powers[0] * SecureField::from(BaseField::from(rel_id)) - z;
    let mut ptrs = Vec::with_capacity(slots.len());
    let mut alphas = Vec::with_capacity(slots.len());
    for (j, slot) in slots.iter().enumerate() {
        match slot {
            TupleSlot::Col(col) => {
                ptrs.push(col.device_ptr);
                alphas.push(alpha_powers[1 + j]);
            }
            TupleSlot::Const(c) => {
                base += alpha_powers[1 + j] * SecureField::from(BaseField::from(*c));
            }
        }
    }
    (base, ptrs, alphas)
}

fn slots_of_cols<'a>(cols: &[&'a BaseFieldVec]) -> Vec<TupleSlot<'a>> {
    cols.iter().map(|c| TupleSlot::Col(c)).collect()
}

/// Pair-of-tuples logup column: `d_i = combine([rel_id_i, slots_i...])`,
/// numerator `sign * (d0*m1 + d1*m0)`, denominator `d0*d1`.
#[allow(clippy::too_many_arguments)]
pub fn tuple_pair_logup_slots(
    rel_id0: u32,
    slots0: &[TupleSlot<'_>],
    rel_id1: u32,
    slots1: &[TupleSlot<'_>],
    mult0: Mult<'_>,
    mult1: Mult<'_>,
    negate: bool,
    column_length: usize,
    alpha_powers: &[SecureField],
    z: SecureField,
) -> DeviceRawLogupColumn {
    let (base0, ptrs0, alphas0) = fold_tuple_slots(rel_id0, slots0, alpha_powers, z);
    let (base1, ptrs1, alphas1) = fold_tuple_slots(rel_id1, slots1, alpha_powers, z);
    let table0 = UploadedDevicePointerVec::upload(&ptrs0);
    let table1 = UploadedDevicePointerVec::upload(&ptrs1);
    let alphas0_dev = crate::columns::SecureFieldVec::from_vec(alphas0);
    let alphas1_dev = crate::columns::SecureFieldVec::from_vec(alphas1);
    let (m0_ptr, e0) = mult0.encode();
    let (m1_ptr, e1) = mult1.encode();
    let out = new_device_raw_column(column_length);
    unsafe {
        stwo_backend_cuda_kernels::raw::tuple_pair_logup(
            CudaSecureField::from(base0).into_raw(),
            table0.as_ptr(),
            alphas0_dev.device_ptr,
            ptrs0.len() as u32,
            CudaSecureField::from(base1).into_raw(),
            table1.as_ptr(),
            alphas1_dev.device_ptr,
            ptrs1.len() as u32,
            m0_ptr,
            e0,
            m1_ptr,
            e1,
            negate as u32,
            column_length as u32,
            out.denominator.device_ptr,
            out.numerator[0].device_ptr,
            out.numerator[1].device_ptr,
            out.numerator[2].device_ptr,
            out.numerator[3].device_ptr,
        );
    }
    out
}

/// Column-only convenience over [`tuple_pair_logup_slots`].
#[allow(clippy::too_many_arguments)]
pub fn tuple_pair_logup(
    rel_id0: u32,
    cols0: &[&BaseFieldVec],
    rel_id1: u32,
    cols1: &[&BaseFieldVec],
    mult0: Mult<'_>,
    mult1: Mult<'_>,
    negate: bool,
    column_length: usize,
    alpha_powers: &[SecureField],
    z: SecureField,
) -> DeviceRawLogupColumn {
    tuple_pair_logup_slots(
        rel_id0,
        &slots_of_cols(cols0),
        rel_id1,
        &slots_of_cols(cols1),
        mult0,
        mult1,
        negate,
        column_length,
        alpha_powers,
        z,
    )
}

/// Single-tuple logup column: numerator `(sign * m, 0, 0, 0)`, denominator
/// `combine([rel_id, slots...])`.
#[allow(clippy::too_many_arguments)]
pub fn tuple_single_logup_slots(
    rel_id: u32,
    slots: &[TupleSlot<'_>],
    mult: Mult<'_>,
    negate: bool,
    column_length: usize,
    alpha_powers: &[SecureField],
    z: SecureField,
) -> DeviceRawLogupColumn {
    let (base, ptrs, alphas) = fold_tuple_slots(rel_id, slots, alpha_powers, z);
    let table = UploadedDevicePointerVec::upload(&ptrs);
    let alphas_dev = crate::columns::SecureFieldVec::from_vec(alphas);
    let (m_ptr, e) = mult.encode();
    let out = new_device_raw_column(column_length);
    unsafe {
        stwo_backend_cuda_kernels::raw::tuple_single_logup(
            CudaSecureField::from(base).into_raw(),
            table.as_ptr(),
            alphas_dev.device_ptr,
            ptrs.len() as u32,
            m_ptr,
            e,
            negate as u32,
            column_length as u32,
            out.denominator.device_ptr,
            out.numerator[0].device_ptr,
            out.numerator[1].device_ptr,
            out.numerator[2].device_ptr,
            out.numerator[3].device_ptr,
        );
    }
    out
}

/// Column-only convenience over [`tuple_single_logup_slots`].
#[allow(clippy::too_many_arguments)]
pub fn tuple_single_logup(
    rel_id: u32,
    cols: &[&BaseFieldVec],
    mult: Mult<'_>,
    negate: bool,
    column_length: usize,
    alpha_powers: &[SecureField],
    z: SecureField,
) -> DeviceRawLogupColumn {
    tuple_single_logup_slots(
        rel_id,
        &slots_of_cols(cols),
        mult,
        negate,
        column_length,
        alpha_powers,
        z,
    )
}

/// Counts width-W column tuples into relation-indexed count tables through a
/// dense input->row LUT (per-slot bit packing; rc_9_9 = [9,9], rc_7_2_5 =
/// [7,2,5], rc_4_3 = [4,3]). Padding rows included, like every host feed.
#[allow(clippy::too_many_arguments)]
pub fn tuple_count(
    tuple_cols: &[&BaseFieldVec],
    width: usize,
    slot_bits: &[u32],
    n_relations: usize,
    column_length: usize,
    input_to_row_lut: &[u32],
    table_size: usize,
) -> Vec<u32> {
    assert_eq!(slot_bits.len(), width);
    assert!(tuple_cols.len().is_multiple_of(width));
    assert_eq!(
        input_to_row_lut.len(),
        1usize << slot_bits.iter().sum::<u32>()
    );
    bindings::ensure_mem_pool_init();
    let ptrs: Vec<*const u32> = tuple_cols.iter().map(|c| c.device_ptr).collect();
    let table = UploadedDevicePointerVec::upload(&ptrs);
    let bits_dev = unsafe {
        bindings::copy_uint32_t_vec_from_host_to_device(slot_bits.as_ptr(), slot_bits.len() as u32)
    };
    let bits = BaseFieldVec::new(bits_dev, slot_bits.len());
    let lut_dev = unsafe {
        bindings::copy_uint32_t_vec_from_host_to_device(
            input_to_row_lut.as_ptr(),
            input_to_row_lut.len() as u32,
        )
    };
    let lut = BaseFieldVec::new(lut_dev, input_to_row_lut.len());
    let counts = BaseFieldVec::new_zeroes(n_relations * table_size);
    unsafe {
        stwo_backend_cuda_kernels::raw::tuple_count(
            table.as_ptr(),
            (tuple_cols.len() / width) as u32,
            width as u32,
            bits.device_ptr,
            n_relations as u32,
            column_length as u32,
            lut.device_ptr,
            table_size as u32,
            counts.device_ptr.cast_mut(),
        );
    }
    counts.to_vec().into_iter().map(|f| f.0).collect()
}

/// The verify_instruction base trace: 17 trace columns + the 3 staged
/// combination columns its interaction tuples reference.
#[allow(clippy::too_many_arguments)]
pub fn verify_instruction_trace(
    inputs: [&BaseFieldVec; 9],
    column_length: usize,
) -> (Vec<BaseFieldVec>, [BaseFieldVec; 3]) {
    bindings::ensure_mem_pool_init();
    let trace: Vec<BaseFieldVec> = (0..17)
        .map(|_| BaseFieldVec::new_uninitialized(column_length))
        .collect();
    let staged: [BaseFieldVec; 3] =
        std::array::from_fn(|_| BaseFieldVec::new_uninitialized(column_length));
    let trace_ptrs: Vec<*const u32> = trace.iter().map(|c| c.device_ptr).collect();
    let table = UploadedDevicePointerVec::upload(&trace_ptrs);
    unsafe {
        stwo_backend_cuda_kernels::raw::verify_instruction_trace(
            inputs[0].device_ptr,
            inputs[1].device_ptr,
            inputs[2].device_ptr,
            inputs[3].device_ptr,
            inputs[4].device_ptr,
            inputs[5].device_ptr,
            inputs[6].device_ptr,
            inputs[7].device_ptr,
            inputs[8].device_ptr,
            column_length as u32,
            table.as_ptr(),
            staged[0].device_ptr.cast_mut(),
            staged[1].device_ptr.cast_mut(),
            staged[2].device_ptr.cast_mut(),
        );
    }
    (trace, staged)
}

// ---------------------------------------------------------------------------
// Opcode cohort (opcodes.cu): per-opcode base-trace kernels with the memory
// deduce lookups fused as gathers against prove-wide device tables.
// ---------------------------------------------------------------------------

/// The memory tables every opcode base kernel gathers from, uploaded once per
/// prove (raw words, not field elements): the address-ordered raw id table,
/// the big-value words (8 per value) and the small-value words (4 per value,
/// u128 LE).
pub struct DeviceMemTables {
    pub addr_to_id: Uint32Vec,
    pub big_words: Uint32Vec,
    pub small_words: Uint32Vec,
}

impl DeviceMemTables {
    pub fn upload(addr_to_id: Vec<u32>, big_words: Vec<u32>, small_words: Vec<u32>) -> Self {
        bindings::ensure_mem_pool_init();
        Self {
            addr_to_id: Uint32Vec::from_vec(addr_to_id),
            big_words: Uint32Vec::from_vec(big_words),
            small_words: Uint32Vec::from_vec(small_words),
        }
    }
}

/// ret_opcode base trace: 16 trace columns plus 4 staged columns
/// (read addresses fp-1 / fp-2, and the next_pc / next_fp recombinations).
pub fn ret_opcode_trace(
    inputs: [&BaseFieldVec; 3], // pc, ap, fp (padded to column_length)
    tables: &DeviceMemTables,
    n_rows: usize,
    column_length: usize,
) -> (Vec<BaseFieldVec>, [BaseFieldVec; 4]) {
    bindings::ensure_mem_pool_init();
    let trace: Vec<BaseFieldVec> = (0..16)
        .map(|_| BaseFieldVec::new_uninitialized(column_length))
        .collect();
    let staged: [BaseFieldVec; 4] =
        std::array::from_fn(|_| BaseFieldVec::new_uninitialized(column_length));
    let trace_ptrs: Vec<*const u32> = trace.iter().map(|c| c.device_ptr).collect();
    let table = UploadedDevicePointerVec::upload(&trace_ptrs);
    unsafe {
        stwo_backend_cuda_kernels::raw::ret_opcode_trace(
            inputs[0].device_ptr,
            inputs[1].device_ptr,
            inputs[2].device_ptr,
            tables.addr_to_id.device_ptr,
            tables.big_words.device_ptr,
            tables.small_words.device_ptr,
            n_rows as u32,
            column_length as u32,
            table.as_ptr(),
            staged[0].device_ptr,
            staged[1].device_ptr,
            staged[2].device_ptr,
            staged[3].device_ptr,
        );
    }
    (trace, staged)
}
