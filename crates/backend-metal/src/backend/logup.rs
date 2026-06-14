use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::N_LANES;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_backend_metal_sys::metal::{LogupFractionChainDescriptor, U32Buffer};
use stwo_constraint_framework::{LogupFinalizeBackend, RawLogupTrace};

use super::MetalBackend;
use crate::columns::BaseFieldVec;

pub struct MetalRawLogupColumn {
    numerator: [BaseFieldVec; 4],
    denominator_packed: U32Buffer,
    params: Option<U32Buffer>,
}

fn upload_secure_fields(values: &[SecureField]) -> U32Buffer {
    let raw = values
        .iter()
        .flat_map(|value| value.to_m31_array().map(|limb| limb.0))
        .collect::<Vec<_>>();
    U32Buffer::from_slice_private(&raw).expect("Metal secure-field parameter upload should succeed")
}

pub fn memory_rc_pair_logup(
    limbs: [&BaseFieldVec; 4],
    rel_id0: u32,
    rel_id1: u32,
    alpha_powers: &[SecureField],
    z: SecureField,
) -> MetalRawLogupColumn {
    assert_eq!(
        alpha_powers.len(),
        3,
        "memory rc-pair logup expects 3 alpha powers"
    );
    let alpha_powers = upload_secure_fields(alpha_powers);
    let (numerator, denominator_packed) = U32Buffer::witness_memory_rc_pair_logup_async(
        [
            limbs[0].gpu_buffer(),
            limbs[1].gpu_buffer(),
            limbs[2].gpu_buffer(),
            limbs[3].gpu_buffer(),
        ],
        rel_id0,
        rel_id1,
        &alpha_powers,
        z.to_m31_array().map(|limb| limb.0),
    )
    .expect("Metal memory rc-pair logup should succeed");
    MetalRawLogupColumn {
        numerator: numerator.map(BaseFieldVec::from_buffer),
        denominator_packed,
        params: Some(alpha_powers),
    }
}

pub fn memory_logup_inputs(
    limbs: &[BaseFieldVec],
    mults: &BaseFieldVec,
    relation_id: u32,
    id_offset: u32,
    id_tag: u32,
    alpha_powers: &[SecureField],
    z: SecureField,
) -> MetalRawLogupColumn {
    assert_eq!(
        alpha_powers.len(),
        limbs.len() + 2,
        "memory logup input alpha-powers length mismatch"
    );
    let alpha_powers = upload_secure_fields(alpha_powers);
    let limb_refs = limbs.iter().map(|col| col.gpu_buffer()).collect::<Vec<_>>();
    let (numerator, denominator_packed) = U32Buffer::witness_memory_logup_inputs_async(
        &limb_refs,
        mults.gpu_buffer(),
        relation_id,
        id_offset,
        id_tag,
        &alpha_powers,
        z.to_m31_array().map(|limb| limb.0),
    )
    .expect("Metal memory logup inputs should succeed");
    MetalRawLogupColumn {
        numerator: numerator.map(BaseFieldVec::from_buffer),
        denominator_packed,
        params: Some(alpha_powers),
    }
}

pub fn ret_opcode_logup_inputs(
    trace_cols: &[BaseFieldVec],
    alpha_powers: &[SecureField],
    z: SecureField,
) -> Vec<MetalRawLogupColumn> {
    assert!(
        trace_cols.len() >= 16,
        "ret opcode interaction expects 16 trace columns"
    );
    assert!(
        alpha_powers.len() >= 30,
        "ret opcode interaction expects at least 30 alpha powers"
    );
    let alpha_powers = upload_secure_fields(alpha_powers);
    let trace_col_refs = trace_cols
        .iter()
        .take(16)
        .map(|col| col.gpu_buffer())
        .collect::<Vec<_>>();
    U32Buffer::witness_ret_opcode_interaction_raw_async(
        &trace_col_refs,
        &alpha_powers,
        z.to_m31_array().map(|limb| limb.0),
    )
    .expect("Metal ret opcode interaction raw logup should succeed")
    .into_iter()
    .enumerate()
    .map(|(i, (numerator, denominator_packed))| MetalRawLogupColumn {
        numerator: numerator.map(BaseFieldVec::from_buffer),
        denominator_packed,
        params: (i == 0).then(|| alpha_powers.clone()),
    })
    .collect()
}

pub fn memory_addr_to_id_logup_inputs(
    trace_cols: &[BaseFieldVec],
    relation_id: u32,
    alpha_powers: &[SecureField],
    z: SecureField,
    split: u32,
) -> Vec<MetalRawLogupColumn> {
    assert!(
        split > 0 && split % 2 == 0,
        "memory_address_to_id split must be positive and even"
    );
    assert!(
        trace_cols.len() >= 2 * split as usize,
        "memory_address_to_id interaction expects 2 * split trace columns"
    );
    assert!(
        alpha_powers.len() >= 3,
        "memory_address_to_id interaction expects at least 3 alpha powers"
    );
    let alpha_powers = upload_secure_fields(&alpha_powers[..3]);
    let trace_col_refs = trace_cols
        .iter()
        .take(2 * split as usize)
        .map(|col| col.gpu_buffer())
        .collect::<Vec<_>>();
    U32Buffer::witness_memory_addr_to_id_interaction_raw_async(
        &trace_col_refs,
        relation_id,
        &alpha_powers,
        z.to_m31_array().map(|limb| limb.0),
        split,
    )
    .expect("Metal memory_address_to_id interaction raw logup should succeed")
    .into_iter()
    .enumerate()
    .map(|(i, (numerator, denominator_packed))| MetalRawLogupColumn {
        numerator: numerator.map(BaseFieldVec::from_buffer),
        denominator_packed,
        params: (i == 0).then(|| alpha_powers.clone()),
    })
    .collect()
}

pub fn add_opcode_small_logup_inputs(
    trace_flat: &U32Buffer,
    alpha_powers: &[SecureField],
    z: SecureField,
    n_rows: u32,
    column_length: u32,
) -> Vec<MetalRawLogupColumn> {
    assert!(
        alpha_powers.len() >= 30,
        "add_opcode_small interaction expects at least 30 alpha powers"
    );
    let alpha_powers = upload_secure_fields(alpha_powers);
    U32Buffer::witness_add_opcode_small_interaction_raw_async(
        trace_flat,
        &alpha_powers,
        z.to_m31_array().map(|limb| limb.0),
        n_rows,
        column_length,
    )
    .expect("Metal add_opcode_small interaction raw logup should succeed")
    .into_iter()
    .enumerate()
    .map(|(i, (numerator, denominator_packed))| MetalRawLogupColumn {
        numerator: numerator.map(BaseFieldVec::from_buffer),
        denominator_packed,
        params: (i == 0).then(|| alpha_powers.clone()),
    })
    .collect()
}

pub fn add_opcode_small_logup_inputs_from_cols(
    trace_cols: &[BaseFieldVec],
    alpha_powers: &[SecureField],
    z: SecureField,
    n_rows: u32,
    column_length: u32,
) -> Vec<MetalRawLogupColumn> {
    assert!(
        trace_cols.len() >= 39,
        "add_opcode_small interaction expects 39 trace columns"
    );
    assert!(
        alpha_powers.len() >= 30,
        "add_opcode_small interaction expects at least 30 alpha powers"
    );
    let alpha_powers = upload_secure_fields(alpha_powers);
    let trace_col_refs = trace_cols
        .iter()
        .take(39)
        .map(|col| col.gpu_buffer())
        .collect::<Vec<_>>();
    U32Buffer::witness_add_opcode_small_interaction_raw_columns_async(
        &trace_col_refs,
        &alpha_powers,
        z.to_m31_array().map(|limb| limb.0),
        n_rows,
        column_length,
    )
    .expect("Metal add_opcode_small column interaction raw logup should succeed")
    .into_iter()
    .enumerate()
    .map(|(i, (numerator, denominator_packed))| MetalRawLogupColumn {
        numerator: numerator.map(BaseFieldVec::from_buffer),
        denominator_packed,
        params: (i == 0).then(|| alpha_powers.clone()),
    })
    .collect()
}

pub fn assert_eq_opcode_logup_inputs(
    trace_flat: &U32Buffer,
    alpha_powers: &[SecureField],
    z: SecureField,
    n_rows: u32,
    column_length: u32,
) -> Vec<MetalRawLogupColumn> {
    assert!(
        alpha_powers.len() >= 8,
        "assert_eq_opcode interaction expects at least 8 alpha powers"
    );
    let alpha_powers = upload_secure_fields(alpha_powers);
    U32Buffer::witness_assert_eq_opcode_interaction_raw_async(
        trace_flat,
        &alpha_powers,
        z.to_m31_array().map(|limb| limb.0),
        n_rows,
        column_length,
    )
    .expect("Metal assert_eq_opcode interaction raw logup should succeed")
    .into_iter()
    .enumerate()
    .map(|(i, (numerator, denominator_packed))| MetalRawLogupColumn {
        numerator: numerator.map(BaseFieldVec::from_buffer),
        denominator_packed,
        params: (i == 0).then(|| alpha_powers.clone()),
    })
    .collect()
}

pub fn assert_eq_opcode_logup_inputs_from_cols(
    trace_cols: &[BaseFieldVec],
    alpha_powers: &[SecureField],
    z: SecureField,
    n_rows: u32,
    column_length: u32,
) -> Vec<MetalRawLogupColumn> {
    assert!(
        trace_cols.len() >= 12,
        "assert_eq_opcode interaction expects 12 trace columns"
    );
    assert!(
        alpha_powers.len() >= 8,
        "assert_eq_opcode interaction expects at least 8 alpha powers"
    );
    let alpha_powers = upload_secure_fields(alpha_powers);
    let trace_col_refs = trace_cols
        .iter()
        .take(12)
        .map(|col| col.gpu_buffer())
        .collect::<Vec<_>>();
    U32Buffer::witness_assert_eq_opcode_interaction_raw_columns_async(
        &trace_col_refs,
        &alpha_powers,
        z.to_m31_array().map(|limb| limb.0),
        n_rows,
        column_length,
    )
    .expect("Metal assert_eq_opcode column interaction raw logup should succeed")
    .into_iter()
    .enumerate()
    .map(|(i, (numerator, denominator_packed))| MetalRawLogupColumn {
        numerator: numerator.map(BaseFieldVec::from_buffer),
        denominator_packed,
        params: (i == 0).then(|| alpha_powers.clone()),
    })
    .collect()
}

/// Finalizes a raw logup trace on Metal. The returned interaction columns are
/// born as Metal buffers instead of going through `finalize_on_simd` and
/// `from_simd_evals`.
pub fn finalize_device_raw_logup(
    log_size: u32,
    columns: Vec<MetalRawLogupColumn>,
) -> (
    ColumnVec<CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();
    let mut finalized = Vec::<[BaseFieldVec; 4]>::with_capacity(columns.len());
    let mut pending_raw_buffers = Vec::with_capacity(columns.len());

    for raw_col in columns {
        let MetalRawLogupColumn {
            numerator,
            denominator_packed,
            params,
        } = raw_col;
        let [c0, c1, c2, c3] = numerator;
        assert_eq!(
            c0.len(),
            size,
            "raw logup numerator length must match log_size"
        );
        assert_eq!(
            c1.len(),
            size,
            "raw logup numerator length must match log_size"
        );
        assert_eq!(
            c2.len(),
            size,
            "raw logup numerator length must match log_size"
        );
        assert_eq!(
            c3.len(),
            size,
            "raw logup numerator length must match log_size"
        );
        finalized.push([c0, c1, c2, c3]);
        pending_raw_buffers.push((denominator_packed, params));
    }

    let descriptors = (0..finalized.len())
        .map(|i| {
            let nums = finalized[i]
                .each_ref()
                .map(|coord| coord.gpu_buffer().opaque_ptr());
            let prev = if i == 0 {
                [std::ptr::null_mut(); 4]
            } else {
                finalized[i - 1]
                    .each_ref()
                    .map(|coord| coord.gpu_buffer().opaque_ptr())
            };
            LogupFractionChainDescriptor {
                nums,
                denom_packed: pending_raw_buffers[i].0.opaque_ptr(),
                prev,
            }
        })
        .collect::<Vec<_>>();
    unsafe {
        U32Buffer::logup_fraction_chain_packed_batch_raw(
            &descriptors,
            size.try_into()
                .expect("raw logup size should fit in u32 for Metal"),
        )
    }
    .expect("Metal raw-logup fraction-chain batch should succeed");

    let last = finalized
        .last_mut()
        .expect("raw logup trace has no columns");
    let [last0, last1, last2, last3] = last;
    let coordinate_sums = U32Buffer::reduce_sum_m31_4col([
        last0.gpu_buffer(),
        last1.gpu_buffer(),
        last2.gpu_buffer(),
        last3.gpu_buffer(),
    ])
    .expect("Metal raw-logup claimed-sum reduction should succeed");
    let claimed_sum = SecureField::from_u32_unchecked(
        coordinate_sums[0],
        coordinate_sums[1],
        coordinate_sums[2],
        coordinate_sums[3],
    );
    let cumsum_shift = claimed_sum / BaseField::from_u32_unchecked(1 << log_size);
    U32Buffer::subtract_m31_4col(
        [
            last0.gpu_buffer_mut(),
            last1.gpu_buffer_mut(),
            last2.gpu_buffer_mut(),
            last3.gpu_buffer_mut(),
        ],
        cumsum_shift.to_m31_array().map(|limb| limb.0),
    )
    .expect("Metal raw-logup cumsum shift should succeed");
    U32Buffer::inclusive_prefix_sum_bit_rev_circle_domain_4col([
        last0.gpu_buffer_mut(),
        last1.gpu_buffer_mut(),
        last2.gpu_buffer_mut(),
        last3.gpu_buffer_mut(),
    ])
    .expect("Metal raw-logup 4-coordinate prefix sum should succeed");

    let trace = finalized
        .into_iter()
        .flat_map(|coords| coords.map(|col| CircleEvaluation::new(domain, col)))
        .collect();
    (trace, claimed_sum)
}

/// Finalizes a host-written raw logup trace on Metal. The returned interaction
/// columns are born as Metal buffers instead of going through `finalize_on_simd`
/// and `from_simd_evals`.
pub fn finalize_raw_logup(
    raw: RawLogupTrace,
) -> (
    ColumnVec<CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let log_size = raw.log_size;
    let size = 1usize << log_size;
    let columns = raw
        .columns
        .into_iter()
        .map(|raw_col| {
            let numerator = raw_col.numerator;
            let denominator = raw_col.denominator;
            assert_eq!(
                denominator.data.len() * N_LANES,
                size,
                "raw logup denominator length must match the log size"
            );

            let numerator = numerator
                .columns
                .map(|column| BaseFieldVec::from_packed_m31_slice_private(&column.data));
            let byte_len = std::mem::size_of_val(denominator.data.as_slice());
            assert_eq!(
                byte_len % std::mem::size_of::<u32>(),
                0,
                "PackedSecureField storage should be u32-aligned"
            );
            let raw_len = byte_len / std::mem::size_of::<u32>();
            let raw = unsafe {
                std::slice::from_raw_parts(denominator.data.as_ptr().cast::<u32>(), raw_len)
            };
            let denominator_packed = U32Buffer::from_slice_private(raw)
                .expect("Metal raw-logup denominator upload should succeed");
            MetalRawLogupColumn {
                numerator,
                denominator_packed,
                params: None,
            }
        })
        .collect();
    finalize_device_raw_logup(log_size, columns)
}

impl LogupFinalizeBackend for MetalBackend {
    fn finalize_raw_logup(
        raw: RawLogupTrace,
    ) -> (
        ColumnVec<CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        SecureField,
    ) {
        finalize_raw_logup(raw)
    }
}
