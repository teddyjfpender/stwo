use num_traits::identities::Zero;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::simd::m31::{PackedM31, LOG_N_LANES, N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::Column;
use stwo_backend_metal::{
    add_opcode_small_logup_inputs, add_opcode_small_logup_inputs_from_cols,
    assert_eq_opcode_logup_inputs, assert_eq_opcode_logup_inputs_from_cols,
    finalize_device_raw_logup, memory_addr_to_id_logup_inputs, memory_logup_inputs,
    memory_rc_pair_logup, BaseFieldVec,
};
use stwo_backend_metal_sys::metal::{metal_runtime_support, MetalRuntimeSupport, U32Buffer};
use stwo_constraint_framework::RawLogupTraceGenerator;

const LIMB_BITS: u32 = 9;
const LIMB_MASK: u32 = (1 << LIMB_BITS) - 1;

fn require_metal() -> bool {
    if metal_runtime_support() == MetalRuntimeSupport::Available {
        return true;
    }
    eprintln!("skipping Metal witness differential: Metal runtime unavailable");
    false
}

fn as_base(values: &[u32]) -> Vec<BaseField> {
    values
        .iter()
        .copied()
        .map(BaseField::from_u32_unchecked)
        .collect()
}

fn split_le_9bit(words: &[u32], n_limbs: usize) -> Vec<u32> {
    let mut limbs = Vec::with_capacity(n_limbs);
    let mut n_bits_in_word = 32u32;
    let mut word_i = 0usize;
    let mut word = words.first().copied().unwrap_or(0);
    for _ in 0..n_limbs {
        if n_bits_in_word > LIMB_BITS {
            limbs.push(word & LIMB_MASK);
            word >>= LIMB_BITS;
            n_bits_in_word -= LIMB_BITS;
            continue;
        }

        let mut limb = word;
        word_i += 1;
        word = words.get(word_i).copied().unwrap_or(0);
        if n_bits_in_word < LIMB_BITS {
            limb |= (word << n_bits_in_word) & LIMB_MASK;
            word >>= LIMB_BITS - n_bits_in_word;
        }
        n_bits_in_word += 32 - LIMB_BITS;
        limbs.push(limb);
    }
    limbs
}

fn pack_le_9bit(limbs: &[u32], n_words: usize) -> Vec<u32> {
    let mut words = vec![0u32; n_words];
    for (i, &limb) in limbs.iter().enumerate() {
        assert!(limb <= LIMB_MASK);
        let bit_offset = i * LIMB_BITS as usize;
        let word = bit_offset / 32;
        let shift = bit_offset % 32;
        if word < n_words {
            words[word] |= limb << shift;
        }
        if shift > 32 - LIMB_BITS as usize && word + 1 < n_words {
            words[word + 1] |= limb >> (32 - shift);
        }
    }
    words
}

fn expected_memory_id_columns(
    values: &[u32],
    mults: &[u32],
    n_values: usize,
    column_length: usize,
    n_words: usize,
    n_limbs: usize,
) -> Vec<Vec<BaseField>> {
    let mut columns = vec![vec![BaseField::from_u32_unchecked(0); column_length]; n_limbs + 1];
    for row in 0..column_length {
        let row_words = if row < n_values {
            &values[row * n_words..(row + 1) * n_words]
        } else {
            &[]
        };
        for (limb, value) in split_le_9bit(row_words, n_limbs).into_iter().enumerate() {
            columns[limb][row] = BaseField::from_u32_unchecked(value);
        }
        if row < n_values {
            columns[n_limbs][row] = BaseField::from_u32_unchecked(mults[row]);
        }
    }
    columns
}

fn host_rc99_counts(
    columns: &[Vec<BaseField>],
    input_to_row_lut: &[u32],
    column_length: usize,
    n_pairs: usize,
    rc_table_size: usize,
) -> Vec<u32> {
    let mut counts = vec![0u32; 8 * rc_table_size];
    for row in 0..column_length {
        for pair in 0..n_pairs {
            let v0 = columns[2 * pair][row].0;
            let v1 = columns[2 * pair + 1][row].0;
            let rc_row = input_to_row_lut[((v0 << LIMB_BITS) | v1) as usize] as usize;
            counts[(pair % 8) * rc_table_size + rc_row] += 1;
        }
    }
    counts
}

fn pack_m31(values: &[u32]) -> PackedM31 {
    assert_eq!(values.len(), N_LANES);
    PackedM31::from_array(std::array::from_fn(|i| {
        BaseField::from_u32_unchecked(values[i])
    }))
}

fn combine_packed(
    values: &[PackedM31],
    alphas: &[SecureField],
    z: SecureField,
) -> PackedSecureField {
    let mut acc = PackedSecureField::zero();
    for (i, value) in values.iter().enumerate() {
        acc += PackedSecureField::broadcast(alphas[i]) * *value;
    }
    acc - PackedSecureField::broadcast(z)
}

fn add_small_i2b_values(
    id: PackedM31,
    limb0: PackedM31,
    limb1: PackedM31,
    limb2: PackedM31,
    rem: PackedM31,
    msb: PackedM31,
    mid_set: PackedM31,
) -> [PackedM31; 30] {
    let c = |value| PackedM31::broadcast(BaseField::from_u32_unchecked(value));
    let dss2 = mid_set * c(508);
    let dss3 = mid_set * c(511);
    let dss4 = msb * c(136) - mid_set;
    let dss5 = msb * c(256);
    [
        c(1_662_111_297),
        id,
        limb0,
        limb1,
        limb2,
        rem + dss2,
        dss3,
        dss3,
        dss3,
        dss3,
        dss3,
        dss3,
        dss3,
        dss3,
        dss3,
        dss3,
        dss3,
        dss3,
        dss3,
        dss3,
        dss3,
        dss3,
        dss3,
        dss4,
        c(0),
        c(0),
        c(0),
        c(0),
        c(0),
        dss5,
    ]
}

#[test]
fn memory_addr_to_id_trace_columns_match_chunks() {
    if !require_metal() {
        return;
    }

    let split = 4u32;
    let column_length = 8u32;
    let total = split * column_length;
    let ids = (0..total).map(|value| value + 100).collect::<Vec<_>>();
    let mults = (0..total).map(|value| total - value).collect::<Vec<_>>();
    let ids_dev = BaseFieldVec::from_u32_slice_private(&ids);
    let mults_dev = BaseFieldVec::from_u32_slice_private(&mults);

    let trace = BaseFieldVec::witness_memory_addr_to_id_trace_columns(
        &ids_dev,
        &mults_dev,
        total,
        column_length,
        split,
    );

    assert_eq!(trace.len(), 2 * split as usize);
    for chunk in 0..split as usize {
        let start = chunk * column_length as usize;
        let end = start + column_length as usize;
        assert_eq!(trace[2 * chunk].to_vec(), as_base(&ids[start..end]));
        assert_eq!(trace[2 * chunk + 1].to_vec(), as_base(&mults[start..end]));
    }
}

#[test]
fn memory_id_to_big_trace_columns_match_cairo_split() {
    if !require_metal() {
        return;
    }

    let n_values = 19usize;
    let column_length = 32usize;
    let values = (0..n_values * 8)
        .map(|i| {
            (i as u32)
                .wrapping_mul(0x9e37_79b9)
                .rotate_left((i % 31) as u32)
                ^ 0xa5a5_5a5a
        })
        .collect::<Vec<_>>();
    let mults = (0..n_values)
        .map(|i| 17u32.wrapping_mul(i as u32).wrapping_add(3))
        .collect::<Vec<_>>();
    let values_dev = BaseFieldVec::from_u32_slice_private(&values);
    let mults_dev = BaseFieldVec::from_u32_slice_private(&mults);

    let trace = BaseFieldVec::witness_memory_id_to_big_trace_columns(
        &values_dev,
        &mults_dev,
        n_values as u32,
        column_length as u32,
    );
    let expected = expected_memory_id_columns(&values, &mults, n_values, column_length, 8, 28);

    assert_eq!(trace.len(), 29);
    for (actual, expected) in trace.iter().zip(expected.iter()) {
        assert_eq!(actual.to_vec(), *expected);
    }
}

#[test]
fn memory_id_to_big_small_trace_columns_match_cairo_split() {
    if !require_metal() {
        return;
    }

    let n_values = 23usize;
    let column_length = 32usize;
    let values = (0..n_values * 4)
        .map(|i| {
            (i as u32)
                .wrapping_mul(0x85eb_ca6b)
                .rotate_left((i % 29) as u32)
                ^ 0x3c6e_f372
        })
        .collect::<Vec<_>>();
    let mults = (0..n_values)
        .map(|i| 23u32.wrapping_mul(i as u32).wrapping_add(5))
        .collect::<Vec<_>>();
    let values_dev = BaseFieldVec::from_u32_slice_private(&values);
    let mults_dev = BaseFieldVec::from_u32_slice_private(&mults);

    let trace = BaseFieldVec::witness_memory_id_to_big_small_trace_columns(
        &values_dev,
        &mults_dev,
        n_values as u32,
        column_length as u32,
    );
    let expected = expected_memory_id_columns(&values, &mults, n_values, column_length, 4, 8);

    assert_eq!(trace.len(), 9);
    for (actual, expected) in trace.iter().zip(expected.iter()) {
        assert_eq!(actual.to_vec(), *expected);
    }
}

#[test]
fn memory_rc99_count_matches_host_counts() {
    if !require_metal() {
        return;
    }

    let n_values = 19usize;
    let column_length = 32usize;
    let rc_table_size = 1024usize;
    let values = (0..n_values * 8)
        .map(|i| {
            (i as u32)
                .wrapping_mul(0x9e37_79b9)
                .rotate_left((i % 31) as u32)
                ^ 0xa5a5_5a5a
        })
        .collect::<Vec<_>>();
    let mults = (0..n_values)
        .map(|i| 17u32.wrapping_mul(i as u32).wrapping_add(3))
        .collect::<Vec<_>>();
    let values_dev = BaseFieldVec::from_u32_slice_private(&values);
    let mults_dev = BaseFieldVec::from_u32_slice_private(&mults);
    let trace = BaseFieldVec::witness_memory_id_to_big_trace_columns(
        &values_dev,
        &mults_dev,
        n_values as u32,
        column_length as u32,
    );
    let lut = (0..(1usize << (2 * LIMB_BITS)))
        .map(|key| ((key * 13 + 7) % rc_table_size) as u32)
        .collect::<Vec<_>>();
    let lut_dev = BaseFieldVec::from_u32_slice_private(&lut);

    let counts = BaseFieldVec::witness_memory_rc99_count(
        &trace[..28],
        &lut_dev,
        column_length as u32,
        14,
        rc_table_size as u32,
    );
    let host_columns = trace[..28]
        .iter()
        .map(BaseFieldVec::to_vec)
        .collect::<Vec<_>>();

    assert_eq!(
        counts,
        host_rc99_counts(&host_columns, &lut, column_length, 14, rc_table_size)
    );
}

#[test]
fn memory_logup_kernels_finalize_like_host_raw_logup() {
    if !require_metal() {
        return;
    }

    let log_size = 5u32;
    let column_length = 1usize << log_size;
    let rel0 = 517_791_011u32;
    let rel1 = 1_897_792_095u32;
    let memory_rel = 1_662_111_297u32;
    let id_offset = 17u32;
    let id_tag = 1 << 30;
    let alphas = (0..30)
        .map(|i| SecureField::from_u32_unchecked(3 + i * 11, 5 + i * 13, 7 + i * 17, 9 + i * 19))
        .collect::<Vec<_>>();
    let z = SecureField::from_u32_unchecked(101, 103, 107, 109);
    let limbs = (0..4)
        .map(|col| {
            (0..column_length)
                .map(|row| ((row * (col + 3) + col * 17) % 512) as u32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mults = (0..column_length)
        .map(|row| ((row * 5 + 3) % 97) as u32)
        .collect::<Vec<_>>();

    let metal_limbs = limbs
        .iter()
        .map(|values| BaseFieldVec::from_u32_slice_private(values))
        .collect::<Vec<_>>();
    let metal_mults = BaseFieldVec::from_u32_slice_private(&mults);
    let device_raws = vec![
        memory_rc_pair_logup(
            [
                &metal_limbs[0],
                &metal_limbs[1],
                &metal_limbs[2],
                &metal_limbs[3],
            ],
            rel0,
            rel1,
            &alphas[..3],
            z,
        ),
        memory_logup_inputs(
            &metal_limbs,
            &metal_mults,
            memory_rel,
            id_offset,
            id_tag,
            &alphas[..metal_limbs.len() + 2],
            z,
        ),
    ];
    let (metal_trace, metal_sum) = finalize_device_raw_logup(log_size, device_raws);

    let mut host = RawLogupTraceGenerator::new(log_size);
    {
        let mut rc_col = host.new_col();
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let start = vec_row * N_LANES;
            let end = start + N_LANES;
            let z_packed = PackedSecureField::broadcast(z);
            let limb0 = pack_m31(&limbs[0][start..end]);
            let limb1 = pack_m31(&limbs[1][start..end]);
            let limb2 = pack_m31(&limbs[2][start..end]);
            let limb3 = pack_m31(&limbs[3][start..end]);
            let d0 = PackedSecureField::broadcast(alphas[0]) * BaseField::from_u32_unchecked(rel0)
                + PackedSecureField::broadcast(alphas[1]) * limb0
                + PackedSecureField::broadcast(alphas[2]) * limb1
                - z_packed;
            let d1 = PackedSecureField::broadcast(alphas[0]) * BaseField::from_u32_unchecked(rel1)
                + PackedSecureField::broadcast(alphas[1]) * limb2
                + PackedSecureField::broadcast(alphas[2]) * limb3
                - z_packed;
            rc_col.write_frac(vec_row, d0 + d1, d0 * d1);
        }
        rc_col.finalize_col();
    }
    {
        let mut mem_col = host.new_col();
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let start = vec_row * N_LANES;
            let end = start + N_LANES;
            let z_packed = PackedSecureField::broadcast(z);
            let ids = PackedM31::from_array(std::array::from_fn(|lane| {
                BaseField::from_u32_unchecked((id_offset + (start + lane) as u32) | id_tag)
            }));
            let mut denom = PackedSecureField::broadcast(alphas[0])
                * BaseField::from_u32_unchecked(memory_rel)
                + PackedSecureField::broadcast(alphas[1]) * ids;
            for (i, limb_values) in limbs.iter().enumerate() {
                denom += PackedSecureField::broadcast(alphas[i + 2])
                    * pack_m31(&limb_values[start..end]);
            }
            denom -= z_packed;
            let mult = pack_m31(&mults[start..end]);
            mem_col.write_frac(vec_row, PackedSecureField::from(-mult), denom);
        }
        mem_col.finalize_col();
    }
    let (host_trace, host_sum) = host.into_raw().finalize_on_simd();

    assert_eq!(metal_sum, host_sum);
    assert_eq!(metal_trace.len(), host_trace.len());
    for (metal, host) in metal_trace.iter().zip(host_trace.iter()) {
        assert_eq!(metal.values.to_vec(), host.values.to_cpu());
    }
}

#[test]
fn memory_addr_to_id_logup_kernel_finalizes_like_host_raw_logup() {
    if !require_metal() {
        return;
    }

    let log_size = 5u32;
    let column_length = 1usize << log_size;
    let split = 4u32;
    let relation_id = 1_444_891_767u32;
    let alphas = (0..3)
        .map(|i| SecureField::from_u32_unchecked(11 + i * 7, 13 + i * 11, 17 + i * 13, 19 + i * 17))
        .collect::<Vec<_>>();
    let z = SecureField::from_u32_unchecked(101, 103, 107, 109);
    let total = split as usize * column_length;
    let ids = (0..total)
        .map(|i| ((i * 17 + 31) % 997) as u32)
        .collect::<Vec<_>>();
    let mults = (0..total)
        .map(|i| ((i * 19 + 5) % 113) as u32)
        .collect::<Vec<_>>();
    let ids_dev = BaseFieldVec::from_u32_slice_private(&ids);
    let mults_dev = BaseFieldVec::from_u32_slice_private(&mults);
    let trace_cols = BaseFieldVec::witness_memory_addr_to_id_trace_columns(
        &ids_dev,
        &mults_dev,
        total as u32,
        column_length as u32,
        split,
    );
    let device_raws = memory_addr_to_id_logup_inputs(&trace_cols, relation_id, &alphas, z, split);
    let (metal_trace, metal_sum) = finalize_device_raw_logup(log_size, device_raws);

    let mut host = RawLogupTraceGenerator::new(log_size);
    for pair in 0..split as usize / 2 {
        let mut col = host.new_col();
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let start = vec_row * N_LANES;
            let end = start + N_LANES;
            let chunk0 = 2 * pair;
            let chunk1 = 2 * pair + 1;
            let base0 = chunk0 * column_length;
            let base1 = chunk1 * column_length;
            let z_packed = PackedSecureField::broadcast(z);
            let addr0 = PackedM31::from_array(std::array::from_fn(|lane| {
                BaseField::from_u32_unchecked((base0 + start + lane + 1) as u32)
            }));
            let addr1 = PackedM31::from_array(std::array::from_fn(|lane| {
                BaseField::from_u32_unchecked((base1 + start + lane + 1) as u32)
            }));
            let id0 = pack_m31(&ids[base0 + start..base0 + end]);
            let id1 = pack_m31(&ids[base1 + start..base1 + end]);
            let mult0 = pack_m31(&mults[base0 + start..base0 + end]);
            let mult1 = pack_m31(&mults[base1 + start..base1 + end]);
            let d0 = PackedSecureField::broadcast(alphas[0])
                * BaseField::from_u32_unchecked(relation_id)
                + PackedSecureField::broadcast(alphas[1]) * addr0
                + PackedSecureField::broadcast(alphas[2]) * id0
                - z_packed;
            let d1 = PackedSecureField::broadcast(alphas[0])
                * BaseField::from_u32_unchecked(relation_id)
                + PackedSecureField::broadcast(alphas[1]) * addr1
                + PackedSecureField::broadcast(alphas[2]) * id1
                - z_packed;
            col.write_frac(
                vec_row,
                d0 * PackedSecureField::from(-mult1) + d1 * PackedSecureField::from(-mult0),
                d1 * d0,
            );
        }
        col.finalize_col();
    }
    let (host_trace, host_sum) = host.into_raw().finalize_on_simd();

    assert_eq!(metal_sum, host_sum);
    assert_eq!(metal_trace.len(), host_trace.len());
    for (metal, host) in metal_trace.iter().zip(host_trace.iter()) {
        assert_eq!(metal.values.to_vec(), host.values.to_cpu());
    }
}

#[test]
fn add_opcode_small_logup_kernel_finalizes_like_host_raw_logup() {
    if !require_metal() {
        return;
    }

    let log_size = LOG_N_LANES;
    let column_length = N_LANES;
    let n_rows = 1usize;
    let alphas = (0..30)
        .map(|i| SecureField::from_u32_unchecked(3 + i * 5, 7 + i * 11, 11 + i * 13, 13 + i * 17))
        .collect::<Vec<_>>();
    let z = SecureField::from_u32_unchecked(101, 103, 107, 109);
    let large_id_base = 1u32 << 30;
    let inputs = [1u32, 10, 20];
    let mut address_to_id = vec![0u32; 10];
    address_to_id[0] = large_id_base;
    address_to_id[9] = 0;
    let mut instruction_limbs = vec![0u32; 28];
    instruction_limbs[1] = 64;
    instruction_limbs[3] = 16;
    instruction_limbs[5] = 4;
    let big_values = pack_le_9bit(&instruction_limbs, 8);
    let small_values = [5u32, 0, 0, 0];

    let inputs_dev = U32Buffer::from_slice(&inputs).expect("inputs upload");
    let address_to_id_dev = U32Buffer::from_slice(&address_to_id).expect("address table upload");
    let big_values_dev = U32Buffer::from_slice(&big_values).expect("big values upload");
    let small_values_dev = U32Buffer::from_slice(&small_values).expect("small values upload");
    let trace_flat = U32Buffer::witness_add_opcode_small_trace(
        &inputs_dev,
        &address_to_id_dev,
        &big_values_dev,
        &small_values_dev,
        n_rows as u32,
        column_length as u32,
    )
    .expect("add_opcode_small trace");
    let device_raws =
        add_opcode_small_logup_inputs(&trace_flat, &alphas, z, n_rows as u32, column_length as u32);
    let (metal_trace, metal_sum) = finalize_device_raw_logup(log_size, device_raws);

    let trace = trace_flat
        .to_vec()
        .expect("add_opcode_small trace readback");
    let col = |idx: usize| pack_m31(&trace[idx * column_length..(idx + 1) * column_length]);
    let c = |value| PackedM31::broadcast(BaseField::from_u32_unchecked(value));
    let input_pc = col(0);
    let input_ap = col(1);
    let input_fp = col(2);
    let offset0 = col(3);
    let offset1 = col(4);
    let offset2 = col(5);
    let dst_base_fp = col(6);
    let op0_base_fp = col(7);
    let op1_imm = col(8);
    let op1_base_fp = col(9);
    let ap_update = col(10);
    let mem_dst_base = col(11);
    let mem0_base = col(12);
    let mem1_base = col(13);
    let dst_id = col(14);
    let dst_msb = col(15);
    let dst_mid_set = col(16);
    let dst_limb0 = col(17);
    let dst_limb1 = col(18);
    let dst_limb2 = col(19);
    let dst_rem = col(20);
    let op0_id = col(22);
    let op0_msb = col(23);
    let op0_mid_set = col(24);
    let op0_limb0 = col(25);
    let op0_limb1 = col(26);
    let op0_limb2 = col(27);
    let op0_rem = col(28);
    let op1_id = col(30);
    let op1_msb = col(31);
    let op1_mid_set = col(32);
    let op1_limb0 = col(33);
    let op1_limb1 = col(34);
    let op1_limb2 = col(35);
    let op1_rem = col(36);
    let enabler = col(38);
    let op1_base_ap = c(1) - op1_imm - op1_base_fp;
    let flags = dst_base_fp * c(8)
        + op0_base_fp * c(16)
        + op1_imm * c(32)
        + op1_base_fp * c(64)
        + op1_base_ap * c(128)
        + c(256);

    let mut host = RawLogupTraceGenerator::new(log_size);
    {
        let mut col_gen = host.new_col();
        let values0 = [
            c(1_719_106_205),
            input_pc,
            offset0,
            offset1,
            offset2,
            flags,
            ap_update * c(32) + c(256),
            c(0),
        ];
        let values1 = [c(1_444_891_767), mem_dst_base + offset0 - c(32768), dst_id];
        let d0 = combine_packed(&values0, &alphas, z);
        let d1 = combine_packed(&values1, &alphas, z);
        col_gen.write_frac(0, d0 + d1, d0 * d1);
        col_gen.finalize_col();
    }
    {
        let mut col_gen = host.new_col();
        let values0 = add_small_i2b_values(
            dst_id,
            dst_limb0,
            dst_limb1,
            dst_limb2,
            dst_rem,
            dst_msb,
            dst_mid_set,
        );
        let values1 = [c(1_444_891_767), mem0_base + offset1 - c(32768), op0_id];
        let d0 = combine_packed(&values0, &alphas, z);
        let d1 = combine_packed(&values1, &alphas, z);
        col_gen.write_frac(0, d0 + d1, d0 * d1);
        col_gen.finalize_col();
    }
    {
        let mut col_gen = host.new_col();
        let values0 = add_small_i2b_values(
            op0_id,
            op0_limb0,
            op0_limb1,
            op0_limb2,
            op0_rem,
            op0_msb,
            op0_mid_set,
        );
        let values1 = [c(1_444_891_767), mem1_base + offset2 - c(32768), op1_id];
        let d0 = combine_packed(&values0, &alphas, z);
        let d1 = combine_packed(&values1, &alphas, z);
        col_gen.write_frac(0, d0 + d1, d0 * d1);
        col_gen.finalize_col();
    }
    {
        let mut col_gen = host.new_col();
        let values0 = add_small_i2b_values(
            op1_id,
            op1_limb0,
            op1_limb1,
            op1_limb2,
            op1_rem,
            op1_msb,
            op1_mid_set,
        );
        let values1 = [c(428_564_188), input_pc, input_ap, input_fp];
        let d0 = combine_packed(&values0, &alphas, z);
        let d1 = combine_packed(&values1, &alphas, z);
        col_gen.write_frac(0, d0 * enabler + d1, d0 * d1);
        col_gen.finalize_col();
    }
    {
        let mut col_gen = host.new_col();
        let values = [
            c(428_564_188),
            input_pc + c(1) + op1_imm,
            input_ap + ap_update,
            input_fp,
        ];
        let denom = combine_packed(&values, &alphas, z);
        col_gen.write_frac(0, PackedSecureField::from(-enabler), denom);
        col_gen.finalize_col();
    }
    let (host_trace, host_sum) = host.into_raw().finalize_on_simd();

    assert_eq!(metal_sum, host_sum);
    assert_eq!(metal_trace.len(), host_trace.len());
    for (metal, host) in metal_trace.iter().zip(host_trace.iter()) {
        assert_eq!(metal.values.to_vec(), host.values.to_cpu());
    }

    let trace_cols = U32Buffer::witness_add_opcode_small_trace_columns(
        &inputs_dev,
        &address_to_id_dev,
        &big_values_dev,
        &small_values_dev,
        n_rows as u32,
        column_length as u32,
    )
    .expect("add_opcode_small trace columns")
    .into_iter()
    .map(BaseFieldVec::from_buffer)
    .collect::<Vec<_>>();
    let device_raws_from_cols = add_opcode_small_logup_inputs_from_cols(
        &trace_cols,
        &alphas,
        z,
        n_rows as u32,
        column_length as u32,
    );
    let (metal_trace_from_cols, metal_sum_from_cols) =
        finalize_device_raw_logup(log_size, device_raws_from_cols);
    assert_eq!(metal_sum_from_cols, host_sum);
    assert_eq!(metal_trace_from_cols.len(), host_trace.len());
    for (metal, host) in metal_trace_from_cols.iter().zip(host_trace.iter()) {
        assert_eq!(metal.values.to_vec(), host.values.to_cpu());
    }
}

#[test]
fn assert_eq_opcode_logup_kernel_finalizes_like_host_raw_logup() {
    if !require_metal() {
        return;
    }

    let log_size = LOG_N_LANES;
    let column_length = N_LANES;
    let n_rows = 1usize;
    let alphas = (0..8)
        .map(|i| SecureField::from_u32_unchecked(19 + i * 3, 23 + i * 5, 29 + i * 7, 31 + i * 11))
        .collect::<Vec<_>>();
    let z = SecureField::from_u32_unchecked(131, 137, 139, 149);
    let large_id_base = 1u32 << 30;
    let inputs = [1u32, 10, 20];
    let mut address_to_id = vec![0u32; 10];
    address_to_id[0] = large_id_base;
    address_to_id[9] = 0;
    let mut instruction_limbs = vec![0u32; 28];
    instruction_limbs[1] = 64;
    instruction_limbs[5] = 4;
    let big_values = pack_le_9bit(&instruction_limbs, 8);
    let small_values = [5u32, 0, 0, 0];

    let inputs_dev = U32Buffer::from_slice(&inputs).expect("inputs upload");
    let address_to_id_dev = U32Buffer::from_slice(&address_to_id).expect("address table upload");
    let big_values_dev = U32Buffer::from_slice(&big_values).expect("big values upload");
    let small_values_dev = U32Buffer::from_slice(&small_values).expect("small values upload");
    let trace_flat = U32Buffer::witness_assert_eq_opcode_trace(
        &inputs_dev,
        &address_to_id_dev,
        &big_values_dev,
        &small_values_dev,
        n_rows as u32,
        column_length as u32,
    )
    .expect("assert_eq_opcode trace");
    let device_raws =
        assert_eq_opcode_logup_inputs(&trace_flat, &alphas, z, n_rows as u32, column_length as u32);
    let (metal_trace, metal_sum) = finalize_device_raw_logup(log_size, device_raws);

    let trace = trace_flat
        .to_vec()
        .expect("assert_eq_opcode trace readback");
    let col = |idx: usize| pack_m31(&trace[idx * column_length..(idx + 1) * column_length]);
    let c = |value| PackedM31::broadcast(BaseField::from_u32_unchecked(value));
    let pc = col(0);
    let ap = col(1);
    let fp = col(2);
    let offset0 = col(3);
    let offset2 = col(4);
    let dst_base_fp = col(5);
    let op1_base_fp = col(6);
    let ap_update = col(7);
    let mem_dst_base = col(8);
    let mem1_base = col(9);
    let dst_id = col(10);
    let enabler = col(11);
    let vi_felt5 = dst_base_fp * c(8) + c(16) + op1_base_fp * c(64) + (c(1) - op1_base_fp) * c(128);
    let vi_felt6 = ap_update * c(32) + c(256);

    let mut host = RawLogupTraceGenerator::new(log_size);
    {
        let mut col_gen = host.new_col();
        let values0 = [
            c(1_719_106_205),
            pc,
            offset0,
            c(32_767),
            offset2,
            vi_felt5,
            vi_felt6,
            c(0),
        ];
        let values1 = [c(1_444_891_767), mem_dst_base + offset0 - c(32_768), dst_id];
        let d0 = combine_packed(&values0, &alphas, z);
        let d1 = combine_packed(&values1, &alphas, z);
        col_gen.write_frac(0, d0 + d1, d0 * d1);
        col_gen.finalize_col();
    }
    {
        let mut col_gen = host.new_col();
        let values0 = [c(1_444_891_767), mem1_base + offset2 - c(32_768), dst_id];
        let values1 = [c(428_564_188), pc, ap, fp];
        let d0 = combine_packed(&values0, &alphas, z);
        let d1 = combine_packed(&values1, &alphas, z);
        col_gen.write_frac(0, d0 * enabler + d1, d0 * d1);
        col_gen.finalize_col();
    }
    {
        let mut col_gen = host.new_col();
        let values = [c(428_564_188), pc + c(1), ap + ap_update, fp];
        let denom = combine_packed(&values, &alphas, z);
        col_gen.write_frac(0, PackedSecureField::from(-enabler), denom);
        col_gen.finalize_col();
    }
    let (host_trace, host_sum) = host.into_raw().finalize_on_simd();

    assert_eq!(metal_sum, host_sum);
    assert_eq!(metal_trace.len(), host_trace.len());
    for (metal, host) in metal_trace.iter().zip(host_trace.iter()) {
        assert_eq!(metal.values.to_vec(), host.values.to_cpu());
    }

    let trace_cols = U32Buffer::witness_assert_eq_opcode_trace_columns(
        &inputs_dev,
        &address_to_id_dev,
        &big_values_dev,
        &small_values_dev,
        n_rows as u32,
        column_length as u32,
    )
    .expect("assert_eq_opcode trace columns")
    .into_iter()
    .map(BaseFieldVec::from_buffer)
    .collect::<Vec<_>>();
    let device_raws_from_cols = assert_eq_opcode_logup_inputs_from_cols(
        &trace_cols,
        &alphas,
        z,
        n_rows as u32,
        column_length as u32,
    );
    let (metal_trace_from_cols, metal_sum_from_cols) =
        finalize_device_raw_logup(log_size, device_raws_from_cols);
    assert_eq!(metal_sum_from_cols, host_sum);
    assert_eq!(metal_trace_from_cols.len(), host_trace.len());
    for (metal, host) in metal_trace_from_cols.iter().zip(host_trace.iter()) {
        assert_eq!(metal.values.to_vec(), host.values.to_cpu());
    }
}
