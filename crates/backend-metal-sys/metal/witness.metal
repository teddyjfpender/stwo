#include "secure_field_support.h"

#define STWO_METAL_FELT252_BITS_PER_WORD 9u
#define STWO_METAL_LIMB_MASK 511u
#define STWO_METAL_LOGUP_LANE_BLOCK 16u

static inline StwoMetalQm31 stwo_metal_qm31_mul_m31(StwoMetalQm31 value, uint scalar) {
    return stwo_metal_qm31_mul_base(value, scalar);
}

static inline StwoMetalQm31 stwo_metal_load_qm31_from_constant(
    constant uint *values,
    uint index
) {
    uint base = index * 4u;
    return StwoMetalQm31 {
        values[base + 0u],
        values[base + 1u],
        values[base + 2u],
        values[base + 3u],
    };
}

static inline void stwo_metal_store_packed_lane_qm31(
    device uint *packed,
    uint index,
    StwoMetalQm31 value
) {
    uint lane = index & (STWO_METAL_LOGUP_LANE_BLOCK - 1u);
    uint base = (index / STWO_METAL_LOGUP_LANE_BLOCK) *
        (4u * STWO_METAL_LOGUP_LANE_BLOCK) + lane;
    packed[base] = value.a;
    packed[base + STWO_METAL_LOGUP_LANE_BLOCK] = value.b;
    packed[base + 2u * STWO_METAL_LOGUP_LANE_BLOCK] = value.c;
    packed[base + 3u * STWO_METAL_LOGUP_LANE_BLOCK] = value.d;
}

static inline void stwo_metal_split_le_9bit(
    thread const uint *words,
    uint n_words,
    thread uint *limbs,
    uint n_limbs
) {
    uint n_bits_in_word = 32u;
    uint word_i = 0u;
    uint word = n_words == 0u ? 0u : words[0];
    for (uint limb = 0u; limb < n_limbs; ++limb) {
        if (n_bits_in_word > STWO_METAL_FELT252_BITS_PER_WORD) {
            limbs[limb] = word & STWO_METAL_LIMB_MASK;
            word >>= STWO_METAL_FELT252_BITS_PER_WORD;
            n_bits_in_word -= STWO_METAL_FELT252_BITS_PER_WORD;
            continue;
        }

        limbs[limb] = word;
        word_i += 1u;
        word = word_i < n_words ? words[word_i] : 0u;
        if (n_bits_in_word < STWO_METAL_FELT252_BITS_PER_WORD) {
            limbs[limb] |= (word << n_bits_in_word) & STWO_METAL_LIMB_MASK;
            word >>= STWO_METAL_FELT252_BITS_PER_WORD - n_bits_in_word;
        }
        n_bits_in_word += 32u - STWO_METAL_FELT252_BITS_PER_WORD;
    }
}

kernel void witness_memory_id_to_big_trace(
    device const uint *big_values [[buffer(0)]],
    device const uint *mults [[buffer(1)]],
    device uint *trace [[buffer(2)]],
    constant uint &n_values [[buffer(3)]],
    constant uint &column_length [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    thread uint words[8];
    for (uint i = 0u; i < 8u; ++i) {
        words[i] = row < n_values ? big_values[row * 8u + i] : 0u;
    }

    thread uint limbs[28];
    stwo_metal_split_le_9bit(words, 8u, limbs, 28u);
    for (uint i = 0u; i < 28u; ++i) {
        trace[i * column_length + row] = limbs[i];
    }
    trace[28u * column_length + row] = row < n_values ? mults[row] : 0u;
}

kernel void witness_memory_id_to_big_trace_columns(
    device const uint *big_values [[buffer(0)]],
    device const uint *mults [[buffer(1)]],
    device const uint64_t *trace_col_addrs [[buffer(2)]],
    constant uint &n_values [[buffer(3)]],
    constant uint &column_length [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    thread uint words[8];
    for (uint i = 0u; i < 8u; ++i) {
        words[i] = row < n_values ? big_values[row * 8u + i] : 0u;
    }

    thread uint limbs[28];
    stwo_metal_split_le_9bit(words, 8u, limbs, 28u);
    for (uint i = 0u; i < 28u; ++i) {
        device uint *col = (device uint *)trace_col_addrs[i];
        col[row] = limbs[i];
    }
    device uint *mult_col = (device uint *)trace_col_addrs[28];
    mult_col[row] = row < n_values ? mults[row] : 0u;
}

kernel void witness_memory_id_to_big_small_trace(
    device const uint *small_values [[buffer(0)]],
    device const uint *mults [[buffer(1)]],
    device uint *trace [[buffer(2)]],
    constant uint &n_values [[buffer(3)]],
    constant uint &column_length [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    thread uint words[4];
    for (uint i = 0u; i < 4u; ++i) {
        words[i] = row < n_values ? small_values[row * 4u + i] : 0u;
    }

    thread uint limbs[8];
    stwo_metal_split_le_9bit(words, 4u, limbs, 8u);
    for (uint i = 0u; i < 8u; ++i) {
        trace[i * column_length + row] = limbs[i];
    }
    trace[8u * column_length + row] = row < n_values ? mults[row] : 0u;
}

kernel void witness_memory_id_to_big_small_trace_columns(
    device const uint *small_values [[buffer(0)]],
    device const uint *mults [[buffer(1)]],
    device const uint64_t *trace_col_addrs [[buffer(2)]],
    constant uint &n_values [[buffer(3)]],
    constant uint &column_length [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    thread uint words[4];
    for (uint i = 0u; i < 4u; ++i) {
        words[i] = row < n_values ? small_values[row * 4u + i] : 0u;
    }

    thread uint limbs[8];
    stwo_metal_split_le_9bit(words, 4u, limbs, 8u);
    for (uint i = 0u; i < 8u; ++i) {
        device uint *col = (device uint *)trace_col_addrs[i];
        col[row] = limbs[i];
    }
    device uint *mult_col = (device uint *)trace_col_addrs[8];
    mult_col[row] = row < n_values ? mults[row] : 0u;
}

kernel void witness_memory_rc99_count(
    device const uint64_t *limb_col_addrs [[buffer(0)]],
    device const uint *input_to_row_lut [[buffer(1)]],
    device atomic_uint *counts [[buffer(2)]],
    constant uint &n_pairs [[buffer(3)]],
    constant uint &column_length [[buffer(4)]],
    constant uint &rc_table_size [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    for (uint pair = 0u; pair < n_pairs; ++pair) {
        device const uint *limb0 = (device const uint *)limb_col_addrs[2u * pair];
        device const uint *limb1 = (device const uint *)limb_col_addrs[2u * pair + 1u];
        uint key = (limb0[row] << STWO_METAL_FELT252_BITS_PER_WORD) | limb1[row];
        uint rc_row = input_to_row_lut[key];
        atomic_fetch_add_explicit(
            &counts[(pair % 8u) * rc_table_size + rc_row],
            1u,
            memory_order_relaxed
        );
    }
}

kernel void witness_memory_rc_pair_logup(
    device const uint *limb_a [[buffer(0)]],
    device const uint *limb_b [[buffer(1)]],
    device const uint *limb_c [[buffer(2)]],
    device const uint *limb_d [[buffer(3)]],
    device uint *denom_packed [[buffer(4)]],
    device uint *num0 [[buffer(5)]],
    device uint *num1 [[buffer(6)]],
    device uint *num2 [[buffer(7)]],
    device uint *num3 [[buffer(8)]],
    device const uint *alpha_powers [[buffer(9)]],
    constant uint *z_limbs [[buffer(10)]],
    constant uint &rel_id0 [[buffer(11)]],
    constant uint &rel_id1 [[buffer(12)]],
    constant uint &column_length [[buffer(13)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    StwoMetalQm31 alpha0 = stwo_metal_load_qm31(alpha_powers, 0u);
    StwoMetalQm31 alpha1 = stwo_metal_load_qm31(alpha_powers, 1u);
    StwoMetalQm31 alpha2 = stwo_metal_load_qm31(alpha_powers, 2u);
    StwoMetalQm31 z = stwo_metal_load_qm31_from_constant(z_limbs, 0u);

    StwoMetalQm31 d0 = stwo_metal_qm31_mul_m31(alpha0, rel_id0);
    d0 = stwo_metal_qm31_add(d0, stwo_metal_qm31_mul_m31(alpha1, limb_a[row]));
    d0 = stwo_metal_qm31_add(d0, stwo_metal_qm31_mul_m31(alpha2, limb_b[row]));
    d0 = stwo_metal_qm31_sub(d0, z);

    StwoMetalQm31 d1 = stwo_metal_qm31_mul_m31(alpha0, rel_id1);
    d1 = stwo_metal_qm31_add(d1, stwo_metal_qm31_mul_m31(alpha1, limb_c[row]));
    d1 = stwo_metal_qm31_add(d1, stwo_metal_qm31_mul_m31(alpha2, limb_d[row]));
    d1 = stwo_metal_qm31_sub(d1, z);

    StwoMetalQm31 numerator = stwo_metal_qm31_add(d0, d1);
    StwoMetalQm31 denominator = stwo_metal_qm31_mul(d0, d1);
    num0[row] = numerator.a;
    num1[row] = numerator.b;
    num2[row] = numerator.c;
    num3[row] = numerator.d;
    stwo_metal_store_packed_lane_qm31(denom_packed, row, denominator);
}

kernel void witness_memory_logup_inputs(
    device const uint64_t *limb_col_addrs [[buffer(0)]],
    device const uint *mults [[buffer(1)]],
    device uint *denom_packed [[buffer(2)]],
    device uint *num0 [[buffer(3)]],
    device uint *num1 [[buffer(4)]],
    device uint *num2 [[buffer(5)]],
    device uint *num3 [[buffer(6)]],
    device const uint *alpha_powers [[buffer(7)]],
    constant uint *z_limbs [[buffer(8)]],
    constant uint &relation_id [[buffer(9)]],
    constant uint &id_offset [[buffer(10)]],
    constant uint &id_tag [[buffer(11)]],
    constant uint &n_limbs [[buffer(12)]],
    constant uint &column_length [[buffer(13)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    StwoMetalQm31 acc = stwo_metal_qm31_mul_m31(
        stwo_metal_load_qm31(alpha_powers, 0u),
        relation_id
    );
    uint id = (id_offset + row) | id_tag;
    acc = stwo_metal_qm31_add(
        acc,
        stwo_metal_qm31_mul_m31(stwo_metal_load_qm31(alpha_powers, 1u), id)
    );
    for (uint limb = 0u; limb < n_limbs; ++limb) {
        device const uint *limb_col = (device const uint *)limb_col_addrs[limb];
        acc = stwo_metal_qm31_add(
            acc,
            stwo_metal_qm31_mul_m31(
                stwo_metal_load_qm31(alpha_powers, 2u + limb),
                limb_col[row]
            )
        );
    }
    StwoMetalQm31 z = stwo_metal_load_qm31_from_constant(z_limbs, 0u);
    StwoMetalQm31 denominator = stwo_metal_qm31_sub(acc, z);

    num0[row] = stwo_metal_m31_neg(mults[row]);
    num1[row] = 0u;
    num2[row] = 0u;
    num3[row] = 0u;
    stwo_metal_store_packed_lane_qm31(denom_packed, row, denominator);
}

kernel void witness_memory_addr_to_id_trace(
    device const uint *ids [[buffer(0)]],
    device const uint *mults [[buffer(1)]],
    device uint *trace [[buffer(2)]],
    constant uint &n_ids [[buffer(3)]],
    constant uint &column_length [[buffer(4)]],
    constant uint &split [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    uint total = split * column_length;
    if (index >= total) {
        return;
    }

    uint chunk = index / column_length;
    uint row = index - chunk * column_length;
    uint src_index = chunk * column_length + row;
    uint id = src_index < n_ids ? ids[src_index] : 0u;
    uint mult = src_index < n_ids ? mults[src_index] : 0u;

    uint trace_base = chunk * 2u * column_length + row;
    trace[trace_base] = id;
    trace[trace_base + column_length] = mult;
}

kernel void witness_memory_addr_to_id_trace_columns(
    device const uint *ids [[buffer(0)]],
    device const uint *mults [[buffer(1)]],
    device const uint64_t *trace_col_addrs [[buffer(2)]],
    constant uint &n_ids [[buffer(3)]],
    constant uint &column_length [[buffer(4)]],
    constant uint &split [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    uint total = split * column_length;
    if (index >= total) {
        return;
    }

    uint chunk = index / column_length;
    uint row = index - chunk * column_length;
    uint src_index = chunk * column_length + row;
    uint id = src_index < n_ids ? ids[src_index] : 0u;
    uint mult = src_index < n_ids ? mults[src_index] : 0u;

    device uint *id_col = (device uint *)trace_col_addrs[2u * chunk];
    device uint *mult_col = (device uint *)trace_col_addrs[2u * chunk + 1u];
    id_col[row] = id;
    mult_col[row] = mult;
}

static inline StwoMetalQm31 stwo_metal_memory_addr_to_id_denom(
    device const uint *alpha_powers,
    constant uint *z_limbs,
    uint relation_id,
    uint address,
    uint id
) {
    StwoMetalQm31 acc = stwo_metal_qm31_mul_m31(
        stwo_metal_load_qm31(alpha_powers, 0u),
        relation_id
    );
    acc = stwo_metal_qm31_add(
        acc,
        stwo_metal_qm31_mul_m31(stwo_metal_load_qm31(alpha_powers, 1u), address)
    );
    acc = stwo_metal_qm31_add(
        acc,
        stwo_metal_qm31_mul_m31(stwo_metal_load_qm31(alpha_powers, 2u), id)
    );
    return stwo_metal_qm31_sub(acc, stwo_metal_load_qm31_from_constant(z_limbs, 0u));
}

static inline void stwo_metal_memory_addr_to_id_store_raw_logup_col(
    device const uint64_t *num_col_addrs,
    device const uint64_t *denom_col_addrs,
    uint col,
    uint row,
    StwoMetalQm31 numerator,
    StwoMetalQm31 denominator
) {
    device uint *num0 = (device uint *)num_col_addrs[col * 4u + 0u];
    device uint *num1 = (device uint *)num_col_addrs[col * 4u + 1u];
    device uint *num2 = (device uint *)num_col_addrs[col * 4u + 2u];
    device uint *num3 = (device uint *)num_col_addrs[col * 4u + 3u];
    device uint *denom = (device uint *)denom_col_addrs[col];
    num0[row] = numerator.a;
    num1[row] = numerator.b;
    num2[row] = numerator.c;
    num3[row] = numerator.d;
    stwo_metal_store_packed_lane_qm31(denom, row, denominator);
}

kernel void witness_memory_addr_to_id_interaction_raw(
    device const uint64_t *trace_col_addrs [[buffer(0)]],
    device const uint64_t *num_col_addrs [[buffer(1)]],
    device const uint64_t *denom_col_addrs [[buffer(2)]],
    device const uint *alpha_powers [[buffer(3)]],
    constant uint *z_limbs [[buffer(4)]],
    constant uint &relation_id [[buffer(5)]],
    constant uint &column_length [[buffer(6)]],
    constant uint &n_logup_cols [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    uint total = column_length * n_logup_cols;
    if (index >= total) {
        return;
    }

    uint col = index / column_length;
    uint row = index - col * column_length;

    device const uint *id0_col = (device const uint *)trace_col_addrs[4u * col + 0u];
    device const uint *mult0_col = (device const uint *)trace_col_addrs[4u * col + 1u];
    device const uint *id1_col = (device const uint *)trace_col_addrs[4u * col + 2u];
    device const uint *mult1_col = (device const uint *)trace_col_addrs[4u * col + 3u];

    uint addr0 = row + 1u + (2u * col) * column_length;
    uint addr1 = row + 1u + (2u * col + 1u) * column_length;
    StwoMetalQm31 denom0 = stwo_metal_memory_addr_to_id_denom(
        alpha_powers,
        z_limbs,
        relation_id,
        addr0,
        id0_col[row]
    );
    StwoMetalQm31 denom1 = stwo_metal_memory_addr_to_id_denom(
        alpha_powers,
        z_limbs,
        relation_id,
        addr1,
        id1_col[row]
    );

    StwoMetalQm31 numerator = stwo_metal_qm31_add(
        stwo_metal_qm31_mul_m31(denom0, stwo_metal_m31_neg(mult1_col[row])),
        stwo_metal_qm31_mul_m31(denom1, stwo_metal_m31_neg(mult0_col[row]))
    );
    StwoMetalQm31 denominator = stwo_metal_qm31_mul(denom1, denom0);

    stwo_metal_memory_addr_to_id_store_raw_logup_col(
        num_col_addrs,
        denom_col_addrs,
        col,
        row,
        numerator,
        denominator
    );
}

static inline uint stwo_metal_opcode_mem_addr_to_id(
    device const uint *address_to_id,
    uint address
) {
    return address_to_id[address - 1u];
}

static inline void stwo_metal_opcode_mem_id_to_limbs(
    uint id,
    device const uint *big_values,
    device const uint *small_values,
    thread uint *limbs,
    uint n_limbs
) {
    uint tag = id >> 30u;
    uint value_id = id & 0x3fffffffu;
    thread uint words[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

    if (tag == 1u) {
        for (uint i = 0u; i < 8u; ++i) {
            words[i] = big_values[value_id * 8u + i];
        }
    } else {
        for (uint i = 0u; i < 4u; ++i) {
            words[i] = small_values[value_id * 4u + i];
        }
    }

    stwo_metal_split_le_9bit(words, 8u, limbs, n_limbs);
}

static inline uint stwo_metal_opcode_recombine_29(thread const uint *limbs) {
    return stwo_metal_m31_add(
        stwo_metal_m31_add(limbs[0], stwo_metal_m31_mul(limbs[1], 512u)),
        stwo_metal_m31_add(
            stwo_metal_m31_mul(limbs[2], 262144u),
            stwo_metal_m31_mul(limbs[3], 134217728u)
        )
    );
}

struct StwoMetalDecodedInstruction {
    uint offset0;
    uint offset1;
    uint offset2;
    uint dst_base_fp;
    uint op0_base_fp;
    uint op1_imm;
    uint op1_base_fp;
    uint ap_update_add_1;
};

static inline StwoMetalDecodedInstruction stwo_metal_opcode_decode_instruction(
    thread const uint *limbs
) {
    uint flags_word = (limbs[5] >> 3u) + (limbs[6] << 6u);
    return StwoMetalDecodedInstruction {
        limbs[0] + ((limbs[1] & 0x7fu) << 9u),
        (limbs[1] >> 7u) + (limbs[2] << 2u) + ((limbs[3] & 0x1fu) << 11u),
        (limbs[3] >> 5u) + (limbs[4] << 4u) + ((limbs[5] & 0x7u) << 13u),
        (flags_word >> 0u) & 1u,
        (flags_word >> 1u) & 1u,
        (flags_word >> 2u) & 1u,
        (flags_word >> 3u) & 1u,
        (flags_word >> 11u) & 1u,
    };
}

struct StwoMetalSmallSign {
    uint msb;
    uint mid_limbs_set;
};

static inline StwoMetalSmallSign stwo_metal_opcode_decode_small_sign(
    thread const uint *limbs
) {
    uint msb = limbs[27] == 256u ? 1u : 0u;
    uint mid_limbs_set = (limbs[20] == 511u ? 1u : 0u) & msb;
    return StwoMetalSmallSign { msb, mid_limbs_set };
}

static inline void stwo_metal_opcode_write_small_read_trace(
    device uint *trace,
    uint column_length,
    uint row,
    uint base_col,
    uint id,
    thread const uint *limbs
) {
    StwoMetalSmallSign sign = stwo_metal_opcode_decode_small_sign(limbs);
    uint remainder_bits = limbs[3] & 3u;
    trace[(base_col + 0u) * column_length + row] = id;
    trace[(base_col + 1u) * column_length + row] = sign.msb;
    trace[(base_col + 2u) * column_length + row] = sign.mid_limbs_set;
    trace[(base_col + 3u) * column_length + row] = limbs[0];
    trace[(base_col + 4u) * column_length + row] = limbs[1];
    trace[(base_col + 5u) * column_length + row] = limbs[2];
    trace[(base_col + 6u) * column_length + row] = remainder_bits;
    trace[(base_col + 7u) * column_length + row] = (remainder_bits & 2u) >> 1u;
}

kernel void witness_add_opcode_small_trace(
    device const uint *inputs [[buffer(0)]],
    device const uint *address_to_id [[buffer(1)]],
    device const uint *big_values [[buffer(2)]],
    device const uint *small_values [[buffer(3)]],
    device uint *trace [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    constant uint &column_length [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    uint src_row = row < n_rows ? row : 0u;
    uint input_base = src_row * 3u;
    uint pc = inputs[input_base];
    uint ap = inputs[input_base + 1u];
    uint fp = inputs[input_base + 2u];

    trace[0u * column_length + row] = pc;
    trace[1u * column_length + row] = ap;
    trace[2u * column_length + row] = fp;

    uint instr_id = stwo_metal_opcode_mem_addr_to_id(address_to_id, pc);
    thread uint instr_limbs[28];
    stwo_metal_opcode_mem_id_to_limbs(instr_id, big_values, small_values, instr_limbs, 28u);
    StwoMetalDecodedInstruction instr = stwo_metal_opcode_decode_instruction(instr_limbs);

    trace[3u * column_length + row] = instr.offset0;
    trace[4u * column_length + row] = instr.offset1;
    trace[5u * column_length + row] = instr.offset2;
    trace[6u * column_length + row] = instr.dst_base_fp;
    trace[7u * column_length + row] = instr.op0_base_fp;
    trace[8u * column_length + row] = instr.op1_imm;
    trace[9u * column_length + row] = instr.op1_base_fp;
    trace[10u * column_length + row] = instr.ap_update_add_1;

    uint op1_base_ap = stwo_metal_m31_sub(
        stwo_metal_m31_sub(1u, instr.op1_imm),
        instr.op1_base_fp
    );
    uint offset0 = stwo_metal_m31_sub(instr.offset0, 32768u);
    uint offset1 = stwo_metal_m31_sub(instr.offset1, 32768u);
    uint offset2 = stwo_metal_m31_sub(instr.offset2, 32768u);
    uint mem_dst_base = stwo_metal_m31_add(
        stwo_metal_m31_mul(instr.dst_base_fp, fp),
        stwo_metal_m31_mul(stwo_metal_m31_sub(1u, instr.dst_base_fp), ap)
    );
    uint mem0_base = stwo_metal_m31_add(
        stwo_metal_m31_mul(instr.op0_base_fp, fp),
        stwo_metal_m31_mul(stwo_metal_m31_sub(1u, instr.op0_base_fp), ap)
    );
    uint mem1_base = stwo_metal_m31_add(
        stwo_metal_m31_add(
            stwo_metal_m31_mul(instr.op1_imm, pc),
            stwo_metal_m31_mul(instr.op1_base_fp, fp)
        ),
        stwo_metal_m31_mul(op1_base_ap, ap)
    );

    trace[11u * column_length + row] = mem_dst_base;
    trace[12u * column_length + row] = mem0_base;
    trace[13u * column_length + row] = mem1_base;

    uint dst_id = stwo_metal_opcode_mem_addr_to_id(
        address_to_id,
        stwo_metal_m31_add(mem_dst_base, offset0)
    );
    thread uint dst_limbs[28];
    stwo_metal_opcode_mem_id_to_limbs(dst_id, big_values, small_values, dst_limbs, 28u);
    stwo_metal_opcode_write_small_read_trace(trace, column_length, row, 14u, dst_id, dst_limbs);

    uint op0_id = stwo_metal_opcode_mem_addr_to_id(
        address_to_id,
        stwo_metal_m31_add(mem0_base, offset1)
    );
    thread uint op0_limbs[28];
    stwo_metal_opcode_mem_id_to_limbs(op0_id, big_values, small_values, op0_limbs, 28u);
    stwo_metal_opcode_write_small_read_trace(trace, column_length, row, 22u, op0_id, op0_limbs);

    uint op1_id = stwo_metal_opcode_mem_addr_to_id(
        address_to_id,
        stwo_metal_m31_add(mem1_base, offset2)
    );
    thread uint op1_limbs[28];
    stwo_metal_opcode_mem_id_to_limbs(op1_id, big_values, small_values, op1_limbs, 28u);
    stwo_metal_opcode_write_small_read_trace(trace, column_length, row, 30u, op1_id, op1_limbs);

    trace[38u * column_length + row] = row < n_rows ? 1u : 0u;
}

static inline void stwo_metal_opcode_store_trace_col(
    device const uint64_t *trace_col_addrs,
    uint col,
    uint row,
    uint value
) {
    device uint *trace_col = (device uint *)trace_col_addrs[col];
    trace_col[row] = value;
}

static inline void stwo_metal_opcode_write_small_read_trace_cols(
    device const uint64_t *trace_col_addrs,
    uint row,
    uint base_col,
    uint id,
    thread const uint *limbs
) {
    StwoMetalSmallSign sign = stwo_metal_opcode_decode_small_sign(limbs);
    uint remainder_bits = limbs[3] & 3u;
    stwo_metal_opcode_store_trace_col(trace_col_addrs, base_col + 0u, row, id);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, base_col + 1u, row, sign.msb);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, base_col + 2u, row, sign.mid_limbs_set);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, base_col + 3u, row, limbs[0]);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, base_col + 4u, row, limbs[1]);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, base_col + 5u, row, limbs[2]);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, base_col + 6u, row, remainder_bits);
    stwo_metal_opcode_store_trace_col(
        trace_col_addrs,
        base_col + 7u,
        row,
        (remainder_bits & 2u) >> 1u
    );
}

kernel void witness_add_opcode_small_trace_columns(
    device const uint *inputs [[buffer(0)]],
    device const uint *address_to_id [[buffer(1)]],
    device const uint *big_values [[buffer(2)]],
    device const uint *small_values [[buffer(3)]],
    device const uint64_t *trace_col_addrs [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    constant uint &column_length [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    uint src_row = row < n_rows ? row : 0u;
    uint input_base = src_row * 3u;
    uint pc = inputs[input_base];
    uint ap = inputs[input_base + 1u];
    uint fp = inputs[input_base + 2u];

    stwo_metal_opcode_store_trace_col(trace_col_addrs, 0u, row, pc);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 1u, row, ap);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 2u, row, fp);

    uint instr_id = stwo_metal_opcode_mem_addr_to_id(address_to_id, pc);
    thread uint instr_limbs[28];
    stwo_metal_opcode_mem_id_to_limbs(instr_id, big_values, small_values, instr_limbs, 28u);
    StwoMetalDecodedInstruction instr = stwo_metal_opcode_decode_instruction(instr_limbs);

    stwo_metal_opcode_store_trace_col(trace_col_addrs, 3u, row, instr.offset0);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 4u, row, instr.offset1);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 5u, row, instr.offset2);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 6u, row, instr.dst_base_fp);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 7u, row, instr.op0_base_fp);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 8u, row, instr.op1_imm);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 9u, row, instr.op1_base_fp);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 10u, row, instr.ap_update_add_1);

    uint op1_base_ap = stwo_metal_m31_sub(
        stwo_metal_m31_sub(1u, instr.op1_imm),
        instr.op1_base_fp
    );
    uint offset0 = stwo_metal_m31_sub(instr.offset0, 32768u);
    uint offset1 = stwo_metal_m31_sub(instr.offset1, 32768u);
    uint offset2 = stwo_metal_m31_sub(instr.offset2, 32768u);
    uint mem_dst_base = stwo_metal_m31_add(
        stwo_metal_m31_mul(instr.dst_base_fp, fp),
        stwo_metal_m31_mul(stwo_metal_m31_sub(1u, instr.dst_base_fp), ap)
    );
    uint mem0_base = stwo_metal_m31_add(
        stwo_metal_m31_mul(instr.op0_base_fp, fp),
        stwo_metal_m31_mul(stwo_metal_m31_sub(1u, instr.op0_base_fp), ap)
    );
    uint mem1_base = stwo_metal_m31_add(
        stwo_metal_m31_add(
            stwo_metal_m31_mul(instr.op1_imm, pc),
            stwo_metal_m31_mul(instr.op1_base_fp, fp)
        ),
        stwo_metal_m31_mul(op1_base_ap, ap)
    );

    stwo_metal_opcode_store_trace_col(trace_col_addrs, 11u, row, mem_dst_base);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 12u, row, mem0_base);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 13u, row, mem1_base);

    uint dst_id = stwo_metal_opcode_mem_addr_to_id(
        address_to_id,
        stwo_metal_m31_add(mem_dst_base, offset0)
    );
    thread uint dst_limbs[28];
    stwo_metal_opcode_mem_id_to_limbs(dst_id, big_values, small_values, dst_limbs, 28u);
    stwo_metal_opcode_write_small_read_trace_cols(
        trace_col_addrs,
        row,
        14u,
        dst_id,
        dst_limbs
    );

    uint op0_id = stwo_metal_opcode_mem_addr_to_id(
        address_to_id,
        stwo_metal_m31_add(mem0_base, offset1)
    );
    thread uint op0_limbs[28];
    stwo_metal_opcode_mem_id_to_limbs(op0_id, big_values, small_values, op0_limbs, 28u);
    stwo_metal_opcode_write_small_read_trace_cols(
        trace_col_addrs,
        row,
        22u,
        op0_id,
        op0_limbs
    );

    uint op1_id = stwo_metal_opcode_mem_addr_to_id(
        address_to_id,
        stwo_metal_m31_add(mem1_base, offset2)
    );
    thread uint op1_limbs[28];
    stwo_metal_opcode_mem_id_to_limbs(op1_id, big_values, small_values, op1_limbs, 28u);
    stwo_metal_opcode_write_small_read_trace_cols(
        trace_col_addrs,
        row,
        30u,
        op1_id,
        op1_limbs
    );

    stwo_metal_opcode_store_trace_col(trace_col_addrs, 38u, row, row < n_rows ? 1u : 0u);
}

kernel void witness_assert_eq_opcode_trace(
    device const uint *inputs [[buffer(0)]],
    device const uint *address_to_id [[buffer(1)]],
    device const uint *big_values [[buffer(2)]],
    device const uint *small_values [[buffer(3)]],
    device uint *trace [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    constant uint &column_length [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    uint src_row = row < n_rows ? row : 0u;
    uint input_base = src_row * 3u;
    uint pc = inputs[input_base];
    uint ap = inputs[input_base + 1u];
    uint fp = inputs[input_base + 2u];

    trace[0u * column_length + row] = pc;
    trace[1u * column_length + row] = ap;
    trace[2u * column_length + row] = fp;

    uint instr_id = stwo_metal_opcode_mem_addr_to_id(address_to_id, pc);
    thread uint instr_limbs[28];
    stwo_metal_opcode_mem_id_to_limbs(instr_id, big_values, small_values, instr_limbs, 28u);
    StwoMetalDecodedInstruction instr = stwo_metal_opcode_decode_instruction(instr_limbs);

    trace[3u * column_length + row] = instr.offset0;
    trace[4u * column_length + row] = instr.offset2;
    trace[5u * column_length + row] = instr.dst_base_fp;
    trace[6u * column_length + row] = instr.op1_base_fp;
    trace[7u * column_length + row] = instr.ap_update_add_1;

    uint mem_dst_base = stwo_metal_m31_add(
        stwo_metal_m31_mul(instr.dst_base_fp, fp),
        stwo_metal_m31_mul(stwo_metal_m31_sub(1u, instr.dst_base_fp), ap)
    );
    uint mem1_base = stwo_metal_m31_add(
        stwo_metal_m31_mul(instr.op1_base_fp, fp),
        stwo_metal_m31_mul(stwo_metal_m31_sub(1u, instr.op1_base_fp), ap)
    );

    trace[8u * column_length + row] = mem_dst_base;
    trace[9u * column_length + row] = mem1_base;

    uint dst_id = stwo_metal_opcode_mem_addr_to_id(
        address_to_id,
        stwo_metal_m31_add(mem_dst_base, stwo_metal_m31_sub(instr.offset0, 32768u))
    );
    trace[10u * column_length + row] = dst_id;
    trace[11u * column_length + row] = row < n_rows ? 1u : 0u;
}

kernel void witness_assert_eq_opcode_trace_columns(
    device const uint *inputs [[buffer(0)]],
    device const uint *address_to_id [[buffer(1)]],
    device const uint *big_values [[buffer(2)]],
    device const uint *small_values [[buffer(3)]],
    device const uint64_t *trace_col_addrs [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    constant uint &column_length [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    uint src_row = row < n_rows ? row : 0u;
    uint input_base = src_row * 3u;
    uint pc = inputs[input_base];
    uint ap = inputs[input_base + 1u];
    uint fp = inputs[input_base + 2u];

    stwo_metal_opcode_store_trace_col(trace_col_addrs, 0u, row, pc);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 1u, row, ap);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 2u, row, fp);

    uint instr_id = stwo_metal_opcode_mem_addr_to_id(address_to_id, pc);
    thread uint instr_limbs[28];
    stwo_metal_opcode_mem_id_to_limbs(instr_id, big_values, small_values, instr_limbs, 28u);
    StwoMetalDecodedInstruction instr = stwo_metal_opcode_decode_instruction(instr_limbs);

    stwo_metal_opcode_store_trace_col(trace_col_addrs, 3u, row, instr.offset0);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 4u, row, instr.offset2);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 5u, row, instr.dst_base_fp);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 6u, row, instr.op1_base_fp);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 7u, row, instr.ap_update_add_1);

    uint mem_dst_base = stwo_metal_m31_add(
        stwo_metal_m31_mul(instr.dst_base_fp, fp),
        stwo_metal_m31_mul(stwo_metal_m31_sub(1u, instr.dst_base_fp), ap)
    );
    uint mem1_base = stwo_metal_m31_add(
        stwo_metal_m31_mul(instr.op1_base_fp, fp),
        stwo_metal_m31_mul(stwo_metal_m31_sub(1u, instr.op1_base_fp), ap)
    );

    stwo_metal_opcode_store_trace_col(trace_col_addrs, 8u, row, mem_dst_base);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 9u, row, mem1_base);

    uint dst_id = stwo_metal_opcode_mem_addr_to_id(
        address_to_id,
        stwo_metal_m31_add(mem_dst_base, stwo_metal_m31_sub(instr.offset0, 32768u))
    );
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 10u, row, dst_id);
    stwo_metal_opcode_store_trace_col(trace_col_addrs, 11u, row, row < n_rows ? 1u : 0u);
}

kernel void witness_ret_opcode_trace(
    device const uint *inputs [[buffer(0)]],
    device const uint *address_to_id [[buffer(1)]],
    device const uint *big_values [[buffer(2)]],
    device const uint *small_values [[buffer(3)]],
    device uint *trace [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    constant uint &column_length [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    uint input_base = row * 3u;
    uint pc = inputs[input_base];
    uint ap = inputs[input_base + 1u];
    uint fp = inputs[input_base + 2u];

    trace[0u * column_length + row] = pc;
    trace[1u * column_length + row] = ap;
    trace[2u * column_length + row] = fp;

    uint addr0 = stwo_metal_m31_sub(fp, 1u);
    uint id0 = stwo_metal_opcode_mem_addr_to_id(address_to_id, addr0);
    thread uint limbs0[4];
    stwo_metal_opcode_mem_id_to_limbs(id0, big_values, small_values, limbs0, 4u);
    trace[3u * column_length + row] = id0;
    trace[4u * column_length + row] = limbs0[0];
    trace[5u * column_length + row] = limbs0[1];
    trace[6u * column_length + row] = limbs0[2];
    trace[7u * column_length + row] = limbs0[3];
    trace[8u * column_length + row] = (limbs0[3] & 2u) >> 1u;

    uint addr1 = stwo_metal_m31_sub(fp, 2u);
    uint id1 = stwo_metal_opcode_mem_addr_to_id(address_to_id, addr1);
    thread uint limbs1[4];
    stwo_metal_opcode_mem_id_to_limbs(id1, big_values, small_values, limbs1, 4u);
    trace[9u * column_length + row] = id1;
    trace[10u * column_length + row] = limbs1[0];
    trace[11u * column_length + row] = limbs1[1];
    trace[12u * column_length + row] = limbs1[2];
    trace[13u * column_length + row] = limbs1[3];
    trace[14u * column_length + row] = (limbs1[3] & 2u) >> 1u;

    trace[15u * column_length + row] = row < n_rows ? 1u : 0u;
}

kernel void witness_ret_opcode_trace_columns(
    device const uint *inputs [[buffer(0)]],
    device const uint *address_to_id [[buffer(1)]],
    device const uint *big_values [[buffer(2)]],
    device const uint *small_values [[buffer(3)]],
    device const uint64_t *trace_col_addrs [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    constant uint &column_length [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    uint input_base = row * 3u;
    uint pc = inputs[input_base];
    uint ap = inputs[input_base + 1u];
    uint fp = inputs[input_base + 2u];

    device uint *pc_col = (device uint *)trace_col_addrs[0];
    device uint *ap_col = (device uint *)trace_col_addrs[1];
    device uint *fp_col = (device uint *)trace_col_addrs[2];
    device uint *id0_col = (device uint *)trace_col_addrs[3];
    device uint *limb0_0_col = (device uint *)trace_col_addrs[4];
    device uint *limb0_1_col = (device uint *)trace_col_addrs[5];
    device uint *limb0_2_col = (device uint *)trace_col_addrs[6];
    device uint *limb0_3_col = (device uint *)trace_col_addrs[7];
    device uint *partial0_col = (device uint *)trace_col_addrs[8];
    device uint *id1_col = (device uint *)trace_col_addrs[9];
    device uint *limb1_0_col = (device uint *)trace_col_addrs[10];
    device uint *limb1_1_col = (device uint *)trace_col_addrs[11];
    device uint *limb1_2_col = (device uint *)trace_col_addrs[12];
    device uint *limb1_3_col = (device uint *)trace_col_addrs[13];
    device uint *partial1_col = (device uint *)trace_col_addrs[14];
    device uint *enabler_col = (device uint *)trace_col_addrs[15];

    pc_col[row] = pc;
    ap_col[row] = ap;
    fp_col[row] = fp;

    uint addr0 = stwo_metal_m31_sub(fp, 1u);
    uint id0 = stwo_metal_opcode_mem_addr_to_id(address_to_id, addr0);
    thread uint limbs0[4];
    stwo_metal_opcode_mem_id_to_limbs(id0, big_values, small_values, limbs0, 4u);
    id0_col[row] = id0;
    limb0_0_col[row] = limbs0[0];
    limb0_1_col[row] = limbs0[1];
    limb0_2_col[row] = limbs0[2];
    limb0_3_col[row] = limbs0[3];
    partial0_col[row] = (limbs0[3] & 2u) >> 1u;

    uint addr1 = stwo_metal_m31_sub(fp, 2u);
    uint id1 = stwo_metal_opcode_mem_addr_to_id(address_to_id, addr1);
    thread uint limbs1[4];
    stwo_metal_opcode_mem_id_to_limbs(id1, big_values, small_values, limbs1, 4u);
    id1_col[row] = id1;
    limb1_0_col[row] = limbs1[0];
    limb1_1_col[row] = limbs1[1];
    limb1_2_col[row] = limbs1[2];
    limb1_3_col[row] = limbs1[3];
    partial1_col[row] = (limbs1[3] & 2u) >> 1u;

    enabler_col[row] = row < n_rows ? 1u : 0u;
}

static inline StwoMetalQm31 stwo_metal_ret_combine_values(
    device const uint *alpha_powers,
    constant uint *z_limbs,
    thread const uint *values,
    uint n_values
) {
    StwoMetalQm31 acc = StwoMetalQm31 {0u, 0u, 0u, 0u};
    for (uint i = 0u; i < n_values; ++i) {
        acc = stwo_metal_qm31_add(
            acc,
            stwo_metal_qm31_mul_m31(stwo_metal_load_qm31(alpha_powers, i), values[i])
        );
    }
    return stwo_metal_qm31_sub(acc, stwo_metal_load_qm31_from_constant(z_limbs, 0u));
}

static inline StwoMetalQm31 stwo_metal_ret_memory_id_to_big_denom(
    device const uint *alpha_powers,
    constant uint *z_limbs,
    uint id,
    uint limb0,
    uint limb1,
    uint limb2,
    uint limb3
) {
    thread uint values[6] = {
        1662111297u,
        id,
        limb0,
        limb1,
        limb2,
        limb3,
    };
    return stwo_metal_ret_combine_values(alpha_powers, z_limbs, values, 6u);
}

static inline StwoMetalQm31 stwo_metal_ret_memory_address_to_id_denom(
    device const uint *alpha_powers,
    constant uint *z_limbs,
    uint address,
    uint id
) {
    thread uint values[3] = {1444891767u, address, id};
    return stwo_metal_ret_combine_values(alpha_powers, z_limbs, values, 3u);
}

static inline StwoMetalQm31 stwo_metal_ret_opcodes_denom(
    device const uint *alpha_powers,
    constant uint *z_limbs,
    uint pc,
    uint ap,
    uint fp
) {
    thread uint values[4] = {428564188u, pc, ap, fp};
    return stwo_metal_ret_combine_values(alpha_powers, z_limbs, values, 4u);
}

static inline StwoMetalQm31 stwo_metal_ret_verify_instruction_denom(
    device const uint *alpha_powers,
    constant uint *z_limbs,
    uint pc
) {
    thread uint values[8] = {
        1719106205u,
        pc,
        32766u,
        32767u,
        32767u,
        88u,
        130u,
        0u,
    };
    return stwo_metal_ret_combine_values(alpha_powers, z_limbs, values, 8u);
}

static inline void stwo_metal_ret_store_raw_logup_col(
    device const uint64_t *num_col_addrs,
    device const uint64_t *denom_col_addrs,
    uint col,
    uint row,
    uint column_length,
    StwoMetalQm31 numerator,
    StwoMetalQm31 denominator
) {
    device uint *num0 = (device uint *)num_col_addrs[col * 4u + 0u];
    device uint *num1 = (device uint *)num_col_addrs[col * 4u + 1u];
    device uint *num2 = (device uint *)num_col_addrs[col * 4u + 2u];
    device uint *num3 = (device uint *)num_col_addrs[col * 4u + 3u];
    device uint *denom = (device uint *)denom_col_addrs[col];
    num0[row] = numerator.a;
    num1[row] = numerator.b;
    num2[row] = numerator.c;
    num3[row] = numerator.d;
    stwo_metal_store_packed_lane_qm31(denom, row, denominator);
}

kernel void witness_ret_opcode_interaction_raw(
    device const uint64_t *trace_col_addrs [[buffer(0)]],
    device const uint64_t *num_col_addrs [[buffer(1)]],
    device const uint64_t *denom_col_addrs [[buffer(2)]],
    device const uint *alpha_powers [[buffer(3)]],
    constant uint *z_limbs [[buffer(4)]],
    constant uint &column_length [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    uint total = column_length * 4u;
    if (index >= total) {
        return;
    }

    uint col = index / column_length;
    uint row = index - col * column_length;
    device const uint *pc_col = (device const uint *)trace_col_addrs[0];
    device const uint *ap_col = (device const uint *)trace_col_addrs[1];
    device const uint *fp_col = (device const uint *)trace_col_addrs[2];
    device const uint *id0_col = (device const uint *)trace_col_addrs[3];
    device const uint *limb0_0_col = (device const uint *)trace_col_addrs[4];
    device const uint *limb0_1_col = (device const uint *)trace_col_addrs[5];
    device const uint *limb0_2_col = (device const uint *)trace_col_addrs[6];
    device const uint *limb0_3_col = (device const uint *)trace_col_addrs[7];
    device const uint *id1_col = (device const uint *)trace_col_addrs[9];
    device const uint *limb1_0_col = (device const uint *)trace_col_addrs[10];
    device const uint *limb1_1_col = (device const uint *)trace_col_addrs[11];
    device const uint *limb1_2_col = (device const uint *)trace_col_addrs[12];
    device const uint *limb1_3_col = (device const uint *)trace_col_addrs[13];
    device const uint *enabler_col = (device const uint *)trace_col_addrs[15];

    uint pc = pc_col[row];
    uint ap = ap_col[row];
    uint fp = fp_col[row];
    uint id0 = id0_col[row];
    uint id1 = id1_col[row];
    uint limb0_0 = limb0_0_col[row];
    uint limb0_1 = limb0_1_col[row];
    uint limb0_2 = limb0_2_col[row];
    uint limb0_3 = limb0_3_col[row];
    uint limb1_0 = limb1_0_col[row];
    uint limb1_1 = limb1_1_col[row];
    uint limb1_2 = limb1_2_col[row];
    uint limb1_3 = limb1_3_col[row];
    uint enabler = enabler_col[row];

    StwoMetalQm31 denom0;
    StwoMetalQm31 denom1;
    StwoMetalQm31 numerator;
    StwoMetalQm31 denominator;

    if (col == 0u) {
        denom0 = stwo_metal_ret_verify_instruction_denom(alpha_powers, z_limbs, pc);
        denom1 = stwo_metal_ret_memory_address_to_id_denom(
            alpha_powers,
            z_limbs,
            stwo_metal_m31_sub(fp, 1u),
            id0
        );
        numerator = stwo_metal_qm31_add(denom0, denom1);
        denominator = stwo_metal_qm31_mul(denom0, denom1);
    } else if (col == 1u) {
        denom0 = stwo_metal_ret_memory_id_to_big_denom(
            alpha_powers,
            z_limbs,
            id0,
            limb0_0,
            limb0_1,
            limb0_2,
            limb0_3
        );
        denom1 = stwo_metal_ret_memory_address_to_id_denom(
            alpha_powers,
            z_limbs,
            stwo_metal_m31_sub(fp, 2u),
            id1
        );
        numerator = stwo_metal_qm31_add(denom0, denom1);
        denominator = stwo_metal_qm31_mul(denom0, denom1);
    } else if (col == 2u) {
        denom0 = stwo_metal_ret_memory_id_to_big_denom(
            alpha_powers,
            z_limbs,
            id1,
            limb1_0,
            limb1_1,
            limb1_2,
            limb1_3
        );
        denom1 = stwo_metal_ret_opcodes_denom(alpha_powers, z_limbs, pc, ap, fp);
        numerator = stwo_metal_qm31_add(
            stwo_metal_qm31_mul_m31(denom0, enabler),
            denom1
        );
        denominator = stwo_metal_qm31_mul(denom0, denom1);
    } else {
        uint next_pc = stwo_metal_m31_add(
            stwo_metal_m31_add(limb0_0, stwo_metal_m31_mul(limb0_1, 512u)),
            stwo_metal_m31_add(
                stwo_metal_m31_mul(limb0_2, 262144u),
                stwo_metal_m31_mul(limb0_3, 134217728u)
            )
        );
        uint next_fp = stwo_metal_m31_add(
            stwo_metal_m31_add(limb1_0, stwo_metal_m31_mul(limb1_1, 512u)),
            stwo_metal_m31_add(
                stwo_metal_m31_mul(limb1_2, 262144u),
                stwo_metal_m31_mul(limb1_3, 134217728u)
            )
        );
        denom0 = stwo_metal_ret_opcodes_denom(alpha_powers, z_limbs, next_pc, ap, next_fp);
        numerator = stwo_metal_qm31_from_base(stwo_metal_m31_neg(enabler));
        denominator = denom0;
    }

    stwo_metal_ret_store_raw_logup_col(
        num_col_addrs,
        denom_col_addrs,
        col,
        row,
        column_length,
        numerator,
        denominator
    );
}

static inline uint stwo_metal_add_small_trace_col(
    device const uint *trace,
    uint col,
    uint row,
    uint column_length
) {
    return trace[col * column_length + row];
}

static inline void stwo_metal_add_small_build_id_to_big_lookup(
    thread uint *values,
    uint id,
    uint limb0,
    uint limb1,
    uint limb2,
    uint remainder_bits,
    uint msb,
    uint mid_limbs_set
) {
    uint dss2 = stwo_metal_m31_mul(mid_limbs_set, 508u);
    uint dss3 = stwo_metal_m31_mul(mid_limbs_set, 511u);
    uint dss4 = stwo_metal_m31_sub(stwo_metal_m31_mul(msb, 136u), mid_limbs_set);
    uint dss5 = stwo_metal_m31_mul(msb, 256u);

    values[0] = 1662111297u;
    values[1] = id;
    values[2] = limb0;
    values[3] = limb1;
    values[4] = limb2;
    values[5] = stwo_metal_m31_add(remainder_bits, dss2);
    for (uint i = 6u; i < 23u; ++i) {
        values[i] = dss3;
    }
    values[23] = dss4;
    for (uint i = 24u; i < 29u; ++i) {
        values[i] = 0u;
    }
    values[29] = dss5;
}

static inline void stwo_metal_add_small_store_pair_raw(
    device const uint64_t *num_col_addrs,
    device const uint64_t *denom_col_addrs,
    uint col,
    uint row,
    StwoMetalQm31 denom0,
    StwoMetalQm31 denom1,
    uint mult0,
    uint mult1
) {
    StwoMetalQm31 numerator = stwo_metal_qm31_add(
        stwo_metal_qm31_mul_m31(denom0, mult1),
        stwo_metal_qm31_mul_m31(denom1, mult0)
    );
    StwoMetalQm31 denominator = stwo_metal_qm31_mul(denom0, denom1);
    stwo_metal_ret_store_raw_logup_col(
        num_col_addrs,
        denom_col_addrs,
        col,
        row,
        0u,
        numerator,
        denominator
    );
}

kernel void witness_add_opcode_small_interaction_raw(
    device const uint *trace [[buffer(0)]],
    device const uint64_t *num_col_addrs [[buffer(1)]],
    device const uint64_t *denom_col_addrs [[buffer(2)]],
    device const uint *alpha_powers [[buffer(3)]],
    constant uint *z_limbs [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    constant uint &column_length [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    uint input_pc = stwo_metal_add_small_trace_col(trace, 0u, row, column_length);
    uint input_ap = stwo_metal_add_small_trace_col(trace, 1u, row, column_length);
    uint input_fp = stwo_metal_add_small_trace_col(trace, 2u, row, column_length);
    uint offset0 = stwo_metal_add_small_trace_col(trace, 3u, row, column_length);
    uint offset1 = stwo_metal_add_small_trace_col(trace, 4u, row, column_length);
    uint offset2 = stwo_metal_add_small_trace_col(trace, 5u, row, column_length);
    uint dst_base_fp = stwo_metal_add_small_trace_col(trace, 6u, row, column_length);
    uint op0_base_fp = stwo_metal_add_small_trace_col(trace, 7u, row, column_length);
    uint op1_imm = stwo_metal_add_small_trace_col(trace, 8u, row, column_length);
    uint op1_base_fp = stwo_metal_add_small_trace_col(trace, 9u, row, column_length);
    uint ap_update = stwo_metal_add_small_trace_col(trace, 10u, row, column_length);
    uint mem_dst_base = stwo_metal_add_small_trace_col(trace, 11u, row, column_length);
    uint mem0_base = stwo_metal_add_small_trace_col(trace, 12u, row, column_length);
    uint mem1_base = stwo_metal_add_small_trace_col(trace, 13u, row, column_length);
    uint dst_id = stwo_metal_add_small_trace_col(trace, 14u, row, column_length);
    uint dst_msb = stwo_metal_add_small_trace_col(trace, 15u, row, column_length);
    uint dst_mid_set = stwo_metal_add_small_trace_col(trace, 16u, row, column_length);
    uint dst_limb0 = stwo_metal_add_small_trace_col(trace, 17u, row, column_length);
    uint dst_limb1 = stwo_metal_add_small_trace_col(trace, 18u, row, column_length);
    uint dst_limb2 = stwo_metal_add_small_trace_col(trace, 19u, row, column_length);
    uint dst_rem = stwo_metal_add_small_trace_col(trace, 20u, row, column_length);
    uint op0_id = stwo_metal_add_small_trace_col(trace, 22u, row, column_length);
    uint op0_msb = stwo_metal_add_small_trace_col(trace, 23u, row, column_length);
    uint op0_mid_set = stwo_metal_add_small_trace_col(trace, 24u, row, column_length);
    uint op0_limb0 = stwo_metal_add_small_trace_col(trace, 25u, row, column_length);
    uint op0_limb1 = stwo_metal_add_small_trace_col(trace, 26u, row, column_length);
    uint op0_limb2 = stwo_metal_add_small_trace_col(trace, 27u, row, column_length);
    uint op0_rem = stwo_metal_add_small_trace_col(trace, 28u, row, column_length);
    uint op1_id = stwo_metal_add_small_trace_col(trace, 30u, row, column_length);
    uint op1_msb = stwo_metal_add_small_trace_col(trace, 31u, row, column_length);
    uint op1_mid_set = stwo_metal_add_small_trace_col(trace, 32u, row, column_length);
    uint op1_limb0 = stwo_metal_add_small_trace_col(trace, 33u, row, column_length);
    uint op1_limb1 = stwo_metal_add_small_trace_col(trace, 34u, row, column_length);
    uint op1_limb2 = stwo_metal_add_small_trace_col(trace, 35u, row, column_length);
    uint op1_rem = stwo_metal_add_small_trace_col(trace, 36u, row, column_length);
    uint enabler = row < n_rows ? 1u : 0u;
    uint op1_base_ap = stwo_metal_m31_sub(stwo_metal_m31_sub(1u, op1_imm), op1_base_fp);
    uint offset0_signed = stwo_metal_m31_sub(offset0, 32768u);
    uint offset1_signed = stwo_metal_m31_sub(offset1, 32768u);
    uint offset2_signed = stwo_metal_m31_sub(offset2, 32768u);

    thread uint values0[30];
    thread uint values1[30];

    uint flags_val = stwo_metal_m31_add(
        stwo_metal_m31_add(
            stwo_metal_m31_add(
                stwo_metal_m31_mul(dst_base_fp, 8u),
                stwo_metal_m31_mul(op0_base_fp, 16u)
            ),
            stwo_metal_m31_add(
                stwo_metal_m31_mul(op1_imm, 32u),
                stwo_metal_m31_mul(op1_base_fp, 64u)
            )
        ),
        stwo_metal_m31_add(stwo_metal_m31_mul(op1_base_ap, 128u), 256u)
    );

    values0[0] = 1719106205u;
    values0[1] = input_pc;
    values0[2] = offset0;
    values0[3] = offset1;
    values0[4] = offset2;
    values0[5] = flags_val;
    values0[6] = stwo_metal_m31_add(stwo_metal_m31_mul(ap_update, 32u), 256u);
    values0[7] = 0u;
    values1[0] = 1444891767u;
    values1[1] = stwo_metal_m31_add(mem_dst_base, offset0_signed);
    values1[2] = dst_id;
    stwo_metal_add_small_store_pair_raw(
        num_col_addrs,
        denom_col_addrs,
        0u,
        row,
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values0, 8u),
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values1, 3u),
        1u,
        1u
    );

    stwo_metal_add_small_build_id_to_big_lookup(
        values0,
        dst_id,
        dst_limb0,
        dst_limb1,
        dst_limb2,
        dst_rem,
        dst_msb,
        dst_mid_set
    );
    values1[0] = 1444891767u;
    values1[1] = stwo_metal_m31_add(mem0_base, offset1_signed);
    values1[2] = op0_id;
    stwo_metal_add_small_store_pair_raw(
        num_col_addrs,
        denom_col_addrs,
        1u,
        row,
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values0, 30u),
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values1, 3u),
        1u,
        1u
    );

    stwo_metal_add_small_build_id_to_big_lookup(
        values0,
        op0_id,
        op0_limb0,
        op0_limb1,
        op0_limb2,
        op0_rem,
        op0_msb,
        op0_mid_set
    );
    values1[0] = 1444891767u;
    values1[1] = stwo_metal_m31_add(mem1_base, offset2_signed);
    values1[2] = op1_id;
    stwo_metal_add_small_store_pair_raw(
        num_col_addrs,
        denom_col_addrs,
        2u,
        row,
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values0, 30u),
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values1, 3u),
        1u,
        1u
    );

    stwo_metal_add_small_build_id_to_big_lookup(
        values0,
        op1_id,
        op1_limb0,
        op1_limb1,
        op1_limb2,
        op1_rem,
        op1_msb,
        op1_mid_set
    );
    values1[0] = 428564188u;
    values1[1] = input_pc;
    values1[2] = input_ap;
    values1[3] = input_fp;
    stwo_metal_add_small_store_pair_raw(
        num_col_addrs,
        denom_col_addrs,
        3u,
        row,
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values0, 30u),
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values1, 4u),
        1u,
        enabler
    );

    values0[0] = 428564188u;
    values0[1] = stwo_metal_m31_add(stwo_metal_m31_add(input_pc, 1u), op1_imm);
    values0[2] = stwo_metal_m31_add(input_ap, ap_update);
    values0[3] = input_fp;
    StwoMetalQm31 numerator = stwo_metal_qm31_from_base(stwo_metal_m31_neg(enabler));
    StwoMetalQm31 denominator = stwo_metal_ret_combine_values(
        alpha_powers,
        z_limbs,
        values0,
        4u
    );
    stwo_metal_ret_store_raw_logup_col(
        num_col_addrs,
        denom_col_addrs,
        4u,
        row,
        0u,
        numerator,
        denominator
    );
}

static inline void stwo_metal_add_small_interaction_raw_from_values(
    thread const uint *t,
    device const uint64_t *num_col_addrs,
    device const uint64_t *denom_col_addrs,
    device const uint *alpha_powers,
    constant uint *z_limbs,
    uint n_rows,
    uint column_length,
    uint row
) {
    uint input_pc = t[0];
    uint input_ap = t[1];
    uint input_fp = t[2];
    uint offset0 = t[3];
    uint offset1 = t[4];
    uint offset2 = t[5];
    uint dst_base_fp = t[6];
    uint op0_base_fp = t[7];
    uint op1_imm = t[8];
    uint op1_base_fp = t[9];
    uint ap_update = t[10];
    uint mem_dst_base = t[11];
    uint mem0_base = t[12];
    uint mem1_base = t[13];
    uint dst_id = t[14];
    uint dst_msb = t[15];
    uint dst_mid_set = t[16];
    uint dst_limb0 = t[17];
    uint dst_limb1 = t[18];
    uint dst_limb2 = t[19];
    uint dst_rem = t[20];
    uint op0_id = t[22];
    uint op0_msb = t[23];
    uint op0_mid_set = t[24];
    uint op0_limb0 = t[25];
    uint op0_limb1 = t[26];
    uint op0_limb2 = t[27];
    uint op0_rem = t[28];
    uint op1_id = t[30];
    uint op1_msb = t[31];
    uint op1_mid_set = t[32];
    uint op1_limb0 = t[33];
    uint op1_limb1 = t[34];
    uint op1_limb2 = t[35];
    uint op1_rem = t[36];
    uint enabler = row < n_rows ? 1u : 0u;
    uint op1_base_ap = stwo_metal_m31_sub(stwo_metal_m31_sub(1u, op1_imm), op1_base_fp);
    uint offset0_signed = stwo_metal_m31_sub(offset0, 32768u);
    uint offset1_signed = stwo_metal_m31_sub(offset1, 32768u);
    uint offset2_signed = stwo_metal_m31_sub(offset2, 32768u);

    thread uint values0[30];
    thread uint values1[30];

    uint flags_val = stwo_metal_m31_add(
        stwo_metal_m31_add(
            stwo_metal_m31_add(
                stwo_metal_m31_mul(dst_base_fp, 8u),
                stwo_metal_m31_mul(op0_base_fp, 16u)
            ),
            stwo_metal_m31_add(
                stwo_metal_m31_mul(op1_imm, 32u),
                stwo_metal_m31_mul(op1_base_fp, 64u)
            )
        ),
        stwo_metal_m31_add(stwo_metal_m31_mul(op1_base_ap, 128u), 256u)
    );

    values0[0] = 1719106205u;
    values0[1] = input_pc;
    values0[2] = offset0;
    values0[3] = offset1;
    values0[4] = offset2;
    values0[5] = flags_val;
    values0[6] = stwo_metal_m31_add(stwo_metal_m31_mul(ap_update, 32u), 256u);
    values0[7] = 0u;
    values1[0] = 1444891767u;
    values1[1] = stwo_metal_m31_add(mem_dst_base, offset0_signed);
    values1[2] = dst_id;
    stwo_metal_add_small_store_pair_raw(
        num_col_addrs,
        denom_col_addrs,
        0u,
        row,
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values0, 8u),
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values1, 3u),
        1u,
        1u
    );

    stwo_metal_add_small_build_id_to_big_lookup(
        values0,
        dst_id,
        dst_limb0,
        dst_limb1,
        dst_limb2,
        dst_rem,
        dst_msb,
        dst_mid_set
    );
    values1[0] = 1444891767u;
    values1[1] = stwo_metal_m31_add(mem0_base, offset1_signed);
    values1[2] = op0_id;
    stwo_metal_add_small_store_pair_raw(
        num_col_addrs,
        denom_col_addrs,
        1u,
        row,
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values0, 30u),
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values1, 3u),
        1u,
        1u
    );

    stwo_metal_add_small_build_id_to_big_lookup(
        values0,
        op0_id,
        op0_limb0,
        op0_limb1,
        op0_limb2,
        op0_rem,
        op0_msb,
        op0_mid_set
    );
    values1[0] = 1444891767u;
    values1[1] = stwo_metal_m31_add(mem1_base, offset2_signed);
    values1[2] = op1_id;
    stwo_metal_add_small_store_pair_raw(
        num_col_addrs,
        denom_col_addrs,
        2u,
        row,
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values0, 30u),
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values1, 3u),
        1u,
        1u
    );

    stwo_metal_add_small_build_id_to_big_lookup(
        values0,
        op1_id,
        op1_limb0,
        op1_limb1,
        op1_limb2,
        op1_rem,
        op1_msb,
        op1_mid_set
    );
    values1[0] = 428564188u;
    values1[1] = input_pc;
    values1[2] = input_ap;
    values1[3] = input_fp;
    stwo_metal_add_small_store_pair_raw(
        num_col_addrs,
        denom_col_addrs,
        3u,
        row,
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values0, 30u),
        stwo_metal_ret_combine_values(alpha_powers, z_limbs, values1, 4u),
        1u,
        enabler
    );

    values0[0] = 428564188u;
    values0[1] = stwo_metal_m31_add(stwo_metal_m31_add(input_pc, 1u), op1_imm);
    values0[2] = stwo_metal_m31_add(input_ap, ap_update);
    values0[3] = input_fp;
    StwoMetalQm31 numerator = stwo_metal_qm31_from_base(stwo_metal_m31_neg(enabler));
    StwoMetalQm31 denominator = stwo_metal_ret_combine_values(
        alpha_powers,
        z_limbs,
        values0,
        4u
    );
    stwo_metal_ret_store_raw_logup_col(
        num_col_addrs,
        denom_col_addrs,
        4u,
        row,
        0u,
        numerator,
        denominator
    );
}

kernel void witness_add_opcode_small_interaction_raw_columns(
    device const uint64_t *trace_col_addrs [[buffer(0)]],
    device const uint64_t *num_col_addrs [[buffer(1)]],
    device const uint64_t *denom_col_addrs [[buffer(2)]],
    device const uint *alpha_powers [[buffer(3)]],
    constant uint *z_limbs [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    constant uint &column_length [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    thread uint t[39];
    for (uint col = 0u; col < 39u; ++col) {
        device const uint *trace_col = (device const uint *)trace_col_addrs[col];
        t[col] = trace_col[row];
    }
    stwo_metal_add_small_interaction_raw_from_values(
        t,
        num_col_addrs,
        denom_col_addrs,
        alpha_powers,
        z_limbs,
        n_rows,
        column_length,
        row
    );
}

kernel void witness_assert_eq_opcode_interaction_raw(
    device const uint *trace [[buffer(0)]],
    device const uint64_t *num_col_addrs [[buffer(1)]],
    device const uint64_t *denom_col_addrs [[buffer(2)]],
    device const uint *alpha_powers [[buffer(3)]],
    constant uint *z_limbs [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    constant uint &column_length [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    uint pc = stwo_metal_add_small_trace_col(trace, 0u, row, column_length);
    uint ap = stwo_metal_add_small_trace_col(trace, 1u, row, column_length);
    uint fp = stwo_metal_add_small_trace_col(trace, 2u, row, column_length);
    uint offset0 = stwo_metal_add_small_trace_col(trace, 3u, row, column_length);
    uint offset2 = stwo_metal_add_small_trace_col(trace, 4u, row, column_length);
    uint dst_base_fp = stwo_metal_add_small_trace_col(trace, 5u, row, column_length);
    uint op1_base_fp = stwo_metal_add_small_trace_col(trace, 6u, row, column_length);
    uint ap_update = stwo_metal_add_small_trace_col(trace, 7u, row, column_length);
    uint mem_dst_base = stwo_metal_add_small_trace_col(trace, 8u, row, column_length);
    uint mem1_base = stwo_metal_add_small_trace_col(trace, 9u, row, column_length);
    uint dst_id = stwo_metal_add_small_trace_col(trace, 10u, row, column_length);
    uint enabler = row < n_rows ? 1u : 0u;

    uint vi_felt5 = stwo_metal_m31_add(
        stwo_metal_m31_add(stwo_metal_m31_mul(dst_base_fp, 8u), 16u),
        stwo_metal_m31_add(
            stwo_metal_m31_mul(op1_base_fp, 64u),
            stwo_metal_m31_mul(stwo_metal_m31_sub(1u, op1_base_fp), 128u)
        )
    );
    uint vi_felt6 = stwo_metal_m31_add(stwo_metal_m31_mul(ap_update, 32u), 256u);
    uint dst_addr = stwo_metal_m31_add(
        mem_dst_base,
        stwo_metal_m31_sub(offset0, 32768u)
    );
    uint op1_addr = stwo_metal_m31_add(
        mem1_base,
        stwo_metal_m31_sub(offset2, 32768u)
    );

    thread uint values0[8];
    thread uint values1[4];

    values0[0] = 1719106205u;
    values0[1] = pc;
    values0[2] = offset0;
    values0[3] = 32767u;
    values0[4] = offset2;
    values0[5] = vi_felt5;
    values0[6] = vi_felt6;
    values0[7] = 0u;
    values1[0] = 1444891767u;
    values1[1] = dst_addr;
    values1[2] = dst_id;
    StwoMetalQm31 denom0 = stwo_metal_ret_combine_values(alpha_powers, z_limbs, values0, 8u);
    StwoMetalQm31 denom1 = stwo_metal_ret_combine_values(alpha_powers, z_limbs, values1, 3u);
    stwo_metal_add_small_store_pair_raw(
        num_col_addrs,
        denom_col_addrs,
        0u,
        row,
        denom0,
        denom1,
        1u,
        1u
    );

    values0[0] = 1444891767u;
    values0[1] = op1_addr;
    values0[2] = dst_id;
    values1[0] = 428564188u;
    values1[1] = pc;
    values1[2] = ap;
    values1[3] = fp;
    denom0 = stwo_metal_ret_combine_values(alpha_powers, z_limbs, values0, 3u);
    denom1 = stwo_metal_ret_combine_values(alpha_powers, z_limbs, values1, 4u);
    stwo_metal_add_small_store_pair_raw(
        num_col_addrs,
        denom_col_addrs,
        1u,
        row,
        denom0,
        denom1,
        1u,
        enabler
    );

    values1[0] = 428564188u;
    values1[1] = stwo_metal_m31_add(pc, 1u);
    values1[2] = stwo_metal_m31_add(ap, ap_update);
    values1[3] = fp;
    StwoMetalQm31 denominator = stwo_metal_ret_combine_values(
        alpha_powers,
        z_limbs,
        values1,
        4u
    );
    StwoMetalQm31 numerator = stwo_metal_qm31_from_base(stwo_metal_m31_neg(enabler));
    stwo_metal_ret_store_raw_logup_col(
        num_col_addrs,
        denom_col_addrs,
        2u,
        row,
        0u,
        numerator,
        denominator
    );
}

kernel void witness_assert_eq_opcode_interaction_raw_columns(
    device const uint64_t *trace_col_addrs [[buffer(0)]],
    device const uint64_t *num_col_addrs [[buffer(1)]],
    device const uint64_t *denom_col_addrs [[buffer(2)]],
    device const uint *alpha_powers [[buffer(3)]],
    constant uint *z_limbs [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    constant uint &column_length [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= column_length) {
        return;
    }

    device const uint *col0 = (device const uint *)trace_col_addrs[0];
    device const uint *col1 = (device const uint *)trace_col_addrs[1];
    device const uint *col2 = (device const uint *)trace_col_addrs[2];
    device const uint *col3 = (device const uint *)trace_col_addrs[3];
    device const uint *col4 = (device const uint *)trace_col_addrs[4];
    device const uint *col5 = (device const uint *)trace_col_addrs[5];
    device const uint *col6 = (device const uint *)trace_col_addrs[6];
    device const uint *col7 = (device const uint *)trace_col_addrs[7];
    device const uint *col8 = (device const uint *)trace_col_addrs[8];
    device const uint *col9 = (device const uint *)trace_col_addrs[9];
    device const uint *col10 = (device const uint *)trace_col_addrs[10];

    uint pc = col0[row];
    uint ap = col1[row];
    uint fp = col2[row];
    uint offset0 = col3[row];
    uint offset2 = col4[row];
    uint dst_base_fp = col5[row];
    uint op1_base_fp = col6[row];
    uint ap_update = col7[row];
    uint mem_dst_base = col8[row];
    uint mem1_base = col9[row];
    uint dst_id = col10[row];
    uint enabler = row < n_rows ? 1u : 0u;

    uint vi_felt5 = stwo_metal_m31_add(
        stwo_metal_m31_add(stwo_metal_m31_mul(dst_base_fp, 8u), 16u),
        stwo_metal_m31_add(
            stwo_metal_m31_mul(op1_base_fp, 64u),
            stwo_metal_m31_mul(stwo_metal_m31_sub(1u, op1_base_fp), 128u)
        )
    );
    uint vi_felt6 = stwo_metal_m31_add(stwo_metal_m31_mul(ap_update, 32u), 256u);
    uint dst_addr = stwo_metal_m31_add(
        mem_dst_base,
        stwo_metal_m31_sub(offset0, 32768u)
    );
    uint op1_addr = stwo_metal_m31_add(
        mem1_base,
        stwo_metal_m31_sub(offset2, 32768u)
    );

    thread uint values0[8];
    thread uint values1[4];

    values0[0] = 1719106205u;
    values0[1] = pc;
    values0[2] = offset0;
    values0[3] = 32767u;
    values0[4] = offset2;
    values0[5] = vi_felt5;
    values0[6] = vi_felt6;
    values0[7] = 0u;
    values1[0] = 1444891767u;
    values1[1] = dst_addr;
    values1[2] = dst_id;
    StwoMetalQm31 denom0 = stwo_metal_ret_combine_values(alpha_powers, z_limbs, values0, 8u);
    StwoMetalQm31 denom1 = stwo_metal_ret_combine_values(alpha_powers, z_limbs, values1, 3u);
    stwo_metal_add_small_store_pair_raw(
        num_col_addrs,
        denom_col_addrs,
        0u,
        row,
        denom0,
        denom1,
        1u,
        1u
    );

    values0[0] = 1444891767u;
    values0[1] = op1_addr;
    values0[2] = dst_id;
    values1[0] = 428564188u;
    values1[1] = pc;
    values1[2] = ap;
    values1[3] = fp;
    denom0 = stwo_metal_ret_combine_values(alpha_powers, z_limbs, values0, 3u);
    denom1 = stwo_metal_ret_combine_values(alpha_powers, z_limbs, values1, 4u);
    stwo_metal_add_small_store_pair_raw(
        num_col_addrs,
        denom_col_addrs,
        1u,
        row,
        denom0,
        denom1,
        1u,
        enabler
    );

    values1[0] = 428564188u;
    values1[1] = stwo_metal_m31_add(pc, 1u);
    values1[2] = stwo_metal_m31_add(ap, ap_update);
    values1[3] = fp;
    StwoMetalQm31 denominator = stwo_metal_ret_combine_values(
        alpha_powers,
        z_limbs,
        values1,
        4u
    );
    StwoMetalQm31 numerator = stwo_metal_qm31_from_base(stwo_metal_m31_neg(enabler));
    stwo_metal_ret_store_raw_logup_col(
        num_col_addrs,
        denom_col_addrs,
        2u,
        row,
        0u,
        numerator,
        denominator
    );
}
