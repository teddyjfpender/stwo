typedef unsigned long long u64;

#define STWO_M31_P 2147483647u

__device__ __forceinline__ unsigned stwo_m31_add(unsigned lhs, unsigned rhs) {
    unsigned sum = lhs + rhs;
    return sum >= STWO_M31_P ? sum - STWO_M31_P : sum;
}

__device__ __forceinline__ unsigned stwo_m31_sub(unsigned lhs, unsigned rhs) {
    return lhs >= rhs ? lhs - rhs : lhs + STWO_M31_P - rhs;
}

__device__ __forceinline__ unsigned stwo_m31_neg(unsigned value) {
    unsigned negated = STWO_M31_P - value;
    return negated == STWO_M31_P ? 0u : negated;
}

__device__ __forceinline__ unsigned stwo_m31_mul(unsigned lhs, unsigned rhs) {
    u64 product = (u64)lhs * (u64)rhs;
    u64 reduced = (((((product >> 31) + product + 1u) >> 31) + product) & (u64)STWO_M31_P);
    return (unsigned)reduced;
}

static __device__ __forceinline__ unsigned stwo_m31_inverse(unsigned a) {
    unsigned result = a;                 // consumes exponent bit 30
    for (int bit = 29; bit >= 0; --bit) {
        result = stwo_m31_mul(result, result);
        if (bit != 1) { result = stwo_m31_mul(result, a); }
    }
    return result;
}

static __device__ __forceinline__ unsigned stwo_wit_deduce_limb(
    const unsigned *const *tb, const unsigned *ts, unsigned id, unsigned limb) {
    unsigned tag = id >> 30u;
    unsigned val = id & 0x3FFFFFFFu;
    if (tag == 1u) { return val < ts[1] ? tb[1u + limb][val] : 0u; }
    return (limb < 8u && val < ts[2]) ? tb[29u + limb][val] : 0u;
}

extern "C" __global__ void __launch_bounds__(256) stwo_jit_witness_b37424b6011f268e(
    const unsigned *const *input_cols,   // [n_inputs][row]
    const unsigned *const *table_bases,  // deduce_output LUTs, per table
    const unsigned *table_strides,       // words per key, per table
    unsigned *const *out_cols,           // [n_cols][row]
    unsigned *const *mult_counts,        // atomic count tables, per mult table
    unsigned *lookup_words,              // [k * row_count + row] (word-major)
    unsigned *sub_words,                 // [k * row_count + row] (word-major)
    unsigned row_count
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) { return; }

    unsigned r0 = 0u;
    unsigned r1 = 1u;
    unsigned r2 = 8u;
    unsigned r3 = 16u;
    unsigned r4 = 32u;
    unsigned r5 = 136u;
    unsigned r6 = 256u;
    unsigned r7 = 508u;
    unsigned r8 = 511u;
    unsigned r9 = 512u;
    unsigned r10 = 32767u;
    unsigned r11 = 32768u;
    unsigned r12 = 32769u;
    unsigned r13 = 262144u;
    unsigned r14 = 134217728u;
    unsigned r15 = 428564188u;
    unsigned r16 = 536870912u;
    unsigned r17 = 1444891767u;
    unsigned r18 = 1662111297u;
    unsigned r19 = 1719106205u;
    unsigned r20 = 2147483646u;
    unsigned r21 = input_cols[0u][row];
    out_cols[0u][row] = r21;
    unsigned r22 = input_cols[1u][row];
    out_cols[1u][row] = r22;
    unsigned r23 = input_cols[2u][row];
    out_cols[2u][row] = r23;
    unsigned r24 = (r21 < table_strides[0u] ? table_bases[0u][r21] : 0u);
    unsigned r25 = stwo_wit_deduce_limb(table_bases, table_strides, r24, 0u);
    unsigned r26 = (r25 & 0xFFFFu);
    unsigned r27 = stwo_wit_deduce_limb(table_bases, table_strides, r24, 1u);
    unsigned r28 = (r27 & 0xFFFFu);
    unsigned r29 = (r28 & 127u);
    unsigned r30 = ((r29 << 9u) & 0xFFFFu);
    unsigned r31 = ((r26 + r30) & 0xFFFFu);
    unsigned r32 = (r31 % STWO_M31_P);
    out_cols[3u][row] = r32;
    unsigned r33 = stwo_wit_deduce_limb(table_bases, table_strides, r24, 5u);
    unsigned r34 = (r33 & 0xFFFFu);
    unsigned r35 = ((r34 & 0xFFFFu) >> 3u);
    unsigned r36 = stwo_wit_deduce_limb(table_bases, table_strides, r24, 6u);
    unsigned r37 = (r36 & 0xFFFFu);
    unsigned r38 = ((r37 << 6u) & 0xFFFFu);
    unsigned r39 = ((r35 + r38) & 0xFFFFu);
    unsigned r40 = ((r39 & 0xFFFFu) >> 0u);
    unsigned r41 = (r40 & 1u);
    unsigned r42 = (r41 % STWO_M31_P);
    out_cols[4u][row] = r42;
    unsigned r43 = stwo_wit_deduce_limb(table_bases, table_strides, r24, 5u);
    unsigned r44 = (r43 & 0xFFFFu);
    unsigned r45 = ((r44 & 0xFFFFu) >> 3u);
    unsigned r46 = stwo_wit_deduce_limb(table_bases, table_strides, r24, 6u);
    unsigned r47 = (r46 & 0xFFFFu);
    unsigned r48 = ((r47 << 6u) & 0xFFFFu);
    unsigned r49 = ((r45 + r48) & 0xFFFFu);
    unsigned r50 = ((r49 & 0xFFFFu) >> 11u);
    unsigned r51 = (r50 & 1u);
    unsigned r52 = (r51 % STWO_M31_P);
    out_cols[5u][row] = r52;
    unsigned r53 = stwo_m31_mul(r42, r2);
    unsigned r54 = stwo_m31_add(r53, r3);
    unsigned r55 = stwo_m31_add(r54, r4);
    unsigned r56 = stwo_m31_mul(r52, r4);
    unsigned r57 = stwo_m31_add(r2, r56);
    sub_words[0u * row_count + row] = r21;
    sub_words[1u * row_count + row] = r32;
    sub_words[2u * row_count + row] = r10;
    sub_words[3u * row_count + row] = r12;
    sub_words[4u * row_count + row] = r55;
    sub_words[5u * row_count + row] = r57;
    sub_words[6u * row_count + row] = r0;
    lookup_words[0u * row_count + row] = r19;
    lookup_words[1u * row_count + row] = r21;
    lookup_words[2u * row_count + row] = r32;
    lookup_words[3u * row_count + row] = r10;
    lookup_words[4u * row_count + row] = r12;
    unsigned r58 = stwo_m31_mul(r42, r2);
    unsigned r59 = stwo_m31_add(r58, r3);
    unsigned r60 = stwo_m31_add(r59, r4);
    lookup_words[5u * row_count + row] = r60;
    unsigned r61 = stwo_m31_mul(r52, r4);
    unsigned r62 = stwo_m31_add(r2, r61);
    lookup_words[6u * row_count + row] = r62;
    lookup_words[7u * row_count + row] = r0;
    unsigned r63 = stwo_m31_sub(r32, r11);
    unsigned r64 = stwo_m31_mul(r42, r23);
    unsigned r65 = stwo_m31_sub(r1, r42);
    unsigned r66 = stwo_m31_mul(r65, r22);
    unsigned r67 = stwo_m31_add(r64, r66);
    out_cols[6u][row] = r67;
    unsigned r68 = stwo_m31_add(r67, r63);
    unsigned r69 = (r68 < table_strides[0u] ? table_bases[0u][r68] : 0u);
    out_cols[7u][row] = r69;
    unsigned r70 = stwo_m31_add(r67, r63);
    sub_words[7u * row_count + row] = r70;
    lookup_words[8u * row_count + row] = r17;
    unsigned r71 = stwo_m31_add(r67, r63);
    lookup_words[9u * row_count + row] = r71;
    lookup_words[10u * row_count + row] = r69;
    unsigned r72 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 0u);
    out_cols[8u][row] = r72;
    unsigned r73 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 1u);
    out_cols[9u][row] = r73;
    unsigned r74 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 2u);
    out_cols[10u][row] = r74;
    unsigned r75 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 3u);
    out_cols[11u][row] = r75;
    unsigned r76 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 4u);
    out_cols[12u][row] = r76;
    unsigned r77 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 5u);
    out_cols[13u][row] = r77;
    unsigned r78 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 6u);
    out_cols[14u][row] = r78;
    unsigned r79 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 7u);
    out_cols[15u][row] = r79;
    unsigned r80 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 8u);
    out_cols[16u][row] = r80;
    unsigned r81 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 9u);
    out_cols[17u][row] = r81;
    unsigned r82 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 10u);
    out_cols[18u][row] = r82;
    unsigned r83 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 11u);
    out_cols[19u][row] = r83;
    unsigned r84 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 12u);
    out_cols[20u][row] = r84;
    unsigned r85 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 13u);
    out_cols[21u][row] = r85;
    unsigned r86 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 14u);
    out_cols[22u][row] = r86;
    unsigned r87 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 15u);
    out_cols[23u][row] = r87;
    unsigned r88 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 16u);
    out_cols[24u][row] = r88;
    unsigned r89 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 17u);
    out_cols[25u][row] = r89;
    unsigned r90 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 18u);
    out_cols[26u][row] = r90;
    unsigned r91 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 19u);
    out_cols[27u][row] = r91;
    unsigned r92 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 20u);
    out_cols[28u][row] = r92;
    unsigned r93 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 21u);
    out_cols[29u][row] = r93;
    unsigned r94 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 22u);
    out_cols[30u][row] = r94;
    unsigned r95 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 23u);
    out_cols[31u][row] = r95;
    unsigned r96 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 24u);
    out_cols[32u][row] = r96;
    unsigned r97 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 25u);
    out_cols[33u][row] = r97;
    unsigned r98 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 26u);
    out_cols[34u][row] = r98;
    unsigned r99 = stwo_wit_deduce_limb(table_bases, table_strides, r69, 27u);
    out_cols[35u][row] = r99;
    sub_words[9u * row_count + row] = r69;
    lookup_words[11u * row_count + row] = r18;
    lookup_words[12u * row_count + row] = r69;
    lookup_words[13u * row_count + row] = r72;
    lookup_words[14u * row_count + row] = r73;
    lookup_words[15u * row_count + row] = r74;
    lookup_words[16u * row_count + row] = r75;
    lookup_words[17u * row_count + row] = r76;
    lookup_words[18u * row_count + row] = r77;
    lookup_words[19u * row_count + row] = r78;
    lookup_words[20u * row_count + row] = r79;
    lookup_words[21u * row_count + row] = r80;
    lookup_words[22u * row_count + row] = r81;
    lookup_words[23u * row_count + row] = r82;
    lookup_words[24u * row_count + row] = r83;
    lookup_words[25u * row_count + row] = r84;
    lookup_words[26u * row_count + row] = r85;
    lookup_words[27u * row_count + row] = r86;
    lookup_words[28u * row_count + row] = r87;
    lookup_words[29u * row_count + row] = r88;
    lookup_words[30u * row_count + row] = r89;
    lookup_words[31u * row_count + row] = r90;
    lookup_words[32u * row_count + row] = r91;
    lookup_words[33u * row_count + row] = r92;
    lookup_words[34u * row_count + row] = r93;
    lookup_words[35u * row_count + row] = r94;
    lookup_words[36u * row_count + row] = r95;
    lookup_words[37u * row_count + row] = r96;
    lookup_words[38u * row_count + row] = r97;
    lookup_words[39u * row_count + row] = r98;
    lookup_words[40u * row_count + row] = r99;
    unsigned r100 = stwo_m31_add(r73, r74);
    unsigned r101 = stwo_m31_add(r100, r75);
    unsigned r102 = stwo_m31_add(r101, r76);
    unsigned r103 = stwo_m31_add(r102, r77);
    unsigned r104 = stwo_m31_add(r103, r78);
    unsigned r105 = stwo_m31_add(r104, r79);
    unsigned r106 = stwo_m31_add(r105, r80);
    unsigned r107 = stwo_m31_add(r106, r81);
    unsigned r108 = stwo_m31_add(r107, r82);
    unsigned r109 = stwo_m31_add(r108, r83);
    unsigned r110 = stwo_m31_add(r109, r84);
    unsigned r111 = stwo_m31_add(r110, r85);
    unsigned r112 = stwo_m31_add(r111, r86);
    unsigned r113 = stwo_m31_add(r112, r87);
    unsigned r114 = stwo_m31_add(r113, r88);
    unsigned r115 = stwo_m31_add(r114, r89);
    unsigned r116 = stwo_m31_add(r115, r90);
    unsigned r117 = stwo_m31_add(r116, r91);
    unsigned r118 = stwo_m31_add(r117, r92);
    unsigned r119 = stwo_m31_add(r118, r94);
    unsigned r120 = stwo_m31_add(r119, r95);
    unsigned r121 = stwo_m31_add(r120, r96);
    unsigned r122 = stwo_m31_add(r121, r97);
    unsigned r123 = stwo_m31_add(r122, r98);
    unsigned r124 = stwo_m31_add(r72, r93);
    unsigned r125 = stwo_m31_add(r124, r99);
    unsigned r126 = stwo_m31_add(r123, r125);
    unsigned r127 = stwo_m31_inverse(r126);
    out_cols[36u][row] = r127;
    unsigned r128 = stwo_m31_sub(r72, r1);
    unsigned r129 = stwo_m31_sub(r93, r5);
    unsigned r130 = stwo_m31_sub(r99, r6);
    unsigned r131 = stwo_m31_mul(r128, r128);
    unsigned r132 = stwo_m31_mul(r129, r129);
    unsigned r133 = stwo_m31_add(r131, r132);
    unsigned r134 = stwo_m31_mul(r130, r130);
    unsigned r135 = stwo_m31_add(r133, r134);
    unsigned r136 = stwo_m31_add(r123, r135);
    unsigned r137 = stwo_m31_inverse(r136);
    out_cols[37u][row] = r137;
    unsigned r138 = stwo_m31_add(r21, r1);
    unsigned r139 = (r138 < table_strides[0u] ? table_bases[0u][r138] : 0u);
    out_cols[38u][row] = r139;
    unsigned r140 = stwo_m31_add(r21, r1);
    sub_words[8u * row_count + row] = r140;
    lookup_words[41u * row_count + row] = r17;
    unsigned r141 = stwo_m31_add(r21, r1);
    lookup_words[42u * row_count + row] = r141;
    lookup_words[43u * row_count + row] = r139;
    unsigned r142 = stwo_wit_deduce_limb(table_bases, table_strides, r139, 27u);
    unsigned r143 = (r142 == r6 ? 1u : 0u);
    out_cols[39u][row] = r143;
    unsigned r144 = stwo_wit_deduce_limb(table_bases, table_strides, r139, 20u);
    unsigned r145 = (r144 == r8 ? 1u : 0u);
    unsigned r146 = stwo_m31_mul(r145, r143);
    out_cols[40u][row] = r146;
    unsigned r147 = stwo_m31_mul(r146, r7);
    unsigned r148 = stwo_m31_mul(r146, r8);
    unsigned r149 = stwo_m31_mul(r143, r5);
    unsigned r150 = stwo_m31_sub(r149, r146);
    unsigned r151 = stwo_m31_mul(r143, r6);
    unsigned r152 = stwo_wit_deduce_limb(table_bases, table_strides, r139, 0u);
    out_cols[41u][row] = r152;
    unsigned r153 = stwo_wit_deduce_limb(table_bases, table_strides, r139, 1u);
    out_cols[42u][row] = r153;
    unsigned r154 = stwo_wit_deduce_limb(table_bases, table_strides, r139, 2u);
    out_cols[43u][row] = r154;
    unsigned r155 = stwo_wit_deduce_limb(table_bases, table_strides, r139, 3u);
    unsigned r156 = (r155 & 0xFFFFu);
    unsigned r157 = (r156 & 3u);
    unsigned r158 = (r157 % STWO_M31_P);
    out_cols[44u][row] = r158;
    unsigned r159 = (r158 & 0xFFFFu);
    unsigned r160 = (r159 & 2u);
    unsigned r161 = ((r160 & 0xFFFFu) >> 1u);
    unsigned r162 = (r161 % STWO_M31_P);
    out_cols[45u][row] = r162;
    sub_words[10u * row_count + row] = r139;
    lookup_words[44u * row_count + row] = r18;
    lookup_words[45u * row_count + row] = r139;
    lookup_words[46u * row_count + row] = r152;
    lookup_words[47u * row_count + row] = r153;
    lookup_words[48u * row_count + row] = r154;
    unsigned r163 = stwo_m31_add(r158, r147);
    lookup_words[49u * row_count + row] = r163;
    lookup_words[50u * row_count + row] = r148;
    lookup_words[51u * row_count + row] = r148;
    lookup_words[52u * row_count + row] = r148;
    lookup_words[53u * row_count + row] = r148;
    lookup_words[54u * row_count + row] = r148;
    lookup_words[55u * row_count + row] = r148;
    lookup_words[56u * row_count + row] = r148;
    lookup_words[57u * row_count + row] = r148;
    lookup_words[58u * row_count + row] = r148;
    lookup_words[59u * row_count + row] = r148;
    lookup_words[60u * row_count + row] = r148;
    lookup_words[61u * row_count + row] = r148;
    lookup_words[62u * row_count + row] = r148;
    lookup_words[63u * row_count + row] = r148;
    lookup_words[64u * row_count + row] = r148;
    lookup_words[65u * row_count + row] = r148;
    lookup_words[66u * row_count + row] = r148;
    lookup_words[67u * row_count + row] = r150;
    lookup_words[68u * row_count + row] = r0;
    lookup_words[69u * row_count + row] = r0;
    lookup_words[70u * row_count + row] = r0;
    lookup_words[71u * row_count + row] = r0;
    lookup_words[72u * row_count + row] = r0;
    lookup_words[73u * row_count + row] = r151;
    unsigned r164 = stwo_m31_mul(r153, r9);
    unsigned r165 = stwo_m31_add(r152, r164);
    unsigned r166 = stwo_m31_mul(r154, r13);
    unsigned r167 = stwo_m31_add(r165, r166);
    unsigned r168 = stwo_m31_mul(r158, r14);
    unsigned r169 = stwo_m31_add(r167, r168);
    unsigned r170 = stwo_m31_sub(r169, r143);
    unsigned r171 = stwo_m31_mul(r16, r146);
    unsigned r172 = stwo_m31_sub(r170, r171);
    unsigned r173 = input_cols[3u][row];
    out_cols[46u][row] = r173;
    lookup_words[74u * row_count + row] = r15;
    lookup_words[75u * row_count + row] = r21;
    lookup_words[76u * row_count + row] = r22;
    lookup_words[77u * row_count + row] = r23;
    lookup_words[78u * row_count + row] = r15;
    unsigned r174 = stwo_m31_add(r21, r172);
    lookup_words[79u * row_count + row] = r174;
    unsigned r175 = stwo_m31_add(r22, r52);
    lookup_words[80u * row_count + row] = r175;
    lookup_words[81u * row_count + row] = r23;
    lookup_words[82u * row_count + row] = r1;
    lookup_words[83u * row_count + row] = r173;
}
