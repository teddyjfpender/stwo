// Stack-free native Blake-G producer/count feed.
//
// This file is included inside blake_witness.cu's anonymous namespace after
// the resident ABI structs and lo16/hi16 helpers are defined.  The generic
// recorded/legacy writer intentionally keeps its indexed c[73] compatibility
// plane.  This fused instantiation does not: every value is a named scalar,
// every output index is a template constant, and each phase is retired before
// the next one.  That makes a non-zero ptxas stack frame a release blocker for
// this kernel rather than accepting hidden thread-local row traffic.

#ifndef BLAKE_G_FUSED_SCALAR_CUH
#define BLAKE_G_FUSED_SCALAR_CUH

template <uint32_t Column>
DEVICE_FORCEINLINE void bg_trace(BlakeGResidentOutputs outputs, uint32_t row,
                                 uint32_t value) {
    static_assert(Column < BG_N_TRACE, "Blake-G trace column is out of range");
    outputs.trace[Column][row] = value;
}

template <uint32_t Word>
DEVICE_FORCEINLINE void bg_lookup(BlakeGResidentOutputs outputs, uint32_t row,
                                  uint32_t column_length, uint32_t value) {
    static_assert(Word < 87u, "Blake-G lookup word is out of range");
    if (outputs.lookup != nullptr) {
        outputs.lookup[(size_t)Word * column_length + row] = value;
    }
}

template <uint32_t Column>
DEVICE_FORCEINLINE void bg_aux(BlakeGResidentOutputs outputs, uint32_t row,
                               uint32_t column_length, uint32_t value) {
    static_assert(Column < BG_N_AUX, "Blake-G auxiliary column is out of range");
    if (outputs.aux != nullptr) {
        outputs.aux[(size_t)Column * column_length + row] = value;
    }
}

template <uint32_t Tuple, uint32_t Relation>
DEVICE_FORCEINLINE void bg_lookup_tuple(BlakeGResidentOutputs outputs,
                                        uint32_t row,
                                        uint32_t column_length, uint32_t a,
                                        uint32_t b, uint32_t x) {
    static_assert(Tuple < 16u, "Blake-G xor tuple is out of range");
    constexpr uint32_t base = 4u * Tuple;
    bg_lookup<base>(outputs, row, column_length, Relation);
    bg_lookup<base + 1u>(outputs, row, column_length, a);
    bg_lookup<base + 2u>(outputs, row, column_length, b);
    bg_lookup<base + 3u>(outputs, row, column_length, x);
}

template <uint32_t Lut, uint32_t Destination, uint32_t Bits,
          uint32_t Relation>
DEVICE_FORCEINLINE void bg_count_lut(BlakeGFusedFeed feed, uint32_t a,
                                     uint32_t b) {
    static_assert(Lut < 4u && Destination < 5u,
                  "Blake-G count binding is out of range");
    static_assert(Bits > 0u && Bits <= 15u,
                  "Blake-G xor table width is invalid");
    constexpr uint32_t table_size = 1u << (2u * Bits);
    if ((a | b) < (1u << Bits)) {
        uint32_t index = feed.luts[Lut][(a << Bits) | b];
        if (index < table_size) {
            atomicAdd(&feed.counts[Destination]
                                   [(size_t)Relation * table_size + index],
                      1u);
        }
    }
}

DEVICE_FORCEINLINE void bg_count_xor12(BlakeGFusedFeed feed, uint32_t a,
                                       uint32_t b) {
    if ((a | b) < (1u << 12)) {
        uint32_t column = ((a >> 10) << 2) | (b >> 10);
        uint32_t table_row = ((a & 0x3ffu) << 10) | (b & 0x3ffu);
        atomicAdd(&feed.counts[1][(size_t)column * (1u << 20) + table_row],
                  1u);
    }
}

__global__ void blake_g_write_trace_fused_scalar_kernel(
    BlakeGColumnInputs inputs, uint32_t n_rows, uint32_t column_length,
    BlakeGResidentOutputs outputs, BlakeGFusedFeed feed) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= column_length) {
        return;
    }

    const uint32_t in0 = inputs.columns[0][row];
    const uint32_t in1 = inputs.columns[1][row];
    const uint32_t in2 = inputs.columns[2][row];
    const uint32_t in3 = inputs.columns[3][row];
    const uint32_t in4 = inputs.columns[4][row];
    const uint32_t in5 = inputs.columns[5][row];

    // Input limbs. Recompute these cheap projections for the final tuple so
    // twelve limb values do not remain live through the complete G function.
    bg_trace<0>(outputs, row, lo16(in0));
    bg_trace<1>(outputs, row, hi16(in0));
    bg_trace<2>(outputs, row, lo16(in1));
    bg_trace<3>(outputs, row, hi16(in1));
    bg_trace<4>(outputs, row, lo16(in2));
    bg_trace<5>(outputs, row, hi16(in2));
    bg_trace<6>(outputs, row, lo16(in3));
    bg_trace<7>(outputs, row, hi16(in3));
    bg_trace<8>(outputs, row, lo16(in4));
    bg_trace<9>(outputs, row, hi16(in4));
    bg_trace<10>(outputs, row, lo16(in5));
    bg_trace<11>(outputs, row, hi16(in5));

    // Triple Sum 32 and R16 xor. Emit the first four tuples and count edges
    // while their operands are live.
    const uint32_t ts0 = in0 + in1 + in4;
    const uint32_t ts0_lo = lo16(ts0);
    const uint32_t ts0_hi = hi16(ts0);
    bg_trace<12>(outputs, row, ts0_lo);
    bg_trace<13>(outputs, row, ts0_hi);

    const uint32_t ts0_lo_ms8 = ts0_lo >> 8;
    const uint32_t ts0_lo_ls8 = ts0_lo - ts0_lo_ms8 * 256u;
    const uint32_t ts0_hi_ms8 = ts0_hi >> 8;
    const uint32_t ts0_hi_ls8 = ts0_hi - ts0_hi_ms8 * 256u;
    const uint32_t in3_lo_ms8 = lo16(in3) >> 8;
    const uint32_t in3_lo_ls8 = lo16(in3) - in3_lo_ms8 * 256u;
    const uint32_t in3_hi_ms8 = hi16(in3) >> 8;
    const uint32_t in3_hi_ls8 = hi16(in3) - in3_hi_ms8 * 256u;
    const uint32_t xor16_0 = ts0_lo_ls8 ^ in3_lo_ls8;
    const uint32_t xor16_1 = ts0_lo_ms8 ^ in3_lo_ms8;
    const uint32_t xor16_2 = ts0_hi_ls8 ^ in3_hi_ls8;
    const uint32_t xor16_3 = ts0_hi_ms8 ^ in3_hi_ms8;
    bg_aux<0>(outputs, row, column_length, ts0_lo_ls8);
    bg_aux<1>(outputs, row, column_length, ts0_hi_ls8);
    bg_aux<2>(outputs, row, column_length, in3_lo_ls8);
    bg_aux<3>(outputs, row, column_length, in3_hi_ls8);
    bg_trace<14>(outputs, row, ts0_lo_ms8);
    bg_trace<15>(outputs, row, ts0_hi_ms8);
    bg_trace<16>(outputs, row, in3_lo_ms8);
    bg_trace<17>(outputs, row, in3_hi_ms8);
    bg_trace<18>(outputs, row, xor16_0);
    bg_trace<19>(outputs, row, xor16_1);
    bg_trace<20>(outputs, row, xor16_2);
    bg_trace<21>(outputs, row, xor16_3);
    bg_lookup_tuple<0, 112558620>(outputs, row, column_length, ts0_lo_ls8,
                                  in3_lo_ls8, xor16_0);
    bg_lookup_tuple<1, 112558620>(outputs, row, column_length, ts0_lo_ms8,
                                  in3_lo_ms8, xor16_1);
    bg_lookup_tuple<2, 521092554>(outputs, row, column_length, ts0_hi_ls8,
                                  in3_hi_ls8, xor16_2);
    bg_lookup_tuple<3, 521092554>(outputs, row, column_length, ts0_hi_ms8,
                                  in3_hi_ms8, xor16_3);
    bg_count_lut<0, 0, 8, 0>(feed, ts0_lo_ls8, in3_lo_ls8);
    bg_count_lut<0, 0, 8, 0>(feed, ts0_lo_ms8, in3_lo_ms8);
    bg_count_lut<0, 0, 8, 1>(feed, ts0_hi_ls8, in3_hi_ls8);
    bg_count_lut<0, 0, 8, 1>(feed, ts0_hi_ms8, in3_hi_ms8);
    const uint32_t xr16_lo = xor16_2 + xor16_3 * 256u;
    const uint32_t xr16_hi = xor16_0 + xor16_1 * 256u;
    const uint32_t xr16 = xr16_lo + (xr16_hi << 16);

    // Triple Sum 32 and R12 xor.
    const uint32_t ts22 = in2 + xr16;
    const uint32_t ts22_lo = lo16(ts22);
    const uint32_t ts22_hi = hi16(ts22);
    bg_trace<22>(outputs, row, ts22_lo);
    bg_trace<23>(outputs, row, ts22_hi);

    const uint32_t in1_lo_ms4 = lo16(in1) >> 12;
    const uint32_t in1_lo_ls12 = lo16(in1) - in1_lo_ms4 * 4096u;
    const uint32_t in1_hi_ms4 = hi16(in1) >> 12;
    const uint32_t in1_hi_ls12 = hi16(in1) - in1_hi_ms4 * 4096u;
    const uint32_t ts22_lo_ms4 = ts22_lo >> 12;
    const uint32_t ts22_lo_ls12 = ts22_lo - ts22_lo_ms4 * 4096u;
    const uint32_t ts22_hi_ms4 = ts22_hi >> 12;
    const uint32_t ts22_hi_ls12 = ts22_hi - ts22_hi_ms4 * 4096u;
    const uint32_t xor12_0 = in1_lo_ls12 ^ ts22_lo_ls12;
    const uint32_t xor12_1 = in1_lo_ms4 ^ ts22_lo_ms4;
    const uint32_t xor12_2 = in1_hi_ls12 ^ ts22_hi_ls12;
    const uint32_t xor12_3 = in1_hi_ms4 ^ ts22_hi_ms4;
    bg_aux<4>(outputs, row, column_length, in1_lo_ls12);
    bg_aux<5>(outputs, row, column_length, in1_hi_ls12);
    bg_aux<6>(outputs, row, column_length, ts22_lo_ls12);
    bg_aux<7>(outputs, row, column_length, ts22_hi_ls12);
    bg_trace<24>(outputs, row, in1_lo_ms4);
    bg_trace<25>(outputs, row, in1_hi_ms4);
    bg_trace<26>(outputs, row, ts22_lo_ms4);
    bg_trace<27>(outputs, row, ts22_hi_ms4);
    bg_trace<28>(outputs, row, xor12_0);
    bg_trace<29>(outputs, row, xor12_1);
    bg_trace<30>(outputs, row, xor12_2);
    bg_trace<31>(outputs, row, xor12_3);
    bg_lookup_tuple<4, 648362599>(outputs, row, column_length, in1_lo_ls12,
                                  ts22_lo_ls12, xor12_0);
    bg_lookup_tuple<5, 45448144>(outputs, row, column_length, in1_lo_ms4,
                                 ts22_lo_ms4, xor12_1);
    bg_lookup_tuple<6, 648362599>(outputs, row, column_length, in1_hi_ls12,
                                  ts22_hi_ls12, xor12_2);
    bg_lookup_tuple<7, 45448144>(outputs, row, column_length, in1_hi_ms4,
                                 ts22_hi_ms4, xor12_3);
    bg_count_xor12(feed, in1_lo_ls12, ts22_lo_ls12);
    bg_count_xor12(feed, in1_hi_ls12, ts22_hi_ls12);
    bg_count_lut<1, 2, 4, 0>(feed, in1_lo_ms4, ts22_lo_ms4);
    bg_count_lut<1, 2, 4, 0>(feed, in1_hi_ms4, ts22_hi_ms4);
    const uint32_t xr12_lo = xor12_1 + xor12_2 * 16u;
    const uint32_t xr12_hi = xor12_3 + xor12_0 * 16u;
    const uint32_t xr12 = xr12_lo + (xr12_hi << 16);

    // Triple Sum 32 and R8 xor.
    const uint32_t ts44 = ts0 + xr12 + in5;
    const uint32_t ts44_lo = lo16(ts44);
    const uint32_t ts44_hi = hi16(ts44);
    bg_trace<32>(outputs, row, ts44_lo);
    bg_trace<33>(outputs, row, ts44_hi);

    const uint32_t ts44_lo_ms8 = ts44_lo >> 8;
    const uint32_t ts44_lo_ls8 = ts44_lo - ts44_lo_ms8 * 256u;
    const uint32_t ts44_hi_ms8 = ts44_hi >> 8;
    const uint32_t ts44_hi_ls8 = ts44_hi - ts44_hi_ms8 * 256u;
    const uint32_t xr16_lo_ms8 = xr16_lo >> 8;
    const uint32_t xr16_lo_ls8 = xr16_lo - xr16_lo_ms8 * 256u;
    const uint32_t xr16_hi_ms8 = xr16_hi >> 8;
    const uint32_t xr16_hi_ls8 = xr16_hi - xr16_hi_ms8 * 256u;
    const uint32_t xor8_0 = ts44_lo_ls8 ^ xr16_lo_ls8;
    const uint32_t xor8_1 = ts44_lo_ms8 ^ xr16_lo_ms8;
    const uint32_t xor8_2 = ts44_hi_ls8 ^ xr16_hi_ls8;
    const uint32_t xor8_3 = ts44_hi_ms8 ^ xr16_hi_ms8;
    bg_aux<8>(outputs, row, column_length, ts44_lo_ls8);
    bg_aux<9>(outputs, row, column_length, ts44_hi_ls8);
    bg_aux<10>(outputs, row, column_length, xr16_lo_ls8);
    bg_aux<11>(outputs, row, column_length, xr16_hi_ls8);
    bg_trace<34>(outputs, row, ts44_lo_ms8);
    bg_trace<35>(outputs, row, ts44_hi_ms8);
    bg_trace<36>(outputs, row, xr16_lo_ms8);
    bg_trace<37>(outputs, row, xr16_hi_ms8);
    bg_trace<38>(outputs, row, xor8_0);
    bg_trace<39>(outputs, row, xor8_1);
    bg_trace<40>(outputs, row, xor8_2);
    bg_trace<41>(outputs, row, xor8_3);
    bg_lookup_tuple<8, 112558620>(outputs, row, column_length, ts44_lo_ls8,
                                  xr16_lo_ls8, xor8_0);
    bg_lookup_tuple<9, 112558620>(outputs, row, column_length, ts44_lo_ms8,
                                  xr16_lo_ms8, xor8_1);
    bg_lookup_tuple<10, 521092554>(outputs, row, column_length, ts44_hi_ls8,
                                   xr16_hi_ls8, xor8_2);
    bg_lookup_tuple<11, 521092554>(outputs, row, column_length, ts44_hi_ms8,
                                   xr16_hi_ms8, xor8_3);
    bg_count_lut<0, 0, 8, 0>(feed, ts44_lo_ls8, xr16_lo_ls8);
    bg_count_lut<0, 0, 8, 0>(feed, ts44_lo_ms8, xr16_lo_ms8);
    bg_count_lut<0, 0, 8, 1>(feed, ts44_hi_ls8, xr16_hi_ls8);
    bg_count_lut<0, 0, 8, 1>(feed, ts44_hi_ms8, xr16_hi_ms8);
    const uint32_t xr8_lo = xor8_1 + xor8_2 * 256u;
    const uint32_t xr8_hi = xor8_3 + xor8_0 * 256u;
    const uint32_t xr8 = xr8_lo + (xr8_hi << 16);

    // Triple Sum 32 and R7 xor.
    const uint32_t ts66 = ts22 + xr8;
    const uint32_t ts66_lo = lo16(ts66);
    const uint32_t ts66_hi = hi16(ts66);
    bg_trace<42>(outputs, row, ts66_lo);
    bg_trace<43>(outputs, row, ts66_hi);

    const uint32_t xr12_lo_ms9 = xr12_lo >> 7;
    const uint32_t xr12_lo_ls7 = xr12_lo - xr12_lo_ms9 * 128u;
    const uint32_t xr12_hi_ms9 = xr12_hi >> 7;
    const uint32_t xr12_hi_ls7 = xr12_hi - xr12_hi_ms9 * 128u;
    const uint32_t ts66_lo_ms9 = ts66_lo >> 7;
    const uint32_t ts66_lo_ls7 = ts66_lo - ts66_lo_ms9 * 128u;
    const uint32_t ts66_hi_ms9 = ts66_hi >> 7;
    const uint32_t ts66_hi_ls7 = ts66_hi - ts66_hi_ms9 * 128u;
    const uint32_t xor7_0 = xr12_lo_ls7 ^ ts66_lo_ls7;
    const uint32_t xor7_1 = xr12_lo_ms9 ^ ts66_lo_ms9;
    const uint32_t xor7_2 = xr12_hi_ls7 ^ ts66_hi_ls7;
    const uint32_t xor7_3 = xr12_hi_ms9 ^ ts66_hi_ms9;
    bg_aux<12>(outputs, row, column_length, xr12_lo_ls7);
    bg_aux<13>(outputs, row, column_length, xr12_hi_ls7);
    bg_aux<14>(outputs, row, column_length, ts66_lo_ls7);
    bg_aux<15>(outputs, row, column_length, ts66_hi_ls7);
    bg_trace<44>(outputs, row, xr12_lo_ms9);
    bg_trace<45>(outputs, row, xr12_hi_ms9);
    bg_trace<46>(outputs, row, ts66_lo_ms9);
    bg_trace<47>(outputs, row, ts66_hi_ms9);
    bg_trace<48>(outputs, row, xor7_0);
    bg_trace<49>(outputs, row, xor7_1);
    bg_trace<50>(outputs, row, xor7_2);
    bg_trace<51>(outputs, row, xor7_3);
    bg_lookup_tuple<12, 62225763>(outputs, row, column_length, xr12_lo_ls7,
                                  ts66_lo_ls7, xor7_0);
    bg_lookup_tuple<13, 95781001>(outputs, row, column_length, xr12_lo_ms9,
                                  ts66_lo_ms9, xor7_1);
    bg_lookup_tuple<14, 62225763>(outputs, row, column_length, xr12_hi_ls7,
                                  ts66_hi_ls7, xor7_2);
    bg_lookup_tuple<15, 95781001>(outputs, row, column_length, xr12_hi_ms9,
                                  ts66_hi_ms9, xor7_3);
    bg_count_lut<2, 3, 7, 0>(feed, xr12_lo_ls7, ts66_lo_ls7);
    bg_count_lut<2, 3, 7, 0>(feed, xr12_hi_ls7, ts66_hi_ls7);
    bg_count_lut<3, 4, 9, 0>(feed, xr12_lo_ms9, ts66_lo_ms9);
    bg_count_lut<3, 4, 9, 0>(feed, xr12_hi_ms9, ts66_hi_ms9);
    const uint32_t xr7_lo = xor7_1 + xor7_2 * 512u;
    const uint32_t xr7_hi = xor7_3 + xor7_0 * 512u;
    bg_aux<16>(outputs, row, column_length, xr7_lo);
    bg_aux<17>(outputs, row, column_length, xr7_hi);
    bg_aux<18>(outputs, row, column_length, xr8_lo);
    bg_aux<19>(outputs, row, column_length, xr8_hi);

    const uint32_t enabler = row < n_rows ? 1u : 0u;
    bg_trace<52>(outputs, row, enabler);

    // Final blake_g relation tuple and multiplicities.
    bg_lookup<64>(outputs, row, column_length, 1139985212u);
    bg_lookup<65>(outputs, row, column_length, lo16(in0));
    bg_lookup<66>(outputs, row, column_length, hi16(in0));
    bg_lookup<67>(outputs, row, column_length, lo16(in1));
    bg_lookup<68>(outputs, row, column_length, hi16(in1));
    bg_lookup<69>(outputs, row, column_length, lo16(in2));
    bg_lookup<70>(outputs, row, column_length, hi16(in2));
    bg_lookup<71>(outputs, row, column_length, lo16(in3));
    bg_lookup<72>(outputs, row, column_length, hi16(in3));
    bg_lookup<73>(outputs, row, column_length, lo16(in4));
    bg_lookup<74>(outputs, row, column_length, hi16(in4));
    bg_lookup<75>(outputs, row, column_length, lo16(in5));
    bg_lookup<76>(outputs, row, column_length, hi16(in5));
    bg_lookup<77>(outputs, row, column_length, ts44_lo);
    bg_lookup<78>(outputs, row, column_length, ts44_hi);
    bg_lookup<79>(outputs, row, column_length, xr7_lo);
    bg_lookup<80>(outputs, row, column_length, xr7_hi);
    bg_lookup<81>(outputs, row, column_length, ts66_lo);
    bg_lookup<82>(outputs, row, column_length, ts66_hi);
    bg_lookup<83>(outputs, row, column_length, xr8_lo);
    bg_lookup<84>(outputs, row, column_length, xr8_hi);
    bg_lookup<85>(outputs, row, column_length, 1u);
    bg_lookup<86>(outputs, row, column_length, enabler);
}

#endif  // BLAKE_G_FUSED_SCALAR_CUH
