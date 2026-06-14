use std::time::Instant;

use stwo_backend_metal_sys::metal::{metal_runtime_support, MetalRuntimeSupport, U32Buffer};

const LIMB_BITS: u32 = 9;
const LIMB_MASK: u32 = (1 << LIMB_BITS) - 1;
const M31_P: u32 = 2_147_483_647;

fn require_metal() -> bool {
    if metal_runtime_support() == MetalRuntimeSupport::Available {
        return true;
    }
    eprintln!("skipping Metal witness memory test: Metal runtime unavailable");
    false
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

fn m31_sub(lhs: u32, rhs: u32) -> u32 {
    if lhs >= rhs {
        lhs - rhs
    } else {
        lhs + M31_P - rhs
    }
}

fn expected_ret_opcode_trace(
    inputs: &[u32],
    address_to_id: &[u32],
    big_values: &[u32],
    small_values: &[u32],
    n_rows: usize,
    column_length: usize,
) -> Vec<u32> {
    let mut trace = vec![0; 16 * column_length];
    for row in 0..column_length {
        let pc = inputs[row * 3];
        let ap = inputs[row * 3 + 1];
        let fp = inputs[row * 3 + 2];
        trace[row] = pc;
        trace[column_length + row] = ap;
        trace[2 * column_length + row] = fp;

        let addr0 = m31_sub(fp, 1);
        let id0 = address_to_id[addr0 as usize - 1];
        let words0 = if id0 >> 30 == 1 {
            &big_values[((id0 & 0x3fff_ffff) as usize) * 8..][..8]
        } else {
            &small_values[((id0 & 0x3fff_ffff) as usize) * 4..][..4]
        };
        let limbs0 = split_le_9bit(words0, 4);
        trace[3 * column_length + row] = id0;
        trace[4 * column_length + row] = limbs0[0];
        trace[5 * column_length + row] = limbs0[1];
        trace[6 * column_length + row] = limbs0[2];
        trace[7 * column_length + row] = limbs0[3];
        trace[8 * column_length + row] = (limbs0[3] & 2) >> 1;

        let addr1 = m31_sub(fp, 2);
        let id1 = address_to_id[addr1 as usize - 1];
        let words1 = if id1 >> 30 == 1 {
            &big_values[((id1 & 0x3fff_ffff) as usize) * 8..][..8]
        } else {
            &small_values[((id1 & 0x3fff_ffff) as usize) * 4..][..4]
        };
        let limbs1 = split_le_9bit(words1, 4);
        trace[9 * column_length + row] = id1;
        trace[10 * column_length + row] = limbs1[0];
        trace[11 * column_length + row] = limbs1[1];
        trace[12 * column_length + row] = limbs1[2];
        trace[13 * column_length + row] = limbs1[3];
        trace[14 * column_length + row] = (limbs1[3] & 2) >> 1;
        trace[15 * column_length + row] = u32::from(row < n_rows);
    }
    trace
}

fn expected_add_opcode_small_trace(n_rows: usize, column_length: usize) -> Vec<u32> {
    let mut trace = vec![0; 39 * column_length];
    for row in 0..column_length {
        trace[row] = 1;
        trace[column_length + row] = 10;
        trace[2 * column_length + row] = 20;
        trace[3 * column_length + row] = 32768;
        trace[4 * column_length + row] = 32768;
        trace[5 * column_length + row] = 32768;
        trace[11 * column_length + row] = 10;
        trace[12 * column_length + row] = 10;
        trace[13 * column_length + row] = 10;
        for base in [14usize, 22, 30] {
            trace[(base + 3) * column_length + row] = 5;
        }
        trace[38 * column_length + row] = u32::from(row < n_rows);
    }
    trace
}

fn expected_assert_eq_opcode_trace(n_rows: usize, column_length: usize) -> Vec<u32> {
    let mut trace = vec![0; 12 * column_length];
    for row in 0..column_length {
        trace[row] = 1;
        trace[column_length + row] = 10;
        trace[2 * column_length + row] = 20;
        trace[3 * column_length + row] = 32768;
        trace[4 * column_length + row] = 32768;
        trace[8 * column_length + row] = 10;
        trace[9 * column_length + row] = 10;
        trace[11 * column_length + row] = u32::from(row < n_rows);
    }
    trace
}

fn deterministic_words(n_rows: usize, n_words: usize) -> Vec<u32> {
    (0..n_rows * n_words)
        .map(|i| {
            let value = i as u32;
            value.wrapping_mul(0x9e37_79b9).rotate_left((i % 31) as u32) ^ 0xa5a5_5a5a
        })
        .collect()
}

fn deterministic_mults(n_rows: usize) -> Vec<u32> {
    (0..n_rows)
        .map(|i| (17u32.wrapping_mul(i as u32).wrapping_add(3)) & 0x7fff_ffff)
        .collect()
}

fn expected_trace(
    values: &[u32],
    mults: &[u32],
    n_values: usize,
    column_length: usize,
    n_words: usize,
    n_limbs: usize,
) -> Vec<u32> {
    let mut trace = vec![0; (n_limbs + 1) * column_length];
    for row in 0..column_length {
        let row_words = if row < n_values {
            &values[row * n_words..(row + 1) * n_words]
        } else {
            &[]
        };
        for (limb, value) in split_le_9bit(row_words, n_limbs).into_iter().enumerate() {
            trace[limb * column_length + row] = value;
        }
        if row < n_values {
            trace[n_limbs * column_length + row] = mults[row];
        }
    }
    trace
}

fn host_rc99_counts(
    columns: &[Vec<u32>],
    input_to_row_lut: &[u32],
    column_length: usize,
    n_pairs: usize,
    rc_table_size: usize,
) -> Vec<u32> {
    let mut counts = vec![0u32; 8 * rc_table_size];
    for row in 0..column_length {
        for pair in 0..n_pairs {
            let v0 = columns[2 * pair][row];
            let v1 = columns[2 * pair + 1][row];
            let rc_row = input_to_row_lut[((v0 << LIMB_BITS) | v1) as usize] as usize;
            counts[(pair % 8) * rc_table_size + rc_row] += 1;
        }
    }
    counts
}

#[test]
fn add_opcode_small_trace_matches_hand_built_case() {
    if !require_metal() {
        return;
    }

    let n_rows = 1usize;
    let column_length = 16usize;
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
    let trace = U32Buffer::witness_add_opcode_small_trace(
        &inputs_dev,
        &address_to_id_dev,
        &big_values_dev,
        &small_values_dev,
        n_rows as u32,
        column_length as u32,
    )
    .expect("add_opcode_small trace");

    assert_eq!(
        trace.to_vec().expect("add_opcode_small trace readback"),
        expected_add_opcode_small_trace(n_rows, column_length),
    );

    let trace_cols = U32Buffer::witness_add_opcode_small_trace_columns(
        &inputs_dev,
        &address_to_id_dev,
        &big_values_dev,
        &small_values_dev,
        n_rows as u32,
        column_length as u32,
    )
    .expect("add_opcode_small trace columns");
    let expected = expected_add_opcode_small_trace(n_rows, column_length);
    for (i, col) in trace_cols.iter().enumerate() {
        assert_eq!(
            col.to_vec()
                .expect("add_opcode_small trace column readback"),
            expected[i * column_length..(i + 1) * column_length],
        );
    }
}

#[test]
fn assert_eq_opcode_trace_matches_hand_built_case() {
    if !require_metal() {
        return;
    }

    let n_rows = 1usize;
    let column_length = 16usize;
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
    let trace = U32Buffer::witness_assert_eq_opcode_trace(
        &inputs_dev,
        &address_to_id_dev,
        &big_values_dev,
        &small_values_dev,
        n_rows as u32,
        column_length as u32,
    )
    .expect("assert_eq_opcode trace");

    assert_eq!(
        trace.to_vec().expect("assert_eq_opcode trace readback"),
        expected_assert_eq_opcode_trace(n_rows, column_length),
    );

    let trace_cols = U32Buffer::witness_assert_eq_opcode_trace_columns(
        &inputs_dev,
        &address_to_id_dev,
        &big_values_dev,
        &small_values_dev,
        n_rows as u32,
        column_length as u32,
    )
    .expect("assert_eq_opcode trace columns");
    let expected = expected_assert_eq_opcode_trace(n_rows, column_length);
    for (i, col) in trace_cols.iter().enumerate() {
        assert_eq!(
            col.to_vec()
                .expect("assert_eq_opcode trace column readback"),
            expected[i * column_length..(i + 1) * column_length],
        );
    }
}

#[test]
fn memory_id_to_big_trace_matches_cairo_split() {
    if !require_metal() {
        return;
    }

    let n_values = 19usize;
    let column_length = 32usize;
    let values = deterministic_words(n_values, 8);
    let mults = deterministic_mults(n_values);
    let values_dev = U32Buffer::from_slice(&values).expect("big values upload");
    let mults_dev = U32Buffer::from_slice(&mults).expect("big mults upload");

    let trace = U32Buffer::witness_memory_id_to_big_trace(
        &values_dev,
        &mults_dev,
        n_values as u32,
        column_length as u32,
    )
    .expect("big memory witness split");
    let trace_cols = U32Buffer::witness_memory_id_to_big_trace_columns(
        &values_dev,
        &mults_dev,
        n_values as u32,
        column_length as u32,
    )
    .expect("big memory witness column split");
    let expected = expected_trace(&values, &mults, n_values, column_length, 8, 28);

    assert_eq!(trace.to_vec().expect("big trace readback"), expected,);
    for (i, col) in trace_cols.iter().enumerate() {
        assert_eq!(
            col.to_vec().expect("big trace column readback"),
            expected[i * column_length..(i + 1) * column_length],
        );
    }
}

#[test]
fn memory_id_to_big_small_trace_matches_cairo_split() {
    if !require_metal() {
        return;
    }

    let n_values = 23usize;
    let column_length = 32usize;
    let values = deterministic_words(n_values, 4);
    let mults = deterministic_mults(n_values);
    let values_dev = U32Buffer::from_slice(&values).expect("small values upload");
    let mults_dev = U32Buffer::from_slice(&mults).expect("small mults upload");

    let trace = U32Buffer::witness_memory_id_to_big_small_trace(
        &values_dev,
        &mults_dev,
        n_values as u32,
        column_length as u32,
    )
    .expect("small memory witness split");
    let trace_cols = U32Buffer::witness_memory_id_to_big_small_trace_columns(
        &values_dev,
        &mults_dev,
        n_values as u32,
        column_length as u32,
    )
    .expect("small memory witness column split");
    let expected = expected_trace(&values, &mults, n_values, column_length, 4, 8);

    assert_eq!(trace.to_vec().expect("small trace readback"), expected,);
    for (i, col) in trace_cols.iter().enumerate() {
        assert_eq!(
            col.to_vec().expect("small trace column readback"),
            expected[i * column_length..(i + 1) * column_length],
        );
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
    let values = deterministic_words(n_values, 8);
    let mults = deterministic_mults(n_values);
    let values_dev = U32Buffer::from_slice(&values).expect("big values upload");
    let mults_dev = U32Buffer::from_slice(&mults).expect("big mults upload");
    let trace_cols = U32Buffer::witness_memory_id_to_big_trace_columns(
        &values_dev,
        &mults_dev,
        n_values as u32,
        column_length as u32,
    )
    .expect("big memory witness column split");
    let lut = (0..1usize << (2 * LIMB_BITS))
        .map(|key| ((key * 13 + 7) % rc_table_size) as u32)
        .collect::<Vec<_>>();
    let lut_dev = U32Buffer::from_slice(&lut).expect("rc99 lut upload");
    let limb_refs = trace_cols[..28].iter().collect::<Vec<_>>();

    let counts = U32Buffer::witness_memory_rc99_count(
        &limb_refs,
        &lut_dev,
        column_length as u32,
        14,
        rc_table_size as u32,
    )
    .expect("rc99 count");
    let host_columns = trace_cols[..28]
        .iter()
        .map(|col| col.to_vec().expect("limb col readback"))
        .collect::<Vec<_>>();

    assert_eq!(
        counts.to_vec().expect("rc99 count readback"),
        host_rc99_counts(&host_columns, &lut, column_length, 14, rc_table_size)
    );
}

#[test]
fn ret_opcode_trace_matches_host_rows() {
    if !require_metal() {
        return;
    }

    let n_rows = 3usize;
    let column_length = 4usize;
    let inputs = vec![
        10, 20, 4, // row 0: reads addresses 3 and 2.
        11, 21, 5, // row 1: reads addresses 4 and 3.
        12, 22, 6, // row 2: reads addresses 5 and 4.
        10, 20, 4, // padded row, same convention as the host writer.
    ];
    let address_to_id = vec![0, 1, 2, 3, 4, 5];
    let mut small_values = Vec::new();
    for i in 0..6u32 {
        small_values.extend_from_slice(&[
            0x0012_3450u32.wrapping_add(i * 17),
            0x0000_0200u32.wrapping_add(i),
            i,
            0,
        ]);
    }
    let big_values = vec![0u32; 8];

    let inputs_dev = U32Buffer::from_slice(&inputs).expect("ret inputs upload");
    let address_to_id_dev = U32Buffer::from_slice(&address_to_id).expect("addr table upload");
    let big_values_dev = U32Buffer::from_slice(&big_values).expect("big table upload");
    let small_values_dev = U32Buffer::from_slice(&small_values).expect("small table upload");

    let trace = U32Buffer::witness_ret_opcode_trace(
        &inputs_dev,
        &address_to_id_dev,
        &big_values_dev,
        &small_values_dev,
        n_rows as u32,
        column_length as u32,
    )
    .expect("ret opcode trace");

    assert_eq!(
        trace.to_vec().expect("ret trace readback"),
        expected_ret_opcode_trace(
            &inputs,
            &address_to_id,
            &big_values,
            &small_values,
            n_rows,
            column_length,
        )
    );

    let trace_cols = U32Buffer::witness_ret_opcode_trace_columns(
        &inputs_dev,
        &address_to_id_dev,
        &big_values_dev,
        &small_values_dev,
        n_rows as u32,
        column_length as u32,
    )
    .expect("ret opcode trace columns");
    let expected = expected_ret_opcode_trace(
        &inputs,
        &address_to_id,
        &big_values,
        &small_values,
        n_rows,
        column_length,
    );
    for (i, col) in trace_cols.iter().enumerate() {
        assert_eq!(
            col.to_vec().expect("ret trace column readback"),
            expected[i * column_length..(i + 1) * column_length],
        );
    }
}

#[test]
#[ignore = "benchmark; run explicitly with --ignored"]
fn bench_memory_id_to_big_split_sys() {
    if !require_metal() {
        return;
    }

    let log_n_values = std::env::var("BENCH_LOG_N_VALUES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(20);
    let n_values = 1usize << log_n_values;
    let values = deterministic_words(n_values, 8);
    let mults = deterministic_mults(n_values);
    let values_dev = U32Buffer::from_slice(&values).expect("big values upload");
    let mults_dev = U32Buffer::from_slice(&mults).expect("big mults upload");

    let mut timings = Vec::new();
    for iter in 0..5 {
        let start = Instant::now();
        let trace = U32Buffer::witness_memory_id_to_big_trace(
            &values_dev,
            &mults_dev,
            n_values as u32,
            n_values as u32,
        )
        .expect("big memory witness split");
        std::hint::black_box(trace);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        timings.push(elapsed);
        println!(
            "sys_memory_id_to_big_split log_n_values={log_n_values} iter={iter} ms={elapsed:.3} rows_per_s={:.0} mhz={:.3}",
            n_values as f64 / (elapsed / 1000.0),
            n_values as f64 / (elapsed / 1000.0) / 1e6
        );
    }

    let warm_best = timings[1..].iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "RESULT sys_memory_id_to_big_split log_n_values={log_n_values} warm_best_ms={warm_best:.3} warm_rows_per_s={:.0} warm_mhz={:.3}",
        n_values as f64 / (warm_best / 1000.0),
        n_values as f64 / (warm_best / 1000.0) / 1e6
    );

    let mut column_timings = Vec::new();
    for iter in 0..5 {
        let start = Instant::now();
        let trace = U32Buffer::witness_memory_id_to_big_trace_columns(
            &values_dev,
            &mults_dev,
            n_values as u32,
            n_values as u32,
        )
        .expect("big memory witness column split");
        std::hint::black_box(trace);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        column_timings.push(elapsed);
        println!(
            "sys_memory_id_to_big_split_columns log_n_values={log_n_values} iter={iter} ms={elapsed:.3} rows_per_s={:.0} mhz={:.3}",
            n_values as f64 / (elapsed / 1000.0),
            n_values as f64 / (elapsed / 1000.0) / 1e6
        );
    }

    let column_warm_best = column_timings[1..]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    println!(
        "RESULT sys_memory_id_to_big_split_columns log_n_values={log_n_values} warm_best_ms={column_warm_best:.3} warm_rows_per_s={:.0} warm_mhz={:.3}",
        n_values as f64 / (column_warm_best / 1000.0),
        n_values as f64 / (column_warm_best / 1000.0) / 1e6
    );
}
