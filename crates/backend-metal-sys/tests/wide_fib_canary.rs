use std::time::Instant;

use stwo_backend_metal_sys::metal::{metal_runtime_support, MetalRuntimeSupport, U32Buffer};

const P: u32 = 2_147_483_647;

fn require_metal() -> bool {
    if metal_runtime_support() == MetalRuntimeSupport::Available {
        return true;
    }
    eprintln!("skipping Metal wide-fib canary: Metal runtime unavailable");
    false
}

fn m31_add(lhs: u32, rhs: u32) -> u32 {
    let sum = lhs + rhs;
    if sum >= P {
        sum - P
    } else {
        sum
    }
}

fn m31_square(value: u32) -> u32 {
    ((value as u64 * value as u64) % P as u64) as u32
}

fn input_column(len: usize, offset: u32) -> Vec<u32> {
    (0..len)
        .map(|i| ((i as u64 * 1_103 + offset as u64) % P as u64) as u32)
        .collect()
}

fn expected_trace(input_a: &[u32], input_b: &[u32], n_columns: usize) -> Vec<u32> {
    let len = input_a.len();
    let mut trace = vec![0u32; len * n_columns];
    trace[..len].copy_from_slice(input_a);
    trace[len..2 * len].copy_from_slice(input_b);
    for column in 2..n_columns {
        for row in 0..len {
            trace[column * len + row] = m31_add(
                m31_square(trace[(column - 2) * len + row]),
                m31_square(trace[(column - 1) * len + row]),
            );
        }
    }
    trace
}

#[test]
fn wide_fibonacci_sys_path_is_present_and_correct() {
    if !require_metal() {
        return;
    }

    let log_n_rows = 5u32;
    let n_rows = 1usize << log_n_rows;
    let n_columns = 16u32;
    let input_a = input_column(n_rows, 3);
    let input_b = input_column(n_rows, 7);
    let input_a_dev = U32Buffer::from_slice(&input_a).expect("input_a upload");
    let input_b_dev = U32Buffer::from_slice(&input_b).expect("input_b upload");

    let trace = U32Buffer::generate_wide_fibonacci_trace(&input_a_dev, &input_b_dev, n_columns)
        .expect("wide-fib trace generation");
    assert_eq!(
        trace.to_vec().expect("trace readback"),
        expected_trace(&input_a, &input_b, n_columns as usize)
    );

    let n_constraints = n_columns - 2;
    let random_coeffs = vec![1u32, 0, 0, 0].repeat(n_constraints as usize);
    let random_coeffs_dev = U32Buffer::from_slice(&random_coeffs).expect("coeff upload");
    let denominator_inverses = U32Buffer::from_slice(&[1]).expect("denom upload");
    let quotient = U32Buffer::accumulate_wide_fibonacci_quotients(
        &trace,
        &random_coeffs_dev,
        &denominator_inverses,
        log_n_rows,
        log_n_rows,
        n_constraints,
    )
    .expect("wide-fib quotient accumulation");
    assert!(quotient
        .to_vec()
        .expect("quotient readback")
        .into_iter()
        .all(|value| value == 0));
}

#[test]
#[ignore = "benchmark; run explicitly with --ignored"]
fn bench_wide_fibonacci_sys_canary() {
    if !require_metal() {
        return;
    }

    let log_n_rows = std::env::var("BENCH_LOG_N_ROWS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(20);
    let n_rows = 1usize << log_n_rows;
    let n_columns = std::env::var("BENCH_N_COLUMNS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(16);
    let n_constraints = n_columns - 2;
    let input_a = input_column(n_rows, 3);
    let input_b = input_column(n_rows, 7);
    let input_a_dev = U32Buffer::from_slice(&input_a).expect("input_a upload");
    let input_b_dev = U32Buffer::from_slice(&input_b).expect("input_b upload");
    let random_coeffs = vec![1u32, 0, 0, 0].repeat(n_constraints as usize);
    let random_coeffs_dev = U32Buffer::from_slice(&random_coeffs).expect("coeff upload");
    let denominator_inverses = U32Buffer::from_slice(&[1]).expect("denom upload");

    let mut timings = Vec::new();
    for iter in 0..5 {
        let start = Instant::now();
        let trace = U32Buffer::generate_wide_fibonacci_trace(&input_a_dev, &input_b_dev, n_columns)
            .expect("wide-fib trace generation");
        let quotient = U32Buffer::accumulate_wide_fibonacci_quotients(
            &trace,
            &random_coeffs_dev,
            &denominator_inverses,
            log_n_rows,
            log_n_rows,
            n_constraints,
        )
        .expect("wide-fib quotient accumulation");
        std::hint::black_box(quotient);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        timings.push(elapsed);
        println!(
            "sys_wide_fib log_n_rows={log_n_rows} n_columns={n_columns} iter={iter} ms={elapsed:.3} rows_per_s={:.0} mhz={:.3}",
            n_rows as f64 / (elapsed / 1000.0),
            n_rows as f64 / (elapsed / 1000.0) / 1e6
        );
    }

    let warm_best = timings[1..].iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "RESULT sys_wide_fib log_n_rows={log_n_rows} n_columns={n_columns} warm_best_ms={warm_best:.3} warm_rows_per_s={:.0} warm_mhz={:.3}",
        n_rows as f64 / (warm_best / 1000.0),
        n_rows as f64 / (warm_best / 1000.0) / 1e6
    );
}
