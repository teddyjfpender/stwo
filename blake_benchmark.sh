#!/bin/bash
# Blake2s end-to-end proving benchmark (steps/sec proxy: one instance = one Blake2s
# compression). Prints tracing span timings for each proving stage.
#
# Usage: LOG_N_INSTANCES=18 ./blake_benchmark.sh
LOG_N_INSTANCES=${LOG_N_INSTANCES:-18} RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info \
    RUSTFLAGS="-C target-cpu=native" \
    cargo test --release test_simd_blake_prove --features "parallel,slow-tests" -- --nocapture
