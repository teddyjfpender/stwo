#!/bin/bash
# Poseidon end-to-end proving benchmark.
#
# WARNING: `test_simd_poseidon_prove` is currently marked `#[ignore]` ("AIRs with constraint
# degree >= 2 are not supported yet in the lifted protocol"), so this benchmark runs 0 tests
# on such branches. The script fails loudly in that case instead of silently reporting
# success. For a working end-to-end proving benchmark, use `blake_benchmark.sh`.
set -uo pipefail

OUT=$(LOG_N_INSTANCES=${LOG_N_INSTANCES:-18} RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info \
    RUSTFLAGS="-C target-cpu=native" \
    cargo test --release test_simd_poseidon_prove --features parallel -- --nocapture)
echo "$OUT"
if echo "$OUT" | grep -q "1 ignored"; then
    echo "ERROR: test_simd_poseidon_prove is ignored on this branch; no benchmark was run." >&2
    echo "Use ./blake_benchmark.sh instead." >&2
    exit 1
fi
