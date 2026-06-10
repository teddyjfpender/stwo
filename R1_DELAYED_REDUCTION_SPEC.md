# R1 Spec: Delayed-Reduction Accumulation in Prover Hot Loops

Status: APPROVED FOR IMPLEMENTATION (supervised tier — see process requirements below)
Origin: PERFORMANCE_PROPOSALS.md § "Round 2 candidates", item R1.
Target branch: `perf-optimizations`.

## Goal

Reduce end-to-end prove time by ~10–15% by eliminating per-multiplication modular
reductions in the prover's accumulation-heavy inner loops, reducing once per loop (or once
per bounded sub-batch) instead. This is the pattern Plonky3 applies to M31 dot products
(github.com/Plonky3/Plonky3 issue #252, reported ~2.6x on reduction-heavy loops).

Measured motivation (blake 2^18, Apple M4 Pro 14-core, see PERFORMANCE_PROPOSALS.md Round 2
baseline): composition phase = 26.7% of prove; `PackedQM31` multiplies ≈ 9.6% of all on-CPU
samples, `LookupElements::combine` ≈ 6.3%, `finalize_logup_batched` ≈ 4.6%, plus the FRI
quotient numerator loops (quotients phase = 6.9% of prove).

## Mathematical basis (the invariant)

Work in M31, p = 2^31 − 1, so 2^31 ≡ 1 (mod p).

For reduced-or-boundary inputs a, b ∈ [0, p] (NOTE: the `PackedM31` representation invariant
allows the value p itself — see the kernel doc comments in
`crates/stwo/src/prover/backend/simd/m31.rs`):

- Full product: a·b ≤ p² = 2^62 − 2^32 + 1 < 2^62.
- Mersenne split: a·b = hi·2^31 + lo with hi, lo < 2^31, and a·b ≡ hi + lo (mod p),
  where hi + lo < 2^32.

Two sound accumulation strategies (implementer chooses per site, benchmark-driven):

1. **Unreduced u64 accumulation of full products.** Σ of k products < k·2^62; overflows u64
   at k = 4. Requires folding (one Mersenne split of the accumulator) at least every 3 adds.
   Lowest per-term cost, tight bound discipline.
2. **Per-product partial reduction (hi + lo), u64 accumulation.** Each term < 2^32, so up to
   2^32 terms fit in u64 — effectively unbounded for our loop lengths (≤ a few thousand).
   One extra add per product, but removes the compare/min/select reduction step per product.

Either way the final accumulator is reduced to [0, p) (or [0, p] where the consumer accepts
the boundary representation) exactly once.

**Invariant to preserve:** every changed code path computes the *identical field element* it
computes today. Proof bytes must be byte-identical for fixed inputs/channel state. This is a
pure re-association of exact integer arithmetic — no rounding, no probabilistic argument.

## Scope: target sites (in priority order)

### Site A — `LookupElements::combine` SIMD dot product

`crates/constraint-framework/src/logup.rs:84-99`. Generic fold:
`acc + EF::from(power) * value` per element — with `EF = PackedQM31`, `F = PackedM31`, each
step is a full QM31×M31 multiply (4 independent reduced M31 muls) plus a QM31 add.

Mathematically this is **4 independent M31 dot products**: each QM31 coordinate c of the
result is Σ_j alpha_power_j.coord_c · value_j (alpha powers are scalar `SecureField`
constants, values are `PackedM31` lanes), followed by the `- z` subtraction.

Required shape: add a SIMD dot-product primitive (see "New primitives" below) and a
specialized `combine` path for the packed types that uses it. The generic scalar fold stays
for non-SIMD callers (the verifier-side / CPU path must remain untouched in behavior).
This site is hot in BOTH interaction (LogUp) trace generation (8.8% of prove) and
composition (via `SimdDomainEvaluator`).

### Site B — FRI quotient numerator accumulation

`crates/stwo/src/prover/backend/simd/quotients.rs:212-262`
(`accumulate_numerators_on_subdomain`): inner loop `*acc += c_broadcast * val` where
`c_broadcast: PackedSecureField`, `val: PackedM31`, iterated over all columns in the sample
batch. Same structure as Site A: 4 independent M31 dot products per lane-coordinate,
accumulated across `quotient_coeffs.len()` terms. The existing `b_sum` hoisting stays.

### Site C — quotient combine loop (secondary)

`crates/stwo/src/prover/backend/simd/quotients.rs:141-176`: the per-chunk accumulation over
sample batches multiplying `PackedSecureField` terms. Full QM31×QM31 products (9 M31 muls
via two-level Karatsuba, `qm31.rs:116-138`, `cm31.rs:90-128`). Delayed reduction here means
keeping the 9 cross-term M31 products unreduced through the Karatsuba recombination and the
chunk accumulation. This is more invasive — only do it if Sites A+B land cleanly and the
microbench shows ≥15% on the kernel; otherwise record as follow-up.

### Site D — LogUp batched fraction sums (optional follow-up)

`crates/constraint-framework/src/lib.rs:185+` (`finalize_logup_batched` proxy macro):
fraction adds (a/b + c/d = (ad + cb, bd)) in batches (typical batch_size 1–2). Short
chains; gains likely small. Investigate only after A–C; do not block on it.

## New primitives — design requirements

Add to `crates/stwo/src/prover/backend/simd/m31.rs` (and a thin QM31-coordinate wrapper
where it helps call sites), e.g.:

```rust
/// Dot product Σ a_i · b_i over PackedM31 lanes with delayed modular reduction.
/// Output is fully reduced. Inputs may be in the unreduced [0, P] representation.
pub fn dot_delayed(a: &[PackedM31], b: &[PackedM31]) -> PackedM31;
/// Variant where one side is a scalar M31 constant per term (alpha powers).
pub fn dot_delayed_scalar(coeffs: &[M31], values: &[PackedM31]) -> PackedM31;
```

(Exact names/signatures at implementer's discretion; match surrounding naming style.)

Rules:

1. **Do not change the semantics, signatures, or implementations of any existing field
   op** (`Mul`/`Add` impls, `mul_neon`, `mul_doubled_*`, `reduce`, etc.). Delayed reduction
   enters only via new, clearly named functions whose doc comments state: input range
   assumptions, the accumulation bound, and where the single reduction happens.
2. **Portable-first.** A portable implementation (std::simd u64 widening, or the existing
   `mul_simd` hi/lo decomposition) must be correct on every arch the crate builds for
   (aarch64 NEON, x86_64 with/without AVX2/AVX-512, wasm, scalar fallback). Arch-specialized
   kernels (e.g. NEON `vmlal_u32`-style widening accumulate, or reuse of `vqdmulhq` hi
   extraction) only where the microbench shows a win, following the existing
   `cfg(target_feature)` dispatch pattern in `m31.rs`.
3. **Bound discipline.** Every unreduced accumulator carries a comment deriving its
   worst-case value from input ranges, and a `debug_assert!` guarding the bound where
   cheaply expressible. If strategy 1 (full-product accumulation) is used, the fold
   frequency must be a named constant with the overflow arithmetic written out next to it.
4. The dot-product length cap (if any) must be asserted against at the call site, not
   silently truncated.

## Process requirements (CLAUDE.md supervised tier)

- All changed files are prover-side (`crates/stwo/src/prover/`, `crates/constraint-framework/`).
  **Nothing under `crates/stwo/src/core/` may change** — no verifier, no Fiat-Shamir, no
  constraint *definitions*. `LookupElements::combine` in `logup.rs` is
  [SOUNDNESS-CRITICAL]-adjacent because the same function body services prover and verifier:
  the specialization must be additive (new code path for packed types) and the generic path
  must remain byte-for-byte identical in behavior. State this explicitly in the PR.
- No new `unsafe` unless an arch kernel requires intrinsics; then follow the existing
  documented-justification style of `m31.rs` (safety comment deriving why the
  representation invariant holds on output).
- Before starting, restate in the PR description: the invariant ("identical field elements,
  re-associated integer arithmetic, single deferred reduction with bound B"), and the test
  that verifies it (below).

## Tests (gates — all must pass before benchmarking matters)

1. **Equivalence tests** for each new primitive vs. the naive reduced fold, over:
   - 10k+ random inputs (seeded `SmallRng`, like existing tests in `m31.rs`),
   - edge lanes: 0, 1, p−1, p (the boundary representation), and mixtures across lanes,
   - dot lengths 0, 1, 2, 3, 4, 5 (around the fold boundary if strategy 1), and a long
     length matching the largest real call site (longest `LookupElements` tuple in
     `crates/examples/` — find it and test at that length + 1).
2. **Call-site equivalence:** `LookupElements::combine` packed vs. scalar-generic results
   on random inputs (extend the existing logup tests).
3. **Proof regression:** existing prove+verify tests for blake, plonk, wide-fibonacci pass
   unchanged (`cargo test --release --features "slow-tests,prover,parallel"`). Where a test
   asserts proof bytes/roots, those must be identical — if any test only checks
   verify-success, add one byte-equality assertion for a small fixed-seed proof to pin
   exactness (pattern: `test_pruned_commit_decommit_equivalence`).
4. Verifier-only build still compiles: `cd ensure-verifier-no_std && cargo build -r`, and
   `cargo test --no-default-features --package stwo` and
   `--package stwo-constraint-framework`.
5. `scripts/clippy.sh` and `scripts/rust_fmt.sh --check` clean.

## Benchmarks & acceptance criteria

Baseline protocol (already captured — see `R1_BASELINE.md` at repo root; criterion baseline
saved under the name `r1-pre` in `target/criterion`; note `cargo clean` destroys it, in
which case re-create it from `dev`+this-branch-pre-R1 per R1_BASELINE.md instructions):

```bash
# e2e (3 runs, report each):
LOG_N_INSTANCES=18 ./blake_benchmark.sh        # phase spans + total prove
# microbenches (same RUSTFLAGS as e2e):
RUSTFLAGS="-C target-cpu=native" cargo bench -p stwo --features "prover,parallel" \
  --bench fri_quotients -- --save-baseline r1-pre
RUSTFLAGS="-C target-cpu=native" cargo bench -p stwo --features "prover,parallel" \
  --bench field -- --save-baseline r1-pre
```

After implementation, compare with `-- --baseline r1-pre`, and add a criterion microbench
for the new dot primitive (e.g. in `benches/field.rs`: lengths 4 / 16 / 64, naive vs
delayed) so the kernel win is tracked independently of e2e noise.

Accept when, on this machine (plugged in, quiet background):

- e2e blake 2^18 total prove improves ≥8% (median of 3 runs) with composition + interaction
  + quotients phase spans showing the gain (i.e. the win is where the theory says);
- `fri_quotients` criterion bench improves on the accumulation cases;
- no criterion case regresses >2% beyond noise;
- peak RSS (from `/usr/bin/time -l` on the test binary) unchanged ±2%.

If e2e lands below 8% but kernel microbenches show the expected win, stop and report — the
bottleneck model needs revisiting, don't force further sites.

## Out of scope

FFT loops, Merkle hashing, OODS/barycentric (R3), chunk-size retuning and parallelism
(R2 — keep R1 changes orthogonal so the two can be measured independently), any change to
proof structure, security parameters, or verifier logic.
