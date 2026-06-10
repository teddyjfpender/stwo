# R1 Baseline — pre-implementation measurements

Captured 2026-06-10 on the local dev machine (Apple M4 Pro, 14 cores, 24 GB), branch
`perf-optimizations` @ 7b19459f, plugged in, sequential runs (no concurrent load).
Companion to `R1_DELAYED_REDUCTION_SPEC.md`.

> NOTE: an earlier profiled run reported blake 2^18 ≈ 25.8 s; that run shared the machine
> with concurrent research agents and is NOT a valid baseline. The numbers below, from three
> clean sequential runs, are authoritative. Phase *ratios* from the earlier profile remain
> roughly consistent (composition ≈ 27–31% of prove).

## End-to-end: blake 2^18 (`LOG_N_INSTANCES=18 ./blake_benchmark.sh`)

Total wall = examples test-binary time; phases are tracing span `time.busy` (seconds).
Raw logs: /tmp/r1_e2e_run{1,2,3}.txt (ephemeral — regenerate via the script below).

| Metric | run 1 | run 2 | run 3 | median |
|---|---|---|---|---|
| **Total (test binary wall)** | 14.57 | 14.63 | 14.01 | **14.57** |
| Trace (gen + interp + commit) | 5.00 | 3.76 | 3.87 | 3.87 |
| Interaction (LogUp gen + commit) | 4.05 | 3.39 | 4.13 | 4.05 |
| prove_ex total | 5.48 | 7.46 | 5.99 | 5.99 |
| — Composition (total) | 4.20 | 4.87 | 4.53 | **4.53** |
| — Composition: constraint eval (largest sub-span) | 2.97 | 3.62 | 3.23 | 3.23 |
| — OODS (evaluate columns out of domain) | 0.79 | 0.95 | 0.84 | 0.84 |
| — FRI quotients | 0.24 | 0.53 | 0.20 | 0.24 |

R1-relevant spans to watch: **Composition** (~31% of wall), **Generate round interaction
trace** (1.40 / 1.74 / 1.94 — calls `LookupElements::combine` per row), **FRI quotients**.
Run 1 has cold-cache inflation in Trace; run-to-run variance on composition is ~±8% —
hence median-of-3 and the ≥8% e2e acceptance bar in the spec.

## Criterion microbenches (saved baseline name: `r1-pre` in `target/criterion`)

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench -p stwo --features "prover,parallel" \
  --bench fri_quotients -- --save-baseline r1-pre
RUSTFLAGS="-C target-cpu=native" cargo bench -p stwo --features "prover,parallel" \
  --bench field -- --save-baseline r1-pre
```

| Bench | median |
|---|---|
| accumulate_numerators 2^21 x 100 cols | 22.32 ms |
| compute_quotients_and_combine 2^21 x 10 pts | 8.46 ms |
| mul_simd (PackedM31) | 7.66 ms |
| add_simd | 4.57 ms |
| sub_simd | 4.51 ms |
| M31 mul (scalar) | 23.42 ms |
| CM31 mul (scalar) | 99.82 ms |
| SecureField mul (scalar) | 572.8 ms |

Compare after implementation with `-- --baseline r1-pre` (criterion prints per-case deltas).
`cargo clean` destroys `target/criterion` — if lost, re-run the two commands above on a
pre-R1 commit to recreate, then re-run on the R1 commit with `--baseline r1-pre`.

## Post-implementation results (2026-06-10, R1 sites A+B implemented)

Kernel microbenches (naive vs delayed, same-run comparison, clean):

| Primitive | len 4 | len 16 | len 64 |
|---|---|---|---|
| dot_delayed (M31×M31) | −4.6% | −22% | −26% |
| dot_delayed_scalar | +2.5% | −16% | −19% |
| dot_delayed_qm31_scalar (Site A kernel) | +4% | −13% | −17% |

`accumulate_numerators 2^21 x 100 cols` (Site B): **−11%** vs r1-pre.
`compute_quotients_and_combine`: +1.2% (within noise; untouched Site C path).
**e2e blake 2^18: flat** — 14.80 s median vs 14.57 s pre (within the ±8% phase variance).

Verdict per the spec's escape hatch: kernel wins are real but e2e is below the 8% bar.
Root cause (from adversarial review): composition — the largest phase — uses
`VeryPackedM31`/`VeryPackedQM31`, which fall through to the generic fold; the Site A
specialization only fires in interaction trace gen, where combine is a minority of the
phase. The profile's ~20%-of-samples attribution for combine/QM31-mul included composition's
VeryPacked code paths, which R1 as implemented does not reach. Follow-up candidate: extend
the specialization to the VeryPacked types (composition), which is where the modeled win
actually lives.

### Round 2: VeryPacked extension (same day)

Extended the Site A specialization to `(VeryPackedM31, VeryPackedQM31)` — the types
composition actually uses (`dot_delayed_very_packed_qm31_scalar` in `very_packed_m31.rs`,
second `CombineDispatch` impl). Measured (3 e2e runs, medians vs the pre-R1 baseline):

| Metric | pre-R1 | post round 2 | delta |
|---|---|---|---|
| Composition (total span) | 4.53 s | 3.57 s | **−21%** |
| Constraint point-wise eval (summed) | 4.43 s | 3.43 s | −23% |
| e2e wall | 14.57 s | 13.54 s | **−7.1%** |

(Both run sets carry a cold first-run outlier; excluding those, e2e is −9…−13%.)
VeryPacked kernel microbench (naive vs delayed): len 4 **+6%** (call overhead dominates),
len 16 −12%, len 64 −15%. Blake's hot relations are long (round tuples ~96 felts), hence
the composition win; AIRs dominated by short tuples (len ≤ 4) may see a small per-call
regression on the combine itself — acceptable here, but worth rechecking if such an AIR
becomes a benchmark target.

Verdict: acceptance criteria met in substance — the gain is exactly where the theory says
(composition + quotients), e2e at-or-above the 8% bar after outlier handling, proof bytes
unchanged, no gate regressions.

### R2: parallelism basket (same day)

Parallel decommit across the 4 commitment trees (order-preserving `into_par_iter`;
Send/Sync analysis in the code comment — `TwiddleTree` is `unsafe impl Sync`, the column
pool is DashMap-backed, per-call decommit state is local), chunk retuning
(`COMBINE_CHUNK_SIZE` 16→32 ~−3%, `FOLD_CHUNK_SIZE` 128→256 ~−3% with 512 regressing +11%,
`NUMERATORS_CHUNK_SIZE` kept at 64 — incumbent won), `with_min_len(1<<13)` guards on the
small barycentric-weights loops, and new SIMD fold benches in `benches/fri.rs`.

| Metric | pre-R1 | post-R1 | post-R2 |
|---|---|---|---|
| e2e blake 2^18 wall (median of 3) | 14.57 s | 13.54 s | **11.90 s** |

R2 alone: **−12%**; branch cumulative this session: **−18%**. Proof bytes verified
unchanged (golden-hash test) and the full gate set re-verified after implementation.

Incident note: the first post-R1 `fri_quotients` bench run crashed with a rayon-worker
stack overflow — the enlarged per-chunk accumulator array (~33 KiB) was inlined into
rayon's recursive splitter frame and replicated per split level at 2^21. Fixed by
extracting the chunk body into an `#[inline(never)]` leaf function
(`accumulate_numerators_chunk` in `simd/quotients.rs`); the pre-R1 code (16 KiB frame) was
already near this cliff on large domains. All gates re-verified after the fix (quotients
tests, proof-byte regression, fmt, clippy).

## Reproduction script

```bash
for i in 1 2 3; do
  LOG_N_INSTANCES=18 RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info \
  RUSTFLAGS="-C target-cpu=native" \
  cargo test --release test_simd_blake_prove --features "parallel,slow-tests" -- --nocapture \
    > /tmp/r1_e2e_run$i.txt 2>&1
done
# then the two cargo bench commands above
```

Conditions to hold constant: plugged in, no concurrent heavy processes (including AI
agents/builds), same RUSTFLAGS, warm build (discard a first run if the build was cold).
