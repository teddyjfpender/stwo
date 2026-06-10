# Performance Proposals Requiring Approval (Supervised Tier)

Proposals below touch soundness/security-critical components or change algorithmic behavior, so
per `CLAUDE.md` they need explicit human approval before implementation. Each is grounded in
measurements taken on 2026-06-09 (Apple M5 Max, 18 cores, 64 GB; baselines in
`/tmp/stwo-baseline/`). E2e baselines: blake 2^18 = 12.5 s / 21.1 GB peak RSS;
plonk 2^20 (blowup 2^4) = 1.4 s / 11.1 GB peak RSS.
After the implemented rounds 1–3: blake 2^18 = 7.1 s / 20.0 GB; plonk 2^20 = 1.07 s / 6.55 GB.

## P0. "Low-memory mode" — retain only trace-domain evaluations (largest remaining RSS lever)

Blake-shaped AIRs (blowup 1, thousands of columns) hold ~15 GB of LDE evaluations until
decommit. They are needed for (a) Merkle leaves — already consumed at commit; (b) quotient
accumulation — uses only the trace-domain bit-reversed prefix; (c) OODS evaluation — could use
the subdomain (P3); (d) decommit queried values + pruned-layer recompute — the only consumer
that genuinely needs LDE rows. A `low_memory` option on `CommitmentSchemeProver` could retain
only the subdomain prefix (halving eval memory at blowup 1, ÷16 at blowup 2^4) and re-extend
columns (IFFT+FFT per column, parallel) at decommit. Trades ~1–3 s prove time for −7 GB+ on
blake-shaped workloads. Needs design sign-off: touches the commitment scheme prover flow.

## P1. Stop retaining bottom Merkle hash layers until decommit — **IMPLEMENTED**

Implemented as `MerkleProverLifted::commit_pruned` (opt-in; used by `CommitmentTreeProver` for
the four commitment trees; the packed FRI path and the default `commit()` are unchanged). The
bottom 4 layers are dropped eagerly during commit and recomputed per-query at decommit via a
memoized generic node resolver. Verified: byte-identical decommitments
(`test_pruned_commit_decommit_equivalence`) + full prove/verify matrix.
**Measured:** plonk 2^20 (blowup 2^4) peak RSS 11.1 GB → 6.55 GB (−41%); blake 2^18 −1 GB.
**Note for review:** the pub `layers` field of trees built with `commit_pruned` no longer
contains the bottom 4 layers; in-repo consumers were migrated to the new `log_size()` accessor.
**Possible follow-up:** deeper pruning (k>4) is a pure constant change
(`N_UNRETAINED_BOTTOM_LAYERS`); recompute cost grows 2^k per query.

## P2. NEON PackedM31 multiplication kernel

**Where:** `crates/stwo/src/prover/backend/simd/m31.rs` (`mul_neon`, `mul_doubled_neon`) — field
arithmetic, approval required by the operations boundary even though verifier behavior is
untouched.
**Problem:** The current kernel decomposes u32x16 into 8× 2-lane `vqdmull_s32` + 8 deinterleaves
+ 4 shifts. A plonky3-style kernel via `vqdmulhq_s32`/`vmulq_u32` on 4-lane registers needs
roughly half the instructions. M31 multiplication dominates FFT butterflies (iFFT 2^24 = 55.5 ms
single-core) and quotient accumulation (309 ms for 2^21×100 cols single-core).
**Invariant preserved:** For all a, b ∈ [0, P]: result ≡ a·b (mod P) and lies in [0, P].
**Verification:** Exhaustive edge-case + randomized differential tests vs scalar `M31` mul
(extend `multiplication_works` with 0, 1, P−1, P boundary values), plus the full proof test
suite.
**Expected:** 15–30% on FFT-heavy stages on aarch64.

## P3. Out-of-domain barycentric evaluation on the trace subdomain

**Where:** `prover/pcs/mod.rs::prove_values`, `barycentric_weights` callers.
**Problem:** OODS evaluation builds barycentric weight columns sized to the *full LDE domain*
(16 B/point) per distinct (log_size, point) pair, and scans full LDE evals. The polynomial is
fully determined by its trace-subdomain values, which are a bit-reversed prefix of the committed
evals (the quotient accumulation already exploits exactly this).
**Proposal:** Compute weights and the barycentric sum over the subdomain prefix only.
**Invariant preserved:** Identical sampled values (same polynomial, exact field arithmetic;
different summation set, mathematically equal).
**Verification:** New test asserting full-domain vs subdomain evaluation equality across sizes,
plus existing CPU/SIMD barycentric consistency tests.
**Expected:** OODS stage (720 ms in blake 2^18) shrinks ~2× at blowup 1, much more at higher
blowups; weights memory shrinks by the blowup factor. (Round-1 change already drops the weights
map right after use; this proposal shrinks it while alive.)

## P4. Wire the SIMD Keccak-f[1600] primitive into Keccak256 Merkle ops

**Where:** `prover/backend/simd/keccak256.rs` (+ `keccak256_permutation.rs` from PR #1398).
**Problem:** The SIMD-parallel Keccak permutation was merged but Keccak256 Merkle commitment
still hashes scalar-per-node with per-node allocations.
**Invariant preserved:** Identical hashes/roots (same function, vectorized).
**Verification:** Existing CPU↔SIMD root-equality tests for the Keccak merkle hasher.
**Expected:** Large speedup for Keccak-channel users; no effect on Blake2s channels.

## P5. FRI inner-layer leaf packing (`fold_step`)

**Where:** FRI prover/verifier + config — protocol-shape adjacent; needs design review, not just
approval. Today each FRI layer commits one QM31 per leaf; packing (fold_step > 1) reduces FRI
tree count/sizes and decommitment cost but changes proof structure and verifier logic.
**Recommendation:** Discuss before any code.

## P6. x86_64 builds silently miss AVX2/AVX-512 kernels

**Where:** build/docs. The SIMD dispatch is compile-time (`cfg(target_feature)`); a plain
`cargo build --release` on x86_64 (no `-C target-cpu=native` / `target-feature`) falls back to
the portable kernel for M31 multiplication and the FFT — a large silent regression for library
consumers on x86.
**Options:** (a) document required RUSTFLAGS prominently in README build section (autonomous,
low effort); (b) add runtime feature detection + multiversioned kernels (larger change in
field-arithmetic dispatch — needs approval).

## Deprioritized (autonomous but not in current hot paths)

- GKR/sumcheck/MLE backend has zero rayon coverage (`prover/lookups/`): real gap, but the
  examples' logup path doesn't route through GKR, so no e2e effect today.
- Poseidon example `gen_trace` is single-threaded (same row-disjoint shape as the blake
  generators parallelized in this round); the e2e test is currently `#[ignore]`d so there is no
  measurable benefit until it is re-enabled.

## Pre-existing footgun worth an upstream fix (not a perf item)

`VeryPackedSecureColumnByCoords::transform_under_mut` produces a view whose inner `Vec` lengths
are still counted in `PackedBaseField` units (4x the real `VeryPacked` element count). Its
`chunks_mut`/`par_chunks_mut` therefore yield slices that extend past the owned allocation; the
constraint-framework consumer compensated by zipping against a correctly-sized range. The
consumer now documents and bounds this explicitly, but the type itself remains easy to misuse.
