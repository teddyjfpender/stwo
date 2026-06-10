# Performance Proposals (Supervised Tier)

Status after the approved implementation round (branch `perf-optimizations`). Measurements on
Apple M5 Max, 18 cores, 64 GB; baselines in `/tmp/stwo-baseline/`.
Original baselines: blake 2^18 = 12.5 s / 21.1 GB peak RSS; plonk 2^20 (blowup 2^4) = 1.4 s /
11.1 GB. Current: blake 2^18 = ~7.1 s / 20.0 GB; plonk 2^20 = ~1.07 s / 6.55 GB.

## P0. Low-memory mode — **IMPLEMENTED** (opt-in), with one honest caveat

`CommitmentSchemeProver::set_low_memory()`: after the FRI quotients, each owned tree's columns
are compacted to at most half their size (in-place interpolation, verified-zero upper
coefficient half dropped); decommit regenerates each column transiently and bit-exactly and
reads only the gathered rows. Proof bytes verified identical across blowup configs.

**Caveat (measured):** peak RSS for blowup-1, degree-2 AIRs (blake-shaped) is *unchanged*,
because the peak occurs at composition time, when constraint evaluation genuinely reads the
full LDE of every column. The mode reduces the FRI-through-decommit phase footprint (e.g.
plonk 2^20: 4.4 GB after OODS instead of holding ~6.5 GB to the end) at ~20–30% extra prove
time — useful when pipelining proofs or co-running provers.

**Follow-up that would move peak RSS (needs design work):** streamed/extension-on-demand
constraint evaluation — evaluate the composition subdomain-chunk by subdomain-chunk so full
LDEs of all columns never coexist; or early eval-truncation driven by a caller-provided
max-constraint-degree hint (sound for blowup > degree headroom, e.g. plonk-shaped AIRs).

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

## P2. NEON PackedM31 multiplication kernel — **IMPLEMENTED**

`vqdmulhq_s32`-based kernel (hi = (2ab)>>32 exactly; lo = low 31 bits), 5 instructions per 4
lanes with no cross-lane shuffles. Exhaustive edge-case tests over the unreduced `[0, P]`
boundary (including `P` itself) for both the plain and doubled-twiddle variants.
**Measured:** `mul_simd` −12.3%, simd iFFT 2^20–2^24 −9.3…−9.7% (single-core).

## P3. Out-of-domain barycentric evaluation on the trace subdomain — **OPEN, de-scoped**

Implementation note from this round: the bit-reversed prefix subdomain of a canonic domain is
*not itself canonic*, and both barycentric-weights implementations hardcode canonic structure
(generator initial index, the S_i conjugation argument). Generalizing them correctly to split
subdomains is exactly the kind of math-adjacent change that deserves its own focused review;
the memory half of the motivation is meanwhile served by P0 (coeffs-based OODS when
`store_polynomials_coefficients` is set). Remaining value: ~0.7 s of blake 2^18 in default
mode.

### Original analysis

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

## P4. Wire the SIMD Keccak-f[1600] primitive into Keccak256 Merkle ops — **SCOPED, NOT DONE**

On inspection, `prover/backend/simd/keccak256.rs` is an intentional correctness-first CPU
delegate (it copies all columns to the CPU backend). "Wiring" the `keccak_f1600x8` primitive
is a full SIMD implementation of the lifted leaf sponge semantics (incremental absorption
across size groups, 136-byte rate vs the 16-felt chunking) — a standalone kernel project with
its own correctness surface. No benchmarked path uses the Keccak channel today. Recommend
scheduling separately; the existing CPU↔SIMD root-equality test is the gate.

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
**Options:** (a) document required RUSTFLAGS prominently in README build section — **DONE**
(see "Performance builds" in the README); (b) add runtime feature detection + multiversioned
kernels (larger change in field-arithmetic dispatch — still open).

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
