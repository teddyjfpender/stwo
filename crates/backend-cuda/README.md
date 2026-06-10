# stwo-backend-cuda

A CUDA prover backend for stwo, ported from the
[`stwo-cuda`](https://github.com/starkware-libs/stwo-cuda) prototype (Nethermind-lineage
kernels, staged and hardware-validated in `crates/backend-cuda-kernels`) and adapted to
this repository's backend extension points.

## Status

- **Conformance-green on real hardware** (RTX 3090, CUDA 11.8, RunPod): the full
  `stwo-backend-testkit` suite passes for both `Blake2sMerkleChannel` and
  `Blake2sM31MerkleChannel` — op-level differentials vs `CpuBackend` plus end-to-end
  **proof byte-equality** and repeated-prove stability. Both channels passed on the
  first runtime attempt; every lesson from the Metal port (host-side M31 Merkle path,
  no pointer-keyed caches, grind delegation) was baked in up front.
- **Compile-gated**: without `nvcc` the kernels crate builds panicking stubs, this crate
  compiles everywhere, and the conformance tests skip. On a CUDA box the build script
  compiles each kernel (`-dc`), device-links (`nvcc -dlink` — required for `-rdc=true`;
  the Rust linker performs no device linking), and archives everything.

## Performance

End-to-end prove of the testkit reference AIR (16 base columns, degree-2 constraints,
`Blake2sMerkleChannel`), RunPod community RTX 3090 + 128-vCPU host, warm-best of 3
(`tests/bench_prove.rs`):

| rows | CudaBackend (RTX 3090) | SimdBackend (same host CPU) | CUDA advantage |
|---|---|---|---|
| 2^16 | 154 ms (425k rows/s) | 235 ms (279k rows/s) | 1.5× |
| 2^18 | 531 ms (494k rows/s) | 984 ms (266k rows/s) | **1.9×** |

The advantage grows with size: CUDA throughput rises (425k → 494k rows/s) while the
host's SIMD throughput is flat. This is the opposite shape from the Metal backend on
Apple Silicon (see `crates/backend-metal/README.md`), where an exceptionally strong
CPU and per-dispatch overhead leave the GPU behind — on typical x86 cloud hosts the
discrete GPU wins, and these numbers still include all the v1 host roundtrips below.

## Big-trace proving (architecture per the SIMD-vs-CUDA performance review)

Requirements applied from the NitrooZK/AntChain performance review (RTX 5090 deck,
Feb 2026) and their [`NitrooZK-stwo`](https://github.com/AntChainOpenLabs/NitrooZK-stwo)
fork:

- **Never-release memory pool**: the default CUDA mem pool's release threshold is set
  to `UINT64_MAX` at first allocation, so warm proves reuse device allocations instead
  of paying `cudaMalloc`/`cudaFree` per run (the deck reports stable warm VRAM and a
  ~280 ms cold→warm saving on their workload).
- **Low-memory big-trace mode**: their L1 roadmap item ("spill evaluations after each
  tree commit, keep coefficients, regenerate per group at decommit" — the prerequisite
  for proving sn_pie-scale traces that OOM a 32 GB card) is *exactly* this repository's
  generic low-memory machinery (`CommitmentSchemeProver::set_low_memory`): committed
  LDE evaluations are compacted to half-size coefficient columns after commit (the
  blowup-sized buffers return to the pool) and regenerated **bit-exactly** at decommit.
  It is generic over `Backend` and now validated on `CudaBackend`:
  conformance (proof byte-equality, both channels) passes with the mode ON.

Measured on an H100 80GB (224-vCPU host), reference AIR, warm-best, end-of-run pool
footprint (`STWO_BENCH_LOW_MEMORY=1` toggles the mode in the bench/testkit harness):

| rows | CUDA (lm off) | CUDA (lm on) | pool footprint off → on | SIMD (host CPU) |
|---|---|---|---|---|
| 2^18 | 622 ms | 658 ms (+6%) | 752 → 720 MB | 1219 ms |
| 2^20 | 2337 ms | 2543 ms (+9%) | 1168 → 1072 MB | 5281 ms |
| 2^22 | 9393 ms | 10602 ms (+13%) | 2800 → 2416 MB (−14%) | — |

Notes: CUDA is 2.0–2.3× the 224-vCPU SIMD run at these sizes. The VRAM saving is
modest on this 16-column AIR (quotient/FRI working set dominates); the mode targets
many-tree, many-column workloads (Cairo's preprocessed + base + interaction trees)
where committed-LDE retention is the dominant term — the deck's 51 GB interaction
trace is the motivating case. The probe reports the end-of-run pool footprint, not
the true in-flight peak; a high-water-mark probe is future work.

## stwo-cairo GPU witness pieces

- **GPU preprocessed columns** (`GenPreprocessedTrace` hook in the stwo-cairo fork):
  the Seq, RangeCheck, and BitwiseXor families generate directly on device (family
  caches keyed by family *parameters* — content, never pointers); other columns take
  the SIMD path per column, and id-parse failures degrade to SIMD, never to wrong
  values. Validated: the Cairo e2e (which pins the preprocessed root) stays
  byte-equal. `PREPROCESSED_TRACE_GPU_GENERATE=0` forces the SIMD path. Batch-NTT
  commitment interpolation and the Pedersen GPU table are the queued follow-ups.
- **Big-trace mode in prove_cairo**: `STWO_CAIRO_LOW_MEMORY=1` enables the
  compact/regenerate low-memory machinery for the whole Cairo prove — validated
  byte-equal on the all-opcode e2e (+~3% time). True sn_pie-scale (~25M steps, the
  51 GB OOM case) validation still needs the sn_pie input artifact (~130 MB, not in
  this repo).

## v1 caveats (correctness first; known perf headroom)

- **Constraint evaluation runs on the CPU** (`evaluate_constraint_quotients_via_cpu`).
  A native lane analogous to the Metal JIT shader compiler is the natural next step.
- **Per-launch `cudaDeviceSynchronize`** in most kernel wrappers — no stream pipelining.
- **Host roundtrips** kept from the port for byte-equality or simplicity: M31-output
  Merkle leaves/layers hashed on host; `TwiddleBuffer::extract_subdomain_twiddles` and
  `PolyOps::join_at_mid` download/upload; the quotient combine kernel runs on the
  evaluation subdomain with the interpolate/extend tail through `PolyOps`.
- **Grinding runs on GPU for the non-M31 channel** (ported from NitrooZK's
  `grind_blake2s.cu`): chunked `atomicMin` search returning the *lowest* valid nonce,
  nonce-equal with `SimdBackend` (testkit-gated on hardware), ~200× at production
  `pow_bits` per their measurements. The M31-output channel still delegates to
  `SimdBackend` (its PoW hash differs at finalize; NitrooZK does the same).

## Fixes over the prototype

- Three **UB transmutes** between `TwiddleTree<CudaBackend>` and `TwiddleTree<CpuBackend>`
  (their `Twiddles` types have different layouts: device-pointer struct vs `Vec`)
  replaced with a download-and-convert helper.
- The `set_len`-after-`with_capacity` device-download idiom (uninitialized `Vec`
  exposure) replaced with zero-initialized buffers.
- `MerkleOpsLifted` genericized over `IS_M31_OUTPUT` (the prototype only supported the
  non-M31 hasher); `QuotientOps`/`FriOps`/`PolyOps` adapted to the current trait surface
  (alphas-slice `fold_line`, returning `fold_circle_into_line`, subdomain quotient
  semantics, `join_at_mid`, `PackLeavesOps`).
- Dropped lanes: capability/planner registries, framework plan/overlay (bytecode
  constraint dispatch), Poseidon252 channel (kernels staged in
  `backend-cuda-kernels/cuda/`, lane unported), prototype witness generators.

## stwo-cairo

Full Cairo e2e proving benchmarks in the stwo-book format (H100 SXM, secure config,
with same-host SIMD and stwo-book CPU references) live in the stwo-cairo fork at
`gpu_benchmarks/RESULTS.md` — honest verdict: the v1 lane scales with size but does
not yet beat strong CPUs on real many-component Cairo proofs; the headroom items are
listed there.

The fork at `teddyjfpender/stwo-cairo` (branch `generic-backend`) proves real Cairo
programs on this backend: `prove_cairo::<CudaBackend, Blake2sMerkleChannel>` with the
witness generated on `SimdBackend` and transferred via `FromSimdColumns`. Gate (passes
on RTX 3090): `test_prove_verify_all_opcode_components_cuda` proves + verifies the
all-opcode program and asserts the serialized proof felts are **identical** to the
SIMD backend's proof. Note from reviewing NitrooZK's production fork: their base and
interaction traces are also CPU/SIMD-generated — their GPU witness advantage is
preprocessed-column generation and batched NTT, both incremental follow-ups here.

## JIT constraint lane (default GPU path)

Constraint kernels are **generated from this build's own AIR**: the component's
constraint tree is recorded once to bytecode (the same recording-evaluator lane as the
Metal JIT, logup included), emitted as self-contained CUDA C with an **explicit C ABI**
(no Rust struct reads — the failure mode that disqualified the precompiled kernel set
below is impossible by construction), compiled via NVRTC at first use, and cached by
the bytecode's **content semantic hash** (never pointers). Validated on RTX 3090:
conformance byte-equal with the lane engaged, and the Cairo all-opcode e2e passes with
**all 46 components on the JIT lane** and the proof byte-identical to SIMD.

- Lane order per component: precompiled kernels (opt-in) → JIT → CPU pointwise, all on
  one accumulator claim. `STWO_CUDA_DISABLE_JIT` forces CPU;
  `STWO_CUDA_CONSTRAINT_VERIFY=1` differentially checks the JIT lane per component.
- Known cost: logup components bake `claimed_sum` into the bytecode, so first proves of
  a new statement pay NVRTC compiles (~100 ms/component); parameterizing the cumsum
  shift is the queued follow-up.

## Per-component constraint kernels (opt-in)

The NitrooZK constraint lane is ported: ~250 generated per-component kernels with
FNV1a-name dispatch, a driver that derives each component's dispatch name from its type
path (no stwo-cairo edits needed), CPU fallback **on the same accumulator claim**, and a
differential-verify harness (`STWO_CUDA_CONSTRAINT_VERIFY=1`) that runs both lanes and
reports per-component mismatch fingerprints while keeping the CPU result.

**Verdict on this stack**: all 44 dispatched components mismatch on 100% of rows from
row 0 — the eval-struct-layout / AIR-revision skew fingerprint. The kernels were
generated against NitrooZK's stwo v2.1.1 and their stwo-cairo AIR, and read raw Rust
struct layouts from that build; our stack is current-upstream stwo with a regenerated
AIR. The lane is therefore **opt-in** (`STWO_CUDA_ENABLE_CONSTRAINT_KERNELS` or
`STWO_CUDA_CONSTRAINT_ALLOWLIST`) until kernels are regenerated against this AIR
revision; the verify harness is the qualification gate (0 mismatches = promotable).
The dispatch infrastructure and harness are the durable parts: regenerated kernels are
drop-in testable per component.

## Testing

```bash
# Anywhere (stub build; conformance skips without nvcc)
cargo test --release -p stwo-backend-cuda

# On a CUDA box — the decisive gate
STWO_CUDA_NVCC=/usr/local/cuda/bin/nvcc cargo test --release -p stwo-backend-cuda

# Benchmarks
BENCH_LOG_N_ROWS=18 cargo test --release -p stwo-backend-cuda --test bench_prove -- --ignored --nocapture
```
