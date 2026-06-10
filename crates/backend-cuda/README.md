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

## v1 caveats (correctness first; known perf headroom)

- **Constraint evaluation runs on the CPU** (`evaluate_constraint_quotients_via_cpu`).
  A native lane analogous to the Metal JIT shader compiler is the natural next step.
- **Per-launch `cudaDeviceSynchronize`** in most kernel wrappers — no stream pipelining.
- **Host roundtrips** kept from the port for byte-equality or simplicity: M31-output
  Merkle leaves/layers hashed on host; `TwiddleBuffer::extract_subdomain_twiddles` and
  `PolyOps::join_at_mid` download/upload; the quotient combine kernel runs on the
  evaluation subdomain with the interpolate/extend tail through `PolyOps`.
- **Grinding delegates to `SimdBackend`** on both channels (byte-equality: a GPU grind
  finding a different valid nonce changes proof bytes).

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

## Testing

```bash
# Anywhere (stub build; conformance skips without nvcc)
cargo test --release -p stwo-backend-cuda

# On a CUDA box — the decisive gate
STWO_CUDA_NVCC=/usr/local/cuda/bin/nvcc cargo test --release -p stwo-backend-cuda

# Benchmarks
BENCH_LOG_N_ROWS=18 cargo test --release -p stwo-backend-cuda --test bench_prove -- --ignored --nocapture
```
