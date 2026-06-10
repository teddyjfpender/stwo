# stwo-backend-metal

An Apple-GPU (Metal) prover backend for stwo, ported from the standalone
[`stwo-metal`](https://github.com/starkware-libs/stwo-metal) prototype and adapted to this
repository's backend extension points (`FrameworkBackend`, `FromSimdColumns`,
`stwo-backend-testkit`).

## Status

- **macOS / Apple Silicon only.** On other targets the sys crate compiles to a stub and every
  GPU entry point returns an initialization error.
- **Conformance-gated.** `tests/conformance.rs` runs the full
  `stwo_backend_testkit::assert_backend_conformance` suite against `CpuBackend` for both
  `Blake2sMerkleChannel` and `Blake2sM31MerkleChannel`: op-level differentials (column ops,
  poly ops, barycentric, split/join, FRI folds, accumulation, quotients, Merkle including
  pruned-equivalence, grinding, `FromSimdColumns`) plus end-to-end **proof byte-equality**.
- **No Xcode required.** `stwo-backend-metal-sys` compiles shaders ahead of time when
  `xcrun metal` is available, and otherwise embeds preprocessed shader source compiled once
  at startup by the Metal driver (one `MTLLibrary` per translation unit, so file-scope
  helpers don't collide). Command Line Tools are sufficient.

## Architecture

```
stwo-backend-metal-sys     ObjC runtime (runtime.m) + FFI wrapper (metal.rs) + .metal kernels
stwo-backend-metal         MetalBackend: Backend/BackendForChannel trait impls
  src/columns/             Unified-memory columns (BaseFieldVec, SecureFieldVec,
                           Blake2sHashVec) — GPU buffers with zero-copy host views
  src/backend/             Trait impls: poly (FFT/LDE), fri, quotient, accumulation,
                           blake2s (Merkle + grind), lookups (GKR/MLE), column ops
  src/backend/zero_copy_bridge.rs   FromSimdColumns: witness generated on SimdBackend,
                           transferred at the commitment boundary
```

Key design points carried over from the prototype:

- **Unified memory**: columns live in `MTLBuffer`s shared between CPU and GPU
  (`host_slice()` gives a zero-copy host view), so "upload/download" is mostly free on
  Apple Silicon.
- **Async submission with same-queue ordering**: `evaluate_polynomials` submits all LDE
  RFFTs without waiting and drops the completion handles; later GPU work (Merkle hashing,
  quotients) is serialized after them by Metal's FIFO queue. **Contract**: any *host-side*
  read of a possibly-in-flight buffer must first fence the queue via
  `stwo_backend_metal_sys::metal::queue_drain()` — see `materialize_leaf_columns` in
  `src/backend/blake2s.rs` for the canonical example (the M31-output Merkle hasher builds
  leaves on the host).
- **CPU constraint evaluation**: `FrameworkBackend` is implemented via
  `evaluate_constraint_quotients_via_cpu` — trace columns are converted to `CpuBackend`,
  constraints evaluated there, and the quotient accumulated back. This keeps the backend
  correct for *any* AIR without a GPU constraint compiler. (The prototype's bytecode-JIT
  constraint lane — `eval_program_v1` / shader codegen — is the natural future replacement;
  see "Not ported" below.)

## Byte-equality decisions

Two places where the GPU-optimal answer was rejected to preserve proof byte-equality with
the reference backend:

- **Grinding** (`GrindOps`) delegates to `SimdBackend`. The GPU grind kernel is kept as the
  inherent `grind_gpu::<IS_M31_OUTPUT>` but finds a *different valid nonce* (it searches the
  low 32-bit space in 2^24 batches), which would change proof bytes.
- **M31-output Merkle leaves** are hashed on the host (chunked, rayon-parallel when the
  `parallel` feature is on) rather than via the non-M31 GPU fast path, matching the
  reference hasher's exact byte stream.

## Not ported (and why)

From the `stwo-metal` prototype, the following were deliberately left behind:

- **Cairo-specific witness/interaction lanes** (`interaction_trace_*`, `handoff`,
  `commitment_slice`, `proof_slice`): coupled to stwo-cairo's component set; this repo's
  seam for that is `FromSimdColumns` + the generic `prove_cairo<B, MC>` in the stwo-cairo
  fork.
- **Bytecode-JIT constraint evaluation** (`eval_program_v1`, `recording_eval`,
  `shader_compiler_v1`): a promising design (records the AIR once, compiles a fused GPU
  kernel) but unfinished and unsound to adopt without its own conformance story. It is the
  natural path to a native-GPU `FrameworkBackend`.
- **Capability/planner/benchmark scaffolding** (`capability`, `planner`, `execution_plan`,
  `workload`, `benchmark`, `prove_runtime_v1`): runtime auto-tuning machinery, orthogonal to
  a correct backend and a large maintenance surface.
- **Dead kernels** (`barycentric`, `fri_decompose`, `merkle_decommit`): unreferenced by the
  surviving trait surface.

## API delta vs the prototype's vendored stwo

The prototype pinned an older stwo; the port adapts to this repo's traits:

- `FriOps::fold_line` takes an `alphas` slice (`alphas[i] = alpha^(2^i)`) and folds the
  whole chain; `fold_circle_into_line` returns the `LineEvaluation`.
- `QuotientOps` gained `log_blowup_factor` + twiddles with subdomain-accumulate semantics;
  the GPU kernel runs on the evaluation subdomain and the result is interpolated/extended
  with subdomain twiddles extracted from the full tree.
- `PolyOps` requires `split_at_mid`/`join_at_mid` and barycentric evaluation.
- `MerkleOpsLifted` requires `PackLeavesOps`; commitment is pruned
  (`commit_pruned`, bottom 4 layers recomputed at decommit).

## Testing

```bash
# Op-level + proof byte-equality conformance (both channels)
cargo test --release -p stwo-backend-metal

# The same suite any new backend should pass
cargo test --release -p stwo-backend-testkit
```
