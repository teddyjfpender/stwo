# Contributing to GPU lab

GPU lab exists to make proof-backend work exact, fast to iterate on, and pleasant to review. A
change is good only when it improves the system without weakening its semantic identity, evidence,
or feedback loop. “It works” is necessary; it is not the taste bar. This is the authoritative
contribution contract for both `stwo/gpu-lab` and `stwo-cairo/gpu_benchmarks/lab`; the latter owns
orchestration and independent-oracle production but must not fork these rules.

Read the replacement architecture before changing a boundary or adding machinery:

- [`GPU-PROVER-BACKEND-REPLACEMENT-2026-07-13.md`](../../evidence/gpu-prover-backend-redesign-2026-07-13/GPU-PROVER-BACKEND-REPLACEMENT-2026-07-13.md), especially §3–§9;
- [`GPU-PROVER-EXECUTION-CHECKLIST-2026-07-13.md`](../../evidence/gpu-prover-backend-redesign-2026-07-13/GPU-PROVER-EXECUTION-CHECKLIST-2026-07-13.md) for the current gates and metrics; and
- this repository's `AGENTS.md` and `CLAUDE.md` for soundness and review policy.

## Taste canons

1. **Make the correct path the shortest path.** One command should build one changed cubin, run
   independent correctness, and return a trustworthy number. Never make contributors memorize a
   release ritual.
2. **Expose concepts, not plumbing.** Files and APIs should read like the architecture:
   `semantic_fixture`, `build_recipe`, `program_image`, `graph_runner`, `validation`, `metrics`.
   Avoid “manager”, “helper”, and grab-bag `utils` modules.
3. **Prefer deletion to accommodation.** This is a replacement backend. Do not add compatibility
   layers that preserve the architecture being removed. Promote one qualified path and delete the
   displaced path.
4. **Keep identity explicit.** Semantic fixture, source, generator, ABI, toolchain, cubin, execution
   plan, device, and result identities must be machine-checkable. Timestamps and filenames are not
   identities.
5. **Fail closed and explain why.** Missing oracle coverage, a stale hash, an unknown field, a
   changed toolchain, or incomplete timing evidence blocks acceptance with a precise diagnostic.
6. **Optimize ownership before instructions.** Remove passes, copies, retained representations,
   allocations, and launch gaps before polishing an isolated instruction sequence.
7. **Measure the thing we claim.** Cheap GPUs establish correctness and same-device regressions.
   Only the exact qualified H100 establishes the headline or physical wall.
8. **Leave the code easier to understand.** A performance win that obscures an invariant needs a
   better shape, name, or proof—not a larger explanatory essay around tangled code.

## Size and progressive disclosure

- Aim for **≤500 logical lines per handwritten source file**.
- **500–750 lines requires a review note** explaining why one cohesive concept is clearer than a
  split.
- The source-shape gate recognizes that note only as
  `gpu-lab-cohesion-review: <reason>` in the file or in a sibling `<file>.cohesion.md`; the reason
  must be non-empty and specific.
- **More than 750 handwritten lines blocks review.** Split by durable responsibility first, not by
  arbitrary line ranges.
- Generated CUDA, sealed fixture data, and schemas may exceed the limit when splitting would make
  identity or generation worse; label the exception and keep the generator small.
- Entry points should be short narratives. Put parsing, identity, replay I/O, CUDA lifecycle,
  validation, and metrics in named modules beneath them.
- Reveal detail in layers: public contract → invariant-bearing type → implementation. A reviewer
  should not need to understand CUDA events to inspect fixture semantics.
- Comments explain **why, invariants, units, ownership, and hardware constraints**. Do not narrate
  syntax or preserve dead history in comments.

The original Stage-0 `harness/main.cpp` and `tools/lab.py` bootstrap monoliths have been split along
these boundaries. Keep their entry points small and put new behavior in the existing owning module;
do not recombine the dependency graph behind a convenient facade.

## Correctness discipline

- The candidate GPU never generates or refreshes its own golden output.
- Accepted values come from the canonical host/SIMD implementation or another genuinely
  independent evaluator. Kernel-only robustness cases must be labeled separately from proof
  semantics.
- Validate every declared output/effect range, reset writable state outside timing, and compare
  eager and graph execution.
- Keep field values canonical and include adversarial carry, boundary, alias, reset, and shape
  cases. A production-invalid input cannot be presented as independent proof evidence.
- Bind the complete Driver ABI: parameter offsets and sizes, pointer depth, effects, launch shape,
  module globals, supported shapes, and graph policy.
- Arithmetic, transcript, channel, constraint, or proof-format changes require the repository's
  supervised soundness review. Infrastructure must not quietly redefine those semantics.
- Prove the local gate can reject a deliberately wrong candidate before spending GPU time. Run
  schema, identity, oracle, source-shape, and host-reference checks first; CUDA is confirmation of
  a locally justified change, not a substitute for reasoning or review.
- A correctness failure suppresses performance acceptance. A soft benchmark failure may preserve
  diagnostics and the loop/environment record, but it must expose no comparison metrics, fail the
  final performance gate, and remain impossible to promote.

## GPU engineering standards

- Count HBM passes and bytes. State which representation owns each buffer and name its final
  consumer before adding another arena or copy.
- Coalesce global access, make alignment/stride explicit, and use shared memory only when reuse
  repays its synchronization and occupancy cost.
- Occupancy is a constraint, not the objective. Record registers, local memory/spills, shared
  memory, active warps, achieved bandwidth, and limiting dependency before trading one for another.
- Specialize on registered shapes when it removes branches, passes, or generic indexing. Keep an
  unseen same-class fixture to prevent benchmark specialization.
- Time GPU work with CUDA events in the owning stream. Record CPU submission and wall time
  separately. Exclude fixture I/O, allocation, reset, capture, and compilation unless the metric
  explicitly names them.
- Compare eager and graph modes with identical bindings and bytes. Destroy graph executables before
  unloading their modules.
- Use Nsight Systems for topology first, then selected Nsight Compute counters. Do not use a full
  counter replay as the ordinary edit loop or infer a roofline from timing alone.
- Freeze clocks/configuration where permitted and always record GPU SKU, SM, MIG/ECC, power limit,
  driver, toolkit, compiler flags, module hash, and fixture hash.
- Never transfer a cheap-GPU speedup ratio to H100. Recompile for one exact architecture per loop.

## The fast loop is a product requirement

Normal development must perform zero Cargo release builds and zero full proofs:

```bash
python3 gpu-lab/tools/lab.py self-test
gpu-lab/tools/cpu-self-test
gpu-lab/tools/quick-loop witness_pedersen_builtin tiny sm_86
```

Keep these feedback budgets green:

| Path | Budget |
|---|---:|
| No-op build | ≤1 s |
| Ordinary one-cubin build | ≤30 s p50 / ≤45 s p95 |
| Tiny correctness | ≤5 s |
| Representative correctness + short benchmark | ≤45 s |
| Ordinary edit → validated result | ≤2 min p95 |
| Selected Systems profile | ≤3 min |
| Selected Compute counter pass | ≤10 min |
| Full proof/Cargo build in daily loop | **0** |

If a change regresses a budget, fix the dependency graph or workflow before adding more caching.
Persistent caches must be content-addressed; mtime reuse is never an acceptance mechanism.
On a remote pod, source checkouts may live on `/workspace`, but active fixtures/chunks, cubins,
plans, replays, execution manifests, harnesses, results, and profiles must resolve beneath the
`labctl`-declared local root. Never bypass `staging.json` or benchmark directly from a network
volume. Acceptance and close must persist and verify the local outputs before termination.

## Change and review workflow

1. State the semantic boundary, registered shape, ownership change, expected metric, and smallest
   falsifiable vertical slice before coding. Backend progress—not harness surface area—is the unit
   of delivery.
2. Add or identify an independent oracle and a mutation that proves the gate can fail.
3. Change one cohesive module or slab. Compile one translation unit, one architecture, and one
   cubin. Do not rebuild the full prover to answer a kernel-level question.
4. Run CPU/schema/identity tests and request an adversarial review of arithmetic, ownership,
   aliasing, lifetime, transcript, and claim boundaries. Resolve soundness findings before any GPU
   run. Then run the tiny cheap-CUDA correctness case; run sanitizers when memory, aliasing,
   synchronization, or launch geometry changes.
5. Benchmark repeated samples; retain raw data and p5/p50/p95. Compare only matching device UUID/SM,
   driver, stable MIG/ECC/persistence/power-limit configuration, fixture, toolchain/flags, ABI, and
   shape; record the baseline and candidate module hashes.
   A promotable baseline additionally requires stable warmup, at least 30 samples in both eager
   and graph modes, non-exploratory p95s, and complete pre/post `nvidia-smi` environment evidence.
   A missing baseline is information, not a correctness failure.
6. Profile only to answer a named question. Record the hypothesis, counter/timeline evidence, and
   conclusion—including negative results.
7. Update the one-page checklist with one before→after metric and an evidence hash. Do not check a
   stage until its exit gate passes.
8. Run a consolidated full proof only after a meaningful, locally sealed batch; use the exact
   accepted cubins and manifests.

Ship that sequence in small, reviewable commits. Each commit should express one invariant,
ownership change, slab, or evidence update; include its narrow local gate; and leave both worktrees
free of generated products. Do not mix refactors, generated output, benchmark evidence, and a
semantic change in one commit. Commit after every locally green boundary so another contributor can
review, bisect, or continue without reconstructing an uncommitted design.

Every performance commit records a quantitative before→after delta in the units the architecture
claims to improve: passes, launches, bytes, live bytes, allocation count, source generations,
registers/spills, latency distribution, or semantic throughput. Modelled deltas must be labelled
modelled, hardware measurements must bind the complete identity tuple, and neither may be silently
promoted into the other.

Every review should be able to answer:

- What independent evidence would catch this being wrong?
- What became simpler or was deleted?
- Which file/API owns each new lifetime and effect?
- What is the before→after result, sample count, and identity tuple?
- Did the ordinary feedback loop stay within budget?
- Is any claim stronger than the hardware and counters actually support?
