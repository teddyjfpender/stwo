# Stwo GPU lab

Contributors should start with [`CONTRIBUTING.md`](CONTRIBUTING.md). Its correctness, taste,
file-size, evidence, and feedback-loop rules are part of the lab contract.

This is the replacement backend's fast CUDA development loop. It deliberately
does not build Cargo or run a proof. The first vertical slice loads one existing
self-contained generated witness cubin through the CUDA Driver API, checks every
output word against an independent host fixture, checks eager/graph identity,
and then measures it with CUDA events.

The checked-in tiny case is synthetic and is **not** a proof-system
qualification fixture or a performance headline. It proves that the module,
ABI, pointer tables, launch, reset, validation, graph, and timing seams work.

## Production-shaped indexed replays

Real prover-input slices use `stwo.gpu-lab.semantic-fixture-index.v2`. The JSON
index stays small while content-addressed M31 chunks hold the resident memory
table, row inputs, and independently generated expected words. Preparation
fails closed on the exact producer shape, full source-ProverInput identity,
semantic identity, host-source identity, contiguous paired chunk coverage,
file sizes and hashes, and every canonical M31 word.

`prepare` pre-sizes the existing replay-v2 image and transposes at most 4,096
rows at a time directly from row-major chunks into final column-major offsets.
It also writes the independently expected reverse-row proof case without
loading the address table or a complete trace as Python integers. Result
validation reconstructs that canonical replay from the sealed chunks and
compares its bytes; it never accepts the replay merely because a manifest names
its hash.

The `--oracle-index` argument for an indexed fixture is a reviewed
`host-oracle-index.v1` seal wrapper, not the raw v2 oracle artifact. The wrapper
binds the fixture-index hash, raw oracle hash, and complete exporter plus
transitive semantic-source closure. Production preparation remains disabled
until that wrapper hash is reviewed in both the Python and C++ allowlists.

The quick-loop integration hook is intentionally narrow: its case resolver must
pass the selected `fixture-index.json` and sealed wrapper paths to the existing
`prepare` and `validate-result` arguments. It must not pass the raw oracle
artifact, copy chunks outside their content-addressed tree, or add a second
replay converter. Pod-local staging owns path placement and durable persistence.

## Local, GPU-free checks

```bash
python3 gpu-lab/tools/lab.py self-test
gpu-lab/tools/cpu-self-test
python3 gpu-lab/tools/lab.py validate-fixture \
  gpu-lab/cases/tiny/witness_pedersen_builtin.semantic.json
```

`cpu-self-test` compiles and runs the C++ SHA/byte parser, exact plan binding,
effect guards, and benchmark-budget arithmetic without a CUDA toolkit or GPU.

## Consumer-GPU CUDA loop

On an admitted `sm_86`, `sm_89`, or `sm_120` development machine:

```bash
gpu-lab/tools/quick-loop witness_pedersen_builtin tiny sm_89
```

Build and deploy the pinned CUDA 12.8 consumer-development image by immutable digest as documented
in [`docker/README.md`](docker/README.md). It admits RTX 3090 (`sm_86`), RTX 4090 (`sm_89`), and
RTX 5090 (`sm_120`), checks Nsight Systems 2026.1.3 exactly, and runs a compiled kernel on the
selected card before GPU-lab work. Performance-counter permission remains a separate admission
gate.

The command configures Ninja once, builds only one cubin, seals it under its
content hash, prepares a replay image from the independently checked semantic
fixture, runs eager and graph correctness, and emits a short benchmark. On a
`labctl` pod it first copies the fixture and every indexed payload chunk into the
declared pod-local root, verifies every copy, and rejects a build, module, replay,
plan, execution manifest, or harness that resolves under `/workspace`. It also
captures `nvidia-smi` state before and after the run, checks for a matching
same-device baseline, and writes an auditable `loop.json`. Set
`GPU_LAB_DEVICE` when the target is not device zero. `labctl` owns
`GPU_LAB_LOCAL_ROOT`; do not point `GPU_LAB_BUILD_ROOT` at the network volume.

Generated cubins, replay images, execution manifests, and results remain under
the local build directory while the lease is active. `loop.json` embeds the
staging source/destination hashes. `labctl accept` copies an authenticated
content-addressed snapshot of records, results, and profiles to `/workspace`;
`labctl close` freezes that local tree, seals it, verifies persistence, and only
then permits compute termination. A missing or identity-incompatible baseline is reported in
`baseline-comparison.json`; it does not turn a correct run red. A comparison is
made only when device UUID/SM, driver, stable MIG/ECC/persistence/power-limit
configuration, fixture, toolchain flags, ABI, and shape match,
and it records both cubin hashes plus eager/graph latency and semantic
rows/words per second.

An exactly comparable run writes `objective-score.json` only after correctness, statistical,
identity, non-regression, bounded-resource, and ordinary-loop-SLO gates pass. The score is an
auditable lexicographic vector; it has no opaque scalar and never treats unavailable counters as
zero.

If core eager/graph and mutated-input correctness pass but the benchmark cannot
meet its time/sample contract, the harness writes a validated diagnostic result
instead of deleting evidence. `quick-loop` still captures the after-environment,
emits a `non_admissible` comparison with no metrics, and writes `loop.json`; its
final performance gate then exits nonzero. Such a result is useful diagnostic
evidence but can never be promoted as a baseline.

Profiling and sanitizer commands re-verify `staging.json` before launching. A
sealed, mutated, symlinked, network-volume, or incompletely copied artifact fails
before candidate execution.

## Explicit baseline promotion

Baselines are immutable evidence envelopes, not aliases to live build files.
Promotion revalidates immutable top-level snapshots, checks the complete
transitive identity before and after validation, and requires stable warmup and
at least 30 non-exploratory samples in both eager and graph modes. It also
requires complete pre/post power, temperature, clock, MIG, ECC, and PCI
evidence. A device that implements neither MIG nor ECC records those capabilities
as `not_supported`; a missing/query-error reading still blocks promotion with its
reason but does not block correctness work.

After reviewing a run, request its confirmation token (paths below assume the
default `sm_86` build directory):

```bash
build=build/gpu-lab-sm86
run="$build/runs/witness_pedersen_builtin-tiny"
gpu-lab/tools/accept-baseline \
  --result "$run/benchmark.json" \
  --fixture gpu-lab/cases/tiny/witness_pedersen_builtin.semantic.json \
  --oracle-index gpu-lab/cases/tiny/witness_pedersen_builtin.oracle-index.json \
  --module-index "$build/modules/witness_pedersen_builtin.sm86.module.json" \
  --execution "$run/execution.json" --replay "$run/fixture.replay" \
  --plan "$run/execution.plan" \
  --harness "$build/stwo-gpu-lab" \
  --correctness-result "$run/correctness.json" \
  --build-commands "$run/build.commands" \
  --environment-before "$run/environment.before.json" \
  --environment-after "$run/environment.after.json" \
  --loop-record "$run/loop.json" --comparison "$run/baseline-comparison.json"
```

The first invocation changes nothing and prints a token bound to the candidate,
destination, and prior baseline. Re-run the same command with
`--confirm ACCEPT-…` to promote it.
