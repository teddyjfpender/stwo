# Consumer-GPU development image

This image is the pinned `linux/amd64` CUDA 12.8 development environment for the RTX 3090
(`sm_86`), RTX 4090 (`sm_89`), and RTX 5090 (`sm_120`) GPU-lab loop. CUDA 12.8 is intentional:
it supports Blackwell. The qualified RunPod 570.195 driver uses CUDA 12.x minor-version
compatibility; deployment acceptance compiles and executes a kernel on the exact card before the
pod is admitted to the development pool.

Build and publish from the `stwo` repository root:

```bash
IMAGE_TAG=registry.example/stwo-consumer-dev:cuda12.8.2-20260716.2
docker buildx build \
  --platform linux/amd64 \
  --file gpu-lab/docker/Dockerfile.consumer-dev \
  --tag "$IMAGE_TAG" \
  --provenance=mode=max \
  --sbom=true \
  --push \
  .
docker buildx imagetools inspect "$IMAGE_TAG"
```

The production build runner must be externally recorded as native `x86_64`. Target-platform
metadata alone cannot distinguish a native runner from emulation. The build acceptance gate
executes both pinned Rust compilers, and the published image must retain the requested provenance
and SBOM attestations.

Record the resulting `sha256:` manifest digest. RunPod deployments must use
`registry.example/stwo-consumer-dev@sha256:<digest>`, never the mutable tag. Mount the persistent
volume at `/workspace` and set `PUBLIC_KEY` to the SSH public key; the default process is `sshd`
on port 22. The same validated key admits the unprivileged `dev` account and root's forced-command
automation lane. Root has no login or PTY shell and no forwarding; use `labctl shell`, which
connects as `dev`. Published source generations and compiler caches then survive pod replacement
under `/workspace`.
The entrypoint performs the legacy cache/target ownership migration once, seals a root-owned
`DEV_LAYOUT_V1` marker, and thereafter checks only the fixed top-level directories; normal pod
startup is constant in tree size rather than recursively walking a persistent Cargo target.

Source generations are published by root and remain root-owned and read-only. Every real build,
benchmark, sanitizer, and profile must run as `dev`; start the session with the fail-closed identity
gate below. The writable Cargo target and compiler caches are isolated under `/workspace/dev` and
`/workspace/.cache`, so no build needs write access to a published source generation:

```bash
stwo-gpu-dev-accept --require-dev
test "$(id -un)" = dev
printf 'target=%s ccache=%s sccache=%s\n' \
  "$CARGO_TARGET_DIR" "$CCACHE_DIR" "$SCCACHE_DIR"
```

For the replay harness, `labctl` exposes only the exact active local-NVMe root to the `dev` group.
The root itself is traverse-only (`0710`), `records/` stays root-only, and only `build/` and
`fixtures/` are owned by `dev` (`0700`). `LOCAL_ROOT.json` and `DEV_LAYOUT.json` bind those
identities and modes; use `$GPU_LAB_LOCAL_ROOT/build` for run output. The persistent evidence
worker remains root and can read, verify, and seal those files without making the control plane
dev-writable.

Do not target root manually. Root SSH disables login and PTY shells and forwarding, but its
forced command deliberately accepts the arbitrary non-interactive command strings needed by
`labctl`; it is not a command allowlist. The trusted boundary is therefore the local controller and
the holder of `PUBLIC_KEY`. The tiny CUDA counter launched through the `dev` SSH lane by
`labctl accept --profile` is an infrastructure admission probe, not a workload profile or benchmark;
root only binds its exact receipt into the acceptance record.

The build runs the host-only acceptance gate. Re-run it after deployment, then include the GPU
smoke test for the exact card:

```bash
stwo-gpu-dev-accept --require-dev
stwo-gpu-dev-accept --require-dev --gpu --sm 89   # 3090=86, 4090=89, 5090=120
```

The GPU gate compiles one tiny fat binary for all three registered SMs, launches it, and rejects
any other architecture. Profiling-counter permission remains a separate provider acceptance gate.
The recurring container healthcheck uses only a bounded ownership/identity/sshd probe; it never
invokes a compiler, profiler, sanitizer, or GPU, avoiding heavyweight timing interference.

Before publishing a changed entrypoint, run the pinned Ubuntu restart regression. It proves the
one-time ownership migration does not recur and rejects a hostile dev-owned `.ssh` symlink without
modifying its root-owned target:

```bash
bash gpu-lab/docker/test-consumer-dev-entrypoint
```
