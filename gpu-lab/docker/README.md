# Consumer-GPU development image

This image is the pinned `linux/amd64` CUDA 12.8 development environment for the RTX 3090
(`sm_86`), RTX 4090 (`sm_89`), and RTX 5090 (`sm_120`) GPU-lab loop. CUDA 12.8 is intentional:
it supports Blackwell. The qualified RunPod 570.195 driver uses CUDA 12.x minor-version
compatibility; deployment acceptance compiles and executes a kernel on the exact card before the
pod is admitted to the development pool.

Build and publish from the `stwo` repository root:

```bash
IMAGE_TAG=registry.example/stwo-consumer-dev:cuda12.8.2-20260716
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
on port 22. The checkout and compiler caches then survive pod replacement under `/workspace`.

The build runs the host-only acceptance gate. Re-run it after deployment, then include the GPU
smoke test for the exact card:

```bash
stwo-gpu-dev-accept
stwo-gpu-dev-accept --gpu --sm 89   # 3090=86, 4090=89, 5090=120
```

The GPU gate compiles one tiny fat binary for all three registered SMs, launches it, and rejects
any other architecture. Profiling-counter permission remains a separate provider acceptance gate.
