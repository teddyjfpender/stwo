#include "cuda_mem_pool.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>

namespace {

// Resolve the device's default memory pool once and set the never-release
// threshold: warm proves reuse pooled allocations instead of paying
// cudaMalloc/cudaFree on every run (threshold = UINT64_MAX, as in the NitrooZK
// setup). Returns nullptr when stream-ordered allocation is unavailable.
cudaMemPool_t resolve_default_pool() {
    int device_id;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
        return nullptr;
    }
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, device_id) != cudaSuccess || pool == nullptr) {
        return nullptr;
    }
    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    return pool;
}

}  // namespace

cudaMemPool_t stwo_default_mem_pool() {
    static std::once_flag once;
    static cudaMemPool_t pool = nullptr;
    std::call_once(once, [] { pool = resolve_default_pool(); });
    return pool;
}

namespace {

cudaError_t default_pool_current(
    cudaMemPool_t pool,
    size_t* used_current,
    size_t* reserved_current
) {
    if (used_current == nullptr || reserved_current == nullptr) {
        return cudaErrorInvalidValue;
    }
    *used_current = 0;
    *reserved_current = 0;
    if (pool == nullptr) {
        return cudaErrorNotSupported;
    }

    uint64_t used = 0;
    uint64_t reserved = 0;
    cudaError_t err = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
    if (err != cudaSuccess) {
        return err;
    }
    *used_current = static_cast<size_t>(used);
    *reserved_current = static_cast<size_t>(reserved);
    return cudaSuccess;
}

}  // namespace

extern "C" cudaError_t cuda_mem_pool_init() {
    return stwo_default_mem_pool() != nullptr ? cudaSuccess : cudaErrorNotSupported;
}

extern "C" cudaError_t cuda_default_pool_current(
    size_t* used_current,
    size_t* reserved_current
) {
    if (used_current == nullptr || reserved_current == nullptr) {
        return cudaErrorInvalidValue;
    }
    return default_pool_current(stwo_default_mem_pool(), used_current, reserved_current);
}

extern "C" cudaError_t cuda_default_pool_trim(
    size_t min_bytes_to_keep,
    size_t* used_current,
    size_t* reserved_current
) {
    if (used_current == nullptr || reserved_current == nullptr) {
        return cudaErrorInvalidValue;
    }
    *used_current = 0;
    *reserved_current = 0;
    cudaMemPool_t pool = stwo_default_mem_pool();
    if (pool == nullptr) {
        return cudaErrorNotSupported;
    }
    cudaError_t err = cudaStreamSynchronize(0);
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaMemPoolTrimTo(pool, min_bytes_to_keep);
    if (err != cudaSuccess) {
        return err;
    }
    return default_pool_current(pool, used_current, reserved_current);
}

extern "C" cudaError_t cuda_mem_pool_destroy() {
    return cudaSuccess;
}

namespace {
struct StreamPool {
    cudaStream_t streams[STWO_N_POOL_STREAMS];
    cudaEvent_t events[STWO_N_POOL_STREAMS];
};
StreamPool &stream_pool() {
    static StreamPool pool = [] {
        StreamPool p{};
        for (int i = 0; i < STWO_N_POOL_STREAMS; ++i) {
            cudaStreamCreateWithFlags(&p.streams[i], cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&p.events[i], cudaEventDisableTiming);
        }
        return p;
    }();
    return pool;
}
}  // namespace

cudaStream_t stwo_pool_stream(int i) { return stream_pool().streams[i % STWO_N_POOL_STREAMS]; }

void stwo_stream_wait_legacy(int i) {
    StreamPool &p = stream_pool();
    int k = i % STWO_N_POOL_STREAMS;
    // Record the legacy stream's current frontier and make stream k wait on it.
    cudaEventRecord(p.events[k], (cudaStream_t)0);
    cudaStreamWaitEvent(p.streams[k], p.events[k], 0);
}

void stwo_legacy_wait_stream(int i) {
    StreamPool &p = stream_pool();
    int k = i % STWO_N_POOL_STREAMS;
    cudaEventRecord(p.events[k], p.streams[k]);
    cudaStreamWaitEvent((cudaStream_t)0, p.events[k], 0);
}

// ---------------------------------------------------------------------------
// Stage B′ fan-out primitives (STWO_CUDA_STREAM_FANOUT). The per-slot bridges
// above share ONE event per slot: if two host threads bridge the same slot
// concurrently (which the multi-threaded witness fan-out does — rayon workers
// each driving a lane), their `cudaEventRecord` calls race on the same event
// object (undefined behaviour) and silently corrupt the ordering edge. These
// use a FRESH event per call, so concurrent forks/joins from different workers
// never touch a shared event. The event is created timing-disabled and
// destroyed right after the wait is enqueued — CUDA defers the real teardown
// until no pending work references it, so this is safe and cheap (~tens of
// create/destroy pairs per prove).
// ---------------------------------------------------------------------------

// Pool stream `i` (round-robin) as an opaque handle for the Rust stream scope.
extern "C" void *stwo_fanout_stream(int i) {
    return (void *)stream_pool().streams[i % STWO_N_POOL_STREAMS];
}

// FORK: `stream` waits for everything enqueued on the legacy stream so far — its
// inputs (pool allocations stay legacy-ordered, plus any shared read-only tables
// uploaded on legacy). MUST be called before launching the lane's kernel on it.
extern "C" void stwo_fanout_fork(void *stream) {
    cudaEvent_t ev;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventRecord(ev, (cudaStream_t)0);
    cudaStreamWaitEvent((cudaStream_t)stream, ev, 0);
    cudaEventDestroy(ev);
}

// JOIN: the legacy stream waits for `stream`'s work. MUST be called before any
// legacy-stream or host consumer reads the lane's outputs (the closing bridge).
extern "C" void stwo_fanout_join(void *stream) {
    cudaEvent_t ev;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventRecord(ev, (cudaStream_t)stream);
    cudaStreamWaitEvent((cudaStream_t)0, ev, 0);
    cudaEventDestroy(ev);
}

extern "C" uint32_t* cuda_mem_pool_allocate_uint32(size_t count) {
    return cuda_mem_pool_allocate<uint32_t>(count);
}

extern "C" uint32_t* cuda_mem_pool_allocate_zeroes_uint32(size_t count) {
    return cuda_mem_pool_allocate_zeroes<uint32_t>(count);
}

extern "C" void cuda_mem_pool_free_uint32(uint32_t* ptr) {
    cuda_mem_pool_free(ptr);
}
