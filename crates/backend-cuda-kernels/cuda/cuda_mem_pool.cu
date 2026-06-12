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

extern "C" cudaError_t cuda_mem_pool_init() {
    return stwo_default_mem_pool() != nullptr ? cudaSuccess : cudaErrorNotSupported;
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

namespace {
// The bridge events are shared per pool stream and cudaEventRecord OVERWRITES an
// event's captured frontier. Two threads interleaving record/wait pairs on the
// same event can therefore under-synchronize (a wait observing the other
// thread's recording) or build a legacy<->pool-stream wait cycle — the round-8
// deadlock class (P1 moved logup finalize onto a rayon worker, making
// concurrent pool use reachable). Every record+wait pair must hold this lock so
// the pair is atomic with respect to other bridge users.
std::mutex &bridge_mutex() {
    static std::mutex m;
    return m;
}
}  // namespace

void stwo_stream_wait_legacy(int i) {
    StreamPool &p = stream_pool();
    int k = i % STWO_N_POOL_STREAMS;
    std::lock_guard<std::mutex> lock(bridge_mutex());
    // Record the legacy stream's current frontier and make stream k wait on it.
    cudaEventRecord(p.events[k], (cudaStream_t)0);
    cudaStreamWaitEvent(p.streams[k], p.events[k], 0);
}

void stwo_legacy_wait_stream(int i) {
    StreamPool &p = stream_pool();
    int k = i % STWO_N_POOL_STREAMS;
    std::lock_guard<std::mutex> lock(bridge_mutex());
    cudaEventRecord(p.events[k], p.streams[k]);
    cudaStreamWaitEvent((cudaStream_t)0, p.events[k], 0);
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
