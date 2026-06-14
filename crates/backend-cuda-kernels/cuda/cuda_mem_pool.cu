#include "cuda_mem_pool.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
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

// Release pooled-but-unused device memory back to the OS. The pool's release
// threshold stays UINT64_MAX (warm proves keep reusing allocations), so the pool
// otherwise never shrinks: this is an EXPLICIT, opt-in trim called at a spill
// point (the low-memory eval-compaction path) AFTER the freed buffers' deferred
// cudaFreeAsync have made them unused. cudaMemPoolTrimTo(pool, 0) keeps zero
// bytes reserved beyond what is still in use. No-op (returns success) when
// stream-ordered allocation is unavailable.
extern "C" void stwo_cuda_mem_pool_trim() {
    cudaMemPool_t pool = stwo_default_mem_pool();
    if (pool != nullptr) {
        // The just-dropped buffers were freed with stream-ordered cudaFreeAsync on
        // the legacy stream; cudaMemPoolTrimTo only reclaims blocks that are
        // already free, so drain the stream first to make the release
        // deterministic. This runs once at the spill point, never in a hot loop.
        cudaStreamSynchronize((cudaStream_t)0);
        cudaMemPoolTrimTo(pool, 0);
    }
}

// Instantaneous device footprint of THIS process: total - free from
// cudaMemGetInfo. End-of-run footprint only (not a peak); see
// stwo_cuda_vram_peak_bytes for the high-water mark.
extern "C" uint64_t stwo_cuda_vram_used_bytes() {
    size_t free_mem = 0;
    size_t total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
        return 0;
    }
    return (uint64_t)(total_mem - free_mem);
}

// True high-water mark of device memory RESERVED by the allocator over the
// process lifetime, in bytes. With the never-release pool this equals peak
// reserved VRAM: cudaMemPoolAttrReservedMemHigh is a running max the pool
// maintains at every allocation, so it captures the peak even when polled at the
// end of a run (after a trim has already shrunk the live footprint). Returns 0
// when stream-ordered allocation is unavailable.
extern "C" uint64_t stwo_cuda_vram_peak_bytes() {
    cudaMemPool_t pool = stwo_default_mem_pool();
    if (pool == nullptr) {
        return 0;
    }
    uint64_t high = 0;
    if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &high) != cudaSuccess) {
        return 0;
    }
    return high;
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

// ---------------------------------------------------------------------------
// Upload lane (see header contract).
// ---------------------------------------------------------------------------
namespace {
struct UploadLane {
    cudaStream_t stream;
    cudaEvent_t bridge;
    cudaEvent_t half_fence[2];
};
UploadLane &upload_lane() {
    static UploadLane lane = [] {
        UploadLane l{};
        cudaStreamCreateWithFlags(&l.stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&l.bridge, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&l.half_fence[0], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&l.half_fence[1], cudaEventDisableTiming);
        return l;
    }();
    return lane;
}
}  // namespace

extern "C" uint32_t* stwo_upload_alloc_uint32(size_t count) {
    cudaMemPool_t pool = stwo_default_mem_pool();
    uint32_t *ptr = nullptr;
    if (pool != nullptr) {
        cudaError_t err = cudaMallocFromPoolAsync((void**)&ptr, count * sizeof(uint32_t), pool,
                                                  upload_lane().stream);
        if (err == cudaSuccess) {
            return ptr;
        }
        printf("upload-lane pool alloc of %zu words failed: %s\n", count,
               cudaGetErrorString(err));
    }
    // Fallback: plain device alloc (synchronizing, but correct on every path).
    if (cudaMalloc((void**)&ptr, count * sizeof(uint32_t)) != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

extern "C" void stwo_upload_h2d_async(const uint32_t* pinned_src, uint32_t* device_dst,
                                      uint64_t n_words) {
    cudaError_t err = cudaMemcpyAsync(device_dst, pinned_src, n_words * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice, upload_lane().stream);
    if (err != cudaSuccess) {
        printf("stwo_upload_h2d_async(%llu words) failed: %s\n",
               (unsigned long long)n_words, cudaGetErrorString(err));
        std::abort();
    }
}

extern "C" void stwo_upload_record_half(int half) {
    UploadLane &l = upload_lane();
    std::lock_guard<std::mutex> lock(bridge_mutex());
    cudaEventRecord(l.half_fence[half & 1], l.stream);
}

extern "C" void stwo_upload_half_sync(int half) {
    // Synchronizing an unrecorded (fresh) event returns immediately.
    cudaEventSynchronize(upload_lane().half_fence[half & 1]);
}

extern "C" void stwo_legacy_wait_uploads() {
    UploadLane &l = upload_lane();
    std::lock_guard<std::mutex> lock(bridge_mutex());
    cudaEventRecord(l.bridge, l.stream);
    cudaStreamWaitEvent((cudaStream_t)0, l.bridge, 0);
}
