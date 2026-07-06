#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// Resource-owning execution context for ONE resident proof (design §19).
//
// The core prove path is stream-0-centric: every kernel + every pool alloc/free
// is ordered on the legacy default stream (see cuda_mem_pool.cuh). To run two
// proofs concurrently, each must own an independent CUDA stream AND its own
// stream-ordered memory pool: two contexts allocating from DISJOINT pools have no
// cross-stream reuse hazard, whereas two streams sharing the single global default
// pool have no ordering guarantee between one's cudaFreeAsync and the other's
// kernels (the central concurrency hazard in the recon).
//
// A context is NOT just a stream pointer — it owns the stream and the pool, and all
// alloc / free / kernel work for the proof's streamful island routes through its
// stream, so buffers free stream-ordered AFTER their last use on that stream. No
// stream-0 free ever races another stream's kernels (the resource-lifetime bug).
//
// Handles are opaque `void*` (StwoExecContext*), matching the FFI style of the rest
// of this crate. `nullptr` pool => the context falls back to per-stream async
// allocation against the default pool (still stream-ordered, just not isolated).
// ---------------------------------------------------------------------------

namespace {
struct StwoExecContext {
    cudaStream_t stream;
    cudaMemPool_t pool;  // owned; nullptr => use the default pool via cudaMallocAsync
    bool owns_pool;
};
}  // namespace

// Create a context: a non-blocking stream + (best-effort) its own never-release
// memory pool on the current device. Returns nullptr if the stream cannot be
// created; a null pool is tolerated (falls back to cudaMallocAsync on the stream).
extern "C" void *stwo_exec_context_create() {
    StwoExecContext *ctx = new StwoExecContext();
    ctx->stream = nullptr;
    ctx->pool = nullptr;
    ctx->owns_pool = false;

    if (cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking) != cudaSuccess) {
        delete ctx;
        return nullptr;
    }

    int device_id = 0;
    if (cudaGetDevice(&device_id) == cudaSuccess) {
        cudaMemPoolProps props = {};
        props.allocType = cudaMemAllocationTypePinned;
        props.handleTypes = cudaMemHandleTypeNone;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = device_id;
        if (cudaMemPoolCreate(&ctx->pool, &props) == cudaSuccess && ctx->pool != nullptr) {
            // Never release back to the OS between warm proves (as the default pool
            // is configured), so the second proof reuses the first's freed blocks.
            uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(ctx->pool, cudaMemPoolAttrReleaseThreshold, &threshold);
            ctx->owns_pool = true;
        } else {
            ctx->pool = nullptr;
        }
    }
    return ctx;
}

// Sync + tear down. Synchronizes the stream first so no pending free/kernel outlives
// the pool/stream it references.
extern "C" void stwo_exec_context_destroy(void *handle) {
    if (handle == nullptr) {
        return;
    }
    StwoExecContext *ctx = static_cast<StwoExecContext *>(handle);
    if (ctx->stream != nullptr) {
        cudaStreamSynchronize(ctx->stream);
        cudaStreamDestroy(ctx->stream);
    }
    if (ctx->owns_pool && ctx->pool != nullptr) {
        cudaMemPoolDestroy(ctx->pool);
    }
    delete ctx;
}

// Block the host until all work enqueued on the context's stream has completed.
extern "C" void stwo_exec_context_sync(void *handle) {
    if (handle == nullptr) {
        return;
    }
    cudaStreamSynchronize(static_cast<StwoExecContext *>(handle)->stream);
}

// The context's stream as an opaque handle, to pass to `_on(stream)` kernel variants.
extern "C" void *stwo_exec_context_stream(void *handle) {
    if (handle == nullptr) {
        return nullptr;
    }
    return static_cast<StwoExecContext *>(handle)->stream;
}

// Allocate `count` u32 from the context's pool, ordered on its stream. The returned
// pointer is usable by later work on the SAME stream (stream-ordered allocation
// guarantees the memory is backed before any later same-stream consumer runs).
// Returns nullptr on failure.
extern "C" uint32_t *stwo_exec_context_alloc_u32(void *handle, size_t count) {
    StwoExecContext *ctx = static_cast<StwoExecContext *>(handle);
    uint32_t *ptr = nullptr;
    size_t bytes = count * sizeof(uint32_t);
    cudaError_t err;
    if (ctx->pool != nullptr) {
        err = cudaMallocFromPoolAsync((void **)&ptr, bytes, ctx->pool, ctx->stream);
    } else {
        err = cudaMallocAsync((void **)&ptr, bytes, ctx->stream);
    }
    if (err != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

// Free a buffer on the context's stream: ordered AFTER all prior same-stream users,
// so no use-after-free for same-stream work and no cross-stream reuse hazard.
extern "C" void stwo_exec_context_free_u32(void *handle, uint32_t *ptr) {
    if (ptr == nullptr) {
        return;
    }
    cudaFreeAsync(ptr, static_cast<StwoExecContext *>(handle)->stream);
}
