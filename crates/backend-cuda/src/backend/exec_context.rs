//! Per-proof CUDA execution resources: an isolated stream/pool, a stable device
//! arena, and transcript-bounded single-stream graph capture.
//!
//! There is deliberately no default/TLS context. A proof workspace owns one
//! [`DeviceArena`], which in turn owns its [`CudaExecContext`]. Graphs must be
//! destroyed before their arena because captured nodes contain arena addresses.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;
use core::ptr::NonNull;
use std::collections::BTreeMap;
use std::rc::Rc;

const CUDA_SUCCESS: i32 = 0;

/// Checked failure from the CUDA runtime boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaRuntimeError {
    /// This binary was built without the CUDA kernel archive.
    Unavailable,
    /// CUDA returned a non-zero status code.
    Cuda { operation: &'static str, code: i32 },
    /// CUDA reported success without returning the required opaque pointer.
    NullPointer { operation: &'static str },
    /// A requested allocation size overflowed `usize` bytes.
    SizeOverflow,
    /// A graph or arena-backed plan was launched on a different context.
    ContextMismatch,
}

impl core::fmt::Display for CudaRuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unavailable => f.write_str("CUDA kernels are not available in this build"),
            Self::Cuda { operation, code } => {
                write!(f, "CUDA operation {operation} failed with status {code}")
            }
            Self::NullPointer { operation } => {
                write!(f, "CUDA operation {operation} returned a null pointer")
            }
            Self::SizeOverflow => f.write_str("CUDA allocation size overflow"),
            Self::ContextMismatch => f.write_str("CUDA context identity mismatch"),
        }
    }
}

impl std::error::Error for CudaRuntimeError {}

pub(crate) fn check_cuda(operation: &'static str, code: i32) -> Result<(), CudaRuntimeError> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaRuntimeError::Cuda { operation, code })
    }
}

/// One proof's isolated non-blocking CUDA stream and never-release memory pool.
///
/// The native constructor fails closed if the custom pool cannot be created; it
/// never falls back to the process-wide default pool. The context may move to its
/// owning proof thread, but it is not shareable across threads.
pub struct CudaExecContext {
    handle: NonNull<c_void>,
    stream: NonNull<c_void>,
    _not_sync: PhantomData<Cell<()>>,
}

// A context and its stream may be moved to one owning host thread. It is not Sync
// (the Cell marker above), so capture/enqueue cannot be driven concurrently.
unsafe impl Send for CudaExecContext {}

impl CudaExecContext {
    /// Create an isolated CUDA stream/pool context.
    pub fn new() -> Result<Self, CudaRuntimeError> {
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            return Err(CudaRuntimeError::Unavailable);
        }

        let mut raw_handle = core::ptr::null_mut();
        let code =
            unsafe { stwo_backend_cuda_kernels::raw::stwo_exec_context_create(&mut raw_handle) };
        check_cuda("exec_context_create", code)?;
        let handle = NonNull::new(raw_handle).ok_or(CudaRuntimeError::NullPointer {
            operation: "exec_context_create",
        })?;

        let mut raw_stream = core::ptr::null_mut();
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_exec_context_stream(
                handle.as_ptr(),
                &mut raw_stream,
            )
        };
        if let Err(error) = check_cuda("exec_context_stream", code) {
            unsafe {
                stwo_backend_cuda_kernels::raw::stwo_exec_context_destroy(handle.as_ptr());
            }
            return Err(error);
        }
        let Some(stream) = NonNull::new(raw_stream) else {
            unsafe {
                stwo_backend_cuda_kernels::raw::stwo_exec_context_destroy(handle.as_ptr());
            }
            return Err(CudaRuntimeError::NullPointer {
                operation: "exec_context_stream",
            });
        };

        Ok(Self {
            handle,
            stream,
            _not_sync: PhantomData,
        })
    }

    /// Opaque CUDA stream pointer for stream-explicit kernel launch wrappers.
    pub fn stream_raw(&self) -> NonNull<c_void> {
        self.stream
    }

    pub(crate) fn identity_token(&self) -> NonNull<c_void> {
        self.handle
    }

    /// Block until every operation enqueued on this context's stream completes.
    pub fn sync(&self) -> Result<(), CudaRuntimeError> {
        let code =
            unsafe { stwo_backend_cuda_kernels::raw::stwo_exec_context_sync(self.handle.as_ptr()) };
        check_cuda("exec_context_sync", code)
    }

    /// Allocate `count` u32 words from this context's isolated pool.
    pub fn alloc_u32(&self, count: usize) -> Result<NonNull<u32>, CudaRuntimeError> {
        count
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(CudaRuntimeError::SizeOverflow)?;
        let mut raw_ptr = core::ptr::null_mut();
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_exec_context_alloc_u32(
                self.handle.as_ptr(),
                count,
                &mut raw_ptr,
            )
        };
        check_cuda("exec_context_alloc_u32", code)?;
        NonNull::new(raw_ptr).ok_or(CudaRuntimeError::NullPointer {
            operation: "exec_context_alloc_u32",
        })
    }

    /// Free a context allocation, ordered after prior work on this stream.
    ///
    /// # Safety
    ///
    /// `ptr` must have been returned by [`Self::alloc_u32`] on this context and
    /// must not have been freed already.
    pub unsafe fn free_u32(&self, ptr: NonNull<u32>) -> Result<(), CudaRuntimeError> {
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_exec_context_free_u32(
                self.handle.as_ptr(),
                ptr.as_ptr(),
            )
        };
        check_cuda("exec_context_free_u32", code)
    }

    /// Enqueue a byte memset on this context's stream.
    ///
    /// # Safety
    ///
    /// `dst..dst+bytes` must be a live device allocation that remains valid until
    /// this stream has passed the operation.
    pub unsafe fn memset_async(
        &self,
        dst: *mut c_void,
        value: u8,
        bytes: usize,
    ) -> Result<(), CudaRuntimeError> {
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_exec_context_memset_async(
                self.handle.as_ptr(),
                dst,
                i32::from(value),
                bytes,
            )
        };
        check_cuda("exec_context_memset_async", code)
    }

    /// Enqueue an arbitrary u32 fill on this context's stream.
    ///
    /// # Safety
    ///
    /// `dst..dst+count` must be a live device range that remains valid until
    /// this stream has passed the operation.
    pub unsafe fn fill_u32_async(
        &self,
        dst: *mut u32,
        value: u32,
        count: usize,
    ) -> Result<(), CudaRuntimeError> {
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_exec_context_fill_u32_async(
                self.handle.as_ptr(),
                dst,
                value,
                count,
            )
        };
        check_cuda("exec_context_fill_u32_async", code)
    }

    /// Enqueue a device-to-device copy on this context's stream.
    ///
    /// # Safety
    ///
    /// Both ranges must be live, non-overlapping device allocations of at least
    /// `bytes` and remain valid until this stream has passed the operation.
    pub unsafe fn memcpy_d2d_async(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> Result<(), CudaRuntimeError> {
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_exec_context_memcpy_d2d_async(
                self.handle.as_ptr(),
                dst,
                src,
                bytes,
            )
        };
        check_cuda("exec_context_memcpy_d2d_async", code)
    }

    /// Enqueue a host-to-device copy on this context's stream.
    ///
    /// # Safety
    ///
    /// `src` must be readable and `dst` writable for `bytes`; both must remain
    /// valid until the operation completes. Pinned host memory is required for
    /// genuinely asynchronous transfer.
    pub unsafe fn memcpy_h2d_async(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> Result<(), CudaRuntimeError> {
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_exec_context_memcpy_h2d_async(
                self.handle.as_ptr(),
                dst,
                src,
                bytes,
            )
        };
        check_cuda("exec_context_memcpy_h2d_async", code)
    }

    /// Enqueue a device-to-host copy on this context's stream.
    ///
    /// # Safety
    ///
    /// `src` must be readable and `dst` writable for `bytes`; both must remain
    /// valid until the operation completes. The host must not read `dst` before
    /// [`Self::sync`] succeeds.
    pub unsafe fn memcpy_d2h_async(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> Result<(), CudaRuntimeError> {
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_exec_context_memcpy_d2h_async(
                self.handle.as_ptr(),
                dst,
                src,
                bytes,
            )
        };
        check_cuda("exec_context_memcpy_d2h_async", code)
    }

    /// Begin thread-local capture on this context's stream.
    pub fn capture(&self) -> Result<CudaGraphCapture<'_>, CudaRuntimeError> {
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_graph_capture_begin(self.handle.as_ptr())
        };
        check_cuda("graph_capture_begin", code)?;
        Ok(CudaGraphCapture {
            context: self,
            active: true,
            _same_thread: PhantomData,
        })
    }
}

impl Drop for CudaExecContext {
    fn drop(&mut self) {
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_exec_context_destroy(self.handle.as_ptr())
        };
        if code != CUDA_SUCCESS && !std::thread::panicking() {
            eprintln!("stwo-backend-cuda: exec_context_destroy failed with status {code}");
        }
    }
}

/// In-progress capture. Dropping without [`Self::finish`] exits capture mode and
/// discards the captured graph.
pub struct CudaGraphCapture<'a> {
    context: &'a CudaExecContext,
    active: bool,
    // cudaStreamCaptureModeThreadLocal requires begin/end on the same host thread.
    _same_thread: PhantomData<Rc<()>>,
}

impl CudaGraphCapture<'_> {
    /// Finish capture, instantiate it, and return an owned executable graph.
    pub fn finish(mut self) -> Result<CudaGraphExec, CudaRuntimeError> {
        let mut raw_exec = core::ptr::null_mut();
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_graph_capture_end(
                self.context.handle.as_ptr(),
                &mut raw_exec,
            )
        };
        // EndCapture exits capture mode even when instantiation later fails.
        self.active = false;
        check_cuda("graph_capture_end", code)?;
        let handle = NonNull::new(raw_exec).ok_or(CudaRuntimeError::NullPointer {
            operation: "graph_capture_end",
        })?;
        Ok(CudaGraphExec {
            handle,
            context_token: self.context.identity_token(),
        })
    }

    /// Explicitly abort capture and discard any graph produced by EndCapture.
    pub fn abort(mut self) -> Result<(), CudaRuntimeError> {
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_graph_capture_abort(self.context.handle.as_ptr())
        };
        self.active = false;
        check_cuda("graph_capture_abort", code)
    }
}

impl Drop for CudaGraphCapture<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_graph_capture_abort(self.context.handle.as_ptr())
        };
        self.active = false;
        if code != CUDA_SUCCESS && !std::thread::panicking() {
            eprintln!("stwo-backend-cuda: graph capture abort failed with status {code}");
        }
    }
}

/// Instantiated CUDA graph. Captured pointer arguments must outlive this object.
pub struct CudaGraphExec {
    handle: NonNull<c_void>,
    context_token: NonNull<c_void>,
}

unsafe impl Send for CudaGraphExec {}

impl CudaGraphExec {
    /// Enqueue one replay on `context`'s stream.
    pub fn launch(&self, context: &CudaExecContext) -> Result<(), CudaRuntimeError> {
        if context.identity_token() != self.context_token {
            return Err(CudaRuntimeError::ContextMismatch);
        }
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_graph_launch(
                self.handle.as_ptr(),
                context.handle.as_ptr(),
            )
        };
        check_cuda("graph_launch", code)
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        let code =
            unsafe { stwo_backend_cuda_kernels::raw::stwo_graph_destroy(self.handle.as_ptr()) };
        if code != CUDA_SUCCESS && !std::thread::panicking() {
            eprintln!("stwo-backend-cuda: graph_destroy failed with status {code}");
        }
    }
}

/// Stable logical identity for one arena slot.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ArenaSlotId(pub u32);

/// One statically planned, non-overlapping range in the arena slab.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArenaSlotSpec {
    pub id: ArenaSlotId,
    pub offset_words: usize,
    pub len_words: usize,
    /// Required power-of-two alignment, measured in u32 words.
    pub alignment_words: usize,
}

/// Rejected arena-plan condition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArenaError {
    ZeroSizedArena,
    DuplicateSlot(ArenaSlotId),
    EmptySlot(ArenaSlotId),
    InvalidAlignment(ArenaSlotId),
    Misaligned(ArenaSlotId),
    RangeOverflow(ArenaSlotId),
    OutOfBounds(ArenaSlotId),
    Overlap {
        first: ArenaSlotId,
        second: ArenaSlotId,
    },
    UnknownSlot(ArenaSlotId),
    Cuda(CudaRuntimeError),
}

impl core::fmt::Display for ArenaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid CUDA arena: {self:?}")
    }
}

impl std::error::Error for ArenaError {}

impl From<CudaRuntimeError> for ArenaError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

/// Validated stable-address layout for one capture epoch.
#[derive(Clone, Debug)]
pub struct ArenaLayout {
    total_words: usize,
    slots: BTreeMap<ArenaSlotId, ArenaSlotSpec>,
}

impl ArenaLayout {
    /// Validate bounds, power-of-two alignment, duplicate identities, and static
    /// non-overlap before any device allocation occurs.
    pub fn new(total_words: usize, specs: &[ArenaSlotSpec]) -> Result<Self, ArenaError> {
        if total_words == 0 {
            return Err(ArenaError::ZeroSizedArena);
        }

        let mut slots = BTreeMap::new();
        let mut ranges = Vec::with_capacity(specs.len());
        for &spec in specs {
            if spec.len_words == 0 {
                return Err(ArenaError::EmptySlot(spec.id));
            }
            if !spec.alignment_words.is_power_of_two() {
                return Err(ArenaError::InvalidAlignment(spec.id));
            }
            if spec.offset_words % spec.alignment_words != 0 {
                return Err(ArenaError::Misaligned(spec.id));
            }
            let end = spec
                .offset_words
                .checked_add(spec.len_words)
                .ok_or(ArenaError::RangeOverflow(spec.id))?;
            if end > total_words {
                return Err(ArenaError::OutOfBounds(spec.id));
            }
            if slots.insert(spec.id, spec).is_some() {
                return Err(ArenaError::DuplicateSlot(spec.id));
            }
            ranges.push((spec.offset_words, end, spec.id));
        }

        ranges.sort_unstable_by_key(|&(start, end, id)| (start, end, id));
        for pair in ranges.windows(2) {
            let (_, first_end, first_id) = pair[0];
            let (second_start, _, second_id) = pair[1];
            if second_start < first_end {
                return Err(ArenaError::Overlap {
                    first: first_id,
                    second: second_id,
                });
            }
        }

        Ok(Self { total_words, slots })
    }

    pub fn total_words(&self) -> usize {
        self.total_words
    }

    pub fn slot(&self, id: ArenaSlotId) -> Option<ArenaSlotSpec> {
        self.slots.get(&id).copied()
    }
}

/// Non-owning view of a stable arena range. Dropping it never frees memory.
#[derive(Clone, Copy, Debug)]
pub struct ArenaSlice {
    id: ArenaSlotId,
    ptr: NonNull<u32>,
    len_words: usize,
    context_token: NonNull<c_void>,
}

impl ArenaSlice {
    pub fn id(self) -> ArenaSlotId {
        self.id
    }

    pub fn as_u32_ptr(self) -> *mut u32 {
        self.ptr.as_ptr()
    }

    pub fn as_void_ptr(self) -> *mut c_void {
        self.ptr.as_ptr().cast()
    }

    pub fn len_words(self) -> usize {
        self.len_words
    }

    pub fn len_bytes(self) -> usize {
        // Layout construction and allocation already proved this multiplication fits.
        self.len_words * core::mem::size_of::<u32>()
    }

    pub(crate) fn context_token(self) -> NonNull<c_void> {
        self.context_token
    }

    #[cfg(test)]
    pub(crate) fn dangling_for_test(id: u32, len_words: usize) -> Self {
        Self {
            id: ArenaSlotId(id),
            // Distinct, pointer-aligned sentinel addresses. Tests only inspect
            // plan geometry; they never dereference these pointers.
            ptr: NonNull::new((32 + id as usize * 64) as *mut u32).unwrap(),
            len_words,
            context_token: NonNull::dangling(),
        }
    }
}

/// One stable device allocation partitioned by a validated slot plan.
///
/// The slab is allocated once and never moved or individually freed during the
/// capture epoch. [`CudaExecContext`] is owned here to make the free/stream/pool
/// lifetime order structural rather than caller convention.
pub struct DeviceArena {
    context: CudaExecContext,
    base: NonNull<u32>,
    layout: ArenaLayout,
}

unsafe impl Send for DeviceArena {}

impl DeviceArena {
    pub fn new(context: CudaExecContext, layout: ArenaLayout) -> Result<Self, ArenaError> {
        layout
            .total_words
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(ArenaError::Cuda(CudaRuntimeError::SizeOverflow))?;
        let base = context.alloc_u32(layout.total_words)?;
        // Back the async allocation before exposing its address to setup work on
        // any other CUDA API. All subsequent arena work stays on this context.
        context.sync()?;
        Ok(Self {
            context,
            base,
            layout,
        })
    }

    pub fn context(&self) -> &CudaExecContext {
        &self.context
    }

    pub fn base_ptr(&self) -> NonNull<u32> {
        self.base
    }

    pub fn layout(&self) -> &ArenaLayout {
        &self.layout
    }

    pub fn bind(&self, id: ArenaSlotId) -> Result<ArenaSlice, ArenaError> {
        let spec = self.layout.slot(id).ok_or(ArenaError::UnknownSlot(id))?;
        let ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(spec.offset_words)) };
        Ok(ArenaSlice {
            id,
            ptr,
            len_words: spec.len_words,
            context_token: self.context.identity_token(),
        })
    }
}

impl Drop for DeviceArena {
    fn drop(&mut self) {
        let result = unsafe { self.context.free_u32(self.base) };
        if let Err(error) = result {
            if !std::thread::panicking() {
                eprintln!("stwo-backend-cuda: arena free failed: {error}");
            }
        }
        // CudaExecContext drops next: it synchronizes the queued free before
        // destroying the stream and pool.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: ArenaSlotId = ArenaSlotId(1);
    const B: ArenaSlotId = ArenaSlotId(2);

    fn valid_specs() -> [ArenaSlotSpec; 2] {
        [
            ArenaSlotSpec {
                id: A,
                offset_words: 0,
                len_words: 16,
                alignment_words: 8,
            },
            ArenaSlotSpec {
                id: B,
                offset_words: 16,
                len_words: 32,
                alignment_words: 8,
            },
        ]
    }

    #[test]
    fn context_is_unavailable_without_cuda() {
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            assert!(matches!(
                CudaExecContext::new(),
                Err(CudaRuntimeError::Unavailable)
            ));
        }
    }

    #[test]
    fn arena_layout_accepts_aligned_non_overlapping_slots() {
        let layout = ArenaLayout::new(64, &valid_specs()).unwrap();
        assert_eq!(layout.total_words(), 64);
        assert_eq!(layout.slot(A).unwrap().len_words, 16);
        assert_eq!(layout.slot(B).unwrap().offset_words, 16);
    }

    #[test]
    fn arena_layout_rejects_duplicate_misaligned_overlapping_and_oob_slots() {
        let mut specs = valid_specs();
        specs[1].id = A;
        assert_eq!(
            ArenaLayout::new(64, &specs).unwrap_err(),
            ArenaError::DuplicateSlot(A)
        );

        let mut specs = valid_specs();
        specs[1].offset_words = 18;
        assert_eq!(
            ArenaLayout::new(64, &specs).unwrap_err(),
            ArenaError::Misaligned(B)
        );

        let mut specs = valid_specs();
        specs[1].offset_words = 8;
        assert_eq!(
            ArenaLayout::new(64, &specs).unwrap_err(),
            ArenaError::Overlap {
                first: A,
                second: B,
            }
        );

        let mut specs = valid_specs();
        specs[1].offset_words = 40;
        assert_eq!(
            ArenaLayout::new(64, &specs).unwrap_err(),
            ArenaError::OutOfBounds(B)
        );
    }

    /// Native graph gate: one captured D2D node replays against the same arena
    /// addresses while the source slot changes between launches.
    #[cfg(stwo_cuda_link)]
    #[test]
    fn graph_capture_replays_over_stable_arena_slots() {
        let context = CudaExecContext::new().unwrap();
        let layout = ArenaLayout::new(64, &valid_specs()).unwrap();
        let arena = DeviceArena::new(context, layout).unwrap();
        let src = arena.bind(A).unwrap();
        let dst = arena.bind(B).unwrap();
        assert_eq!(arena.base_ptr(), src.ptr);

        let capture = arena.context().capture().unwrap();
        unsafe {
            arena
                .context()
                .memcpy_d2d_async(
                    dst.as_void_ptr(),
                    src.as_void_ptr().cast_const(),
                    src.len_bytes(),
                )
                .unwrap();
        }
        let graph = capture.finish().unwrap();

        let mut host = vec![0_u8; src.len_bytes()];
        for byte in [0x11, 0x7f, 0xa5, 0x00] {
            unsafe {
                arena
                    .context()
                    .memset_async(src.as_void_ptr(), byte, src.len_bytes())
                    .unwrap();
            }
            graph.launch(arena.context()).unwrap();
            unsafe {
                arena
                    .context()
                    .memcpy_d2h_async(
                        host.as_mut_ptr().cast(),
                        dst.as_void_ptr().cast_const(),
                        dst.len_bytes(),
                    )
                    .unwrap();
            }
            arena.context().sync().unwrap();
            assert!(host.iter().all(|&value| value == byte));
            assert_eq!(arena.base_ptr(), src.ptr);
        }

        drop(graph);
        drop(arena);
    }
}
