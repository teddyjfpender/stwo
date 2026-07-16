//! Stable-address whole-allocation VMM storage for one bounded reclaim cycle.
//!
//! The only legal lifecycle is mapped generation 0, unmapped after a complete
//! D2H spill, then mapped generation 1 before H2D restore. Unmapping consumes a
//! proof that every auxiliary lane joined and the main stream completed.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;
use core::ptr::NonNull;

use super::exec_context::{
    check_cuda, CudaExecContext, CudaQuiescence, CudaRuntimeError, JoinedCudaLanes,
};

const INITIAL_GENERATION: u32 = 0;
const RESTORED_GENERATION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmmAllocationState {
    Mapped { generation: u32 },
    Unmapped,
    Poisoned,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VmmAllocationError {
    Cuda(CudaRuntimeError),
    InvalidSize(usize),
    InvalidGeometry {
        bytes: usize,
        granularity: usize,
    },
    SizeMismatch {
        expected: usize,
        actual: usize,
    },
    ContextMismatch,
    InvalidState {
        operation: &'static str,
        state: VmmAllocationState,
    },
    SequenceViolation(&'static str),
}

impl core::fmt::Display for VmmAllocationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Cuda(error) => error.fmt(f),
            Self::InvalidSize(bytes) => write!(f, "invalid CUDA VMM allocation size {bytes}"),
            Self::InvalidGeometry { bytes, granularity } => write!(
                f,
                "invalid CUDA VMM geometry: {bytes} bytes at {granularity}-byte granularity"
            ),
            Self::SizeMismatch { expected, actual } => write!(
                f,
                "CUDA VMM full-allocation transfer requires {expected} bytes, got {actual}"
            ),
            Self::ContextMismatch => f.write_str("CUDA VMM owner context mismatch"),
            Self::InvalidState { operation, state } => {
                write!(
                    f,
                    "CUDA VMM operation {operation} is invalid in state {state:?}"
                )
            }
            Self::SequenceViolation(step) => {
                write!(f, "CUDA VMM quiescence sequence violated at {step}")
            }
        }
    }
}

impl std::error::Error for VmmAllocationError {}

impl From<CudaRuntimeError> for VmmAllocationError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

/// One stable virtual address with exactly one admitted physical reclaim cycle.
///
/// This allocation is independent of [`super::exec_context::DeviceArena`]. A
/// graph may retain `stable_address`, but replay is legal only while `state` is
/// mapped. The owner context and the allocation must remain on one host thread.
pub struct VmmAllocation {
    handle: NonNull<c_void>,
    stable_address: NonNull<c_void>,
    bytes: usize,
    granularity: usize,
    owner_context: NonNull<c_void>,
    state: VmmAllocationState,
    _not_sync: PhantomData<Cell<()>>,
}

unsafe impl Send for VmmAllocation {}

impl VmmAllocation {
    pub fn new(context: &CudaExecContext, bytes: usize) -> Result<Self, VmmAllocationError> {
        if bytes == 0 {
            return Err(VmmAllocationError::InvalidSize(bytes));
        }
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            return Err(CudaRuntimeError::Unavailable.into());
        }

        let mut raw_handle = core::ptr::null_mut();
        let mut raw_address = core::ptr::null_mut();
        let mut mapped_bytes = 0usize;
        let mut granularity = 0usize;
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_vmm_allocation_create(
                context.identity_token().as_ptr(),
                bytes,
                &mut raw_handle,
                &mut raw_address,
                &mut mapped_bytes,
                &mut granularity,
            )
        };
        if let Err(error) = check_cuda("vmm_allocation_create", code) {
            if !raw_handle.is_null() {
                unsafe {
                    stwo_backend_cuda_kernels::raw::stwo_vmm_allocation_destroy(raw_handle);
                }
            }
            return Err(error.into());
        }

        let handle = NonNull::new(raw_handle).ok_or(CudaRuntimeError::NullPointer {
            operation: "vmm_allocation_create_handle",
        })?;
        let Some(stable_address) = NonNull::new(raw_address) else {
            unsafe {
                stwo_backend_cuda_kernels::raw::stwo_vmm_allocation_destroy(handle.as_ptr());
            }
            return Err(CudaRuntimeError::NullPointer {
                operation: "vmm_allocation_create_address",
            }
            .into());
        };
        if mapped_bytes != bytes {
            unsafe {
                stwo_backend_cuda_kernels::raw::stwo_vmm_allocation_destroy(handle.as_ptr());
            }
            return Err(VmmAllocationError::SizeMismatch {
                expected: bytes,
                actual: mapped_bytes,
            });
        }
        if !valid_geometry(mapped_bytes, granularity) {
            unsafe {
                stwo_backend_cuda_kernels::raw::stwo_vmm_allocation_destroy(handle.as_ptr());
            }
            return Err(VmmAllocationError::InvalidGeometry {
                bytes: mapped_bytes,
                granularity,
            });
        }

        Ok(Self {
            handle,
            stable_address,
            bytes,
            granularity,
            owner_context: context.identity_token(),
            state: VmmAllocationState::Mapped {
                generation: INITIAL_GENERATION,
            },
            _not_sync: PhantomData,
        })
    }

    pub fn stable_address(&self) -> NonNull<c_void> {
        self.stable_address
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn granularity(&self) -> usize {
        self.granularity
    }

    pub fn state(&self) -> VmmAllocationState {
        self.state
    }

    /// Spill the complete allocation, join every lane, fence the main stream,
    /// then unmap and release physical backing.
    ///
    /// # Safety
    ///
    /// `host_destination` must name writable pinned host memory of exactly
    /// `bytes`. No producer may retain access beyond the validated lane joins.
    pub unsafe fn spill_to_host(
        &mut self,
        context: &CudaExecContext,
        host_destination: NonNull<c_void>,
        bytes: usize,
    ) -> Result<(), VmmAllocationError> {
        self.require_owner(context)?;
        require_exact_bytes(self.bytes, bytes)?;
        let mut operations = CudaVmmOps::new(self.handle, context);
        unsafe {
            spill_transition(
                &mut self.state,
                &mut operations,
                host_destination,
                self.stable_address,
                bytes,
            )
        }
    }

    /// Remap generation 1 at the same virtual address, then enqueue a complete
    /// H2D restore on the owner main stream.
    ///
    /// # Safety
    ///
    /// `host_source` must name readable pinned host memory of exactly `bytes`
    /// and remain live until later owner-stream synchronization. The caller must
    /// order every lane consumer after the main-stream restore.
    pub unsafe fn restore_from_host(
        &mut self,
        context: &CudaExecContext,
        host_source: NonNull<c_void>,
        bytes: usize,
    ) -> Result<(), VmmAllocationError> {
        self.require_owner(context)?;
        require_exact_bytes(self.bytes, bytes)?;
        let mut operations = CudaVmmOps::new(self.handle, context);
        unsafe {
            restore_transition(
                &mut self.state,
                &mut operations,
                self.stable_address,
                host_source,
                bytes,
            )
        }
    }

    fn require_owner(&self, context: &CudaExecContext) -> Result<(), VmmAllocationError> {
        if context.identity_token() != self.owner_context {
            return Err(VmmAllocationError::ContextMismatch);
        }
        Ok(())
    }
}

impl Drop for VmmAllocation {
    fn drop(&mut self) {
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_vmm_allocation_destroy(self.handle.as_ptr())
        };
        if code != 0 && !std::thread::panicking() {
            eprintln!("stwo-backend-cuda: vmm_allocation_destroy failed with status {code}");
        }
    }
}

fn valid_geometry(bytes: usize, granularity: usize) -> bool {
    granularity != 0 && granularity.is_power_of_two() && bytes != 0 && bytes % granularity == 0
}

fn require_exact_bytes(expected: usize, actual: usize) -> Result<(), VmmAllocationError> {
    if actual == expected {
        Ok(())
    } else {
        Err(VmmAllocationError::SizeMismatch { expected, actual })
    }
}

trait VmmOps {
    fn join_all_lanes(&mut self) -> Result<(), VmmAllocationError>;

    unsafe fn copy_d2h(
        &mut self,
        destination: NonNull<c_void>,
        source: NonNull<c_void>,
        bytes: usize,
    ) -> Result<(), VmmAllocationError>;

    fn sync_main(&mut self) -> Result<(), VmmAllocationError>;
    fn unmap_release(&mut self) -> Result<(), VmmAllocationError>;
    fn remap_generation1(&mut self) -> Result<(), VmmAllocationError>;

    unsafe fn copy_h2d(
        &mut self,
        destination: NonNull<c_void>,
        source: NonNull<c_void>,
        bytes: usize,
    ) -> Result<(), VmmAllocationError>;
}

struct CudaVmmOps<'a> {
    allocation: NonNull<c_void>,
    context: &'a CudaExecContext,
    joined: Option<JoinedCudaLanes<'a>>,
    quiescence: Option<CudaQuiescence>,
}

impl<'a> CudaVmmOps<'a> {
    fn new(allocation: NonNull<c_void>, context: &'a CudaExecContext) -> Self {
        Self {
            allocation,
            context,
            joined: None,
            quiescence: None,
        }
    }
}

impl VmmOps for CudaVmmOps<'_> {
    fn join_all_lanes(&mut self) -> Result<(), VmmAllocationError> {
        if self.joined.is_some() || self.quiescence.is_some() {
            return Err(VmmAllocationError::SequenceViolation("join_all_lanes"));
        }
        self.joined = Some(self.context.join_all_lanes_for_vmm()?);
        Ok(())
    }

    unsafe fn copy_d2h(
        &mut self,
        destination: NonNull<c_void>,
        source: NonNull<c_void>,
        bytes: usize,
    ) -> Result<(), VmmAllocationError> {
        let joined = self
            .joined
            .as_ref()
            .ok_or(VmmAllocationError::SequenceViolation("copy_d2h"))?;
        unsafe {
            joined.memcpy_d2h_async(destination.as_ptr(), source.as_ptr(), bytes)?;
        }
        Ok(())
    }

    fn sync_main(&mut self) -> Result<(), VmmAllocationError> {
        let joined = self
            .joined
            .take()
            .ok_or(VmmAllocationError::SequenceViolation("sync_main"))?;
        self.quiescence = Some(joined.sync_main()?);
        Ok(())
    }

    fn unmap_release(&mut self) -> Result<(), VmmAllocationError> {
        let quiescence = self
            .quiescence
            .take()
            .ok_or(VmmAllocationError::SequenceViolation("unmap_release"))?;
        if quiescence.context_token() != self.context.identity_token() {
            return Err(VmmAllocationError::ContextMismatch);
        }
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_vmm_allocation_unmap_release(
                self.allocation.as_ptr(),
                self.context.identity_token().as_ptr(),
            )
        };
        check_cuda("vmm_allocation_unmap_release", code)?;
        Ok(())
    }

    fn remap_generation1(&mut self) -> Result<(), VmmAllocationError> {
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_vmm_allocation_remap_generation1(
                self.allocation.as_ptr(),
                self.context.identity_token().as_ptr(),
            )
        };
        check_cuda("vmm_allocation_remap_generation1", code)?;
        Ok(())
    }

    unsafe fn copy_h2d(
        &mut self,
        destination: NonNull<c_void>,
        source: NonNull<c_void>,
        bytes: usize,
    ) -> Result<(), VmmAllocationError> {
        unsafe {
            self.context.memcpy_h2d_async(
                destination.as_ptr(),
                source.as_ptr().cast_const(),
                bytes,
            )?;
        }
        Ok(())
    }
}

unsafe fn spill_transition(
    state: &mut VmmAllocationState,
    operations: &mut impl VmmOps,
    host_destination: NonNull<c_void>,
    device_source: NonNull<c_void>,
    bytes: usize,
) -> Result<(), VmmAllocationError> {
    if *state
        != (VmmAllocationState::Mapped {
            generation: INITIAL_GENERATION,
        })
    {
        return Err(VmmAllocationError::InvalidState {
            operation: "spill",
            state: *state,
        });
    }
    if let Err(error) = operations.join_all_lanes() {
        *state = VmmAllocationState::Poisoned;
        return Err(error);
    }
    if let Err(error) = unsafe { operations.copy_d2h(host_destination, device_source, bytes) } {
        *state = VmmAllocationState::Poisoned;
        return Err(error);
    }
    if let Err(error) = operations.sync_main() {
        *state = VmmAllocationState::Poisoned;
        return Err(error);
    }
    if let Err(error) = operations.unmap_release() {
        *state = VmmAllocationState::Poisoned;
        return Err(error);
    }
    *state = VmmAllocationState::Unmapped;
    Ok(())
}

unsafe fn restore_transition(
    state: &mut VmmAllocationState,
    operations: &mut impl VmmOps,
    device_destination: NonNull<c_void>,
    host_source: NonNull<c_void>,
    bytes: usize,
) -> Result<(), VmmAllocationError> {
    if *state != VmmAllocationState::Unmapped {
        return Err(VmmAllocationError::InvalidState {
            operation: "restore",
            state: *state,
        });
    }
    if let Err(error) = operations.remap_generation1() {
        *state = VmmAllocationState::Poisoned;
        return Err(error);
    }
    if let Err(error) = unsafe { operations.copy_h2d(device_destination, host_source, bytes) } {
        *state = VmmAllocationState::Poisoned;
        return Err(error);
    }
    *state = VmmAllocationState::Mapped {
        generation: RESTORED_GENERATION,
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Call {
        JoinAll,
        D2h,
        Sync,
        Unmap,
        Remap,
        H2d,
    }

    #[derive(Default)]
    struct MockOps {
        calls: Vec<Call>,
        fail_at: Option<Call>,
    }

    impl MockOps {
        fn invoke(&mut self, call: Call) -> Result<(), VmmAllocationError> {
            self.calls.push(call);
            if self.fail_at == Some(call) {
                Err(VmmAllocationError::SequenceViolation("injected failure"))
            } else {
                Ok(())
            }
        }
    }

    impl VmmOps for MockOps {
        fn join_all_lanes(&mut self) -> Result<(), VmmAllocationError> {
            self.invoke(Call::JoinAll)
        }

        unsafe fn copy_d2h(
            &mut self,
            _destination: NonNull<c_void>,
            _source: NonNull<c_void>,
            _bytes: usize,
        ) -> Result<(), VmmAllocationError> {
            self.invoke(Call::D2h)
        }

        fn sync_main(&mut self) -> Result<(), VmmAllocationError> {
            self.invoke(Call::Sync)
        }

        fn unmap_release(&mut self) -> Result<(), VmmAllocationError> {
            self.invoke(Call::Unmap)
        }

        fn remap_generation1(&mut self) -> Result<(), VmmAllocationError> {
            self.invoke(Call::Remap)
        }

        unsafe fn copy_h2d(
            &mut self,
            _destination: NonNull<c_void>,
            _source: NonNull<c_void>,
            _bytes: usize,
        ) -> Result<(), VmmAllocationError> {
            self.invoke(Call::H2d)
        }
    }

    fn pointer() -> NonNull<c_void> {
        NonNull::dangling()
    }

    #[test]
    fn lifecycle_is_exact_and_orders_quiescence_before_unmap() {
        let mut state = VmmAllocationState::Mapped { generation: 0 };
        let mut spill = MockOps::default();
        unsafe {
            spill_transition(&mut state, &mut spill, pointer(), pointer(), 64).unwrap();
        }
        assert_eq!(
            spill.calls,
            [Call::JoinAll, Call::D2h, Call::Sync, Call::Unmap]
        );
        assert_eq!(state, VmmAllocationState::Unmapped);

        let mut restore = MockOps::default();
        unsafe {
            restore_transition(&mut state, &mut restore, pointer(), pointer(), 64).unwrap();
        }
        assert_eq!(restore.calls, [Call::Remap, Call::H2d]);
        assert_eq!(state, VmmAllocationState::Mapped { generation: 1 });

        let mut rejected = MockOps::default();
        assert!(
            unsafe { spill_transition(&mut state, &mut rejected, pointer(), pointer(), 64) }
                .is_err()
        );
        assert!(rejected.calls.is_empty());
    }

    #[test]
    fn every_spill_failure_stops_before_later_steps_and_poison_state() {
        let cases = [
            (Call::JoinAll, vec![Call::JoinAll]),
            (Call::D2h, vec![Call::JoinAll, Call::D2h]),
            (Call::Sync, vec![Call::JoinAll, Call::D2h, Call::Sync]),
            (
                Call::Unmap,
                vec![Call::JoinAll, Call::D2h, Call::Sync, Call::Unmap],
            ),
        ];
        for (fail_at, expected) in cases {
            let mut state = VmmAllocationState::Mapped { generation: 0 };
            let mut operations = MockOps {
                fail_at: Some(fail_at),
                ..MockOps::default()
            };
            assert!(unsafe {
                spill_transition(&mut state, &mut operations, pointer(), pointer(), 64)
            }
            .is_err());
            assert_eq!(operations.calls, expected);
            assert_eq!(state, VmmAllocationState::Poisoned);
        }
    }

    #[test]
    fn every_restore_failure_poison_state_and_cannot_retry() {
        for (fail_at, expected) in [
            (Call::Remap, vec![Call::Remap]),
            (Call::H2d, vec![Call::Remap, Call::H2d]),
        ] {
            let mut state = VmmAllocationState::Unmapped;
            let mut operations = MockOps {
                fail_at: Some(fail_at),
                ..MockOps::default()
            };
            assert!(unsafe {
                restore_transition(&mut state, &mut operations, pointer(), pointer(), 64)
            }
            .is_err());
            assert_eq!(operations.calls, expected);
            assert_eq!(state, VmmAllocationState::Poisoned);

            let mut retry = MockOps::default();
            assert!(unsafe {
                restore_transition(&mut state, &mut retry, pointer(), pointer(), 64)
            }
            .is_err());
            assert!(retry.calls.is_empty());
        }
    }

    #[test]
    fn full_allocation_geometry_is_exact() {
        assert!(valid_geometry(128 * 1024, 64 * 1024));
        assert!(!valid_geometry(96 * 1024, 64 * 1024));
        assert!(!valid_geometry(64 * 1024, 48 * 1024));
        assert_eq!(require_exact_bytes(64, 64), Ok(()));
        assert_eq!(
            require_exact_bytes(64, 63),
            Err(VmmAllocationError::SizeMismatch {
                expected: 64,
                actual: 63,
            })
        );
    }
}
