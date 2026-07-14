//! Device registration of the HOST-BUILT pedersen points table (borrowed mode).
//!
//! The deduce-gate oracle falsified the GPU-generated table: 144/256 sampled
//! rows — including section boundaries — differ from the host
//! `PEDERSEN_TABLE_18` (run 20260705T113615Z). Until that generator is fixed
//! and re-gated, the ONLY table the witness-deduce lane may read is the host
//! table itself, uploaded column-for-column: byte-identical by construction.
//!
//! The caller (stwo-cairo, which owns `PEDERSEN_TABLE_18`) supplies columns
//! through a fill closure — one column at a time into a reusable buffer, so the
//! host-side overhead peaks at one padded column (~33MB), not the whole ~1.9GB
//! table. Columns are padded to a power of two (the deduce functions mask row
//! indices with `n_rows - 1`) and registered via `pedersen_table_init`
//! (borrowed mode: the pointers publish to the precompiled module's device
//! globals; JIT modules' per-module globals fill from the same registration at
//! module load).

use std::sync::OnceLock;

use crate::columns::bindings;

/// Column count of the pedersen points table (28 x-limbs + 28 y-limbs).
pub const PEDERSEN_TABLE_N_COLUMNS: usize = 56;

/// One immutable, process-lifetime device column from the registered host table.
///
/// The address is intentionally not publicly constructible: callers may borrow
/// only columns whose backing allocation was installed by
/// [`register_borrowed_pedersen_table`] and deliberately retained for the
/// process lifetime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisteredPedersenColumn {
    index: usize,
    device_address: usize,
    len_words: usize,
}

impl RegisteredPedersenColumn {
    pub const fn index(self) -> usize {
        self.index
    }

    pub fn as_u32_ptr(self) -> *mut u32 {
        self.device_address as *mut u32
    }

    pub const fn len_words(self) -> usize {
        self.len_words
    }
}

/// Exact process-lifetime registration published to precompiled and AOT/JIT
/// witness modules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisteredPedersenTable {
    columns: [RegisteredPedersenColumn; PEDERSEN_TABLE_N_COLUMNS],
    source_n_rows: usize,
    n_rows: usize,
}

/// A registered table failed the exact geometry required by a borrower.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegisteredPedersenTableError {
    SourceRowCount {
        expected: usize,
        actual: usize,
    },
    RowCount {
        expected: usize,
        actual: usize,
    },
    ColumnIndex {
        position: usize,
        actual: usize,
    },
    ColumnLength {
        column: usize,
        expected: usize,
        actual: usize,
    },
    NullColumnPointer {
        column: usize,
    },
}

/// A recoverable, host-visible failure while constructing the process table.
///
/// Once one of these failures is observed, registration is poisoned for the
/// process. Returning the same cause on every later call prevents a second
/// builder from mixing a new allocation set with partially initialized state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PedersenTableRegistrationError {
    CudaUnavailable,
    EmptyTable,
    RowCountOverflow {
        requested_rows: usize,
    },
    NativeRowCountLimit {
        padded_rows: usize,
        max_rows: usize,
    },
    HostAllocationFailed {
        allocation: &'static str,
    },
    FillPanicked {
        column: usize,
    },
    ColumnLength {
        column: usize,
        expected: usize,
        actual: usize,
    },
    PoolInitialization(crate::CudaRuntimeError),
    DeviceUploadReturnedNull {
        column: usize,
    },
    InvalidReadyGeometry(RegisteredPedersenTableError),
    RequestGeometryMismatch {
        requested_padded_rows: usize,
        registered_padded_rows: usize,
    },
    RequestSourceRowCountMismatch {
        requested_source_rows: usize,
        registered_source_rows: usize,
        padded_rows: usize,
    },
}

impl core::fmt::Display for PedersenTableRegistrationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CudaUnavailable => f.write_str("CUDA kernels are not available in this build"),
            Self::EmptyTable => f.write_str("the pedersen table has no rows"),
            Self::RowCountOverflow { requested_rows } => write!(
                f,
                "pedersen row count {requested_rows} overflows power-of-two padding"
            ),
            Self::NativeRowCountLimit {
                padded_rows,
                max_rows,
            } => write!(
                f,
                "padded pedersen row count {padded_rows} exceeds native upload limit {max_rows}"
            ),
            Self::HostAllocationFailed { allocation } => {
                write!(f, "host allocation failed for {allocation}")
            }
            Self::FillPanicked { column } => {
                write!(f, "pedersen column builder panicked at column {column}")
            }
            Self::ColumnLength {
                column,
                expected,
                actual,
            } => write!(
                f,
                "pedersen column {column} has {actual} rows, expected {expected}"
            ),
            Self::PoolInitialization(error) => {
                write!(f, "CUDA pool initialization failed: {error}")
            }
            Self::DeviceUploadReturnedNull { column } => {
                write!(f, "device upload returned null at pedersen column {column}")
            }
            Self::InvalidReadyGeometry(error) => {
                write!(
                    f,
                    "constructed pedersen table has invalid geometry: {error:?}"
                )
            }
            Self::RequestGeometryMismatch {
                requested_padded_rows,
                registered_padded_rows,
            } => write!(
                f,
                "requested pedersen geometry has {requested_padded_rows} padded rows, but the \
                 registered table has {registered_padded_rows}"
            ),
            Self::RequestSourceRowCountMismatch {
                requested_source_rows,
                registered_source_rows,
                padded_rows,
            } => write!(
                f,
                "requested pedersen source has {requested_source_rows} rows, but the registered \
                 source has {registered_source_rows} rows (both pad to {padded_rows})"
            ),
        }
    }
}

impl std::error::Error for PedersenTableRegistrationError {}

/// Snapshot of the process-wide registration state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PedersenTableRegistrationState {
    Uninitialized,
    Ready(RegisteredPedersenTable),
    Poisoned(PedersenTableRegistrationError),
}

impl RegisteredPedersenTable {
    pub const fn source_n_rows(self) -> usize {
        self.source_n_rows
    }

    pub const fn n_rows(self) -> usize {
        self.n_rows
    }

    pub const fn columns(self) -> [RegisteredPedersenColumn; PEDERSEN_TABLE_N_COLUMNS] {
        self.columns
    }

    pub fn column(self, index: usize) -> Option<RegisteredPedersenColumn> {
        self.columns.get(index).copied()
    }

    pub const fn has_exact_rows(self, expected_rows: usize) -> bool {
        self.n_rows == expected_rows
    }

    /// Validate every borrowed pointer before exposing the table to another
    /// execution context. A single malformed column rejects the whole table;
    /// callers must never mix registered and regenerated columns silently.
    pub fn validate_exact_geometry(
        self,
        expected_rows: usize,
    ) -> Result<(), RegisteredPedersenTableError> {
        if self.n_rows != expected_rows {
            return Err(RegisteredPedersenTableError::RowCount {
                expected: expected_rows,
                actual: self.n_rows,
            });
        }
        for (position, column) in self.columns.iter().copied().enumerate() {
            if column.index != position {
                return Err(RegisteredPedersenTableError::ColumnIndex {
                    position,
                    actual: column.index,
                });
            }
            if column.len_words != expected_rows {
                return Err(RegisteredPedersenTableError::ColumnLength {
                    column: position,
                    expected: expected_rows,
                    actual: column.len_words,
                });
            }
            if column.as_u32_ptr().is_null() {
                return Err(RegisteredPedersenTableError::NullColumnPointer { column: position });
            }
        }
        Ok(())
    }

    /// Validate both the host source identity and its padded device geometry.
    pub fn validate_exact_registration_geometry(
        self,
        expected_source_rows: usize,
        expected_padded_rows: usize,
    ) -> Result<(), RegisteredPedersenTableError> {
        if self.source_n_rows != expected_source_rows {
            return Err(RegisteredPedersenTableError::SourceRowCount {
                expected: expected_source_rows,
                actual: self.source_n_rows,
            });
        }
        self.validate_exact_geometry(expected_padded_rows)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RegistrationGeometry {
    source_rows: usize,
    padded_rows: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum StoredRegistrationState {
    Ready(RegisteredPedersenTable),
    Poisoned(PedersenTableRegistrationError),
}

struct RegistrationSlot {
    state: OnceLock<StoredRegistrationState>,
}

impl RegistrationSlot {
    const fn new() -> Self {
        Self {
            state: OnceLock::new(),
        }
    }

    fn try_register(
        &self,
        requested_geometry: Result<RegistrationGeometry, PedersenTableRegistrationError>,
        build: impl FnOnce(
            RegistrationGeometry,
        ) -> Result<RegisteredPedersenTable, PedersenTableRegistrationError>,
    ) -> Result<RegisteredPedersenTable, PedersenTableRegistrationError> {
        let state = self.state.get_or_init(|| {
            let result = requested_geometry.clone().and_then(|geometry| {
                let table = build(geometry)?;
                table
                    .validate_exact_registration_geometry(
                        geometry.source_rows,
                        geometry.padded_rows,
                    )
                    .map_err(PedersenTableRegistrationError::InvalidReadyGeometry)?;
                Ok(table)
            });
            match result {
                Ok(table) => StoredRegistrationState::Ready(table),
                Err(error) => StoredRegistrationState::Poisoned(error),
            }
        });

        match state {
            StoredRegistrationState::Poisoned(error) => Err(error.clone()),
            StoredRegistrationState::Ready(table) => {
                let requested = requested_geometry?;
                if table.n_rows != requested.padded_rows {
                    return Err(PedersenTableRegistrationError::RequestGeometryMismatch {
                        requested_padded_rows: requested.padded_rows,
                        registered_padded_rows: table.n_rows,
                    });
                }
                if table.source_n_rows != requested.source_rows {
                    return Err(
                        PedersenTableRegistrationError::RequestSourceRowCountMismatch {
                            requested_source_rows: requested.source_rows,
                            registered_source_rows: table.source_n_rows,
                            padded_rows: requested.padded_rows,
                        },
                    );
                }
                Ok(*table)
            }
        }
    }

    fn snapshot(&self) -> PedersenTableRegistrationState {
        match self.state.get() {
            None => PedersenTableRegistrationState::Uninitialized,
            Some(StoredRegistrationState::Ready(table)) => {
                PedersenTableRegistrationState::Ready(*table)
            }
            Some(StoredRegistrationState::Poisoned(error)) => {
                PedersenTableRegistrationState::Poisoned(error.clone())
            }
        }
    }

    fn ready(&self) -> Option<RegisteredPedersenTable> {
        match self.state.get() {
            Some(StoredRegistrationState::Ready(table)) => Some(*table),
            None | Some(StoredRegistrationState::Poisoned(_)) => None,
        }
    }
}

struct PendingDeviceColumns {
    pointers: Vec<*mut u32>,
    published: bool,
}

impl PendingDeviceColumns {
    fn new() -> Result<Self, PedersenTableRegistrationError> {
        let mut pointers = Vec::new();
        pointers
            .try_reserve_exact(PEDERSEN_TABLE_N_COLUMNS)
            .map_err(|_| PedersenTableRegistrationError::HostAllocationFailed {
                allocation: "pedersen device-pointer list",
            })?;
        Ok(Self {
            pointers,
            published: false,
        })
    }

    fn mark_published(&mut self) {
        self.published = true;
    }
}

impl Drop for PendingDeviceColumns {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        for pointer in self.pointers.drain(..) {
            unsafe {
                // This is the only available deallocator. It returns no status;
                // native code logs an async-free failure and falls back to a
                // synchronous free.
                bindings::cuda_free_memory(pointer.cast());
            }
        }
    }
}

static REGISTERED: RegistrationSlot = RegistrationSlot::new();

fn requested_geometry(
    n_rows: usize,
) -> Result<RegistrationGeometry, PedersenTableRegistrationError> {
    if n_rows == 0 {
        return Err(PedersenTableRegistrationError::EmptyTable);
    }
    let padded_rows = n_rows.checked_next_power_of_two().ok_or(
        PedersenTableRegistrationError::RowCountOverflow {
            requested_rows: n_rows,
        },
    )?;
    // The legacy upload entry point takes a C `int`, despite the generated
    // Rust declaration using `u32`. Reject values that would become negative.
    let max_rows = i32::MAX as usize;
    if padded_rows > max_rows {
        return Err(PedersenTableRegistrationError::NativeRowCountLimit {
            padded_rows,
            max_rows,
        });
    }
    Ok(RegistrationGeometry {
        source_rows: n_rows,
        padded_rows,
    })
}

fn build_borrowed_pedersen_table(
    geometry: RegistrationGeometry,
    fill_column: &mut impl FnMut(usize, &mut Vec<u32>),
) -> Result<RegisteredPedersenTable, PedersenTableRegistrationError> {
    let RegistrationGeometry {
        source_rows,
        padded_rows,
    } = geometry;
    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        return Err(PedersenTableRegistrationError::CudaUnavailable);
    }
    bindings::try_ensure_mem_pool_init()
        .map_err(PedersenTableRegistrationError::PoolInitialization)?;

    let mut pending = PendingDeviceColumns::new()?;
    let mut buf = Vec::new();
    buf.try_reserve_exact(padded_rows).map_err(|_| {
        PedersenTableRegistrationError::HostAllocationFailed {
            allocation: "padded pedersen column buffer",
        }
    })?;

    for column in 0..PEDERSEN_TABLE_N_COLUMNS {
        buf.clear();
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fill_column(column, &mut buf);
        }))
        .is_err()
        {
            return Err(PedersenTableRegistrationError::FillPanicked { column });
        }
        if buf.len() != source_rows {
            return Err(PedersenTableRegistrationError::ColumnLength {
                column,
                expected: source_rows,
                actual: buf.len(),
            });
        }

        // Padding rows are 0. Real deduce indices never reach them, and raw
        // words must not be canonicalized during transport.
        buf.try_reserve_exact(padded_rows - buf.len())
            .map_err(|_| PedersenTableRegistrationError::HostAllocationFailed {
                allocation: "padded pedersen column buffer",
            })?;
        buf.resize(padded_rows, 0);
        let device_pointer = unsafe {
            bindings::copy_uint32_t_vec_from_host_to_device(buf.as_ptr(), padded_rows as u32)
        };
        if device_pointer.is_null() {
            return Err(PedersenTableRegistrationError::DeviceUploadReturnedNull { column });
        }
        pending.pointers.push(device_pointer.cast_mut());
    }

    let columns = std::array::from_fn(|index| RegisteredPedersenColumn {
        index,
        device_address: pending.pointers[index] as usize,
        len_words: padded_rows,
    });
    let table = RegisteredPedersenTable {
        columns,
        source_n_rows: source_rows,
        n_rows: padded_rows,
    };
    table
        .validate_exact_registration_geometry(source_rows, padded_rows)
        .map_err(PedersenTableRegistrationError::InvalidReadyGeometry)?;

    // These legacy native APIs abort the process on CUDA allocation, copy,
    // launch, or synchronization errors. They do not expose a status that Rust
    // can poison and recover from. If both calls return, publication completed;
    // only then may RAII release ownership and the OnceLock publish `Ready`.
    unsafe {
        stwo_backend_cuda_kernels::raw::pedersen_table_init(
            pending.pointers.as_ptr(),
            padded_rows as u32,
        );
        bindings::stwo_legacy_stream_sync();
    }
    pending.mark_published();
    Ok(table)
}

/// Checked, one-shot upload and publication of the host pedersen table.
///
/// `n_rows` is the unpadded host row count. The fill closure must append exactly
/// that many raw words for each column. A recoverable failure poisons this slot,
/// frees every uploaded but unpublished prefix, and is returned unchanged on
/// later calls without invoking their builders. A ready slot is reusable only
/// for the same unpadded source row count and padded device geometry.
///
/// The legacy upload and publication functions still terminate the process on
/// native CUDA errors; such aborts cannot be represented as a Rust error until
/// those native entry points return status codes.
pub fn try_register_borrowed_pedersen_table(
    n_rows: usize,
    mut fill_column: impl FnMut(usize, &mut Vec<u32>),
) -> Result<RegisteredPedersenTable, PedersenTableRegistrationError> {
    REGISTERED.try_register(requested_geometry(n_rows), |geometry| {
        build_borrowed_pedersen_table(geometry, &mut fill_column)
    })
}

/// Compatibility wrapper for callers that only distinguish device-ready from
/// host fallback.
pub fn register_borrowed_pedersen_table(
    n_rows: usize,
    fill_column: impl FnMut(usize, &mut Vec<u32>),
) -> bool {
    try_register_borrowed_pedersen_table(n_rows, fill_column).is_ok()
}

/// Whether a (successful) registration happened this process.
pub fn pedersen_table_registered() -> bool {
    registered_borrowed_pedersen_table().is_some()
}

/// Borrow the exact process-lifetime table without allocating or copying.
pub fn registered_borrowed_pedersen_table() -> Option<RegisteredPedersenTable> {
    REGISTERED.ready()
}

/// Inspect whether registration has not run, is ready, or is deterministically
/// poisoned by the first recoverable failure.
pub fn pedersen_table_registration_state() -> PedersenTableRegistrationState {
    REGISTERED.snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_build_registers_nothing() {
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            assert!(!register_borrowed_pedersen_table(32, |_c, buf| {
                buf.extend(std::iter::repeat_n(0u32, 32));
            }));
            assert!(!pedersen_table_registered());
            assert!(registered_borrowed_pedersen_table().is_none());
            assert_eq!(
                pedersen_table_registration_state(),
                PedersenTableRegistrationState::Poisoned(
                    PedersenTableRegistrationError::CudaUnavailable
                )
            );
        }
    }

    const fn geometry(source_rows: usize, padded_rows: usize) -> RegistrationGeometry {
        RegistrationGeometry {
            source_rows,
            padded_rows,
        }
    }

    fn table_with_geometry(source_rows: usize, padded_rows: usize) -> RegisteredPedersenTable {
        RegisteredPedersenTable {
            columns: std::array::from_fn(|index| RegisteredPedersenColumn {
                index,
                device_address: 0x1000 + index * 0x100,
                len_words: padded_rows,
            }),
            source_n_rows: source_rows,
            n_rows: padded_rows,
        }
    }

    #[test]
    fn ready_registration_reuses_without_a_second_builder() {
        let slot = RegistrationSlot::new();
        let invocations = std::cell::Cell::new(0);
        let first = slot
            .try_register(Ok(geometry(32, 32)), |_| {
                invocations.set(invocations.get() + 1);
                Ok(table_with_geometry(32, 32))
            })
            .unwrap();
        let second = slot
            .try_register(
                Ok(geometry(32, 32)),
                |_| -> Result<_, PedersenTableRegistrationError> {
                    panic!("ready registration invoked a second builder")
                },
            )
            .unwrap();

        assert_eq!(first, second);
        assert_eq!(invocations.get(), 1);
    }

    #[test]
    fn ready_registration_rejects_request_geometry_drift() {
        let slot = RegistrationSlot::new();
        slot.try_register(Ok(geometry(32, 32)), |_| Ok(table_with_geometry(32, 32)))
            .unwrap();

        assert_eq!(
            slot.try_register(
                Ok(geometry(64, 64)),
                |_| -> Result<_, PedersenTableRegistrationError> {
                    panic!("geometry drift invoked a second builder")
                }
            ),
            Err(PedersenTableRegistrationError::RequestGeometryMismatch {
                requested_padded_rows: 64,
                registered_padded_rows: 32,
            })
        );
    }

    #[test]
    fn ready_registration_rejects_same_padded_different_source_without_rebuilding() {
        let slot = RegistrationSlot::new();
        let invocations = std::cell::Cell::new(0);
        slot.try_register(Ok(geometry(17, 32)), |_| {
            invocations.set(invocations.get() + 1);
            Ok(table_with_geometry(17, 32))
        })
        .unwrap();

        assert_eq!(
            slot.try_register(Ok(geometry(32, 32)), |_| {
                invocations.set(invocations.get() + 1);
                Ok(table_with_geometry(32, 32))
            }),
            Err(
                PedersenTableRegistrationError::RequestSourceRowCountMismatch {
                    requested_source_rows: 32,
                    registered_source_rows: 17,
                    padded_rows: 32,
                }
            )
        );
        assert_eq!(invocations.get(), 1);
    }

    #[test]
    fn malformed_column_poison_is_stable() {
        let slot = RegistrationSlot::new();
        let malformed = PedersenTableRegistrationError::ColumnLength {
            column: 17,
            expected: 32,
            actual: 31,
        };

        assert_eq!(
            slot.try_register(Ok(geometry(32, 32)), |_| Err(malformed.clone())),
            Err(malformed.clone())
        );
        assert_eq!(
            slot.snapshot(),
            PedersenTableRegistrationState::Poisoned(malformed)
        );
        assert!(slot.ready().is_none());
    }

    #[test]
    fn poisoned_registration_never_invokes_another_builder() {
        let slot = RegistrationSlot::new();
        let invocations = std::cell::Cell::new(0);
        let failure = PedersenTableRegistrationError::DeviceUploadReturnedNull { column: 3 };
        let first = slot.try_register(Ok(geometry(32, 32)), |_| {
            invocations.set(invocations.get() + 1);
            Err(failure.clone())
        });
        let second = slot.try_register(Ok(geometry(32, 32)), |_| {
            invocations.set(invocations.get() + 1);
            Ok(table_with_geometry(32, 32))
        });

        assert_eq!(first, Err(failure.clone()));
        assert_eq!(second, Err(failure));
        assert_eq!(invocations.get(), 1);
    }

    #[test]
    fn registered_geometry_is_ordered_and_fails_closed() {
        let table = table_with_geometry(1 << 23, 1 << 23);

        assert_eq!(table.source_n_rows(), 1 << 23);
        assert!(table.has_exact_rows(1 << 23));
        assert!(!table.has_exact_rows(1 << 22));
        assert_eq!(table.column(0).unwrap().index(), 0);
        assert_eq!(table.column(55).unwrap().index(), 55);
        assert_eq!(table.column(55).unwrap().len_words(), 1 << 23);
        assert!(table.column(PEDERSEN_TABLE_N_COLUMNS).is_none());
        assert!(table
            .columns()
            .iter()
            .enumerate()
            .all(|(index, column)| column.index() == index));
        assert_eq!(table.validate_exact_geometry(1 << 23), Ok(()));
        assert_eq!(
            table.validate_exact_registration_geometry(1 << 23, 1 << 23),
            Ok(())
        );

        let mut wrong_source_rows = table;
        wrong_source_rows.source_n_rows -= 1;
        assert!(matches!(
            wrong_source_rows.validate_exact_registration_geometry(1 << 23, 1 << 23),
            Err(RegisteredPedersenTableError::SourceRowCount { .. })
        ));

        let mut wrong_rows = table;
        wrong_rows.n_rows -= 1;
        assert!(matches!(
            wrong_rows.validate_exact_geometry(1 << 23),
            Err(RegisteredPedersenTableError::RowCount { .. })
        ));

        let mut wrong_index = table;
        wrong_index.columns[17].index = 18;
        assert!(matches!(
            wrong_index.validate_exact_geometry(1 << 23),
            Err(RegisteredPedersenTableError::ColumnIndex { position: 17, .. })
        ));

        let mut wrong_length = table;
        wrong_length.columns[31].len_words -= 1;
        assert!(matches!(
            wrong_length.validate_exact_geometry(1 << 23),
            Err(RegisteredPedersenTableError::ColumnLength { column: 31, .. })
        ));

        let mut null_pointer = table;
        null_pointer.columns[55].device_address = 0;
        assert_eq!(
            null_pointer.validate_exact_geometry(1 << 23),
            Err(RegisteredPedersenTableError::NullColumnPointer { column: 55 })
        );
    }
}
