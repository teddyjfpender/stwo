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
    n_rows: usize,
}

/// A registered table failed the exact geometry required by a borrower.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegisteredPedersenTableError {
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

impl RegisteredPedersenTable {
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
}

static REGISTERED: OnceLock<Option<RegisteredPedersenTable>> = OnceLock::new();

/// Upload the host pedersen table and register it as the device table.
/// `n_rows` is the UNPADDED host row count; `fill_column(c, buf)` appends
/// column `c`'s `n_rows` raw words to `buf` (cleared beforehand). Idempotent
/// per process (first call wins). Returns `false` — callers must fall back to
/// host lanes — on a stub build or a malformed column. The uploaded buffers
/// are deliberately leaked: the table lives for the process, like the host
/// static it mirrors.
pub fn register_borrowed_pedersen_table(
    n_rows: usize,
    mut fill_column: impl FnMut(usize, &mut Vec<u32>),
) -> bool {
    REGISTERED
        .get_or_init(|| {
            if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT || n_rows == 0 {
                return None;
            }
            let padded = n_rows.next_power_of_two();
            bindings::ensure_mem_pool_init();

            let mut device_ptrs: Vec<*mut u32> = Vec::with_capacity(PEDERSEN_TABLE_N_COLUMNS);
            let mut buf: Vec<u32> = Vec::with_capacity(padded);
            for c in 0..PEDERSEN_TABLE_N_COLUMNS {
                buf.clear();
                fill_column(c, &mut buf);
                if buf.len() != n_rows {
                    eprintln!(
                        "pedersen table registration: column {c} has {} rows, expected {n_rows}",
                        buf.len()
                    );
                    return None;
                }
                // Padding rows are 0 — the deduce functions' pow2 mask means only
                // garbage inputs can read them; the differential gates own value
                // correctness for real rows. Raw u32 transport (limbs are 9-bit
                // values, but nothing here may canonicalize).
                buf.resize(padded, 0);
                let dev = unsafe {
                    bindings::copy_uint32_t_vec_from_host_to_device(buf.as_ptr(), padded as u32)
                };
                if dev.is_null() {
                    eprintln!("pedersen table registration: device upload failed (column {c})");
                    return None;
                }
                device_ptrs.push(dev.cast_mut());
            }
            unsafe {
                stwo_backend_cuda_kernels::raw::pedersen_table_init(
                    device_ptrs.as_ptr(),
                    padded as u32,
                );
            }
            eprintln!(
            "pedersen table: registered HOST-BUILT table on device ({PEDERSEN_TABLE_N_COLUMNS} \
             cols x {padded} rows, borrowed mode)"
        );
            // The raw CUDA allocations have process lifetime. The temporary host
            // pointer vector may be dropped because the C++ registration copied its
            // pointer values into process-global state.
            let columns = device_ptrs
                .iter()
                .enumerate()
                .map(|(index, &device_ptr)| RegisteredPedersenColumn {
                    index,
                    device_address: device_ptr as usize,
                    len_words: padded,
                })
                .collect::<Vec<_>>()
                .try_into()
                .expect("pedersen registration built exactly 56 columns");
            Some(RegisteredPedersenTable {
                columns,
                n_rows: padded,
            })
        })
        .is_some()
}

/// Whether a (successful) registration happened this process.
pub fn pedersen_table_registered() -> bool {
    registered_borrowed_pedersen_table().is_some()
}

/// Borrow the exact process-lifetime table without allocating or copying.
pub fn registered_borrowed_pedersen_table() -> Option<RegisteredPedersenTable> {
    REGISTERED.get().copied().flatten()
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
        }
    }

    #[test]
    fn registered_geometry_is_ordered_and_fails_closed() {
        let columns = std::array::from_fn(|index| RegisteredPedersenColumn {
            index,
            device_address: 0x1000 + index * 0x100,
            len_words: 1 << 23,
        });
        let table = RegisteredPedersenTable {
            columns,
            n_rows: 1 << 23,
        };

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
