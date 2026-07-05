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

static REGISTERED: OnceLock<bool> = OnceLock::new();

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
    *REGISTERED.get_or_init(|| {
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT || n_rows == 0 {
            return false;
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
                return false;
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
                return false;
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
        // The device buffers and the pointer array live for the process.
        std::mem::forget(device_ptrs);
        true
    })
}

/// Whether a (successful) registration happened this process.
pub fn pedersen_table_registered() -> bool {
    REGISTERED.get().copied().unwrap_or(false)
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
        }
    }
}
