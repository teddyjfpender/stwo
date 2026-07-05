//! The embedded AOT kernel pack (design §4, M3): per-arch cubins for every
//! kernel_emit-generated kernel, compiled offline at -O3 by build.rs and served
//! to `runtime_jit.cu`'s `get_or_compile` tier-0 via [`stwo_aot_lookup`]. A
//! cache-key miss (new/changed recording, unknown arch) falls back to NVRTC —
//! the drift check by construction. Empty on stub builds and until kernel_emit
//! populates `cuda/generated/`.

static AOT_PACK: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/aot_pack.bin"));
include!(concat!(env!("OUT_DIR"), "/aot_index.rs"));

/// C hook for `get_or_compile`: resolve `(cache_key, sm)` to an embedded cubin.
/// Exact-arch match only — SASS is arch-specific; any miss falls back to NVRTC.
///
/// # Safety
/// `out_data`/`out_len` must be valid writable pointers; the returned blob
/// borrows the process-lifetime embedded pack.
#[no_mangle]
pub unsafe extern "C" fn stwo_aot_lookup(
    cache_key: u64,
    sm_major: u32,
    sm_minor: u32,
    out_data: *mut *const u8,
    out_len: *mut usize,
) -> bool {
    let sm = sm_major * 10 + sm_minor;
    let Ok(i) = AOT_INDEX.binary_search_by_key(&(cache_key, sm), |e| (e.0, e.1)) else {
        return false;
    };
    let (_, _, off, len) = AOT_INDEX[i];
    unsafe {
        *out_data = AOT_PACK[off..off + len].as_ptr();
        *out_len = len;
    }
    true
}

/// Number of embedded (kernel, arch) entries — surfaced for logs/tests.
pub fn aot_pack_entries() -> usize {
    AOT_INDEX.len()
}
