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

/// Stable identity of the AOT modules embedded in this binary.
///
/// The graph cache needs the semantic cache keys and target architectures, not
/// process-local module pointers.  An empty/stub pack deliberately returns zero
/// so the GPU-native runtime can fail closed instead of constructing a graph key
/// that would later fall through to NVRTC.
pub fn aot_pack_manifest_hash() -> u64 {
    if AOT_INDEX.is_empty() {
        return 0;
    }
    let mut hash = 0xcbf29ce484222325u64;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    feed(b"stwo-cuda-aot-pack-v2\0");
    feed(&(AOT_CONSTRAINT_MAX_INSTRS as u64).to_le_bytes());
    feed(&(AOT_INDEX.len() as u64).to_le_bytes());
    for &(cache_key, sm, _offset, len) in AOT_INDEX {
        feed(&cache_key.to_le_bytes());
        feed(&sm.to_le_bytes());
        feed(&(len as u64).to_le_bytes());
    }
    hash
}

/// Constraint-lowering split cap used to generate the embedded semantic keys.
/// Zero is returned for an empty/stub pack so strict planning cannot guess a
/// cap that has no corresponding loaded kernels.
pub fn aot_pack_constraint_max_instrs() -> usize {
    if AOT_INDEX.is_empty() {
        0
    } else {
        AOT_CONSTRAINT_MAX_INSTRS
    }
}

/// True when this binary contains at least one kernel for `sm_major.sm_minor`.
/// Full per-proof coverage is still established by fail-closed lookup at every
/// semantic key; this is the cheap admission check before arena allocation.
pub fn aot_pack_supports_arch(sm_major: u32, sm_minor: u32) -> bool {
    let sm = sm_major * 10 + sm_minor;
    AOT_INDEX.iter().any(|entry| entry.1 == sm)
}

#[cfg(test)]
mod manifest_tests {
    use super::*;

    #[test]
    fn empty_pack_is_explicitly_unbound() {
        if AOT_INDEX.is_empty() {
            assert_eq!(aot_pack_manifest_hash(), 0);
            assert_eq!(aot_pack_constraint_max_instrs(), 0);
        } else {
            assert_ne!(aot_pack_manifest_hash(), 0);
            assert_ne!(aot_pack_constraint_max_instrs(), 0);
        }
    }

    #[test]
    fn architecture_admission_matches_embedded_index() {
        for &(_, sm, ..) in AOT_INDEX {
            assert!(aot_pack_supports_arch(sm / 10, sm % 10));
        }
    }
}
