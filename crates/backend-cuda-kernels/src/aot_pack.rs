//! The embedded AOT kernel pack (design §4, M3): per-arch cubins for every
//! kernel_emit-generated kernel, compiled offline at -O3 by build.rs and served
//! to `runtime_jit.cu`'s `get_or_compile` tier-0 via [`stwo_aot_lookup`]. A
//! cache-key miss (new/changed recording, unknown arch) falls back to NVRTC —
//! the drift check by construction. Empty on stub builds and until kernel_emit
//! populates `cuda/generated/`.

const ZERO_IDENTITY: [u8; 32] = [0; 32];
type AotIndexEntry = (u64, u32, usize, usize, [u8; 32]);

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
    if aot_pack_identity() == ZERO_IDENTITY {
        return false;
    }
    let Some(sm) = encode_sm(sm_major, sm_minor) else {
        return false;
    };
    let Ok(i) = AOT_INDEX.binary_search_by_key(&(cache_key, sm), |e| (e.0, e.1)) else {
        return false;
    };
    let (_, _, off, len, cubin_identity) = AOT_INDEX[i];
    if cubin_identity == ZERO_IDENTITY {
        return false;
    }
    let Some(end) = off.checked_add(len) else {
        return false;
    };
    let Some(cubin) = AOT_PACK.get(off..end) else {
        return false;
    };
    unsafe {
        *out_data = cubin.as_ptr();
        *out_len = len;
    }
    true
}

/// Number of embedded (kernel, arch) entries — surfaced for logs/tests.
pub fn aot_pack_entries() -> usize {
    if aot_pack_identity() == ZERO_IDENTITY {
        0
    } else {
        AOT_INDEX.len()
    }
}

/// Number of embedded kernels for one exact device architecture.
pub fn aot_pack_entries_for_arch(sm_major: u32, sm_minor: u32) -> usize {
    if aot_pack_identity() == ZERO_IDENTITY {
        return 0;
    }
    let Some(sm) = encode_sm(sm_major, sm_minor) else {
        return 0;
    };
    AOT_INDEX
        .iter()
        .filter(|entry| entry.1 == sm && entry.4 != ZERO_IDENTITY)
        .count()
}

/// Collision-resistant identity of the exact AOT modules and generation
/// policies embedded in this binary.
/// This binds compiled bytes, not a source/effect contract; the generated
/// manifest does not yet carry collision-resistant effect authority.
///
/// An empty/stub pack deliberately returns all-zero so the GPU-native runtime
/// fails closed instead of constructing authority that later falls through to
/// NVRTC.
pub fn aot_pack_identity() -> [u8; 32] {
    if !aot_pack_is_well_formed() {
        return ZERO_IDENTITY;
    }
    AOT_PACK_IDENTITY
}

/// Collision-resistant identity of one exact `(cache_key, SM)` cubin. A miss,
/// malformed architecture, empty pack, or invalid generated identity returns
/// all-zero. This is binary identity, not proof that the binary implements a
/// separately claimed effect body.
pub fn aot_cubin_identity(cache_key: u64, sm_major: u32, sm_minor: u32) -> [u8; 32] {
    if aot_pack_identity() == ZERO_IDENTITY {
        return ZERO_IDENTITY;
    }
    let Some(sm) = encode_sm(sm_major, sm_minor) else {
        return ZERO_IDENTITY;
    };
    AOT_INDEX
        .binary_search_by_key(&(cache_key, sm), |entry| (entry.0, entry.1))
        .ok()
        .map(|index| AOT_INDEX[index].4)
        .unwrap_or(ZERO_IDENTITY)
}

/// Non-authoritative u64 compatibility/telemetry tag derived from the full
/// pack identity. Never use this value as kernel, graph, or proof authority.
pub fn aot_pack_manifest_hash() -> u64 {
    legacy_telemetry_tag(aot_pack_identity())
}

/// Constraint-lowering split cap used to generate the embedded semantic keys.
/// Zero is returned for an empty/stub pack so strict planning cannot guess a
/// cap that has no corresponding loaded kernels.
pub fn aot_pack_constraint_max_instrs() -> usize {
    if aot_pack_identity() == ZERO_IDENTITY {
        0
    } else {
        AOT_CONSTRAINT_MAX_INSTRS
    }
}

/// Compacted live-u32-lane cap used to generate the embedded constraint split.
/// Zero is returned for an empty/stub pack so runtime admission cannot silently
/// combine a stale AOT pack with a different resource policy.
pub fn aot_pack_constraint_max_live_u32_lanes() -> usize {
    if aot_pack_identity() == ZERO_IDENTITY {
        0
    } else {
        AOT_CONSTRAINT_MAX_LIVE_U32_LANES
    }
}

/// True when this binary contains at least one kernel for `sm_major.sm_minor`.
/// Full per-proof coverage is still established by fail-closed lookup at every
/// semantic key; this is the cheap admission check before arena allocation.
pub fn aot_pack_supports_arch(sm_major: u32, sm_minor: u32) -> bool {
    let Some(sm) = encode_sm(sm_major, sm_minor) else {
        return false;
    };
    aot_pack_identity() != ZERO_IDENTITY
        && AOT_INDEX
            .iter()
            .any(|entry| entry.1 == sm && entry.4 != ZERO_IDENTITY)
}

/// True when this binary contains the exact `(cache_key, architecture)` entry.
/// This only searches the immutable embedded index and never initializes CUDA.
pub fn aot_pack_contains(cache_key: u64, sm_major: u32, sm_minor: u32) -> bool {
    aot_cubin_identity(cache_key, sm_major, sm_minor) != ZERO_IDENTITY
}

fn encode_sm(sm_major: u32, sm_minor: u32) -> Option<u32> {
    if sm_minor > 9 {
        return None;
    }
    sm_major.checked_mul(10)?.checked_add(sm_minor)
}

fn legacy_telemetry_tag(identity: [u8; 32]) -> u64 {
    if identity == ZERO_IDENTITY {
        return 0;
    }
    let folded = identity
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("eight-byte digest chunk")))
        .fold(0, |acc, word| acc ^ word);
    folded.max(1)
}

fn aot_pack_is_well_formed() -> bool {
    static WELL_FORMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *WELL_FORMED.get_or_init(|| {
        AOT_CONSTRAINT_MAX_INSTRS != 0
            && AOT_CONSTRAINT_MAX_LIVE_U32_LANES != 0
            && pack_is_well_formed(AOT_PACK, AOT_PACK_IDENTITY, AOT_INDEX)
    })
}

fn pack_is_well_formed(pack: &[u8], pack_identity: [u8; 32], index: &[AotIndexEntry]) -> bool {
    if index.is_empty() {
        return pack.is_empty() && pack_identity == ZERO_IDENTITY;
    }
    if pack_identity == ZERO_IDENTITY {
        return false;
    }
    let mut expected_offset = 0usize;
    let mut previous_key = None;
    for &(cache_key, sm, offset, len, identity) in index {
        let key = (cache_key, sm);
        if previous_key.is_some_and(|previous| previous >= key)
            || offset != expected_offset
            || len == 0
            || identity == ZERO_IDENTITY
        {
            return false;
        }
        let Some(end) = offset.checked_add(len) else {
            return false;
        };
        if end > pack.len() {
            return false;
        }
        expected_offset = end;
        previous_key = Some(key);
    }
    expected_offset == pack.len()
}

#[cfg(test)]
mod manifest_tests {
    use super::*;
    use crate::aot_identity;

    #[cfg(all(stwo_cuda_link, not(feature = "test-only-empty-aot-pack")))]
    fn generated_source_keys() -> std::collections::BTreeSet<u64> {
        let generated = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("cuda")
            .join("generated");
        std::fs::read_dir(generated)
            .expect("read generated AOT source directory")
            .map(|entry| entry.expect("read generated AOT source entry").path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "cu"))
            .map(|path| {
                let stem = path
                    .file_stem()
                    .expect("generated AOT source has a stem")
                    .to_string_lossy();
                u64::from_str_radix(
                    stem.rsplit('_')
                        .next()
                        .expect("generated AOT source has a cache key"),
                    16,
                )
                .expect("generated AOT source cache key is hexadecimal")
            })
            .collect()
    }

    #[test]
    fn empty_pack_is_explicitly_unbound() {
        if AOT_INDEX.is_empty() {
            assert_eq!(aot_pack_identity(), ZERO_IDENTITY);
            assert_eq!(aot_pack_manifest_hash(), 0);
            assert_eq!(aot_pack_constraint_max_instrs(), 0);
            assert_eq!(aot_pack_constraint_max_live_u32_lanes(), 0);
        } else {
            assert_ne!(aot_pack_identity(), ZERO_IDENTITY);
            assert_ne!(aot_pack_manifest_hash(), 0);
            assert_ne!(aot_pack_constraint_max_instrs(), 0);
            assert_ne!(aot_pack_constraint_max_live_u32_lanes(), 0);
        }
    }

    #[test]
    fn architecture_admission_matches_embedded_index() {
        for &(_, sm, ..) in AOT_INDEX {
            assert!(aot_pack_supports_arch(sm / 10, sm % 10));
        }
    }

    #[test]
    fn exact_membership_matches_embedded_index() {
        for &(cache_key, sm, ..) in AOT_INDEX {
            assert!(aot_pack_contains(cache_key, sm / 10, sm % 10));
            assert_ne!(
                aot_cubin_identity(cache_key, sm / 10, sm % 10),
                ZERO_IDENTITY
            );
        }
        assert!(!aot_pack_contains(0, u32::MAX, u32::MAX));
        assert_eq!(aot_cubin_identity(0, u32::MAX, u32::MAX), ZERO_IDENTITY);
    }

    #[test]
    fn generated_identities_match_exact_embedded_bytes() {
        let inputs = AOT_INDEX
            .iter()
            .map(|&(cache_key, sm, offset, len, identity)| {
                let end = offset
                    .checked_add(len)
                    .expect("generated cubin range does not overflow");
                let bytes = AOT_PACK
                    .get(offset..end)
                    .expect("generated cubin range is inside the embedded pack");
                let input = aot_identity::CubinIdentityInput {
                    cache_key,
                    sm,
                    bytes,
                };
                assert_eq!(identity, aot_identity::cubin_identity(input));
                input
            })
            .collect::<Vec<_>>();
        assert_eq!(
            aot_pack_identity(),
            aot_identity::pack_identity(
                AOT_CONSTRAINT_MAX_INSTRS,
                AOT_CONSTRAINT_MAX_LIVE_U32_LANES,
                &inputs,
            )
        );
    }

    #[test]
    fn malformed_pack_layouts_have_no_authority() {
        const PACK_ID: [u8; 32] = [0x55; 32];
        const CUBIN_ID: [u8; 32] = [0xaa; 32];
        let pack = [1, 2, 3, 4];
        let valid = [(11, 86, 0, 2, CUBIN_ID), (13, 90, 2, 2, CUBIN_ID)];
        assert!(pack_is_well_formed(&pack, PACK_ID, &valid));
        assert!(pack_is_well_formed(&[], ZERO_IDENTITY, &[]));

        assert!(!pack_is_well_formed(&pack, ZERO_IDENTITY, &valid));
        assert!(!pack_is_well_formed(&pack, PACK_ID, &[]));
        assert!(!pack_is_well_formed(&[], PACK_ID, &[]));
        assert!(!pack_is_well_formed(
            &pack,
            PACK_ID,
            &[(13, 90, 0, 2, CUBIN_ID), (11, 86, 2, 2, CUBIN_ID)],
        ));
        assert!(!pack_is_well_formed(
            &pack,
            PACK_ID,
            &[(11, 86, 0, 2, CUBIN_ID), (11, 86, 2, 2, CUBIN_ID)],
        ));
        assert!(!pack_is_well_formed(
            &pack,
            PACK_ID,
            &[(11, 86, 0, 0, CUBIN_ID), (13, 90, 0, 4, CUBIN_ID)],
        ));
        assert!(!pack_is_well_formed(
            &pack,
            PACK_ID,
            &[(11, 86, 0, 2, ZERO_IDENTITY), (13, 90, 2, 2, CUBIN_ID)],
        ));
        assert!(!pack_is_well_formed(
            &pack,
            PACK_ID,
            &[(11, 86, 1, 1, CUBIN_ID), (13, 90, 2, 2, CUBIN_ID)],
        ));
        assert!(!pack_is_well_formed(
            &pack,
            PACK_ID,
            &[(11, 86, 0, 2, CUBIN_ID), (13, 90, 2, 3, CUBIN_ID)],
        ));
        assert!(!pack_is_well_formed(
            &pack,
            PACK_ID,
            &[(11, 86, 0, 2, CUBIN_ID)],
        ));
    }

    #[cfg(all(stwo_cuda_link, not(feature = "test-only-empty-aot-pack")))]
    #[test]
    fn default_cuda_build_embeds_every_generated_source_for_every_arch() {
        use std::collections::BTreeSet;

        let source_keys = generated_source_keys();
        let index_keys: BTreeSet<u64> = AOT_INDEX.iter().map(|entry| entry.0).collect();
        let architectures: BTreeSet<u32> = AOT_INDEX.iter().map(|entry| entry.1).collect();

        assert!(!source_keys.is_empty());
        assert!(!architectures.is_empty());
        assert_eq!(index_keys, source_keys);
        assert_eq!(AOT_INDEX.len(), source_keys.len() * architectures.len());
        for cache_key in source_keys {
            for sm in &architectures {
                assert!(aot_pack_contains(cache_key, sm / 10, sm % 10));
            }
        }
    }

    #[cfg(feature = "test-only-empty-aot-pack")]
    #[test]
    fn test_only_pack_is_empty_and_fails_closed() {
        assert!(AOT_PACK.is_empty());
        assert_eq!(aot_pack_entries(), 0);
        assert_eq!(aot_pack_identity(), ZERO_IDENTITY);
        assert_eq!(aot_pack_manifest_hash(), 0);
        assert_eq!(aot_pack_constraint_max_instrs(), 0);
        assert_eq!(aot_pack_constraint_max_live_u32_lanes(), 0);
        assert!(!aot_pack_supports_arch(8, 6));
        assert!(!aot_pack_contains(0, 8, 6));
        assert_eq!(aot_cubin_identity(0, 8, 6), ZERO_IDENTITY);
    }
}
