//! Canonical collision-resistant identities for the embedded AOT pack.
//!
//! This module is shared by `build.rs` and local tests. Runtime code consumes
//! only the generated constants; it never rehashes cubins during proving.

pub(crate) const ZERO_IDENTITY: [u8; 32] = [0; 32];

const CUBIN_DOMAIN: &[u8] = b"stwo-cuda-aot-cubin-identity-v1\0";
const PACK_DOMAIN: &[u8] = b"stwo-cuda-aot-pack-identity-v1\0";

#[derive(Clone, Copy)]
pub(crate) struct CubinIdentityInput<'a> {
    pub cache_key: u64,
    pub sm: u32,
    pub bytes: &'a [u8],
}

pub(crate) fn cubin_identity(input: CubinIdentityInput<'_>) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CUBIN_DOMAIN);
    hasher.update(&input.cache_key.to_le_bytes());
    hasher.update(&input.sm.to_le_bytes());
    hasher.update(&encoded_len(input.bytes));
    hasher.update(input.bytes);
    *hasher.finalize().as_bytes()
}

pub(crate) fn pack_identity(
    constraint_max_instrs: usize,
    constraint_max_live_u32_lanes: usize,
    entries: &[CubinIdentityInput<'_>],
) -> [u8; 32] {
    if entries.is_empty() || constraint_max_instrs == 0 || constraint_max_live_u32_lanes == 0 {
        return ZERO_IDENTITY;
    }
    let mut canonical = entries.iter().collect::<Vec<_>>();
    canonical.sort_by_key(|entry| (entry.cache_key, entry.sm));
    assert!(
        canonical
            .windows(2)
            .all(|pair| (pair[0].cache_key, pair[0].sm) != (pair[1].cache_key, pair[1].sm)),
        "AOT pack contains duplicate (cache_key, SM) entries"
    );

    let mut hasher = blake3::Hasher::new();
    hasher.update(PACK_DOMAIN);
    hasher.update(&encoded_usize(constraint_max_instrs));
    hasher.update(&encoded_usize(constraint_max_live_u32_lanes));
    hasher.update(&encoded_usize(canonical.len()));
    for entry in canonical {
        hasher.update(&entry.cache_key.to_le_bytes());
        hasher.update(&entry.sm.to_le_bytes());
        hasher.update(&encoded_len(entry.bytes));
        hasher.update(entry.bytes);
    }
    *hasher.finalize().as_bytes()
}

fn encoded_len(bytes: &[u8]) -> [u8; 8] {
    encoded_usize(bytes.len())
}

fn encoded_usize(value: usize) -> [u8; 8] {
    u64::try_from(value)
        .expect("AOT identity length fits in u64")
        .to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: CubinIdentityInput<'static> = CubinIdentityInput {
        cache_key: 0x0123_4567_89ab_cdef,
        sm: 86,
        bytes: b"first exact cubin",
    };
    const B: CubinIdentityInput<'static> = CubinIdentityInput {
        cache_key: 0xfedc_ba98_7654_3210,
        sm: 90,
        bytes: b"second exact cubin",
    };

    #[test]
    fn empty_pack_has_no_authority() {
        assert_eq!(pack_identity(1, 1, &[]), ZERO_IDENTITY);
        assert_eq!(pack_identity(0, 1, &[A]), ZERO_IDENTITY);
        assert_eq!(pack_identity(1, 0, &[A]), ZERO_IDENTITY);
    }

    #[test]
    fn pack_identity_is_canonical_and_covers_every_authority_input() {
        let baseline = pack_identity(1_000, 4_096, &[A, B]);
        assert_ne!(baseline, ZERO_IDENTITY);
        assert_eq!(baseline, pack_identity(1_000, 4_096, &[B, A]));
        assert_ne!(baseline, pack_identity(999, 4_096, &[A, B]));
        assert_ne!(baseline, pack_identity(1_000, 4_095, &[A, B]));
        assert_ne!(baseline, pack_identity(1_000, 4_096, &[A]));
        assert_ne!(
            baseline,
            pack_identity(
                1_000,
                4_096,
                &[
                    CubinIdentityInput {
                        bytes: b"mutated exact cubin",
                        ..A
                    },
                    B,
                ],
            )
        );
    }

    #[test]
    fn cubin_identity_covers_key_architecture_length_and_bytes() {
        let baseline = cubin_identity(A);
        assert_ne!(baseline, ZERO_IDENTITY);
        for changed in [
            CubinIdentityInput {
                cache_key: A.cache_key + 1,
                ..A
            },
            CubinIdentityInput { sm: 89, ..A },
            CubinIdentityInput {
                bytes: b"first exact cubin!",
                ..A
            },
            CubinIdentityInput {
                bytes: b"first exact cubim",
                ..A
            },
        ] {
            assert_ne!(baseline, cubin_identity(changed));
        }
    }

    #[test]
    fn canonical_encoding_has_stable_golden_digests() {
        assert_eq!(
            hex(cubin_identity(A)),
            "6e16a27285eaa5ee1b0cfa61ad9b1e8d64cbfefb2789451da4982cf308a368ad"
        );
        assert_eq!(
            hex(pack_identity(1_000, 4_096, &[B, A])),
            "5dddd0bcfc22d3704800da82ed34ea085b51e18fa7efcb5a28b178f4ccaf6ccd"
        );
    }

    #[test]
    #[should_panic(expected = "duplicate (cache_key, SM)")]
    fn duplicate_lookup_key_is_rejected() {
        let _ = pack_identity(1_000, 4_096, &[A, A]);
    }

    fn hex(identity: [u8; 32]) -> String {
        identity.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}
