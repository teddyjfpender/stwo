//! Canonical collision-resistant identities for the embedded AOT pack.
//!
//! This module is shared by `build.rs` and local tests. Runtime code consumes
//! only the generated constants; it never rehashes cubins during proving.

pub(crate) const ZERO_IDENTITY: [u8; 32] = [0; 32];

const CUBIN_DOMAIN: &[u8] = b"stwo-cuda-aot-cubin-identity-v1\0";
// Shared with build.rs, which is a separate crate; the runtime cannot rehash
// source bytes because they are deliberately not embedded beside the cubins.
#[allow(dead_code)]
const SOURCE_DOMAIN: &[u8] = b"stwo-cuda-aot-source-identity-v1\0";
const ABI_SCHEMA_DOMAIN: &[u8] = b"stwo-cuda-aot-abi-schema-v1\0";
const KERNEL_AUTHORITY_DOMAIN: &[u8] = b"stwo-cuda-aot-kernel-authority-v2\0";
const PACK_DOMAIN: &[u8] = b"stwo-cuda-aot-pack-identity-v1\0";

/// Strength of the generator-owned schema bound into one authority entry.
/// Unsupported families remain exported-symbol-only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
#[repr(u8)]
pub enum AotKernelSchemaScope {
    ExportedSymbolOnly = 1,
    StructuredAbi = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AotKernelAbiKind {
    U32 = 1,
    DevicePointerU32 = 2,
    DevicePointerTableU32 = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AotKernelAbiAccess {
    Read = 1,
    Write = 2,
    ReadWrite = 3,
    LaunchRowCount = 4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AotKernelAbiArgument {
    pub ordinal: u8,
    pub name: &'static str,
    pub kind: AotKernelAbiKind,
    /// Coarse pointee access only. This does not authorize extents or aliases.
    pub access: AotKernelAbiAccess,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AotKernelAbiSchema {
    RecordedWitnessV1,
}

impl AotKernelAbiSchema {
    pub const fn manifest_tag(self) -> &'static str {
        match self {
            Self::RecordedWitnessV1 => "recorded_witness_v1",
        }
    }

    pub const fn family(self) -> &'static str {
        match self {
            Self::RecordedWitnessV1 => "recorded_witness",
        }
    }

    pub const fn version(self) -> u32 {
        match self {
            Self::RecordedWitnessV1 => 1,
        }
    }

    pub const fn arguments(self) -> &'static [AotKernelAbiArgument] {
        match self {
            Self::RecordedWitnessV1 => &RECORDED_WITNESS_V1_ARGUMENTS,
        }
    }

    pub fn identity(self) -> [u8; 32] {
        abi_schema_identity(self)
    }
}

const RECORDED_WITNESS_V1_ARGUMENTS: [AotKernelAbiArgument; 8] = [
    abi_argument(
        0,
        "input_cols",
        AotKernelAbiKind::DevicePointerTableU32,
        AotKernelAbiAccess::Read,
    ),
    abi_argument(
        1,
        "table_bases",
        AotKernelAbiKind::DevicePointerTableU32,
        AotKernelAbiAccess::Read,
    ),
    abi_argument(
        2,
        "table_strides",
        AotKernelAbiKind::DevicePointerU32,
        AotKernelAbiAccess::Read,
    ),
    abi_argument(
        3,
        "out_cols",
        AotKernelAbiKind::DevicePointerTableU32,
        AotKernelAbiAccess::Write,
    ),
    abi_argument(
        4,
        "mult_counts",
        AotKernelAbiKind::DevicePointerTableU32,
        AotKernelAbiAccess::ReadWrite,
    ),
    abi_argument(
        5,
        "lookup_words",
        AotKernelAbiKind::DevicePointerU32,
        AotKernelAbiAccess::Write,
    ),
    abi_argument(
        6,
        "sub_words",
        AotKernelAbiKind::DevicePointerU32,
        AotKernelAbiAccess::Write,
    ),
    abi_argument(
        7,
        "row_count",
        AotKernelAbiKind::U32,
        AotKernelAbiAccess::LaunchRowCount,
    ),
];

const fn abi_argument(
    ordinal: u8,
    name: &'static str,
    kind: AotKernelAbiKind,
    access: AotKernelAbiAccess,
) -> AotKernelAbiArgument {
    AotKernelAbiArgument {
        ordinal,
        name,
        kind,
        access,
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CubinIdentityInput<'a> {
    pub cache_key: u64,
    pub sm: u32,
    pub bytes: &'a [u8],
}

#[derive(Clone, Copy)]
pub(crate) struct KernelAuthorityIdentityInput<'a> {
    pub source_identity: [u8; 32],
    pub kernel_symbol: &'a str,
    pub semantic_hash: u64,
    pub cache_key: u64,
    pub sm: u32,
    pub cubin_identity: [u8; 32],
    pub abi_schema_identity: [u8; 32],
    pub program_identity: [u8; 32],
    pub schema_scope: AotKernelSchemaScope,
}

pub(crate) fn abi_schema_identity(schema: AotKernelAbiSchema) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(ABI_SCHEMA_DOMAIN);
    hasher.update(&encoded_len(schema.family().as_bytes()));
    hasher.update(schema.family().as_bytes());
    hasher.update(&schema.version().to_le_bytes());
    hasher.update(&encoded_usize(schema.arguments().len()));
    for argument in schema.arguments() {
        hasher.update(&[argument.ordinal]);
        hasher.update(&encoded_len(argument.name.as_bytes()));
        hasher.update(argument.name.as_bytes());
        hasher.update(&[argument.kind as u8, argument.access as u8]);
    }
    *hasher.finalize().as_bytes()
}

#[allow(dead_code)]
pub(crate) fn source_identity(source: &[u8]) -> [u8; 32] {
    if source.is_empty() {
        return ZERO_IDENTITY;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(SOURCE_DOMAIN);
    hasher.update(&encoded_len(source));
    hasher.update(source);
    *hasher.finalize().as_bytes()
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

pub(crate) fn kernel_authority_identity(input: KernelAuthorityIdentityInput<'_>) -> [u8; 32] {
    if input.source_identity == ZERO_IDENTITY
        || input.kernel_symbol.is_empty()
        || input.sm == 0
        || input.cubin_identity == ZERO_IDENTITY
        || match input.schema_scope {
            AotKernelSchemaScope::ExportedSymbolOnly => input.abi_schema_identity != ZERO_IDENTITY,
            AotKernelSchemaScope::StructuredAbi => {
                input.abi_schema_identity == ZERO_IDENTITY
                    || input.program_identity == ZERO_IDENTITY
            }
        }
        || (input.schema_scope == AotKernelSchemaScope::ExportedSymbolOnly
            && input.program_identity != ZERO_IDENTITY)
    {
        return ZERO_IDENTITY;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(KERNEL_AUTHORITY_DOMAIN);
    hasher.update(&input.source_identity);
    hasher.update(&encoded_len(input.kernel_symbol.as_bytes()));
    hasher.update(input.kernel_symbol.as_bytes());
    hasher.update(&input.semantic_hash.to_le_bytes());
    hasher.update(&input.cache_key.to_le_bytes());
    hasher.update(&input.sm.to_le_bytes());
    hasher.update(&input.cubin_identity);
    hasher.update(&input.abi_schema_identity);
    hasher.update(&input.program_identity);
    hasher.update(&[input.schema_scope as u8]);
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

    fn authority() -> KernelAuthorityIdentityInput<'static> {
        KernelAuthorityIdentityInput {
            source_identity: source_identity(b"exact generated CUDA source"),
            kernel_symbol: "stwo_jit_fused_example",
            semantic_hash: 0x1234_5678_9abc_def0,
            cache_key: A.cache_key,
            sm: A.sm,
            cubin_identity: cubin_identity(A),
            abi_schema_identity: ZERO_IDENTITY,
            program_identity: ZERO_IDENTITY,
            schema_scope: AotKernelSchemaScope::ExportedSymbolOnly,
        }
    }

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
    fn source_identity_covers_length_and_exact_bytes() {
        let baseline = source_identity(b"exact generated CUDA source");
        assert_ne!(baseline, ZERO_IDENTITY);
        assert_eq!(source_identity(b""), ZERO_IDENTITY);
        assert_ne!(baseline, source_identity(b"exact generated CUDA source!"));
        assert_ne!(baseline, source_identity(b"exact generated CUDA sourcf"));
    }

    #[test]
    fn recorded_witness_schema_is_ordinal_complete_and_typed() {
        let schema = AotKernelAbiSchema::RecordedWitnessV1;
        assert_eq!(schema.family(), "recorded_witness");
        assert_eq!(schema.version(), 1);
        assert_eq!(schema.arguments().len(), 8);
        for (ordinal, argument) in schema.arguments().iter().enumerate() {
            assert_eq!(usize::from(argument.ordinal), ordinal);
        }
        assert_eq!(schema.arguments()[0].name, "input_cols");
        assert_eq!(schema.arguments()[4].access, AotKernelAbiAccess::ReadWrite);
        assert_eq!(schema.arguments()[7].kind, AotKernelAbiKind::U32);
        assert_eq!(
            schema.arguments()[7].access,
            AotKernelAbiAccess::LaunchRowCount
        );
        assert_ne!(abi_schema_identity(schema), ZERO_IDENTITY);
    }

    #[test]
    fn kernel_authority_covers_every_canonical_entry_field() {
        let baseline = kernel_authority_identity(authority());
        assert_ne!(baseline, ZERO_IDENTITY);
        for changed in [
            KernelAuthorityIdentityInput {
                source_identity: source_identity(b"changed generated CUDA source"),
                ..authority()
            },
            KernelAuthorityIdentityInput {
                kernel_symbol: "stwo_jit_fused_changed",
                ..authority()
            },
            KernelAuthorityIdentityInput {
                semantic_hash: authority().semantic_hash + 1,
                ..authority()
            },
            KernelAuthorityIdentityInput {
                cache_key: authority().cache_key + 1,
                ..authority()
            },
            KernelAuthorityIdentityInput {
                sm: 89,
                ..authority()
            },
            KernelAuthorityIdentityInput {
                cubin_identity: cubin_identity(CubinIdentityInput {
                    bytes: b"changed cubin",
                    ..A
                }),
                ..authority()
            },
            KernelAuthorityIdentityInput {
                abi_schema_identity: abi_schema_identity(AotKernelAbiSchema::RecordedWitnessV1),
                program_identity: [7; 32],
                schema_scope: AotKernelSchemaScope::StructuredAbi,
                ..authority()
            },
        ] {
            assert_ne!(baseline, kernel_authority_identity(changed));
        }
        assert_eq!(
            kernel_authority_identity(KernelAuthorityIdentityInput {
                source_identity: ZERO_IDENTITY,
                ..authority()
            }),
            ZERO_IDENTITY
        );
        assert_eq!(
            kernel_authority_identity(KernelAuthorityIdentityInput {
                abi_schema_identity: ZERO_IDENTITY,
                program_identity: [7; 32],
                schema_scope: AotKernelSchemaScope::StructuredAbi,
                ..authority()
            }),
            ZERO_IDENTITY
        );
        assert_eq!(
            kernel_authority_identity(KernelAuthorityIdentityInput {
                abi_schema_identity: abi_schema_identity(AotKernelAbiSchema::RecordedWitnessV1),
                program_identity: ZERO_IDENTITY,
                schema_scope: AotKernelSchemaScope::StructuredAbi,
                ..authority()
            }),
            ZERO_IDENTITY
        );
        assert_eq!(
            kernel_authority_identity(KernelAuthorityIdentityInput {
                abi_schema_identity: abi_schema_identity(AotKernelAbiSchema::RecordedWitnessV1),
                program_identity: [7; 32],
                ..authority()
            }),
            ZERO_IDENTITY
        );
        assert_eq!(
            kernel_authority_identity(KernelAuthorityIdentityInput {
                cubin_identity: ZERO_IDENTITY,
                ..authority()
            }),
            ZERO_IDENTITY
        );
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
        assert_eq!(
            hex(source_identity(b"exact generated CUDA source")),
            "54ff96d5bccdee47e3e6ef59e21d2a954600daf99203864a8e8c6965fceb638b"
        );
        assert_eq!(
            hex(kernel_authority_identity(authority())),
            "a81919754481adaba8dbb3c903f3be0a694794a890ecd258c82887656eec1c41"
        );
        assert_eq!(
            hex(abi_schema_identity(AotKernelAbiSchema::RecordedWitnessV1)),
            "0a9f1fe69907e769f718018a09642ec9a4597179c35d24d75e581ad6ea4f2744"
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
