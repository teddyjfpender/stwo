//! Strict parser for the checked-in AOT source manifest.
//!
//! The generator does not yet emit argument or effect schemas. This module
//! admits only the metadata it does emit and validates the declared exported
//! CUDA symbol against the exact generated source text.

use std::collections::BTreeSet;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SourceManifestEntry {
    pub kind: String,
    pub label: String,
    pub kernel_symbol: String,
    pub cache_key: u64,
    pub semantic_hash: u64,
    pub file: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireEntry {
    kind: String,
    label: String,
    kernel_name: String,
    cache_key: String,
    semantic_hash: String,
    file: String,
}

pub(crate) fn parse_source_manifest(bytes: &[u8]) -> Result<Vec<SourceManifestEntry>, String> {
    let wire = serde_json::from_slice::<Vec<WireEntry>>(bytes)
        .map_err(|error| format!("decode AOT source manifest: {error}"))?;
    if wire.is_empty() {
        return Err("AOT source manifest is empty".to_string());
    }

    let mut entries = Vec::with_capacity(wire.len());
    let mut cache_keys = BTreeSet::new();
    let mut files = BTreeSet::new();
    let mut previous_order = None;
    for (index, entry) in wire.into_iter().enumerate() {
        if !matches!(entry.kind.as_str(), "constraint" | "witness") {
            return Err(format!(
                "AOT source manifest entry {index} has unknown kind {:?}",
                entry.kind
            ));
        }
        if entry.label.is_empty() {
            return Err(format!(
                "AOT source manifest entry {index} has an empty label"
            ));
        }
        if !is_cuda_identifier(&entry.kernel_name) {
            return Err(format!(
                "AOT source manifest entry {index} has an invalid kernel symbol"
            ));
        }
        let cache_key = parse_lower_hex_u64(&entry.cache_key, index, "cache_key")?;
        let semantic_hash = parse_lower_hex_u64(&entry.semantic_hash, index, "semantic_hash")?;
        let expected_file = format!("{}_{}_{cache_key:016x}.cu", entry.kind, entry.label);
        if entry.file != expected_file || !is_plain_file_name(&entry.file) {
            return Err(format!(
                "AOT source manifest entry {index} file does not match its metadata"
            ));
        }
        if !cache_keys.insert(cache_key) {
            return Err(format!(
                "AOT source manifest repeats cache key {cache_key:016x}"
            ));
        }
        if !files.insert(entry.file.clone()) {
            return Err(format!("AOT source manifest repeats file {:?}", entry.file));
        }
        let order = (entry.kind.clone(), entry.label.clone(), cache_key);
        if previous_order
            .as_ref()
            .is_some_and(|previous| previous >= &order)
        {
            return Err("AOT source manifest is not in canonical order".to_string());
        }
        previous_order = Some(order);
        entries.push(SourceManifestEntry {
            kind: entry.kind,
            label: entry.label,
            kernel_symbol: entry.kernel_name,
            cache_key,
            semantic_hash,
            file: entry.file,
        });
    }
    Ok(entries)
}

/// Proves only the source-level exported symbol boundary. Argument types,
/// access ranges and effects remain unavailable until the generator emits a
/// structured schema.
pub(crate) fn validate_exported_kernel_symbol(
    source: &[u8],
    kernel_symbol: &str,
) -> Result<(), String> {
    let source = std::str::from_utf8(source)
        .map_err(|error| format!("generated CUDA source is not UTF-8: {error}"))?;
    let occurrences = source
        .match_indices(kernel_symbol)
        .filter(|(start, _)| {
            let end = start + kernel_symbol.len();
            identifier_boundary(
                start
                    .checked_sub(1)
                    .and_then(|index| source.as_bytes().get(index))
                    .copied(),
            ) && identifier_boundary(source.as_bytes().get(end).copied())
        })
        .map(|(start, _)| start)
        .collect::<Vec<_>>();
    let [symbol_start] = occurrences.as_slice() else {
        return Err(format!(
            "generated CUDA source must contain kernel symbol {kernel_symbol:?} exactly once"
        ));
    };
    let symbol_end = symbol_start + kernel_symbol.len();
    if !source[symbol_end..].trim_start().starts_with('(') {
        return Err(format!(
            "generated CUDA symbol {kernel_symbol:?} is not a function declaration"
        ));
    }
    const EXPORT_PREFIX: &str = "extern \"C\" __global__ void";
    let Some(prefix_start) = source[..*symbol_start].rfind(EXPORT_PREFIX) else {
        return Err(format!(
            "generated CUDA symbol {kernel_symbol:?} is not extern-C global"
        ));
    };
    let between = &source[prefix_start + EXPORT_PREFIX.len()..*symbol_start];
    if between.len() > 128 || between.chars().any(|char| matches!(char, ';' | '{' | '}')) {
        return Err(format!(
            "generated CUDA symbol {kernel_symbol:?} has an unrecognized export declaration"
        ));
    }
    Ok(())
}

fn parse_lower_hex_u64(value: &str, index: usize, field: &'static str) -> Result<u64, String> {
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "AOT source manifest entry {index} has invalid {field}"
        ));
    }
    u64::from_str_radix(value, 16)
        .map_err(|error| format!("parse AOT source manifest {field}: {error}"))
}

fn is_cuda_identifier(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte == b'_' || byte.is_ascii_alphabetic())
        && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
}

fn is_plain_file_name(value: &str) -> bool {
    let path = std::path::Path::new(value);
    path.file_name()
        .is_some_and(|name| name == std::ffi::OsStr::new(value))
        && path.components().count() == 1
        && !value.contains(['/', '\\'])
}

fn identifier_boundary(byte: Option<u8>) -> bool {
    byte.is_none_or(|byte| byte != b'_' && !byte.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Vec<u8> {
        br#"[
          {
            "kind":"constraint",
            "label":"add_ap",
            "kernel_name":"kernel_a",
            "cache_key":"0000000000000001",
            "semantic_hash":"0000000000000002",
            "file":"constraint_add_ap_0000000000000001.cu"
          },
          {
            "kind":"witness",
            "label":"mul",
            "kernel_name":"kernel_b",
            "cache_key":"0000000000000003",
            "semantic_hash":"0000000000000004",
            "file":"witness_mul_0000000000000003.cu"
          }
        ]"#
        .to_vec()
    }

    #[test]
    fn strict_manifest_parser_accepts_only_canonical_complete_entries() {
        let parsed = parse_source_manifest(&manifest()).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].cache_key, 1);
        assert_eq!(parsed[1].semantic_hash, 4);

        for mutation in [
            ("\"cache_key\":\"0000000000000001\"", "\"cache_key\":\"1\""),
            (
                "\"semantic_hash\":\"0000000000000002\"",
                "\"semantic_hash\":\"000000000000000G\"",
            ),
            (
                "\"kernel_name\":\"kernel_a\"",
                "\"kernel_name\":\"bad-kernel\"",
            ),
            ("\"kind\":\"constraint\"", "\"kind\":\"unknown\""),
            (
                "\"file\":\"constraint_add_ap_0000000000000001.cu\"",
                "\"file\":\"../kernel.cu\"",
            ),
        ] {
            let changed = String::from_utf8(manifest())
                .unwrap()
                .replace(mutation.0, mutation.1);
            assert!(parse_source_manifest(changed.as_bytes()).is_err());
        }
    }

    #[test]
    fn duplicate_or_reordered_entries_are_rejected() {
        let original = String::from_utf8(manifest()).unwrap();
        let duplicate_key = original.replace(
            "\"cache_key\":\"0000000000000003\"",
            "\"cache_key\":\"0000000000000001\"",
        );
        assert!(parse_source_manifest(duplicate_key.as_bytes()).is_err());

        let entries = serde_json::from_str::<Vec<serde_json::Value>>(&original).unwrap();
        let reordered = serde_json::to_vec(&[entries[1].clone(), entries[0].clone()]).unwrap();
        assert!(parse_source_manifest(&reordered).is_err());
    }

    #[test]
    fn source_symbol_validation_is_narrow_and_fail_closed() {
        let source = br#"
            extern "C" __global__ void __launch_bounds__(256) kernel_a(
                const unsigned *input,
                unsigned *output) {}
        "#;
        validate_exported_kernel_symbol(source, "kernel_a").unwrap();
        for invalid in [
            br#"__global__ void kernel_a() {}"#.as_slice(),
            br#"extern "C" __global__ void kernel_a;"#.as_slice(),
            br#"extern "C" __global__ void kernel_a() {} kernel_a"#.as_slice(),
        ] {
            assert!(validate_exported_kernel_symbol(invalid, "kernel_a").is_err());
        }
    }

    #[test]
    fn checked_in_manifest_and_every_declared_source_are_exact() {
        let generated = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("cuda")
            .join("generated");
        let manifest = std::fs::read(generated.join("aot_manifest.json")).unwrap();
        let entries = parse_source_manifest(&manifest).unwrap();
        let source_files = std::fs::read_dir(&generated)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "cu"))
            .count();
        assert_eq!(entries.len(), source_files);
        for entry in entries {
            let source = std::fs::read(generated.join(&entry.file)).unwrap();
            validate_exported_kernel_symbol(&source, &entry.kernel_symbol).unwrap();
        }
    }
}
