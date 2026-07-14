"""Focused hostile tests for the non-admitting PIE adapter build receipt."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

from .common import canonical_bytes, require, sha256_bytes
from .pie_adapter_receipt import (
    BUILD_RECEIPT_SCHEMA,
    DISCOVERY_FILES,
    MAX_JSON_BYTES,
    MAX_OUTPUT_BYTES,
    MAX_SOURCES,
    discover_expected_sources,
    generate,
    parse_build_receipt_bytes,
    parse_inventory_bytes,
    parse_source_closure_bytes,
    validate_build_receipt,
    validate_documents,
    validate_files,
    validate_inventory,
    validate_source_closure,
)
from .pie_adapter_source_closure import generate_source_closure


COMMITS = {"stwo": "1" * 40, "stwo-cairo": "2" * 40}


def _commit(root: Path) -> str:
    return COMMITS[root.name]


def _write(path: Path, payload: bytes, mode: int = 0o644) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return path


def _workspace(root: Path) -> tuple[dict, Path]:
    for repository in ("stwo", "stwo-cairo"):
        (root / repository).mkdir(parents=True)
    for repository, relative in DISCOVERY_FILES:
        _write(root / repository / relative, f"{repository}:{relative}\n".encode())

    target = "gpu_benchmarks/lab/pie-adapter/receipt-targets/test-run"
    executable = f"{target}/x86_64-unknown-linux-gnu/release/stwo-gpu-lab-pie-adapter"
    _write(root / "stwo-cairo" / executable, b"host adapter executable\n", 0o755)
    sources = discover_expected_sources(root)
    manifest = {"repository": "stwo-cairo",
                "path": "gpu_benchmarks/lab/pie-adapter/Cargo.toml"}
    inventory = {
        "schema_version": "stwo.gpu-lab.pie-adapter-source-inventory.v2",
        "repository_roots": {"stwo": "stwo", "stwo-cairo": "stwo-cairo"},
        "expected_sources": sources,
        "build": {
            "manifest": manifest,
            "cargo_lock": {"repository": "stwo-cairo",
                           "path": "gpu_benchmarks/lab/pie-adapter/Cargo.lock"},
            "rust_toolchain": {"repository": "stwo-cairo",
                               "path": "gpu_benchmarks/lab/pie-adapter/rust-toolchain.toml"},
            "profile": "release", "features": [], "default_features": True,
            "target": "x86_64-unknown-linux-gnu",
            "target_directory": {"repository": "stwo-cairo", "path": target},
            "working_directory": "<workspace>/stwo-cairo/gpu_benchmarks/lab/pie-adapter",
            "environment": {"CARGO_TARGET_DIR": f"<workspace>/stwo-cairo/{target}"},
            "command": [
                "cargo", "build", "--manifest-path", "Cargo.toml",
                "--release", "--locked", "--offline", "--target",
                "x86_64-unknown-linux-gnu",
            ],
        },
        "generated_outputs": [{"repository": "stwo-cairo", "path": executable}],
        "adapter_executable": {"repository": "stwo-cairo", "path": executable},
    }
    return inventory, root / "stwo-cairo" / executable


def _expect(label: str, operation: Callable[[], object], fragment: str | None = None) -> None:
    try:
        operation()
    except (OSError, ValueError) as error:
        require(fragment is None or fragment in str(error),
                f"{label} failed for the wrong reason: {error}")
        return
    raise ValueError(f"{label} was accepted")


def _inventory_bytes(value: dict) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"


def _rehash(document: dict, body_key: str, hash_key: str) -> dict:
    mutated = copy.deepcopy(document)
    mutated[hash_key] = sha256_bytes(canonical_bytes(mutated[body_key]))
    return mutated


def _schema_checks(inventory: dict, closure: dict, receipt: dict) -> None:
    schema_root = Path(__file__).resolve().parents[2] / "schemas"
    schemas = {
        "inventory": json.loads((schema_root / "pie-adapter-source-inventory-v2.schema.json").read_text()),
        "closure": json.loads((schema_root / "pie-adapter-source-closure-v2.schema.json").read_text()),
        "receipt": json.loads((schema_root / "pie-adapter-build-receipt-v2.schema.json").read_text()),
    }
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        require(all(schema.get("$schema", "").endswith("2020-12/schema")
                    for schema in schemas.values()), "PIE adapter schemas are not Draft 2020-12")
        require(all("pattern" in schema["$defs"]["relative_path"]
                    for schema in schemas.values()), "PIE adapter schema path parity is missing")
        return
    validators = {name: Draft202012Validator(schema) for name, schema in schemas.items()}
    for validator in validators.values():
        validator.check_schema(validator.schema)
    validators["inventory"].validate(inventory)
    validators["closure"].validate(closure)
    validators["receipt"].validate(receipt)
    hostile: list[tuple[str, dict, str]] = []
    for name, base, route in (("inventory path", inventory, "inventory"),
                              ("closure path", closure, "closure"),
                              ("receipt path", receipt, "receipt")):
        value = copy.deepcopy(base)
        if route == "inventory": value["expected_sources"][0]["path"] = "../escape"
        elif route == "closure": value["source_closure"]["sources"][0]["path"] = "a/../b"
        else: value["build_receipt"]["generated_outputs"][0]["path"] = "a//b"
        hostile.append((name, value, route))
    for label, mutate in (
        ("environment extra", lambda value: value["build_receipt"]["build_inputs"][
            "environment"].update({"RUSTFLAGS": "-Ctarget-cpu=native"})),
        ("command control", lambda value: value["build_receipt"]["build_inputs"][
            "command"].__setitem__(0, "cargo\n")),
        ("feature language", lambda value: value["build_receipt"]["build_inputs"].update(
            {"features": ["bad/feature"]})),
        ("mode language", lambda value: value["build_receipt"]["adapter_executable"].update(
            {"mode": "755"})),
    ):
        value = copy.deepcopy(receipt)
        mutate(value)
        hostile.append((label, value, "receipt"))
    for label, value, route in hostile:
        require(not validators[route].is_valid(value),
                f"PIE adapter Draft 2020-12 schema accepted {label}")


def pie_adapter_receipt_self_test(_: Path | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="gpu-lab-pie-receipt-") as temporary:
        root = Path(temporary) / "workspace"
        inventory, executable = _workspace(root)
        inventory_path = Path(temporary) / "inventory.json"
        inventory_path.write_bytes(_inventory_bytes(inventory))
        inventory_sha = sha256_bytes(inventory_path.read_bytes())
        closure_path = Path(temporary) / "closure.json"
        receipt_path = Path(temporary) / "receipt.json"
        executable.unlink()
        prebuild_closure_path = Path(temporary) / "prebuild-closure.json"
        prebuild_closure = generate_source_closure(
            inventory_path, inventory_sha, prebuild_closure_path, root, _commit,
        )
        require(prebuild_closure_path.stat().st_mode & 0o777 == 0o400,
                "PIE adapter prebuild closure is mutable")
        _write(executable, b"host adapter executable\n", 0o755)
        closure, receipt = generate(inventory_path, inventory_sha, closure_path,
                                    receipt_path, root, _commit)
        require(prebuild_closure == closure,
                "PIE adapter prebuild and post-build source closures differ")
        retained = validate_documents(
            inventory_path.read_bytes(), inventory_sha, closure_path.read_bytes(),
            receipt_path.read_bytes(), root, _commit)
        require(retained == (closure, receipt),
                "PIE adapter retained document validation differs")
        _expect("retained inventory hash substitution", lambda: validate_documents(
            inventory_path.read_bytes(), "0" * 64, closure_path.read_bytes(),
            receipt_path.read_bytes(), root, _commit), "out-of-band")
        validate_files(inventory_path, inventory_sha, closure_path, receipt_path, root, _commit)
        require(receipt["schema_version"] == BUILD_RECEIPT_SCHEMA
                and receipt["build_receipt"]["build_execution_attested"] is False
                and receipt["build_receipt"]["unattested_build_inputs"]
                == "cargo-config-compiler-and-user-environment-not-execution-attested-v1"
                and all(receipt["build_receipt"][key] is False for key in
                        ("production_admissible", "correctness_admissible",
                         "performance_admissible")),
                "PIE adapter receipt overclaims evidence")
        _schema_checks(inventory, closure, receipt)

        # A self-consistent attacker cannot erase a mandatory crate entry point from
        # disk, inventory, closure, and receipt: every exported boundary rediscovers.
        missing = root / "stwo-cairo/stwo_cairo_prover/crates/common/src/lib.rs"
        missing_bytes = missing.read_bytes()
        missing.unlink()
        reduced_inventory = copy.deepcopy(inventory)
        reduced_inventory["expected_sources"] = [
            item for item in reduced_inventory["expected_sources"]
            if not (item["repository"] == "stwo-cairo"
                    and item["path"] == "stwo_cairo_prover/crates/common/src/lib.rs")
        ]
        reduced_inventory_bytes = _inventory_bytes(reduced_inventory)
        reduced_inventory_raw_sha = sha256_bytes(reduced_inventory_bytes)
        reduced_inventory_sha = sha256_bytes(canonical_bytes(reduced_inventory))
        reduced_closure = copy.deepcopy(closure)
        reduced_closure["source_closure"]["inventory_sha256"] = reduced_inventory_sha
        reduced_closure["source_closure"]["sources"] = [
            item for item in reduced_closure["source_closure"]["sources"]
            if not (item["repository"] == "stwo-cairo"
                    and item["path"] == "stwo_cairo_prover/crates/common/src/lib.rs")
        ]
        reduced_closure = _rehash(
            reduced_closure, "source_closure", "source_closure_sha256"
        )
        reduced_receipt = copy.deepcopy(receipt)
        reduced_receipt["build_receipt"]["inventory_sha256"] = reduced_inventory_sha
        reduced_receipt["build_receipt"]["source_closure_sha256"] = (
            reduced_closure["source_closure_sha256"]
        )
        reduced_receipt = _rehash(
            reduced_receipt, "build_receipt", "build_receipt_sha256"
        )
        reduced_inventory_path = Path(temporary) / "reduced-inventory.json"
        reduced_closure_path = Path(temporary) / "reduced-closure.json"
        reduced_receipt_path = Path(temporary) / "reduced-receipt.json"
        reduced_inventory_path.write_bytes(reduced_inventory_bytes)
        reduced_closure_path.write_bytes(_inventory_bytes(reduced_closure))
        reduced_receipt_path.write_bytes(_inventory_bytes(reduced_receipt))
        hostile_routes = (
            lambda: validate_inventory(reduced_inventory, root),
            lambda: parse_inventory_bytes(
                reduced_inventory_bytes, reduced_inventory_raw_sha, root),
            lambda: validate_source_closure(
                reduced_closure, reduced_inventory, root, _commit),
            lambda: parse_source_closure_bytes(
                _inventory_bytes(reduced_closure), reduced_inventory, root, _commit),
            lambda: validate_build_receipt(
                reduced_receipt, reduced_inventory, reduced_closure, root, _commit),
            lambda: parse_build_receipt_bytes(
                _inventory_bytes(reduced_receipt), reduced_inventory,
                reduced_closure, root, _commit),
            lambda: generate(
                reduced_inventory_path, reduced_inventory_raw_sha,
                Path(temporary) / "reduced-generated-closure.json",
                Path(temporary) / "reduced-generated-receipt.json", root, _commit),
            lambda: validate_files(
                reduced_inventory_path, reduced_inventory_raw_sha,
                reduced_closure_path, reduced_receipt_path, root, _commit),
        )
        for index, route in enumerate(hostile_routes):
            _expect(f"deleted mandatory source route {index}", route)
        _write(missing, missing_bytes)

        _expect("wrong out-of-band inventory hash", lambda: parse_inventory_bytes(
            inventory_path.read_bytes(), "0" * 64, root), "out-of-band")
        _expect("duplicate inventory JSON key", lambda: parse_inventory_bytes(
            b'{"schema_version":1,"schema_version":2}',
            sha256_bytes(b'{"schema_version":1,"schema_version":2}'), root), "duplicate")
        unknown = copy.deepcopy(inventory)
        unknown["candidate"] = True
        _expect("unknown inventory key", lambda: validate_inventory(unknown, root), "keys differ")
        omitted = copy.deepcopy(inventory)
        omitted["expected_sources"].pop()
        _expect("inventory omission", lambda: validate_inventory(omitted, root), "discovery")
        traversal = copy.deepcopy(inventory)
        traversal["expected_sources"][0]["path"] = "../escape.rs"
        _expect("inventory traversal", lambda: validate_inventory(traversal, root), "ambiguous")
        unsorted = copy.deepcopy(inventory)
        unsorted["expected_sources"][:2] = reversed(unsorted["expected_sources"][:2])
        _expect("unsorted inventory", lambda: validate_inventory(unsorted, root), "sorted")
        duplicate = copy.deepcopy(inventory)
        duplicate["expected_sources"][1] = duplicate["expected_sources"][0]
        _expect("duplicate inventory", lambda: validate_inventory(duplicate, root), "distinct")
        excessive = copy.deepcopy(inventory)
        excessive["expected_sources"] = [
            {"repository": "stwo", "path": f"src/{index:04d}.rs"}
            for index in range(MAX_SOURCES + 1)
        ]
        _expect("excessive inventory", lambda: validate_inventory(excessive, root), "count")
        bad_command = copy.deepcopy(inventory)
        bad_command["build"]["command"].remove("--offline")
        _expect("non-offline build command", lambda: validate_inventory(bad_command, root),
                "canonical")
        bad_environment = copy.deepcopy(inventory)
        bad_environment["build"]["environment"]["RUSTFLAGS"] = "-Ctarget-cpu=native"
        _expect("extra build environment", lambda: validate_inventory(bad_environment, root),
                "environment")
        bad_profile = copy.deepcopy(inventory)
        bad_profile["build"]["profile"] = "release debug"
        _expect("bad profile language", lambda: validate_inventory(bad_profile, root),
                "profile")
        bad_feature = copy.deepcopy(inventory)
        bad_feature["build"]["features"] = ["bad/feature"]
        _expect("bad feature language", lambda: validate_inventory(bad_feature, root),
                "feature")
        bad_target = copy.deepcopy(inventory)
        bad_target["build"]["target"] = "x86_64/escape"
        _expect("bad target language", lambda: validate_inventory(bad_target, root),
                "target")
        fabricated_target = copy.deepcopy(inventory)
        fabricated = "riscv64gc-unknown-linux-gnu"
        fabricated_target["build"]["target"] = fabricated
        fabricated_target["build"]["command"][-1] = fabricated
        old_executable = fabricated_target["adapter_executable"]["path"]
        fabricated_executable = old_executable.replace(
            "x86_64-unknown-linux-gnu", fabricated,
        )
        fabricated_target["generated_outputs"][0]["path"] = fabricated_executable
        fabricated_target["adapter_executable"]["path"] = fabricated_executable
        _expect("fabricated Linux target", lambda: validate_inventory(
            fabricated_target, root), "allowlisted")
        hostile_namespace = copy.deepcopy(inventory)
        hostile_target_dir = ".git/objects/fresh-target"
        hostile_namespace["build"]["target_directory"]["path"] = hostile_target_dir
        hostile_namespace["build"]["environment"]["CARGO_TARGET_DIR"] = (
            f"<workspace>/stwo-cairo/{hostile_target_dir}"
        )
        hostile_executable = (
            f"{hostile_target_dir}/x86_64-unknown-linux-gnu/release/"
            "stwo-gpu-lab-pie-adapter"
        )
        hostile_namespace["generated_outputs"][0]["path"] = hostile_executable
        hostile_namespace["adapter_executable"]["path"] = hostile_executable
        _expect("self-consistent hostile target namespace", lambda: validate_inventory(
            hostile_namespace, root), "receipt-targets namespace")
        wrong_executable = copy.deepcopy(inventory)
        wrong_path = wrong_executable["adapter_executable"]["path"].replace(
            "stwo-gpu-lab-pie-adapter", "alternate-adapter",
        )
        wrong_executable["generated_outputs"][0]["path"] = wrong_path
        wrong_executable["adapter_executable"]["path"] = wrong_path
        _expect("self-consistent alternate executable", lambda: validate_inventory(
            wrong_executable, root), "exact Cargo release output")
        missing_lock = copy.deepcopy(inventory)
        missing_lock["expected_sources"] = [item for item in missing_lock["expected_sources"]
                                            if not item["path"].endswith("Cargo.lock")]
        _expect("omitted Cargo lock", lambda: validate_inventory(missing_lock, root), "discovery")

        extra = root / "stwo" / "crates/stwo/src/new_input.rs"
        _write(extra, b"pub fn new_input() {}\n")
        _expect("live unlisted source", lambda: validate_inventory(inventory, root), "discovery")
        extra.unlink()
        symlink = root / "stwo" / "crates/stwo/src/link.rs"
        symlink.symlink_to(root / "stwo" / "crates/stwo/src/lib.rs")
        _expect("source symlink", lambda: validate_inventory(inventory, root), "non-regular")
        symlink.unlink()
        alias = root / "stwo" / "crates/stwo/src/alias.rs"
        os.link(root / "stwo" / "crates/stwo/src/lib.rs", alias)
        aliased_inventory = copy.deepcopy(inventory)
        aliased_inventory["expected_sources"] = discover_expected_sources(root)
        alias_inventory_path = Path(temporary) / "alias-inventory.json"
        alias_inventory_path.write_bytes(_inventory_bytes(aliased_inventory))
        alias_inventory_sha = sha256_bytes(alias_inventory_path.read_bytes())
        _expect("source inode alias", lambda: generate(
            alias_inventory_path, alias_inventory_sha, Path(temporary) / "alias-closure.json",
            Path(temporary) / "alias-receipt.json", root, _commit), "unsafe")
        alias.unlink()

        unknown_closure = copy.deepcopy(closure)
        unknown_closure["source_closure"]["candidate"] = True
        unknown_closure = _rehash(unknown_closure, "source_closure", "source_closure_sha256")
        _expect("unknown closure field", lambda: validate_source_closure(
            unknown_closure, inventory, root, _commit), "keys differ")
        bad_closure_hash = copy.deepcopy(closure)
        bad_closure_hash["source_closure_sha256"] = "0" * 64
        _expect("bad closure canonical hash", lambda: validate_source_closure(
            bad_closure_hash, inventory, root, _commit), "canonical hash")
        omitted_source = copy.deepcopy(closure)
        omitted_source["source_closure"]["sources"].pop()
        omitted_source = _rehash(omitted_source, "source_closure", "source_closure_sha256")
        _expect("closure source omission", lambda: validate_source_closure(
            omitted_source, inventory, root, _commit), "live expected")
        _expect("repository commit mutation", lambda: validate_source_closure(
            closure, inventory, root,
            lambda repository: "3" * 40 if repository.name == "stwo" else COMMITS[repository.name]),
            "live expected")

        unknown_receipt = copy.deepcopy(receipt)
        unknown_receipt["build_receipt"]["candidate"] = True
        unknown_receipt = _rehash(unknown_receipt, "build_receipt", "build_receipt_sha256")
        _expect("unknown receipt field", lambda: validate_build_receipt(
            unknown_receipt, inventory, closure, root, _commit), "keys differ")
        bad_receipt_hash = copy.deepcopy(receipt)
        bad_receipt_hash["build_receipt_sha256"] = "0" * 64
        _expect("bad receipt canonical hash", lambda: validate_build_receipt(
            bad_receipt_hash, inventory, closure, root, _commit), "canonical hash")
        oversized = copy.deepcopy(receipt)
        oversized["build_receipt"]["generated_outputs"][0]["byte_length"] = MAX_OUTPUT_BYTES + 1
        oversized["build_receipt"]["adapter_executable"]["byte_length"] = MAX_OUTPUT_BYTES + 1
        oversized = _rehash(oversized, "build_receipt", "build_receipt_sha256")
        _expect("oversized output identity", lambda: validate_build_receipt(
            oversized, inventory, closure, root, _commit), "byte bound")
        overclaim = copy.deepcopy(receipt)
        overclaim["build_receipt"]["production_admissible"] = True
        overclaim = _rehash(overclaim, "build_receipt", "build_receipt_sha256")
        _expect("admission overclaim", lambda: validate_build_receipt(
            overclaim, inventory, closure, root, _commit), "overclaims")
        bad_mode = copy.deepcopy(receipt)
        bad_mode["build_receipt"]["generated_outputs"][0]["mode"] = "755"
        bad_mode["build_receipt"]["adapter_executable"]["mode"] = "755"
        bad_mode = _rehash(bad_mode, "build_receipt", "build_receipt_sha256")
        _expect("bad output mode language", lambda: validate_build_receipt(
            bad_mode, inventory, closure, root, _commit), "mode")

        original = executable.read_bytes()
        executable.chmod(0o755)
        executable.write_bytes(original + b"mutation")
        _expect("mutated executable", lambda: validate_files(
            inventory_path, inventory_sha, closure_path, receipt_path, root, _commit), "outputs")
        executable.write_bytes(original)
        executable.chmod(0o755)
        second_output = executable.with_name("adapter-hardlink")
        os.link(executable, second_output)
        output_alias_inventory = copy.deepcopy(inventory)
        output_alias_inventory["generated_outputs"].append({
            "repository": "stwo-cairo",
            "path": str(second_output.relative_to(root / "stwo-cairo")),
        })
        output_alias_inventory["generated_outputs"].sort(
            key=lambda item: (item["repository"], item["path"])
        )
        output_alias_path = Path(temporary) / "output-alias-inventory.json"
        output_alias_path.write_bytes(_inventory_bytes(output_alias_inventory))
        output_alias_sha = sha256_bytes(output_alias_path.read_bytes())
        _expect("generated output inode alias", lambda: generate(
            output_alias_path, output_alias_sha, Path(temporary) / "output-alias-closure.json",
            Path(temporary) / "output-alias-receipt.json", root, _commit), "unsafe")
        second_output.unlink()
        source = root / "stwo" / "crates/stwo/src/lib.rs"
        source_original = source.read_bytes()
        source.write_bytes(source_original + b"// mutation\n")
        _expect("mutated source", lambda: validate_files(
            inventory_path, inventory_sha, closure_path, receipt_path, root, _commit), "live expected")
        source.write_bytes(source_original)

        huge = Path(temporary) / "huge.json"
        with huge.open("wb") as handle:
            handle.truncate(MAX_JSON_BYTES + 1)
        _expect("overlarge inventory file", lambda: validate_files(
            huge, "0" * 64, closure_path, receipt_path, root, _commit), "bounded")

        parse_source_closure_bytes(closure_path.read_bytes(), inventory, root, _commit)
        parse_build_receipt_bytes(receipt_path.read_bytes(), inventory, closure, root, _commit)


if __name__ == "__main__":
    pie_adapter_receipt_self_test()
    print("pie_adapter_receipt_tests: PASS")
