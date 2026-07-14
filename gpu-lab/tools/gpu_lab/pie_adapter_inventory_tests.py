"""Focused generation and mutation tests for the PIE-adapter source inventory."""

from __future__ import annotations

import io
import stat
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from .common import canonical_bytes, require, sha256_bytes
from .immutable_output import write_immutable_bytes as real_write_immutable_bytes
from .pie_adapter_inventory import (
    main,
    make_inventory,
    parse_locator,
    write_inventory,
)
from .pie_adapter_receipt import (
    ADAPTER_BINARY,
    DISCOVERY_FILES,
    DISCOVERY_PACKAGE_ROOTS,
    parse_inventory_bytes,
    validate_inventory,
)


TARGET = "x86_64-unknown-linux-gnu"
TARGET_PATH = "gpu_benchmarks/lab/pie-adapter/receipt-targets/sn2-x86_64"


def _write(path: Path, payload: bytes = b"source\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _workspace(root: Path) -> None:
    for repository in ("stwo", "stwo-cairo"):
        (root / repository).mkdir(parents=True)
    for repository, relative in DISCOVERY_FILES:
        _write(root / repository / relative, f"{repository}:{relative}\n".encode())


def _target(path: str = TARGET_PATH, repository: str = "stwo-cairo") -> dict[str, str]:
    return {"repository": repository, "path": path}


def _executable(path: str = TARGET_PATH, target: str = TARGET,
                repository: str = "stwo-cairo") -> dict[str, str]:
    return {
        "repository": repository,
        "path": f"{path}/{target}/release/{ADAPTER_BINARY}",
    }


def _expect(label: str, operation: Callable[[], object], fragment: str | None = None) -> None:
    try:
        operation()
    except (OSError, ValueError) as error:
        require(fragment is None or fragment in str(error),
                f"{label} failed for the wrong reason: {error}")
        return
    raise ValueError(f"{label} was accepted")


def pie_adapter_inventory_self_test(_: Path | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="gpu-lab-pie-inventory-") as temporary:
        parent = Path(temporary)
        root = parent / "workspace"
        _workspace(root)
        output = parent / "inventory.json"
        inventory, digest = write_inventory(
            output, TARGET, _target(), _executable(), root,
        )
        payload = output.read_bytes()
        require(payload == canonical_bytes(inventory) + b"\n"
                and digest == sha256_bytes(payload),
                "PIE adapter inventory bytes or digest are not canonical")
        require(stat.S_IMODE(output.stat().st_mode) == 0o400
                and output.stat().st_nlink == 1,
                "PIE adapter inventory output is not immutable")
        require(parse_inventory_bytes(payload, digest, root) == inventory,
                "PIE adapter generated inventory does not validate")
        require(inventory["build"]["target"] == TARGET
                and inventory["build"]["target_directory"] == _target()
                and inventory["adapter_executable"] == _executable()
                and inventory["generated_outputs"] == [_executable()],
                "PIE adapter inventory omitted an exact caller locator")

        inode = output.stat().st_ino
        repeated, repeated_digest = write_inventory(
            output, TARGET, _target(), _executable(), root,
        )
        require(repeated == inventory and repeated_digest == digest
                and output.stat().st_ino == inode,
                "identical PIE adapter inventory generation was not idempotent")

        cli_output = parent / "inventory-cli.json"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main([
                "--target", TARGET,
                "--target-directory", f"stwo-cairo:{TARGET_PATH}",
                "--adapter-executable", f"stwo-cairo:{_executable()['path']}",
                "--output", str(cli_output),
            ], workspace_root=root)
        require(status == 0 and stderr.getvalue() == ""
                and "PIE_ADAPTER_INVENTORY=PASS" in stdout.getvalue()
                and digest in stdout.getvalue()
                and "build_execution_attested=false" in stdout.getvalue()
                and "production_admissible=false" in stdout.getvalue()
                and cli_output.read_bytes() == payload,
                "PIE adapter production CLI did not emit the exact inventory")

        for hostile_target in (
            "aarch64-apple-darwin", "aarch64-linux-android",
            "riscv64gc-unknown-linux-gnu", "x-bogus-linux-foo",
        ):
            _expect(f"non-Linux target {hostile_target}", lambda target=hostile_target:
                    make_inventory(target, _target(), _executable(target=target), root),
                    "Linux")
        for locator in ("stwo-cairo", ":path", "unknown:path", "stwo-cairo:"):
            _expect(f"malformed locator {locator!r}",
                    lambda locator=locator: parse_locator(locator, "test"),
                    "REPOSITORY:RELATIVE_PATH")
        _expect("target traversal", lambda: make_inventory(
            TARGET, _target("../receipt-target"),
            _executable("../receipt-target"), root,
        ), "ambiguous")
        _expect("target in wrong repository", lambda: make_inventory(
            TARGET, _target("receipt-targets/run", "stwo"),
            _executable("receipt-targets/run", repository="stwo"), root,
        ), "receipt-targets namespace")
        _expect("noncanonical executable", lambda: make_inventory(
            TARGET, _target(),
            {"repository": "stwo-cairo", "path": f"{TARGET_PATH}/adapter"}, root,
        ), "exact Cargo release output")

        for label, path in (
            ("git metadata", ".git/objects/fresh-target"),
            ("cargo config", ".cargo/fresh-target"),
            ("unrelated package", "stwo_cairo_prover/crates/common/fresh-target"),
            ("nested run path", f"{TARGET_PATH}/nested"),
        ):
            _expect(f"target in {label}", lambda path=path: make_inventory(
                TARGET, _target(path), _executable(path), root,
            ), "receipt-targets namespace")
        hidden_run = "gpu_benchmarks/lab/pie-adapter/receipt-targets/.hidden"
        _expect("hidden target run id", lambda: make_inventory(
            TARGET, _target(hidden_run), _executable(hidden_run), root,
        ), "bounded run id")

        existing = root / "stwo-cairo" / TARGET_PATH
        existing.mkdir(parents=True)
        _expect("existing target directory", lambda: make_inventory(
            TARGET, _target(), _executable(), root,
        ), "must not exist")
        existing.rmdir()
        existing.parent.rmdir()
        link_target = parent / "link-target"
        link_target.mkdir()
        link = root / "stwo-cairo/gpu_benchmarks/lab/pie-adapter/receipt-targets"
        link.symlink_to(link_target, target_is_directory=True)
        _expect("target ancestor symlink", lambda: make_inventory(
            TARGET, _target(), _executable(), root,
        ), "symlink")
        link.unlink()

        baseline = make_inventory(TARGET, _target(), _executable(), root)
        for repository, package_root in DISCOVERY_PACKAGE_ROOTS:
            relative = f"{package_root}/build.rs"
            build_script = root / repository / relative
            _write(build_script, b"fn main() {}\n")
            _expect(f"omitted implicit build script {repository}:{relative}",
                    lambda: validate_inventory(baseline, root), "discovery")
            expanded = make_inventory(TARGET, _target(), _executable(), root)
            require({"repository": repository, "path": relative}
                    in expanded["expected_sources"],
                    f"implicit build script was not inventoried: {repository}:{relative}")
            build_script.unlink()

        baseline = make_inventory(TARGET, _target(), _executable(), root)
        for index, (repository, package_root) in enumerate(DISCOVERY_PACKAGE_ROOTS):
            custom = root / repository / package_root / "generated" / f"custom-{index}.rs"
            _write(custom, b"pub fn custom() {}\n")
            _expect(f"omitted custom package target {repository}:{package_root}",
                    lambda: validate_inventory(baseline, root), "discovery")
            expanded = make_inventory(TARGET, _target(), _executable(), root)
            relative = str(custom.relative_to(root / repository))
            require({"repository": repository, "path": relative}
                    in expanded["expected_sources"],
                    f"custom package target was not inventoried: {repository}:{relative}")
            custom.unlink()
            custom.parent.rmdir()

        _expect("inventory inside target", lambda: write_inventory(
            root / "stwo-cairo" / TARGET_PATH / "inventory.json",
            TARGET, _target(), _executable(), root,
        ), "outside")
        require(not (root / "stwo-cairo" / TARGET_PATH).exists(),
                "rejected inventory output created the fresh target directory")

        original_payload = output.read_bytes()
        alternate_target = "aarch64-unknown-linux-gnu"
        _expect("different inventory at immutable output", lambda: write_inventory(
            output, alternate_target, _target(),
            _executable(target=alternate_target), root,
        ), "differs")
        require(output.read_bytes() == original_payload,
                "failed inventory replacement changed the immutable output")

        mutated_output = parent / "publication-mutation.json"
        new_source = root / "stwo" / "crates/stwo/src/late.rs"

        def mutate_before_publish(destination: Path, data: bytes, sources,
                                  **kwargs) -> None:
            callback = kwargs["_before_publish"]

            def mutate(temporary_path: Path) -> None:
                _write(new_source, b"pub fn late() {}\n")
                callback(temporary_path)

            real_write_immutable_bytes(
                destination, data, sources, _before_publish=mutate,
            )

        with patch("gpu_lab.pie_adapter_inventory.write_immutable_bytes",
                   side_effect=mutate_before_publish):
            _expect("source-set mutation before publication", lambda: write_inventory(
                mutated_output, TARGET, _target(), _executable(), root,
            ), "changed before publication")
        require(not mutated_output.exists(),
                "source-set mutation left an inventory output")
        new_source.unlink()

        failed_output = parent / "failed-cli.json"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main([
                "--target", "aarch64-apple-darwin",
                "--target-directory", f"stwo-cairo:{TARGET_PATH}",
                "--adapter-executable",
                f"stwo-cairo:{_executable(target='aarch64-apple-darwin')['path']}",
                "--output", str(failed_output),
            ], workspace_root=root)
        require(status == 2 and stdout.getvalue() == ""
                and "pie-adapter-inventory: FAIL" in stderr.getvalue()
                and not failed_output.exists(),
                "PIE adapter CLI failure was not fail-closed")


if __name__ == "__main__":
    pie_adapter_inventory_self_test()
    print("pie_adapter_inventory_tests: PASS")
