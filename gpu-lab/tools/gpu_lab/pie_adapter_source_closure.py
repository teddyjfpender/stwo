"""Publish the live PIE-adapter source closure before a fresh build."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import WORKSPACE_ROOT, canonical_bytes, require, sha256_bytes
from .immutable_output import write_immutable_bytes
from .pie_adapter_receipt import (
    CommitReader,
    _closure_document,
    _git_commit,
    _read_bounded,
    _source_body,
    parse_inventory_bytes,
    parse_source_closure_bytes,
    validate_source_closure,
)


def generate_source_closure(inventory_path: Path, raw_inventory_sha256: str,
                            closure_path: Path, workspace_root: Path = WORKSPACE_ROOT,
                            _commit_reader: CommitReader = _git_commit) -> dict[str, Any]:
    inventory_bytes = _read_bounded(inventory_path, "PIE adapter source inventory")
    inventory = parse_inventory_bytes(inventory_bytes, raw_inventory_sha256, workspace_root)
    inventory_sha256 = sha256_bytes(canonical_bytes(inventory))
    source_body, source_paths = _source_body(
        inventory, inventory_sha256, workspace_root, _commit_reader,
    )
    closure = _closure_document(source_body)
    validate_source_closure(closure, inventory, workspace_root, _commit_reader)
    payload = json.dumps(closure, indent=2, sort_keys=True).encode() + b"\n"

    def revalidate(_: Path) -> None:
        expected, _ = _source_body(
            inventory, inventory_sha256, workspace_root, _commit_reader,
        )
        require(expected == source_body,
                "PIE adapter live sources changed before closure publication")

    write_immutable_bytes(
        closure_path, payload, [inventory_path, *source_paths], _before_publish=revalidate,
    )
    return parse_source_closure_bytes(
        _read_bounded(closure_path, "PIE adapter source closure"),
        inventory, workspace_root, _commit_reader,
    )
