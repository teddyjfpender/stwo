#!/usr/bin/env python3
"""GPU-lab build identity, independent tiny oracle, and replay preparation."""

from gpu_lab.cli import main
from gpu_lab.semantics import pedersen_oracle

__all__ = ["main", "pedersen_oracle"]

if __name__ == "__main__":
    raise SystemExit(main())
