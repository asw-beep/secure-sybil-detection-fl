"""Small shared helpers: seed derivation and run-environment metadata."""

from __future__ import annotations

import hashlib
import platform
import sys


def derive_seed(master_seed: int, stage: str) -> int:
    """Derive an independent, reproducible sub-seed for one stage of a run.

    Each stage (partitioning, model init, a client's minibatch order, ...) gets
    its own seed, so changing one stage never shifts another stage's randomness.
    SHA-256 is used only as a stable hash here (Python's built-in hash() is
    salted per process); this is not a security mechanism.
    Returns a 63-bit non-negative int, valid for both NumPy and torch generators.
    """
    digest = hashlib.sha256(f"{master_seed}|{stage}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def environment_info() -> dict[str, str]:
    """Interpreter, OS and library versions, stored with every experiment result."""
    import numpy as np
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
