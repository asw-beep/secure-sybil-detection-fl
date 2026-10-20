"""Secret random projection R_t (k x d) built from a round key (DECISIONS D3).

Entries are +-1/sqrt(k) (a dense Rademacher projection, which satisfies the
Johnson-Lindenstrauss property like a Gaussian one), so each entry has variance 1/k.

generator="shake256" (default, cryptographic): the sign bits are the SHAKE-256 (FIPS 202)
output stream for the input  round_key | "|sybil-fl|projection-matrix|v1|k=<k>|d=<d>".
Entry (i, j) is + if bit i*d + j of the stream is 0 and - if it is 1, bits read MSB first.
Keyed SHAKE-256 with the key first is the sponge-PRF construction that KMAC (SP 800-185)
standardises; hashlib has no KMAC, so this prefix-keyed form is used. Without the round key
the matrix is computationally indistinguishable from uniformly random signs, assuming keyed
SHAKE-256 is a PRF. The shape is part of the input, so matrices of different shapes are unrelated.

generator="pcg64" (comparison only): the same sign distribution from NumPy's PCG64 seeded with
the round key. PCG64 is not a cryptographic generator; it is kept to measure what the
cryptographic stream costs.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

GENERATORS = ("shake256", "pcg64")


def projection_matrix(round_key: bytes, k: int, d: int, generator: str = "shake256") -> torch.Tensor:
    """The k x d float32 projection matrix for one round."""
    if generator not in GENERATORS:
        raise ValueError(f"generator must be one of {GENERATORS}, got {generator!r}")
    if k < 1 or d < 1:
        raise ValueError(f"k and d must be >= 1, got k={k}, d={d}")
    if len(round_key) != 32:
        raise ValueError(f"round key must be 32 bytes, got {len(round_key)}")

    n_bits = k * d
    n_bytes = -(-n_bits // 8)
    if generator == "shake256":
        label = f"|sybil-fl|projection-matrix|v1|k={k}|d={d}".encode("ascii")
        stream = hashlib.shake_256(round_key + label).digest(n_bytes)
    else:
        # Same byte -> bit layout as the SHAKE path, so a timing comparison measures only the generator.
        seed = np.random.SeedSequence(int.from_bytes(round_key, "big"))
        stream = np.random.Generator(np.random.PCG64(seed)).bytes(n_bytes)
    bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))[:n_bits]

    signs = 1.0 - 2.0 * bits.astype(np.float32)
    return torch.from_numpy((signs / np.float32(math.sqrt(k))).reshape(k, d))


def project(vectors: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rows of `vectors` (n x d) mapped to n x k: z_i = R @ v_i."""
    if vectors.ndim != 2 or vectors.shape[1] != R.shape[1]:
        raise ValueError(f"vectors must be n x {R.shape[1]} to match the projection dimension, "
                         f"got shape {tuple(vectors.shape)}")
    if not torch.isfinite(vectors).all():
        raise ValueError("vectors contain non-finite values (NaN/inf)")
    return vectors.to(R.dtype) @ R.T
