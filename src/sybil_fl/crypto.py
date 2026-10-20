"""Server key handling and per-round key derivation (DECISIONS D3, D4).

- The server's master key K (32 bytes) comes from the SYBIL_FL_KEY environment variable
  (64 hex characters) or, if that is absent, is generated fresh with the OS CSPRNG for the
  run. It is never written to results or logs; only its source ("env" / "generated") is.
- Round keys are derived with HKDF-SHA256 (RFC 5869): a fixed protocol salt, and an info
  string naming protocol, purpose, version and round, so every (purpose, round) pair gets
  an independent key. HKDF is implemented here on the standard library's HMAC and checked
  against the RFC 5869 test vectors.

HKDF is a PRF-based key-derivation function: without K, round keys are computationally
indistinguishable from random under the standard assumption that HMAC-SHA256 is a PRF.
That is a statement about the keys, not a proof that the detector resists attacks.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from collections.abc import Mapping

KEY_BYTES = 32
KEY_ENV_VAR = "SYBIL_FL_KEY"
HASH_LEN = 32  # SHA-256 output size
PROTOCOL_SALT = b"sybil-fl|hkdf-salt|v1"


# ---------------------------------------------------------------- HKDF-SHA256 (RFC 5869)

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """PRK = HMAC-SHA256(salt, IKM); an empty salt means HashLen zero bytes (RFC 5869 §2.2)."""
    return hmac.new(salt or bytes(HASH_LEN), ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """OKM = first `length` bytes of T(1) | T(2) | ..., T(i) = HMAC(PRK, T(i-1) | info | i)."""
    if not 0 <= length <= 255 * HASH_LEN:
        raise ValueError(f"HKDF output length must be in [0, {255 * HASH_LEN}], got {length}")
    okm, block = b"", b""
    for i in range(1, -(-length // HASH_LEN) + 1):
        block = hmac.new(prk, block + info + bytes([i]), hashlib.sha256).digest()
        okm += block
    return okm[:length]


def hkdf(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    return hkdf_expand(hkdf_extract(salt, ikm), info, length)


# ---------------------------------------------------------------- keys

def derive_round_key(master_key: bytes, round_t: int, purpose: str = "projection") -> bytes:
    """32-byte key for one (purpose, round): HKDF(K, salt, "sybil-fl|<purpose>|v1|round=<t>")."""
    if len(master_key) != KEY_BYTES:
        raise ValueError(f"master key must be {KEY_BYTES} bytes, got {len(master_key)}")
    if round_t < 0:
        raise ValueError(f"round must be >= 0, got {round_t}")
    if not purpose or "|" in purpose:
        raise ValueError("purpose must be a non-empty label without '|'")
    info = f"sybil-fl|{purpose}|v1|round={round_t}".encode("ascii")
    return hkdf(master_key, PROTOCOL_SALT, info, KEY_BYTES)


def load_server_key(environ: Mapping[str, str] | None = None) -> tuple[bytes, str]:
    """Return (K, source). source is "env" or "generated"; record only that, never K.

    Error messages describe the problem without repeating the supplied value.
    """
    env = os.environ if environ is None else environ
    raw = env.get(KEY_ENV_VAR)
    if raw is None:
        return secrets.token_bytes(KEY_BYTES), "generated"
    try:
        key = bytes.fromhex(raw.strip())
    except ValueError:
        raise ValueError(f"{KEY_ENV_VAR} is not valid hex (value not shown)") from None
    if len(key) != KEY_BYTES:
        raise ValueError(f"{KEY_ENV_VAR} must encode {KEY_BYTES} bytes ({2 * KEY_BYTES} hex characters), "
                         f"got {len(key)} bytes")
    return key, "env"
