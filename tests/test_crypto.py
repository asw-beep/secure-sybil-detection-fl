import pytest

from sybil_fl.crypto import (
    KEY_BYTES,
    derive_round_key,
    hkdf,
    hkdf_expand,
    hkdf_extract,
    load_server_key,
)

TEST_KEY = bytes(range(32))  # obviously fake (DECISIONS D4)


# ---------- HKDF-SHA256 against RFC 5869 Appendix A ----------

def test_hkdf_rfc5869_case1():
    ikm = bytes([0x0B]) * 22
    salt = bytes(range(0x00, 0x0D))
    info = bytes(range(0xF0, 0xFA))
    prk = hkdf_extract(salt, ikm)
    assert prk.hex() == "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
    assert hkdf_expand(prk, info, 42).hex() == (
        "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
    )


def test_hkdf_rfc5869_case3_empty_salt_and_info():
    ikm = bytes([0x0B]) * 22
    assert hkdf_extract(b"", ikm).hex() == "19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04"
    assert hkdf(ikm, salt=b"", info=b"", length=42).hex() == (
        "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8"
    )


def test_hkdf_expand_rejects_too_long_output():
    with pytest.raises(ValueError, match="length"):
        hkdf_expand(bytes(32), b"", 255 * 32 + 1)


# ---------- per-round key derivation ----------

def test_round_key_is_32_bytes_and_deterministic():
    k = derive_round_key(TEST_KEY, round_t=3)
    assert isinstance(k, bytes) and len(k) == 32
    assert derive_round_key(TEST_KEY, round_t=3) == k


def test_round_key_differs_per_round_key_and_purpose():
    base = derive_round_key(TEST_KEY, round_t=1)
    assert derive_round_key(TEST_KEY, round_t=2) != base
    assert derive_round_key(bytes(32), round_t=1) != base
    assert derive_round_key(TEST_KEY, round_t=1, purpose="other") != base


def test_round_key_rejects_bad_inputs():
    with pytest.raises(ValueError, match="round"):
        derive_round_key(TEST_KEY, round_t=-1)
    with pytest.raises(ValueError, match="purpose"):
        derive_round_key(TEST_KEY, round_t=1, purpose="a|b")  # '|' would break the label's structure
    with pytest.raises(ValueError, match="master key"):
        derive_round_key(b"short", round_t=1)


# ---------- server key loading (D4) ----------

def test_key_from_environment():
    key, source = load_server_key({"SYBIL_FL_KEY": TEST_KEY.hex()})
    assert key == TEST_KEY and source == "env"


def test_key_generated_fresh_when_absent():
    k1, s1 = load_server_key({})
    k2, _ = load_server_key({})
    assert s1 == "generated" and len(k1) == KEY_BYTES
    assert k1 != k2


def test_invalid_hex_key_error_does_not_echo_the_value():
    secret_ish = "zz" + TEST_KEY.hex()[2:]
    with pytest.raises(ValueError) as exc:
        load_server_key({"SYBIL_FL_KEY": secret_ish})
    assert secret_ish not in str(exc.value)


def test_wrong_length_key_is_rejected():
    with pytest.raises(ValueError, match="32 bytes"):
        load_server_key({"SYBIL_FL_KEY": "00" * 16})
