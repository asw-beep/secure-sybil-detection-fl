import hashlib
import math

import numpy as np
import pytest
import torch

from sybil_fl.crypto import derive_round_key
from sybil_fl.projection import project, projection_matrix

ROUND_KEY = derive_round_key(bytes(range(32)), round_t=1)


def test_shape_dtype_and_entries_are_plus_minus_one_over_sqrt_k():
    R = projection_matrix(ROUND_KEY, k=16, d=100)
    assert R.shape == (16, 100) and R.dtype == torch.float32
    assert torch.equal(R.abs(), torch.full_like(R, 1 / math.sqrt(16)))


def test_shake_entries_follow_the_documented_bit_layout():
    # Entry (i, j) is + if bit i*d + j of the SHAKE-256 stream is 0, - if 1; bits read MSB first.
    k, d = 4, 10
    stream = hashlib.shake_256(ROUND_KEY + f"|sybil-fl|projection-matrix|v1|k={k}|d={d}".encode()).digest(5)
    bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))[: k * d]
    expected = torch.tensor((1 - 2 * bits.astype(np.float32)).reshape(k, d) / math.sqrt(k))
    assert torch.equal(projection_matrix(ROUND_KEY, k, d), expected)


def test_same_inputs_same_matrix_any_change_gives_a_different_one():
    R = projection_matrix(ROUND_KEY, 8, 64)
    assert torch.equal(projection_matrix(ROUND_KEY, 8, 64), R)
    other_round = derive_round_key(bytes(range(32)), round_t=2)
    assert not torch.equal(projection_matrix(other_round, 8, 64), R)
    assert not torch.equal(projection_matrix(ROUND_KEY, 8, 64, generator="pcg64"), R)


def test_shape_is_domain_separated():
    # A k=8 matrix must not be a prefix of the k=16 one (the shape is part of the SHAKE input).
    small, big = projection_matrix(ROUND_KEY, 8, 64), projection_matrix(ROUND_KEY, 16, 64)
    assert not torch.equal(small * math.sqrt(8), (big * math.sqrt(16))[:8])


def test_signs_are_balanced():
    R = projection_matrix(ROUND_KEY, 64, 4096)
    plus = (R > 0).float().mean().item()
    assert abs(plus - 0.5) < 0.01  # 262,144 bits: std of the mean is ~0.001


def test_pcg64_path_is_deterministic_with_same_distribution():
    R = projection_matrix(ROUND_KEY, 8, 64, generator="pcg64")
    assert torch.equal(projection_matrix(ROUND_KEY, 8, 64, generator="pcg64"), R)
    assert torch.equal(R.abs(), torch.full_like(R, 1 / math.sqrt(8)))


def test_projection_roughly_preserves_norms_and_cosines():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 4096, generator=g)
    x[1] = 0.8 * x[0] + 0.6 * x[1]  # cosine with x[0] about 0.8
    z = project(x, projection_matrix(ROUND_KEY, 256, 4096))
    ratio = z.norm(dim=1) / x.norm(dim=1)
    assert ((ratio > 0.85) & (ratio < 1.15)).all()
    cos = lambda a: torch.nn.functional.cosine_similarity(a[0:1], a[1:2]).item()
    assert abs(cos(z) - cos(x)) < 0.1


def test_project_validates_inputs():
    R = projection_matrix(ROUND_KEY, 4, 10)
    assert project(torch.ones(3, 10), R).shape == (3, 4)
    with pytest.raises(ValueError, match="dimension"):
        project(torch.ones(3, 11), R)
    with pytest.raises(ValueError, match="non-finite"):
        project(torch.tensor([[float("nan")] * 10]), R)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="generator"):
        projection_matrix(ROUND_KEY, 4, 10, generator="mt19937")
    with pytest.raises(ValueError, match="k"):
        projection_matrix(ROUND_KEY, 0, 10)
    with pytest.raises(ValueError, match="round key"):
        projection_matrix(b"short", 4, 10)
