import inspect

import numpy as np
import pytest
import torch

from sybil_fl.crypto import derive_round_key
from sybil_fl.detector import cluster_flags, cosine_similarity_matrix, detect, detector_vectors
from sybil_fl.projection import projection_matrix

D = 2048
R = projection_matrix(derive_round_key(bytes(range(32)), 1), 64, D)


def _clients(n_honest=10, n_sybil=5, noise=0.05, seed=0):
    """Honest: independent random directions. Sybils: one shared direction plus small noise."""
    g = torch.Generator().manual_seed(seed)
    honest = torch.randn(n_honest, D, generator=g)
    base = torch.randn(D, generator=g)
    sybil = base + noise * torch.randn(n_sybil, D, generator=g) * base.norm() / D ** 0.5
    return torch.cat([honest, sybil]), np.array([False] * n_honest + [True] * n_sybil)


# ---------- vectors and similarity ----------

def test_detector_vectors_normalise_copies_and_mark_degenerate_rows():
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0], [float("nan"), 1.0]])
    before = x.clone()
    v, valid = detector_vectors(x)
    assert valid.tolist() == [True, False, False]
    assert torch.allclose(v[0], torch.tensor([0.6, 0.8]))
    assert torch.isfinite(v).all()  # degenerate rows are zeroed, never NaN
    assert torch.equal(x[0], before[0]) and torch.isnan(x[2, 0])  # input untouched


def test_similarity_matrix_is_symmetric_finite_with_unit_diagonal():
    x, _ = _clients()
    v, valid = detector_vectors(x)
    s = cosine_similarity_matrix(v)
    assert torch.allclose(s, s.T) and torch.isfinite(s).all()
    assert torch.allclose(torch.diagonal(s), torch.ones(len(s)), atol=1e-5)
    assert (s <= 1 + 1e-5).all() and (s >= -1 - 1e-5).all()


# ---------- clustering ----------

def test_cluster_flags_finds_the_tight_group_only():
    x, sybil = _clients()
    v, valid = detector_vectors(x)
    flagged = cluster_flags(cosine_similarity_matrix(v), valid, distance_threshold=0.5, min_cluster_size=3)
    assert np.array_equal(flagged, sybil)


def test_small_groups_below_min_cluster_size_are_not_flagged():
    x, _ = _clients(n_honest=10, n_sybil=2)
    v, valid = detector_vectors(x)
    assert not cluster_flags(cosine_similarity_matrix(v), valid, 0.5, min_cluster_size=3).any()


def test_independent_clients_are_not_flagged():
    x, _ = _clients(n_honest=15, n_sybil=0)
    v, valid = detector_vectors(x)
    assert not cluster_flags(cosine_similarity_matrix(v), valid, 0.5, 3).any()


def test_degenerate_rows_are_never_flagged_and_fewer_than_two_valid_flags_nothing():
    x, sybil = _clients()
    x[0] = 0.0
    v, valid = detector_vectors(x)
    assert not cluster_flags(cosine_similarity_matrix(v), valid, 0.5, 3)[0]
    one = torch.zeros(3, D)
    one[0, 0] = 1.0
    v, valid = detector_vectors(one)
    assert not cluster_flags(cosine_similarity_matrix(v), valid, 0.5, 2).any()


# ---------- detect(): full and projected ----------

@pytest.mark.parametrize("method", ["full", "projected"])
def test_detect_flags_sybils_with_both_methods(method):
    x, sybil = _clients()
    res = detect(x, method, distance_threshold=0.5, min_cluster_size=3, R=R if method == "projected" else None)
    assert np.array_equal(res.flagged, sybil)
    assert set(res.seconds) == {"projection", "similarity", "clustering"}


def test_detect_is_order_invariant():
    x, sybil = _clients()
    perm = np.random.default_rng(1).permutation(len(x))
    a = detect(x, "projected", 0.5, 3, R=R).flagged
    b = detect(x[perm], "projected", 0.5, 3, R=R).flagged
    assert np.array_equal(a[perm], b)


def test_detect_validates_method_and_matrix():
    x, _ = _clients()
    with pytest.raises(ValueError, match="R"):
        detect(x, "projected", 0.5, 3)
    with pytest.raises(ValueError, match="method"):
        detect(x, "foolsgold", 0.5, 3)


def test_detect_never_receives_ground_truth():
    # The detector's inputs are the updates and its own settings only (plan §13).
    params = set(inspect.signature(detect).parameters)
    assert params == {"deltas", "method", "distance_threshold", "min_cluster_size", "R"}
