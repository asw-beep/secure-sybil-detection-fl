from pathlib import Path

import numpy as np
import pytest

from sybil_fl.config import DataConfig
from sybil_fl.data import (
    load_mnist,
    make_federated_split,
    make_partition,
    partition_dirichlet,
    partition_iid,
    reserve_attacker_pool,
    validate_partition,
)
from sybil_fl.utils import derive_seed

REPO = Path(__file__).resolve().parents[1]


def _labels(n_per_class=200, n_classes=10, seed=0):
    labels = np.repeat(np.arange(n_classes), n_per_class)
    return np.random.default_rng(seed).permutation(labels)


def _covers_exactly_once(parts, n):
    flat = np.concatenate(parts)
    return len(flat) == n and np.array_equal(np.sort(flat), np.arange(n))


# ---------- seeds ----------

def test_derive_seed_is_deterministic_and_stage_specific():
    assert derive_seed(42, "partition") == derive_seed(42, "partition")
    assert derive_seed(42, "partition") != derive_seed(42, "model_init")
    assert derive_seed(42, "partition") != derive_seed(43, "partition")


# ---------- IID ----------

def test_iid_covers_every_sample_exactly_once():
    parts = partition_iid(1003, num_clients=7, seed=1)
    assert len(parts) == 7
    assert _covers_exactly_once(parts, 1003)


def test_iid_sizes_differ_by_at_most_one():
    sizes = [len(p) for p in partition_iid(1003, num_clients=7, seed=1)]
    assert max(sizes) - min(sizes) <= 1


def test_iid_same_seed_same_partition_different_seed_different():
    a = partition_iid(500, 5, seed=3)
    b = partition_iid(500, 5, seed=3)
    c = partition_iid(500, 5, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


# ---------- Dirichlet ----------

def test_dirichlet_covers_every_sample_exactly_once():
    y = _labels()
    parts = partition_dirichlet(y, num_clients=10, alpha=0.5, seed=0, min_size=20)
    assert len(parts) == 10
    assert _covers_exactly_once(parts, len(y))


def test_dirichlet_respects_min_client_size():
    y = _labels()
    parts = partition_dirichlet(y, num_clients=10, alpha=0.3, seed=0, min_size=40)
    assert min(len(p) for p in parts) >= 40


def test_dirichlet_is_reproducible():
    y = _labels()
    a = partition_dirichlet(y, 10, 0.5, seed=7, min_size=10)
    b = partition_dirichlet(y, 10, 0.5, seed=7, min_size=10)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def _mean_majority_fraction(y, parts):
    return np.mean([np.bincount(y[p], minlength=10).max() / len(p) for p in parts])


def test_smaller_alpha_gives_more_label_skew():
    y = _labels(n_per_class=500)
    skewed = partition_dirichlet(y, 10, alpha=0.1, seed=0, min_size=10)
    near_iid = partition_dirichlet(y, 10, alpha=100.0, seed=0, min_size=10)
    assert _mean_majority_fraction(y, skewed) > _mean_majority_fraction(y, near_iid) + 0.2


def test_dirichlet_impossible_min_size_raises():
    y = _labels(n_per_class=10)  # 100 samples
    with pytest.raises(ValueError, match="min_size"):
        partition_dirichlet(y, num_clients=10, alpha=0.5, seed=0, min_size=11)


# ---------- validation + dispatch ----------

def test_validate_partition_rejects_duplicates():
    with pytest.raises(ValueError, match="more than once"):
        validate_partition([np.array([0, 1]), np.array([1, 2])], n_samples=3)


def test_validate_partition_rejects_missing_samples():
    with pytest.raises(ValueError, match="not assigned"):
        validate_partition([np.array([0]), np.array([2])], n_samples=3)


def test_make_partition_dispatches_on_config():
    y = _labels()
    iid = make_partition(y, DataConfig(num_clients=4, partition="iid"), seed=0)
    dirich = make_partition(
        y, DataConfig(num_clients=4, partition="dirichlet", dirichlet_alpha=0.5, min_client_samples=10), seed=0
    )
    assert _covers_exactly_once(iid, len(y)) and _covers_exactly_once(dirich, len(y))
    sizes = [len(p) for p in iid]
    assert max(sizes) - min(sizes) <= 1


# ---------- real MNIST (skipped if not downloaded) ----------

@pytest.mark.mnist
def test_load_mnist_shapes_and_normalisation():
    data_dir = REPO / "data"
    if not (data_dir / "MNIST" / "raw").exists():
        pytest.skip("MNIST not downloaded; run experiments/run_central.py once")
    (xtr, ytr), (xte, yte) = load_mnist(data_dir, download=False)
    assert tuple(xtr.shape) == (60000, 1, 28, 28) and tuple(xte.shape) == (10000, 1, 28, 28)
    assert ytr.dtype.is_floating_point is False and int(ytr.max()) == 9
    assert abs(float(xtr.mean())) < 0.05 and abs(float(xtr.std()) - 1.0) < 0.05


# ---------- attacker pool (DECISIONS D9) ----------


def test_pool_is_source_class_disjoint_and_covers_with_honest():
    y = _labels()
    pool, honest = reserve_attacker_pool(y, size=50, source_class=7, seed=0)
    assert len(pool) == 50 and (y[pool] == 7).all()
    assert _covers_exactly_once([pool, honest], len(y))


def test_pool_is_reproducible():
    y = _labels()
    a, _ = reserve_attacker_pool(y, 50, 7, seed=1)
    b, _ = reserve_attacker_pool(y, 50, 7, seed=1)
    assert np.array_equal(a, b)


def test_pool_larger_than_source_class_raises():
    y = _labels(n_per_class=20)
    with pytest.raises(ValueError, match="attacker_pool_size"):
        reserve_attacker_pool(y, size=21, source_class=7, seed=0)


def test_federated_split_gives_honest_clients_no_pool_samples():
    y = _labels()
    cfg = DataConfig(num_clients=5, partition="dirichlet", dirichlet_alpha=0.5, min_client_samples=10,
                     attacker_pool_size=60)
    parts, pool = make_federated_split(y, cfg, pool_class=7, partition_seed=0, pool_seed=1)
    assert len(parts) == 5 and len(pool) == 60
    assert _covers_exactly_once(parts + [pool], len(y))


def test_federated_split_without_pool_equals_plain_partition():
    # attacker_pool_size = 0 must reproduce the Phase 2-3 split exactly.
    y = _labels()
    cfg = DataConfig(num_clients=5, partition="dirichlet", dirichlet_alpha=0.5, min_client_samples=10,
                     attacker_pool_size=0)
    parts, pool = make_federated_split(y, cfg, pool_class=7, partition_seed=3, pool_seed=4)
    plain = make_partition(y, cfg, seed=3)
    assert len(pool) == 0
    assert all(np.array_equal(a, b) for a, b in zip(parts, plain))
