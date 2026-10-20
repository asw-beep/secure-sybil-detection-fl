import numpy as np
import pytest
import torch

from sybil_fl.attacks import evasion_perturbations, label_flip_labels, prepare_attack, select_sybils
from sybil_fl.config import AttackConfig, Config, DataConfig
from sybil_fl.crypto import derive_round_key
from sybil_fl.data import partition_iid
from sybil_fl.projection import projection_matrix


def _labels():
    # 12 samples, 3 clients of 4: client 0 = [7,7,1,2], client 1 = [7,3,3,7], client 2 = [0,1,2,4]
    y = torch.tensor([7, 7, 1, 2, 7, 3, 3, 7, 0, 1, 2, 4])
    parts = [np.arange(0, 4), np.arange(4, 8), np.arange(8, 12)]
    return y, parts


# ---------- Sybil selection ----------

def test_select_sybils_count_distinct_in_range_sorted():
    ids = select_sybils(num_clients=20, num_sybils=4, seed=0)
    assert len(ids) == 4 and len(set(ids.tolist())) == 4
    assert ids.min() >= 0 and ids.max() < 20
    assert np.array_equal(ids, np.sort(ids))


def test_select_sybils_is_reproducible():
    assert np.array_equal(select_sybils(20, 4, seed=5), select_sybils(20, 4, seed=5))
    assert not all(np.array_equal(select_sybils(20, 4, seed=5), select_sybils(20, 4, seed=s)) for s in range(6, 10))


def test_select_sybils_rejects_too_many():
    with pytest.raises(ValueError, match="num_sybils"):
        select_sybils(num_clients=3, num_sybils=4, seed=0)


# ---------- label flipping ----------

def test_flip_changes_only_sybil_source_labels():
    y, parts = _labels()
    poisoned = label_flip_labels(y, parts, sybil_ids=[1], source=7, target=1)
    expected = y.clone()
    expected[[4, 7]] = 1  # client 1's two sevens
    assert torch.equal(poisoned, expected)


def test_flip_leaves_honest_clients_untouched_even_with_source_class():
    y, parts = _labels()
    poisoned = label_flip_labels(y, parts, sybil_ids=[1], source=7, target=1)
    assert torch.equal(poisoned[parts[0]], y[parts[0]])  # client 0 has sevens but is honest


def test_flip_does_not_modify_original_labels():
    y, parts = _labels()
    before = y.clone()
    label_flip_labels(y, parts, sybil_ids=[0, 1], source=7, target=1)
    assert torch.equal(y, before)


def test_sybil_without_source_samples_changes_nothing():
    y, parts = _labels()
    assert torch.equal(label_flip_labels(y, parts, sybil_ids=[2], source=7, target=1), y)


def test_flip_rejects_invalid_sybil_ids():
    y, parts = _labels()
    with pytest.raises(ValueError, match="sybil"):
        label_flip_labels(y, parts, sybil_ids=[3], source=7, target=1)
    with pytest.raises(ValueError, match="sybil"):
        label_flip_labels(y, parts, sybil_ids=[1, 1], source=7, target=1)


# ---------- prepare_attack ----------

def test_prepare_attack_none_returns_none():
    y, parts = _labels()
    assert prepare_attack(Config(), y, parts) is None


def test_prepare_attack_label_flip_records_ground_truth_and_flip_counts():
    y = torch.tensor(np.random.default_rng(0).integers(0, 10, size=400))
    parts = partition_iid(400, 5, seed=0)
    cfg = Config(
        seed=3,
        data=DataConfig(num_clients=5, partition="iid"),
        attack=AttackConfig(type="label_flip", num_sybils=2, source_class=7, target_class=1,
                            sybil_data="own_shard"),
    )
    plan = prepare_attack(cfg, y, parts)
    assert len(plan.sybil_ids) == 2
    assert plan.flipped_per_sybil == [int((y[parts[i]] == 7).sum()) for i in plan.sybil_ids]
    assert int((plan.sybil_labels != y).sum()) == sum(plan.flipped_per_sybil)
    again = prepare_attack(cfg, y, parts)
    assert np.array_equal(plan.sybil_ids, again.sybil_ids)
    assert all(np.array_equal(a, b) for a, b in zip(plan.client_indices, parts))  # roster = partition


# ---------- attacker pool (DECISIONS D9) ----------

def _pool_setup(num_sybils=2, samples_per_sybil=30):
    y = torch.tensor(np.random.default_rng(0).integers(0, 10, size=600))
    cfg = Config(
        seed=3,
        data=DataConfig(num_clients=4, partition="iid", attacker_pool_size=50),
        attack=AttackConfig(type="label_flip", num_sybils=num_sybils, source_class=7, target_class=1,
                            sybil_data="attacker_pool", samples_per_sybil=samples_per_sybil),
    )
    sevens = np.flatnonzero(y.numpy() == 7)
    pool = sevens[:50]
    rest = np.setdiff1d(np.arange(600), pool)
    parts = [rest[i::4] for i in range(4)]
    return cfg, y, parts, pool


def test_pool_sybils_join_honest_clients_who_keep_their_shards_in_order():
    cfg, y, parts, pool = _pool_setup()
    plan = prepare_attack(cfg, y, parts, pool=pool)
    assert len(plan.client_indices) == 4 + 2
    honest_positions = [i for i in range(6) if i not in set(plan.sybil_ids.tolist())]
    for pos, shard in zip(honest_positions, parts):
        assert np.array_equal(plan.client_indices[pos], shard)


def test_pool_sybils_sample_distinct_pool_images_all_relabelled():
    cfg, y, parts, pool = _pool_setup()
    plan = prepare_attack(cfg, y, parts, pool=pool)
    for s in plan.sybil_ids:
        idx = plan.client_indices[s]
        assert len(idx) == 30 and len(set(idx.tolist())) == 30
        assert set(idx.tolist()) <= set(pool.tolist())
        assert (plan.sybil_labels[idx] == 1).all()
    assert plan.flipped_per_sybil == [30, 30]


def test_pool_sybils_draw_different_samples():
    cfg, y, parts, pool = _pool_setup()
    plan = prepare_attack(cfg, y, parts, pool=pool)
    a, b = (plan.client_indices[s] for s in plan.sybil_ids)
    assert not np.array_equal(np.sort(a), np.sort(b))


def test_pool_attack_leaves_honest_labels_untouched_and_is_reproducible():
    cfg, y, parts, pool = _pool_setup()
    plan = prepare_attack(cfg, y, parts, pool=pool)
    honest = np.concatenate(parts)
    assert torch.equal(plan.sybil_labels[honest], y[honest])
    again = prepare_attack(cfg, y, parts, pool=pool)
    assert np.array_equal(plan.sybil_ids, again.sybil_ids)
    assert all(np.array_equal(a, b) for a, b in zip(plan.client_indices, again.client_indices))


def test_pool_attack_without_pool_raises():
    cfg, y, parts, _ = _pool_setup()
    with pytest.raises(ValueError, match="pool"):
        prepare_attack(cfg, y, parts, pool=np.array([], dtype=np.int64))


# ---------- evasion (E6) ----------

D_EV, K_EV = 4096, 64
R_TRUE = projection_matrix(derive_round_key(bytes(range(32)), 1), K_EV, D_EV)
R_GUESS = projection_matrix(derive_round_key(bytes(32), 1), K_EV, D_EV)  # attacker's own random matrix


def _sybil_deltas(n=4):
    g = torch.Generator().manual_seed(0)
    base = torch.randn(D_EV, generator=g)
    return base.repeat(n, 1) + 0.01 * torch.randn(n, D_EV, generator=g)


@pytest.mark.parametrize("mode, M", [("gaussian", None), ("known_projection", R_TRUE), ("guessed_projection", R_GUESS)])
def test_perturbation_norm_is_exactly_the_budget(mode, M):
    x = _sybil_deltas()
    p = evasion_perturbations(x, mode, budget=0.05, seed=1, M=M)
    assert p.shape == x.shape
    assert torch.allclose(p.norm(dim=1), 0.05 * x.norm(dim=1), rtol=1e-4)


def test_zero_budget_means_no_perturbation():
    x = _sybil_deltas()
    assert torch.count_nonzero(evasion_perturbations(x, "known_projection", 0.0, seed=1, M=R_TRUE)) == 0


def test_perturbations_are_reproducible_and_differ_between_sybils():
    x = _sybil_deltas()
    a = evasion_perturbations(x, "gaussian", 0.05, seed=1)
    assert torch.equal(a, evasion_perturbations(x, "gaussian", 0.05, seed=1))
    assert not torch.allclose(a[0], a[1])


def test_known_projection_perturbation_is_amplified_by_the_true_matrix():
    # A row-space perturbation is magnified by about sqrt(d/k) = 8 under R; under an unrelated
    # matrix (or for isotropic noise) it is roughly norm-preserving.
    x = _sybil_deltas()
    known = evasion_perturbations(x, "known_projection", 0.05, seed=1, M=R_TRUE)
    guessed = evasion_perturbations(x, "guessed_projection", 0.05, seed=1, M=R_GUESS)
    gain = lambda p: ((p @ R_TRUE.T).norm(dim=1) / p.norm(dim=1)).mean().item()
    assert gain(known) > 0.8 * (D_EV / K_EV) ** 0.5
    assert 0.5 < gain(guessed) < 2.0


def test_known_projection_evasion_breaks_projected_similarity_but_not_full():
    x = _sybil_deltas()
    # Gain sqrt(d/k) = 8 here, so a 20% change becomes ~160% after projection:
    # expected cosines ~1/(1 + 0.2^2) = 0.96 in full and ~1/(1 + 1.6^2) = 0.28 projected.
    p = evasion_perturbations(x, "known_projection", 0.2, seed=1, M=R_TRUE)
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    y = x + p
    assert cos(y[0], y[1]) > 0.9                                    # full-dimensional: still near-identical
    assert cos(R_TRUE @ y[0], R_TRUE @ y[1]) < 0.5                  # projected: no longer similar


def test_projection_modes_need_a_matrix_and_unknown_modes_fail():
    x = _sybil_deltas()
    with pytest.raises(ValueError, match="matrix"):
        evasion_perturbations(x, "known_projection", 0.1, seed=1)
    with pytest.raises(ValueError, match="evasion"):
        evasion_perturbations(x, "mimicry", 0.1, seed=1)
