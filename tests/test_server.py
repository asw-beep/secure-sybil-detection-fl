import csv
import math

import numpy as np
import pytest
import torch

from helpers import synthetic_digits
from sybil_fl.aggregators import fedavg
from sybil_fl.attacks import prepare_attack
from sybil_fl.client import local_update
from sybil_fl.config import AttackConfig, Config, DataConfig, DefenceConfig, EvaluationConfig, FLConfig, TrainConfig
from sybil_fl.crypto import derive_round_key
from sybil_fl.data import make_federated_split, partition_iid
from sybil_fl.detector import detect
from sybil_fl.models import build_model, get_flat_params
from sybil_fl.projection import projection_matrix
from sybil_fl.server import ROUND_FIELDS, run_fl
from sybil_fl.utils import derive_seed


def _setup(rounds=2, num_clients=3, attack=None):
    cfg = Config(
        seed=0,
        data=DataConfig(num_clients=num_clients, partition="iid"),
        train=TrainConfig(batch_size=16, lr=0.05, momentum=0.0),
        fl=FLConfig(rounds=rounds, local_epochs=1),
        attack=attack or AttackConfig(),
    )
    train, test = synthetic_digits(n_per_class=60, seed=0), synthetic_digits(n_per_class=20, seed=1)
    parts = partition_iid(len(train[1]), num_clients, seed=0)
    return cfg, (train, test), parts


def test_short_clean_run_completes_and_learns(tmp_path):
    cfg, data, parts = _setup(rounds=2)
    history = run_fl(cfg, data, parts, build_model(seed=0), tmp_path / "rounds.csv")
    assert [row["round"] for row in history] == [0, 1, 2]
    assert history[-1]["test_acc"] > history[0]["test_acc"] + 0.3
    assert all(row["clients_aggregated"] == 3 for row in history[1:])


def test_csv_has_header_and_one_row_per_round(tmp_path):
    cfg, data, parts = _setup(rounds=2)
    log = tmp_path / "rounds.csv"
    history = run_fl(cfg, data, parts, build_model(seed=0), log)
    with open(log, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == ROUND_FIELDS
    assert len(rows) == 3
    assert [float(r["test_acc"]) for r in rows] == [h["test_acc"] for h in history]
    assert math.isnan(float(rows[0]["mean_train_loss"]))  # nothing trained before round 1


def test_one_round_equals_fedavg_of_client_deltas_applied_to_global(tmp_path):
    # The server must aggregate the clients' original deltas and add the result to the global model.
    cfg, data, parts = _setup(rounds=1)
    (x, y), _ = data
    model = build_model(seed=0)
    start = get_flat_params(model)
    updates = [
        local_update(build_model(seed=0), x, y, idx, client_id=i, train_cfg=cfg.train,
                     local_epochs=1, seed=derive_seed(cfg.seed, f"client_{i}_round_1"))
        for i, idx in enumerate(parts)
    ]
    expected = start + fedavg([u.delta for u in updates], [u.num_samples for u in updates])
    run_fl(cfg, data, parts, model, tmp_path / "rounds.csv")
    assert torch.allclose(get_flat_params(model), expected, atol=1e-6)


def test_repeated_run_is_identical(tmp_path):
    cfg, data, parts = _setup(rounds=2)
    m1, m2 = build_model(seed=0), build_model(seed=0)
    h1 = run_fl(cfg, data, parts, m1, tmp_path / "a.csv")
    h2 = run_fl(cfg, data, parts, m2, tmp_path / "b.csv")
    assert [r["test_acc"] for r in h1] == [r["test_acc"] for r in h2]
    assert torch.equal(get_flat_params(m1), get_flat_params(m2))


def test_on_round_callback_sees_every_row(tmp_path):
    cfg, data, parts = _setup(rounds=2)
    seen = []
    history = run_fl(cfg, data, parts, build_model(seed=0), tmp_path / "rounds.csv", on_round=seen.append)
    assert seen == history
    assert np.isfinite([r["round_seconds"] for r in history]).all()


# ---------- attacks ----------

FLIP = AttackConfig(type="label_flip", num_sybils=1, source_class=7, target_class=1, sybil_data="own_shard")


def test_asr_is_logged_every_round_including_clean_runs(tmp_path):
    cfg, data, parts = _setup(rounds=1)
    history = run_fl(cfg, data, parts, build_model(seed=0), tmp_path / "rounds.csv")
    assert all(0.0 <= row["asr"] <= 1.0 for row in history)


def test_attacked_round_uses_poisoned_labels_for_sybils_only(tmp_path):
    cfg, data, parts = _setup(rounds=1, attack=FLIP)
    (x, y), _ = data
    plan = prepare_attack(cfg, y, parts)
    assert plan.flipped_per_sybil[0] > 0
    sybils = set(plan.sybil_ids.tolist())
    model = build_model(seed=0)
    start = get_flat_params(model)
    updates = [
        local_update(build_model(seed=0), x, plan.sybil_labels if i in sybils else y, idx, client_id=i,
                     train_cfg=cfg.train, local_epochs=1, seed=derive_seed(cfg.seed, f"client_{i}_round_1"))
        for i, idx in enumerate(parts)
    ]
    expected = start + fedavg([u.delta for u in updates], [u.num_samples for u in updates])
    run_fl(cfg, data, parts, model, tmp_path / "rounds.csv", attack=plan)
    assert torch.allclose(get_flat_params(model), expected, atol=1e-6)


def test_attack_config_without_plan_is_rejected(tmp_path):
    # A label_flip config run without its attack plan would silently be a clean run.
    cfg, data, parts = _setup(rounds=1, attack=FLIP)
    with pytest.raises(ValueError, match="attack"):
        run_fl(cfg, data, parts, build_model(seed=0), tmp_path / "rounds.csv")


def test_plan_without_attack_config_is_rejected(tmp_path):
    cfg_attack, data, parts = _setup(rounds=1, attack=FLIP)
    plan = prepare_attack(cfg_attack, data[0][1], parts)
    cfg_clean, _, _ = _setup(rounds=1)
    with pytest.raises(ValueError, match="attack"):
        run_fl(cfg_clean, data, parts, build_model(seed=0), tmp_path / "rounds.csv", attack=plan)


def test_pool_attack_round_trains_honest_shards_plus_sybil_samples(tmp_path):
    # Sybils join the 3 honest clients: 5 updates aggregated, each on its roster entry.
    cfg, data, _ = _setup(rounds=1, attack=AttackConfig(
        type="label_flip", num_sybils=2, source_class=7, target_class=1,
        sybil_data="attacker_pool", samples_per_sybil=20))
    cfg.data.attacker_pool_size = 30
    (x, y), _ = data
    parts, pool = make_federated_split(y.numpy(), cfg.data, pool_class=7, partition_seed=0, pool_seed=1)
    plan = prepare_attack(cfg, y, parts, pool=pool)
    sybils = set(plan.sybil_ids.tolist())
    model = build_model(seed=0)
    start = get_flat_params(model)
    updates = [
        local_update(build_model(seed=0), x, plan.sybil_labels if i in sybils else y, idx, client_id=i,
                     train_cfg=cfg.train, local_epochs=1, seed=derive_seed(cfg.seed, f"client_{i}_round_1"))
        for i, idx in enumerate(plan.client_indices)
    ]
    expected = start + fedavg([u.delta for u in updates], [u.num_samples for u in updates])
    history = run_fl(cfg, data, parts, model, tmp_path / "rounds.csv", attack=plan)
    assert history[-1]["clients_aggregated"] == 5
    assert torch.allclose(get_flat_params(model), expected, atol=1e-6)


# ---------- defence, evasion and shadow evaluation ----------

TEST_KEY = bytes(range(32))
POOL_FLIP = AttackConfig(type="label_flip", num_sybils=3, source_class=7, target_class=1,
                         sybil_data="attacker_pool", samples_per_sybil=20)


def _pool_run_setup(defence=None, evaluation=None, attack=None, rounds=1):
    cfg, data, _ = _setup(rounds=rounds, attack=attack or POOL_FLIP)
    cfg.data.attacker_pool_size = 30
    cfg.defence = defence or DefenceConfig()
    cfg.evaluation = evaluation or EvaluationConfig()
    cfg.validate()
    (x, y), _ = data
    parts, pool = make_federated_split(y.numpy(), cfg.data, pool_class=7, partition_seed=0, pool_seed=1)
    plan = prepare_attack(cfg, y, parts, pool=pool)
    return cfg, data, parts, plan


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_projected_defence_aggregates_exactly_the_unflagged_updates(tmp_path):
    # At round 1 these honest IID clients have cosine ~0.9 with each other and the Sybils ~0.99,
    # so a tight threshold (0.1) flags exactly the Sybils here; the check below holds for any flags.
    cfg, data, parts, plan = _pool_run_setup(DefenceConfig(detector="projected", distance_threshold=0.1))
    (x, y), _ = data
    sybils = set(plan.sybil_ids.tolist())
    model = build_model(seed=0)
    start = get_flat_params(model)
    updates = [
        local_update(build_model(seed=0), x, plan.sybil_labels if i in sybils else y, idx, client_id=i,
                     train_cfg=cfg.train, local_epochs=1, seed=derive_seed(cfg.seed, f"client_{i}_round_1"))
        for i, idx in enumerate(plan.client_indices)
    ]
    deltas = torch.stack([u.delta for u in updates])
    R = projection_matrix(derive_round_key(TEST_KEY, 1), cfg.projection.k, deltas.shape[1])
    flagged = detect(deltas, "projected", 0.1, 3, R=R).flagged
    keep = [i for i in range(len(updates)) if not flagged[i]]
    assert 0 < len(keep) < len(updates)
    expected = start + fedavg([updates[i].delta for i in keep], [updates[i].num_samples for i in keep])
    history = run_fl(cfg, data, parts, model, tmp_path / "rounds.csv", attack=plan, master_key=TEST_KEY)
    row = history[-1]
    assert row["flagged"] == int(flagged.sum()) and row["clients_aggregated"] == len(keep)
    assert row["tp"] + row["fp"] + row["fn"] + row["tn"] == row["clients_total"] == len(updates)
    assert torch.allclose(get_flat_params(model), expected, atol=1e-6)


def test_every_client_flagged_skips_the_round(tmp_path):
    # Threshold near 2 merges everyone into one cluster, so all clients are flagged.
    cfg, data, parts, plan = _pool_run_setup(DefenceConfig(detector="full", distance_threshold=1.99, min_cluster_size=2))
    model = build_model(seed=0)
    start = get_flat_params(model)
    history = run_fl(cfg, data, parts, model, tmp_path / "rounds.csv", attack=plan)
    assert history[-1]["skipped"] == 1 and history[-1]["clients_aggregated"] == 0
    assert torch.equal(get_flat_params(model), start)


def test_projected_defence_requires_the_master_key(tmp_path):
    cfg, data, parts, plan = _pool_run_setup(DefenceConfig(detector="projected"))
    with pytest.raises(ValueError, match="master key"):
        run_fl(cfg, data, parts, build_model(seed=0), tmp_path / "rounds.csv", attack=plan)


def test_evasion_budget_is_logged_and_applied(tmp_path):
    attack = AttackConfig(type="label_flip", num_sybils=3, source_class=7, target_class=1,
                          sybil_data="attacker_pool", samples_per_sybil=20,
                          evasion="known_projection", evasion_budget=0.05)
    cfg, data, parts, plan = _pool_run_setup(attack=attack)
    history = run_fl(cfg, data, parts, build_model(seed=0), tmp_path / "rounds.csv", attack=plan, master_key=TEST_KEY)
    assert abs(history[-1]["evasion_rel_norm"] - 0.05) < 1e-4


def test_shadow_evaluation_writes_one_row_per_method_k_and_threshold(tmp_path):
    ev = EvaluationConfig(shadow_ks=[16, 32], shadow_thresholds=[0.3, 0.6, 0.9])
    cfg, data, parts, plan = _pool_run_setup(evaluation=ev, rounds=2)
    run_fl(cfg, data, parts, build_model(seed=0), tmp_path / "rounds.csv", attack=plan,
           master_key=TEST_KEY, shadow_dir=tmp_path)
    det = _read_csv(tmp_path / "shadow_detection.csv")
    sim = _read_csv(tmp_path / "shadow_similarity.csv")
    assert len(det) == 2 * 3 * 3  # rounds x (full + 2 k) x thresholds
    assert {(r["method"], r["k"]) for r in det} == {("full", "0"), ("projected", "16"), ("projected", "32")}
    assert all(int(r["tp"]) + int(r["fp"]) + int(r["fn"]) + int(r["tn"]) == len(plan.client_indices) for r in det)
    assert len(sim) == 2 * 3 and all(float(r["max_abs_err"]) >= 0 for r in sim)
    assert all(float(r["max_abs_err"]) == 0.0 for r in sim if r["k"] == "0")  # full vs itself
