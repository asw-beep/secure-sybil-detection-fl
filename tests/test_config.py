from pathlib import Path

import pytest

from sybil_fl.config import Config, load_config

REPO = Path(__file__).resolve().parents[1]


def test_default_yaml_loads_into_typed_config():
    cfg = load_config(REPO / "configs" / "default.yaml")
    assert isinstance(cfg, Config)
    assert cfg.seed == 42
    assert cfg.data.num_clients == 20
    assert cfg.data.partition == "dirichlet"
    assert cfg.train.batch_size == 32


def test_unknown_key_is_rejected(tmp_path):
    # A typo in a config key must fail loudly, not silently fall back to a default.
    p = tmp_path / "bad.yaml"
    p.write_text("seed: 1\ndata:\n  num_clinets: 5\n")
    with pytest.raises(ValueError, match="num_clinets"):
        load_config(p)


def test_invalid_partition_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("data:\n  partition: shards\n")
    with pytest.raises(ValueError, match="partition"):
        load_config(p)


def test_nonpositive_alpha_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("data:\n  dirichlet_alpha: 0\n")
    with pytest.raises(ValueError, match="dirichlet_alpha"):
        load_config(p)


def test_default_yaml_has_fl_section():
    cfg = load_config(REPO / "configs" / "default.yaml")
    assert cfg.fl.rounds == 20
    assert cfg.fl.local_epochs == 1


def test_nonpositive_rounds_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("fl:\n  rounds: 0\n")
    with pytest.raises(ValueError, match="rounds"):
        load_config(p)


def test_nonpositive_local_epochs_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("fl:\n  local_epochs: 0\n")
    with pytest.raises(ValueError, match="local_epochs"):
        load_config(p)


def test_smoke_config_differs_from_default_only_in_rounds():
    default = load_config(REPO / "configs" / "default.yaml").to_dict()
    smoke = load_config(REPO / "configs" / "smoke.yaml").to_dict()
    assert smoke["fl"]["rounds"] == 3
    smoke["fl"]["rounds"] = default["fl"]["rounds"]
    assert smoke == default


def test_to_dict_round_trips_for_result_metadata(tmp_path):
    cfg = load_config(REPO / "configs" / "default.yaml")
    d = cfg.to_dict()
    assert d["data"]["dirichlet_alpha"] == cfg.data.dirichlet_alpha
    assert Config.from_dict(d) == cfg


# ---------- attack section ----------

def test_default_yaml_has_no_attack_but_names_asr_classes():
    cfg = load_config(REPO / "configs" / "default.yaml")
    assert cfg.attack.type == "none" and cfg.attack.num_sybils == 0
    assert (cfg.attack.source_class, cfg.attack.target_class) == (7, 1)


def test_label_flip_config_differs_from_default_only_in_attack_section():
    default = load_config(REPO / "configs" / "default.yaml").to_dict()
    flip = load_config(REPO / "configs" / "label_flip.yaml").to_dict()
    assert flip["attack"] == {
        "type": "label_flip", "num_sybils": 8, "source_class": 7, "target_class": 1,
        "sybil_data": "attacker_pool", "samples_per_sybil": 500,
        "evasion": "none", "evasion_budget": 0.0,
    }
    flip["attack"] = default["attack"]
    assert flip == default


def test_demo_configs_differ_from_label_flip_only_in_rounds_and_detector():
    import yaml

    flip = load_config(REPO / "configs" / "label_flip.yaml").to_dict()
    attack = load_config(REPO / "configs" / "demo_attack.yaml").to_dict()
    defended = load_config(REPO / "configs" / "demo_defended.yaml").to_dict()
    assert attack["fl"]["rounds"] == 5
    attack["fl"]["rounds"] = flip["fl"]["rounds"]
    assert attack == flip

    # The demo threshold must stay the tuned one for its k, or the detector flags almost everyone.
    tuned = yaml.safe_load((REPO / "configs" / "tuned_thresholds.yaml").read_text())
    k = defended["projection"]["k"]
    assert defended["defence"]["detector"] == "projected"
    assert defended["defence"]["distance_threshold"] == tuned[f"projected_k{k}"]
    defended["defence"]["detector"] = "none"
    defended["defence"]["distance_threshold"] = flip["defence"]["distance_threshold"]
    defended["fl"]["rounds"] = flip["fl"]["rounds"]
    assert defended == flip


@pytest.mark.parametrize(
    "attack_yaml, message",
    [
        ("type: backdoor", "type"),
        ("type: label_flip\n  num_sybils: 0", "num_sybils"),
        ("type: none\n  num_sybils: 2", "num_sybils"),
        ("type: label_flip\n  num_sybils: 21\n  sybil_data: own_shard", "num_sybils"),
        ("sybil_data: borrowed", "sybil_data"),
        ("type: label_flip\n  num_sybils: 2\n  samples_per_sybil: 0", "samples_per_sybil"),
        ("type: label_flip\n  num_sybils: 2\n  samples_per_sybil: 1001", "attacker_pool_size"),
        ("source_class: 1\n  target_class: 1", "differ"),
        ("source_class: 10", "source_class"),
        ("target_class: -1", "target_class"),
    ],
)
def test_invalid_attack_config_is_rejected(tmp_path, attack_yaml, message):
    p = tmp_path / "bad.yaml"
    p.write_text(f"attack:\n  {attack_yaml}\n")
    with pytest.raises(ValueError, match=message):
        load_config(p)


def test_default_yaml_reserves_attacker_pool_even_for_clean_runs():
    # DECISIONS D9: clean and attacked runs must share the same honest clients.
    assert load_config(REPO / "configs" / "default.yaml").data.attacker_pool_size == 1000


def test_negative_attacker_pool_size_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("data:\n  attacker_pool_size: -1\n")
    with pytest.raises(ValueError, match="attacker_pool_size"):
        load_config(p)


def test_pool_sybils_may_outnumber_honest_clients(tmp_path):
    # With an attacker pool, Sybils are extra identities, so num_sybils is not capped by num_clients.
    p = tmp_path / "ok.yaml"
    p.write_text("data:\n  num_clients: 2\n  attacker_pool_size: 100\n"
                 "attack:\n  type: label_flip\n  num_sybils: 3\n  samples_per_sybil: 10\n")
    assert load_config(p).attack.num_sybils == 3


# ---------- projection section (D3) ----------

def test_default_projection_is_shake256_k64():
    cfg = load_config(REPO / "configs" / "default.yaml")
    assert (cfg.projection.k, cfg.projection.generator) == (64, "shake256")


@pytest.mark.parametrize("yaml_text, message", [
    ("projection:\n  k: 0\n", "k must be >= 1"),
    ("projection:\n  generator: mt19937\n", "generator"),
])
def test_invalid_projection_config_is_rejected(tmp_path, yaml_text, message):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match=message):
        load_config(p)


# ---------- defence, evaluation and evasion ----------

def test_default_defence_is_plain_fedavg_and_evaluation_is_off():
    cfg = load_config(REPO / "configs" / "default.yaml")
    assert (cfg.defence.aggregator, cfg.defence.detector, cfg.defence.fallback) == ("fedavg", "none", "skip")
    assert cfg.defence.min_cluster_size == 3 and 0 < cfg.defence.distance_threshold < 2
    assert cfg.evaluation.shadow_ks == [] and cfg.evaluation.shadow_thresholds == []
    assert (cfg.attack.evasion, cfg.attack.evasion_budget) == ("none", 0.0)


@pytest.mark.parametrize("yaml_text, message", [
    ("defence:\n  aggregator: krum\n", "aggregator"),
    ("defence:\n  detector: foolsgold\n", "detector"),
    ("defence:\n  distance_threshold: 0\n", "distance_threshold"),
    ("defence:\n  distance_threshold: 2\n", "distance_threshold"),
    ("defence:\n  min_cluster_size: 1\n", "min_cluster_size"),
    ("defence:\n  fallback: fedavg\n", "fallback"),
    ("evaluation:\n  shadow_ks: [0]\n", "shadow_ks"),
    ("evaluation:\n  shadow_thresholds: [2.5]\n", "shadow_thresholds"),
    ("attack:\n  evasion: mimicry\n", "evasion must be one of"),
    ("attack:\n  type: label_flip\n  num_sybils: 2\n  evasion: gaussian\n  evasion_budget: -0.1\n",
     "evasion_budget must be >= 0"),
    ("attack:\n  evasion: gaussian\n  evasion_budget: 0.1\n", "evasion needs an attack"),
])
def test_invalid_defence_evaluation_evasion_config_is_rejected(tmp_path, yaml_text, message):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match=message):
        load_config(p)
