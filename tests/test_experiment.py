import json
import math

from helpers import synthetic_digits
from sybil_fl.config import AttackConfig, Config, DataConfig, DefenceConfig, EvaluationConfig, FLConfig, TrainConfig
from sybil_fl.crypto import derive_round_key
from sybil_fl.experiment import run_experiment

TEST_KEY = bytes(range(32))


def _cfg():
    cfg = Config(
        seed=0,
        data=DataConfig(num_clients=3, partition="iid", attacker_pool_size=30),
        train=TrainConfig(batch_size=16),
        fl=FLConfig(rounds=2),
        attack=AttackConfig(type="label_flip", num_sybils=3, samples_per_sybil=20,
                            evasion="known_projection", evasion_budget=0.02),
        defence=DefenceConfig(detector="projected"),
        evaluation=EvaluationConfig(shadow_ks=[16], shadow_thresholds=[0.5]),
    )
    cfg.validate()
    return cfg


def _data():
    return synthetic_digits(n_per_class=60, seed=0), synthetic_digits(n_per_class=20, seed=1)


def test_experiment_writes_rounds_meta_and_shadow_files_with_summary(tmp_path):
    meta = run_experiment(_cfg(), _data(), tmp_path / "run", TEST_KEY, key_source="env")
    files = {p.name for p in (tmp_path / "run").iterdir()}
    assert files == {"meta.json", "rounds.csv", "shadow_detection.csv", "shadow_similarity.csv"}
    saved = json.loads((tmp_path / "run" / "meta.json").read_text())
    assert saved["key_source"] == "env" and saved["summary"] == meta["summary"]
    s = meta["summary"]
    assert s["detection_totals"]["tp"] + s["detection_totals"]["fn"] == 2 * 3  # 3 Sybils x 2 rounds
    assert not math.isnan(s["last_half_mean_test_acc"])


def test_no_key_material_reaches_any_output_file(tmp_path):
    # DECISIONS D4: neither K nor any round key may appear in results, in hex or raw form.
    run_experiment(_cfg(), _data(), tmp_path / "run", TEST_KEY, key_source="env")
    secrets_ = [TEST_KEY] + [derive_round_key(TEST_KEY, t) for t in (1, 2)]
    for path in (tmp_path / "run").iterdir():
        blob = path.read_bytes()
        for s in secrets_:
            assert s not in blob and s.hex().encode() not in blob and s.hex().upper().encode() not in blob, path.name
