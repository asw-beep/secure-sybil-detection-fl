"""Run one configured experiment end to end and write its results.

Outputs in out_dir: rounds.csv (server.ROUND_FIELDS), meta.json (config, seed, environment,
split sizes, attack ground truth, key *source* only, summary) and, with shadow evaluation,
shadow_detection.csv and shadow_similarity.csv. Key material is never written (D4; tested).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .attacks import prepare_attack
from .config import Config
from .data import make_federated_split
from .metrics import detection_scores
from .models import build_model, num_params
from .server import run_fl
from .utils import derive_seed, environment_info


def _mean(values: list[float]) -> float:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(history: list[dict]) -> dict:
    """Run-level summary. Last-half means are the headline numbers: ASR oscillates between rounds."""
    rounds = history[1:]
    half = rounds[len(rounds) // 2:]
    totals = {k: int(sum(r[k] for r in rounds)) for k in ("tp", "fp", "fn", "tn")}
    return {
        "final_test_acc": history[-1]["test_acc"],
        "final_asr": history[-1]["asr"],
        "last_half_mean_test_acc": _mean([r["test_acc"] for r in half]),
        "last_half_mean_asr": _mean([r["asr"] for r in half]),
        "last_half_rounds": [half[0]["round"], half[-1]["round"]] if half else [],
        "max_asr": max((r["asr"] for r in rounds), default=float("nan")),
        "detection_totals": totals,
        "detection_scores": detection_scores(**totals),  # micro-averaged over all rounds
        "skipped_rounds": int(sum(r["skipped"] for r in rounds)),
        "mean_evasion_rel_norm": _mean([r["evasion_rel_norm"] for r in rounds]),
        "mean_seconds": {k: _mean([r[k] for r in rounds]) for k in (
            "t_matrix", "t_detect_projection", "t_detect_similarity", "t_detect_clustering", "round_seconds")},
    }


def run_experiment(
    cfg: Config,
    data,
    out_dir: str | Path,
    master_key: bytes,
    key_source: str,
    config_file: str | None = None,
    label: str | None = None,
    on_round: Callable[[dict], None] | None = None,
) -> dict:
    """data = ((x_train, y_train), (x_test, y_test)). Returns the meta dict (also saved)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (_, y_tr), _ = data

    parts, pool = make_federated_split(
        y_tr.numpy(), cfg.data, pool_class=cfg.attack.source_class,
        partition_seed=derive_seed(cfg.seed, "partition"), pool_seed=derive_seed(cfg.seed, "attacker_pool"),
    )
    model = build_model(seed=derive_seed(cfg.seed, "model_init"))
    plan = prepare_attack(cfg, y_tr, parts, pool=pool)

    meta = {
        "experiment": label or f"fl_{cfg.attack.type}",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config_file": config_file,
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "key_source": key_source,  # "env" or "generated"; the key itself is never stored
        "num_params": num_params(model),
        "partition_sizes": [len(p) for p in parts],
        "attacker_pool_size": len(pool),
    }
    if plan is not None:
        meta["attack"] = {  # ground truth, for evaluation only
            "num_clients_total": len(plan.client_indices),
            "sybil_ids": plan.sybil_ids.tolist(),
            "sybil_sample_sizes": [len(plan.client_indices[i]) for i in plan.sybil_ids],
            "flipped_per_sybil": plan.flipped_per_sybil,
        }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))  # written first so a crashed run still has its config

    t0 = time.perf_counter()
    history = run_fl(
        cfg, data, parts, model, out_dir / "rounds.csv", on_round=on_round, attack=plan,
        master_key=master_key, shadow_dir=out_dir if cfg.evaluation.enabled else None,
    )
    meta["summary"] = summarize(history)
    meta["summary"]["total_seconds"] = time.perf_counter() - t0
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta
