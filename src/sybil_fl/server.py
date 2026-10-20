"""Server round loop: local training, optional Sybil evasion, detection, filtered aggregation.

Each round t:
  1. Every client trains from the current global model and returns its original delta.
     Under an attack the clients are the plan's roster (honest shards plus Sybils, D9) and
     Sybils train on the plan's poisoned labels.
  2. Attacker side (E6, D10): if configured, Sybils add an evasion perturbation to the updates
     they submit. The known-projection attacker is handed the server's R_t (key compromise).
  3. Server side: non-finite updates are rejected; the detector (none | full | projected with
     the round's secret R_t from HKDF + SHAKE-256, D3) flags clients; the configured aggregator
     combines the remaining submitted updates. If every client is excluded the round is
     skipped (fallback 'skip') and the global model is unchanged.
  4. Evaluation: clean accuracy, ASR, detection counts against ground truth, timings, and
     optional shadow scoring (evaluation.py). Ground truth is used only in this step.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .aggregators import aggregate
from .attacks import AttackPlan, attacker_guess_key, evasion_perturbations
from .client import local_update
from .config import Config
from .crypto import derive_round_key
from .data import Partition
from .detector import detect
from .evaluation import DETECTION_FIELDS, SIMILARITY_FIELDS, shadow_round
from .metrics import attack_success_rate, confusion
from .models import get_flat_params, set_flat_params
from .projection import projection_matrix
from .training import evaluate_accuracy
from .utils import derive_seed

# round_seconds: client training + evasion + detection + aggregation (not evaluation or shadow scoring).
# mean_train_loss averages all clients, Sybils included (their loss is on poisoned labels).
# t_matrix: generating R_t; t_detect_*: detector stages (detect_projection includes normalisation).
ROUND_FIELDS = [
    "round", "test_acc", "asr", "mean_train_loss", "clients_total", "clients_aggregated",
    "rejected_non_finite", "flagged", "tp", "fp", "fn", "tn", "skipped", "evasion_rel_norm",
    "t_matrix", "t_detect_projection", "t_detect_similarity", "t_detect_clustering", "round_seconds",
]


def needs_master_key(cfg: Config) -> bool:
    return (
        cfg.defence.detector == "projected"
        or bool(cfg.evaluation.shadow_ks)
        or cfg.attack.evasion == "known_projection"
    )


def run_fl(
    cfg: Config,
    data,
    partition: Partition,
    model: nn.Module,
    log_path: str | Path,
    on_round: Callable[[dict], None] | None = None,
    attack: AttackPlan | None = None,
    master_key: bytes | None = None,
    shadow_dir: str | Path | None = None,
) -> list[dict]:
    """Run cfg.fl.rounds rounds, updating `model` in place; returns one dict per round (row 0 = start).

    partition holds the honest clients' indices; an attack plan (required exactly when
    cfg.attack.type is not 'none') replaces it with its full roster. master_key is the server
    key K (crypto.load_server_key), needed for the projected detector, shadow k's and the
    known-projection attacker; it is never written anywhere. Shadow CSVs go to shadow_dir.
    """
    if (cfg.attack.type != "none") != (attack is not None):
        raise ValueError(
            f"attack type is {cfg.attack.type!r} but an attack plan was "
            f"{'not ' if attack is None else ''}given; build it with attacks.prepare_attack"
        )
    if needs_master_key(cfg) and master_key is None:
        raise ValueError("this configuration needs the server master key (crypto.load_server_key)")
    if cfg.evaluation.enabled and shadow_dir is None:
        raise ValueError("shadow evaluation is configured but no shadow_dir was given")

    (x_tr, y_tr), (x_te, y_te) = data
    clients = attack.client_indices if attack is not None else partition
    sybils = set(attack.sybil_ids.tolist()) if attack is not None else set()
    is_sybil = np.array([i in sybils for i in range(len(clients))])
    sybil_rows = np.flatnonzero(is_sybil)
    source, target = cfg.attack.source_class, cfg.attack.target_class
    d = get_flat_params(model).numel()
    history: list[dict] = []

    shadow_files = []
    if cfg.evaluation.enabled:
        shadow_dir = Path(shadow_dir)
        det_fh = open(shadow_dir / "shadow_detection.csv", "w", newline="", encoding="utf-8")
        sim_fh = open(shadow_dir / "shadow_similarity.csv", "w", newline="", encoding="utf-8")
        shadow_files = [det_fh, sim_fh]
        det_writer = csv.DictWriter(det_fh, fieldnames=DETECTION_FIELDS)
        sim_writer = csv.DictWriter(sim_fh, fieldnames=SIMILARITY_FIELDS)
        det_writer.writeheader()
        sim_writer.writeheader()

    try:
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=ROUND_FIELDS)
            writer.writeheader()

            def record(row: dict) -> None:
                writer.writerow(row)
                fh.flush()
                history.append(row)
                if on_round is not None:
                    on_round(row)

            nan = float("nan")
            record({
                "round": 0, "test_acc": evaluate_accuracy(model, x_te, y_te),
                "asr": attack_success_rate(model, x_te, y_te, source, target), "mean_train_loss": nan,
                "clients_total": len(clients), "clients_aggregated": 0, "rejected_non_finite": 0,
                "flagged": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "skipped": 0, "evasion_rel_norm": nan,
                "t_matrix": 0.0, "t_detect_projection": 0.0, "t_detect_similarity": 0.0,
                "t_detect_clustering": 0.0, "round_seconds": 0.0,
            })

            for t in range(1, cfg.fl.rounds + 1):
                start = time.perf_counter()
                updates = [
                    local_update(
                        model, x_tr, attack.sybil_labels if i in sybils else y_tr, idx, client_id=i,
                        train_cfg=cfg.train, local_epochs=cfg.fl.local_epochs,
                        seed=derive_seed(cfg.seed, f"client_{i}_round_{t}"),
                    )
                    for i, idx in enumerate(clients)
                ]
                submitted = torch.stack([u.delta for u in updates])  # a copy; ClientUpdate.delta is untouched
                counts = [u.num_samples for u in updates]

                round_key = derive_round_key(master_key, t) if master_key is not None else None
                t_matrix = 0.0
                R = None
                if cfg.defence.detector == "projected" or cfg.attack.evasion == "known_projection":
                    t0 = time.perf_counter()
                    R = projection_matrix(round_key, cfg.projection.k, d, cfg.projection.generator)
                    t_matrix = time.perf_counter() - t0

                # attacker side
                rel_norm = nan
                if attack is not None and cfg.attack.evasion != "none" and len(sybil_rows):
                    M = None
                    if cfg.attack.evasion == "known_projection":
                        M = R
                    elif cfg.attack.evasion == "guessed_projection":
                        M = projection_matrix(attacker_guess_key(cfg.seed, t), cfg.projection.k, d,
                                              cfg.projection.generator)
                    p = evasion_perturbations(submitted[sybil_rows], cfg.attack.evasion, cfg.attack.evasion_budget,
                                              seed=derive_seed(cfg.seed, f"evasion_round_{t}"), M=M)
                    norms = submitted[sybil_rows].norm(dim=1)
                    submitted[sybil_rows] = submitted[sybil_rows] + p
                    rel_norm = float((p.norm(dim=1) / norms).mean())

                # server side
                finite = torch.isfinite(submitted).all(dim=1).numpy()
                flagged = np.zeros(len(clients), dtype=bool)
                timings = {"projection": 0.0, "similarity": 0.0, "clustering": 0.0}
                if cfg.defence.detector != "none":
                    res = detect(submitted, cfg.defence.detector, cfg.defence.distance_threshold,
                                 cfg.defence.min_cluster_size, R=R if cfg.defence.detector == "projected" else None)
                    flagged, timings = res.flagged, res.seconds
                keep = np.flatnonzero(finite & ~flagged)
                skipped = 0
                if len(keep):
                    step = aggregate(cfg.defence.aggregator, [submitted[i] for i in keep], [counts[i] for i in keep])
                    set_flat_params(model, get_flat_params(model) + step)
                else:
                    skipped = 1  # fallback 'skip': global model unchanged this round
                seconds = time.perf_counter() - start

                if cfg.evaluation.enabled:
                    mats = {k: projection_matrix(round_key, k, d, cfg.projection.generator)
                            for k in cfg.evaluation.shadow_ks}
                    det_rows, sim_rows = shadow_round(
                        t, submitted[finite], is_sybil[finite], mats,
                        list(cfg.evaluation.shadow_thresholds), cfg.defence.min_cluster_size,
                    )
                    det_writer.writerows(det_rows)
                    sim_writer.writerows(sim_rows)
                    for f in shadow_files:
                        f.flush()

                record({
                    "round": t,
                    "test_acc": evaluate_accuracy(model, x_te, y_te),
                    "asr": attack_success_rate(model, x_te, y_te, source, target),
                    "mean_train_loss": float(np.mean([u.train_loss for u in updates])),
                    "clients_total": len(clients),
                    "clients_aggregated": len(keep),
                    "rejected_non_finite": int((~finite).sum()),
                    "flagged": int(flagged.sum()),
                    **confusion(flagged, is_sybil),
                    "skipped": skipped,
                    "evasion_rel_norm": rel_norm,
                    "t_matrix": t_matrix,
                    "t_detect_projection": timings["projection"],
                    "t_detect_similarity": timings["similarity"],
                    "t_detect_clustering": timings["clustering"],
                    "round_seconds": seconds,
                })
    finally:
        for f in shadow_files:
            f.close()

    return history
