"""E0 sanity check: does the data/model pipeline learn, and is the client split valid?

1. Loads MNIST (downloads it to data/ on first run).
2. Builds the configured client partition and checks it covers every training
   sample exactly once; prints client sizes and label skew.
3. Trains the CNN centrally (no federation) and reports test accuracy per epoch.
4. Saves config, seed, environment versions and metrics to results/.

Usage:  python experiments/run_central.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sybil_fl.config import load_config
from sybil_fl.data import load_mnist, make_partition
from sybil_fl.models import build_model, num_params
from sybil_fl.training import evaluate_accuracy, train_epochs
from sybil_fl.utils import derive_seed, environment_info

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "default.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)

    data_dir = Path(cfg.data.data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO / data_dir
    (x_tr, y_tr), (x_te, y_te) = load_mnist(data_dir, download=True)
    print(f"MNIST: train {tuple(x_tr.shape)}, test {tuple(x_te.shape)}")

    # --- partition check (E0: disjoint + exact coverage) ---
    labels = y_tr.numpy()
    parts = make_partition(labels, cfg.data, seed=derive_seed(cfg.seed, "partition"))  # raises if invalid
    sizes = np.array([len(p) for p in parts])
    majority = np.array([np.bincount(labels[p], minlength=10).max() / len(p) for p in parts])
    print(
        f"partition={cfg.data.partition} (alpha={cfg.data.dirichlet_alpha}), "
        f"{len(parts)} clients: sizes min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}, "
        f"mean majority-class fraction = {majority.mean():.2f}  [covers all {len(labels)} samples exactly once]"
    )

    # --- central training ---
    model = build_model(seed=derive_seed(cfg.seed, "model_init"))
    print(f"model parameters d = {num_params(model)}")
    acc0 = evaluate_accuracy(model, x_te, y_te)
    print(f"epoch 0: test acc {acc0:.4f}")
    history = [{"epoch": 0, "test_acc": acc0}]
    all_idx = np.arange(len(y_tr))
    t0 = time.perf_counter()
    for epoch in range(1, cfg.train.central_epochs + 1):
        loss = train_epochs(
            model, x_tr, y_tr, all_idx, epochs=1,
            batch_size=cfg.train.batch_size, lr=cfg.train.lr, momentum=cfg.train.momentum,
            seed=derive_seed(cfg.seed, f"central_epoch_{epoch}"),
        )
        acc = evaluate_accuracy(model, x_te, y_te)
        history.append({"epoch": epoch, "train_loss": loss, "test_acc": acc})
        print(f"epoch {epoch}: train loss {loss:.4f}, test acc {acc:.4f}")
    train_seconds = time.perf_counter() - t0

    # --- save ---
    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "experiment": "E0_central",
        "timestamp_utc": stamp,
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "num_params": num_params(model),
        "partition_sizes": sizes.tolist(),
        "partition_mean_majority_fraction": float(majority.mean()),
        "history": history,
        "train_seconds": train_seconds,
    }
    out = out_dir / f"e0_central_{stamp}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"saved {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
