"""Run one federated experiment from a config file (clean, attacked, defended, with shadow scoring).

The server key comes from SYBIL_FL_KEY (64 hex characters) or is generated fresh for the run;
only its source is recorded. Results go to results/fl_<attack>_<detector>_<timestamp>/ (or --out):
rounds.csv, meta.json and, if configured, shadow_detection.csv / shadow_similarity.csv.

Usage:  python experiments/run_main.py [--config configs/default.yaml] [--out DIR]
        python experiments/run_main.py --config configs/smoke.yaml       # 3-round smoke test
        python experiments/run_main.py --config configs/label_flip.yaml  # 8 pool Sybils, 7 -> 1
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from sybil_fl.config import load_config
from sybil_fl.crypto import load_server_key
from sybil_fl.data import load_mnist
from sybil_fl.experiment import run_experiment

REPO = Path(__file__).resolve().parents[1]


def show(row: dict) -> None:
    line = f"round {row['round']}: test acc {row['test_acc']:.4f}, ASR {row['asr']:.4f}"
    if row["round"] > 0:
        line += (f", flagged {row['flagged']} (tp {row['tp']}, fp {row['fp']}), "
                 f"aggregated {row['clients_aggregated']}/{row['clients_total']}, {row['round_seconds']:.1f} s")
    print(line, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "default.yaml"))
    ap.add_argument("--out", default=None, help="output directory (default: results/fl_<attack>_<detector>_<timestamp>)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    data_dir = Path(cfg.data.data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO / data_dir
    data = load_mnist(data_dir, download=True)
    key, key_source = load_server_key()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else REPO / "results" / f"fl_{cfg.attack.type}_{cfg.defence.detector}_{stamp}"
    print(f"config {args.config}; server key: {key_source}; output {out}", flush=True)
    meta = run_experiment(cfg, data, out, key, key_source, config_file=args.config, on_round=show)
    s = meta["summary"]
    print(f"last-half mean test acc {s['last_half_mean_test_acc']:.4f}, last-half mean ASR {s['last_half_mean_asr']:.4f}, "
          f"detection {s['detection_scores']}; saved {out}")


if __name__ == "__main__":
    main()
