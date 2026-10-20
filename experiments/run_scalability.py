"""E5: synthetic detector benchmark — full-dimensional vs projected similarity, and matrix cost.

Vectors are i.i.d. standard Gaussian (n x d, float32; the distribution does not affect timing).
For every (n, d, k) setting, each stage is timed as the median of REPEATS runs:
  full_similarity   normalise rows + V V^T over d dimensions          ~ O(n^2 d)
  gen_shake256      the k x d sign matrix from keyed SHAKE-256 (D3)    ~ O(k d)
  gen_pcg64         the same distribution from NumPy PCG64 (comparison only)
  project           normalise + Z = V R^T + normalise                  ~ O(n d k)
  projected_sim     Z Z^T over k dimensions                            ~ O(n^2 k)
  clustering        average-linkage on the n x n distances (identical for both detectors)
Memory: stored detector representations n*d*4 vs n*k*4 bytes, and the k*d*4-byte matrix.

Grids: vary n (d = 46,730, k = 64), vary d (n = 100, k = 64), vary k (n = 100, d = 46,730).
Writes results/suite/e5/scalability.csv and meta.json (environment, thread count).

Usage:  python experiments/run_scalability.py [--threads 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from sybil_fl.crypto import derive_round_key
from sybil_fl.detector import average_linkage, cosine_similarity_matrix, detector_vectors
from sybil_fl.projection import project, projection_matrix
from sybil_fl.utils import environment_info

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "suite" / "e5"
REPEATS = 5
D_MODEL = 46_730
GRID = (
    [("vary_n", n, D_MODEL, 64) for n in (20, 50, 100, 200, 500, 1000)]
    + [("vary_d", 100, d, 64) for d in (10_000, D_MODEL, 100_000, 400_000)]
    + [("vary_k", 100, D_MODEL, k) for k in (16, 32, 64, 128, 256, 512)]
)


def timed(fn) -> tuple[float, object]:
    times, out = [], None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    round_key = derive_round_key(secrets.token_bytes(32), 1)  # throwaway key, never stored

    rows = []
    for grid, n, d, k in GRID:
        x = torch.randn(n, d, generator=torch.Generator().manual_seed(0))
        t_full, _ = timed(lambda: cosine_similarity_matrix(detector_vectors(x)[0]))
        t_shake, R = timed(lambda: projection_matrix(round_key, k, d, "shake256"))
        t_pcg, _ = timed(lambda: projection_matrix(round_key, k, d, "pcg64"))
        v, valid = detector_vectors(x)
        t_proj, z = timed(lambda: detector_vectors(project(v, R))[0])
        t_psim, s = timed(lambda: cosine_similarity_matrix(z))
        t_clu, _ = timed(lambda: average_linkage(s, valid))
        row = {
            "grid": grid, "n": n, "d": d, "k": k,
            "full_similarity": t_full, "gen_shake256": t_shake, "gen_pcg64": t_pcg,
            "project": t_proj, "projected_sim": t_psim, "clustering": t_clu,
            "projected_total_shake256": t_shake + t_proj + t_psim,
            "stored_full_bytes": n * d * 4, "stored_projected_bytes": n * k * 4, "matrix_bytes": k * d * 4,
        }
        rows.append(row)
        print(f"{grid} n={n} d={d} k={k}: full {t_full*1e3:.1f} ms | shake {t_shake*1e3:.1f} pcg64 {t_pcg*1e3:.1f} "
              f"project {t_proj*1e3:.1f} psim {t_psim*1e3:.2f} cluster {t_clu*1e3:.1f} ms", flush=True)

    with open(OUT / "scalability.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (OUT / "meta.json").write_text(json.dumps({
        "experiment": "E5_scalability", "environment": environment_info(), "torch_threads": args.threads,
        "repeats": REPEATS, "vector_distribution": "iid standard Gaussian, float32", "grid": GRID,
    }, indent=2))
    print(f"saved {OUT / 'scalability.csv'}")


if __name__ == "__main__":
    main()
