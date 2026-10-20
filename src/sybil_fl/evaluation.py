"""Shadow evaluation: score detectors on a round's updates without affecting aggregation.

For each round this computes, from the same submitted updates:
- detection counts for the full-dimensional detector (method "full", k = 0) and the projected
  detector at every shadow k, for every shadow threshold (one linkage per method, cut at each
  threshold), and
- how projection distorts pairwise cosine similarity (E1): absolute error against the full
  similarity over all valid client pairs, and the mean similarity within Sybil-Sybil,
  honest-honest and Sybil-honest pairs.

Ground truth (is_sybil) is used only here, to score; detection itself never sees it.
"""

from __future__ import annotations

import numpy as np
import torch

from .detector import average_linkage, cosine_similarity_matrix, detector_vectors, flags_from_linkage
from .metrics import confusion
from .projection import project

DETECTION_FIELDS = ["round", "method", "k", "tau", "tp", "fp", "fn", "tn"]
SIMILARITY_FIELDS = [
    "round", "k", "mean_abs_err", "p95_abs_err", "max_abs_err",
    "cos_sybil_sybil", "cos_honest_honest", "cos_sybil_honest",
]


def _projected_vectors(v: torch.Tensor, valid: np.ndarray, R: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    z = torch.zeros(len(v), R.shape[0])
    if valid.any():
        z[valid] = project(v[valid], R)
    zv, valid_z = detector_vectors(z)
    return zv, valid & valid_z


def _group_mean(s: np.ndarray, mask: np.ndarray) -> float:
    return float(s[mask].mean()) if mask.any() else float("nan")


def shadow_round(
    round_t: int,
    deltas: torch.Tensor,
    is_sybil: np.ndarray,
    matrices: dict[int, torch.Tensor],
    thresholds: list[float],
    min_cluster_size: int,
) -> tuple[list[dict], list[dict]]:
    """Return (detection rows, similarity rows) for one round."""
    is_sybil = np.asarray(is_sybil, dtype=bool)
    v_full, valid_full = detector_vectors(deltas)
    s_full = cosine_similarity_matrix(v_full)

    variants = [("full", 0, s_full, valid_full)]
    for k, R in sorted(matrices.items()):
        zv, valid_k = _projected_vectors(v_full, valid_full, R)
        variants.append(("projected", k, cosine_similarity_matrix(zv), valid_k))

    det_rows, sim_rows = [], []
    n = len(is_sybil)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    for method, k, s, valid in variants:
        link = average_linkage(s, valid)
        for tau in thresholds:
            flags = flags_from_linkage(link, valid, tau, min_cluster_size)
            det_rows.append({"round": round_t, "method": method, "k": k, "tau": tau, **confusion(flags, is_sybil)})

        pairs = upper & np.outer(valid, valid) & np.outer(valid_full, valid_full)
        err = np.abs(s.numpy() - s_full.numpy())[pairs]
        sn = s.numpy()
        sim_rows.append({
            "round": round_t, "k": k,
            "mean_abs_err": float(err.mean()) if err.size else float("nan"),
            "p95_abs_err": float(np.quantile(err, 0.95)) if err.size else float("nan"),
            "max_abs_err": float(err.max()) if err.size else float("nan"),
            "cos_sybil_sybil": _group_mean(sn, pairs & np.outer(is_sybil, is_sybil)),
            "cos_honest_honest": _group_mean(sn, pairs & np.outer(~is_sybil, ~is_sybil)),
            "cos_sybil_honest": _group_mean(sn, pairs & (np.outer(is_sybil, ~is_sybil) | np.outer(~is_sybil, is_sybil))),
        })
    return det_rows, sim_rows
