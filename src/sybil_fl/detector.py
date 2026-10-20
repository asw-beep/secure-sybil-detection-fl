"""Similarity-based Sybil detector, full-dimensional or on the secret projection.

Pipeline for one round (identical for both methods except step 2):
  1. Detector vectors: an L2-normalised *copy* of each update. Zero-norm or non-finite updates
     are marked degenerate: never clustered, never flagged. Original updates are untouched and
     remain what aggregation uses.
  2. method="projected": z_i = R_t u_i with the round's secret matrix (projection.py), then
     normalised again (near-zero projections are degenerate). method="full": skip.
  3. Cosine similarity matrix S = V V^T.
  4. Average-linkage agglomerative clustering on cosine distance 1 - S (scipy), cut at
     `distance_threshold`: two groups end up in one cluster only if their average pairwise
     cosine distance is <= the threshold.
  5. Every member of a cluster with at least `min_cluster_size` clients is flagged.

The detector sees only the updates and its own settings, never which clients are Sybils.
Similarity is a heuristic: honest non-IID clients can look alike, so false positives are measured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .projection import project

METHODS = ("full", "projected")
NORM_EPS = 1e-12


@dataclass
class DetectionResult:
    flagged: np.ndarray        # bool per client
    valid: np.ndarray          # bool per client: False for degenerate vectors
    similarity: torch.Tensor   # n x n cosine similarities (degenerate rows/cols are 0)
    seconds: dict[str, float]  # "projection", "similarity", "clustering"


def detector_vectors(x: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    """Row-normalised copy of x; degenerate rows (zero norm or non-finite) are zeroed and marked invalid."""
    x = x.detach().to(torch.float32)
    finite = torch.isfinite(x).all(dim=1)
    safe = torch.where(finite[:, None], x, torch.zeros_like(x))
    norms = safe.norm(dim=1)
    valid = finite & (norms > NORM_EPS)
    v = torch.zeros_like(safe)
    v[valid] = safe[valid] / norms[valid, None]
    return v, valid.numpy()


def cosine_similarity_matrix(v: torch.Tensor) -> torch.Tensor:
    """S = V V^T for row-normalised V, clamped to [-1, 1] and symmetrised against float error."""
    s = (v @ v.T).clamp(-1.0, 1.0)
    return (s + s.T) / 2


def average_linkage(similarity: torch.Tensor, valid: np.ndarray):
    """Linkage matrix over the valid clients (None if fewer than two)."""
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return None
    dist = (1.0 - similarity[np.ix_(idx, idx)].double()).clamp(min=0.0).numpy()
    np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method="average")


def flags_from_linkage(link, valid: np.ndarray, distance_threshold: float, min_cluster_size: int) -> np.ndarray:
    """Cut the dendrogram at the threshold and flag members of clusters with >= min_cluster_size clients."""
    flagged = np.zeros(len(valid), dtype=bool)
    if link is None:
        return flagged
    labels = fcluster(link, t=distance_threshold, criterion="distance")
    sizes = np.bincount(labels)
    flagged[np.flatnonzero(valid)] = sizes[labels] >= min_cluster_size
    return flagged


def cluster_flags(
    similarity: torch.Tensor, valid: np.ndarray, distance_threshold: float, min_cluster_size: int
) -> np.ndarray:
    return flags_from_linkage(average_linkage(similarity, valid), valid, distance_threshold, min_cluster_size)


def detect(
    deltas: torch.Tensor,
    method: str,
    distance_threshold: float,
    min_cluster_size: int,
    R: torch.Tensor | None = None,
) -> DetectionResult:
    """Flag suspiciously similar groups among one round's updates (rows of `deltas`)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "projected" and R is None:
        raise ValueError("method 'projected' needs the round's projection matrix R")

    t0 = time.perf_counter()
    v, valid = detector_vectors(deltas)
    if method == "projected":
        z = torch.zeros(len(v), R.shape[0])
        if valid.any():
            z[valid] = project(v[valid], R)
        v, valid_z = detector_vectors(z)
        valid = valid & valid_z
    t1 = time.perf_counter()
    s = cosine_similarity_matrix(v)
    t2 = time.perf_counter()
    flagged = cluster_flags(s, valid, distance_threshold, min_cluster_size)
    t3 = time.perf_counter()
    return DetectionResult(
        flagged=flagged, valid=valid, similarity=s,
        seconds={"projection": t1 - t0, "similarity": t2 - t1, "clustering": t3 - t2},
    )
