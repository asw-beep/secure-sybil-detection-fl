"""Aggregation rules. Each takes the clients' *original* flat updates (never normalised copies).

Inputs are validated explicitly: an empty set, unusable weights, mismatched lengths or
non-finite values raise ValueError instead of producing a silently wrong average.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def fedavg(deltas: Sequence[torch.Tensor], num_samples: Sequence[int]) -> torch.Tensor:
    """Sample-count-weighted mean of the updates: sum_i (n_i / sum_j n_j) * delta_i.

    Accumulates in float64 and returns the inputs' dtype. Inputs are not modified.
    """
    if len(deltas) == 0:
        raise ValueError("fedavg: no updates to aggregate")
    if len(num_samples) != len(deltas):
        raise ValueError(f"fedavg: got {len(deltas)} updates but {len(num_samples)} num_samples")
    shape = deltas[0].shape
    for i, d in enumerate(deltas):
        if d.ndim != 1 or d.shape != shape:
            raise ValueError(f"fedavg: update {i} has shape {tuple(d.shape)}, expected 1-D length {shape[0]}")
        if not torch.isfinite(d).all():
            raise ValueError(f"fedavg: update {i} contains non-finite values (NaN/inf)")

    weights = torch.as_tensor(list(num_samples), dtype=torch.float64)
    if (weights < 0).any():
        raise ValueError(f"fedavg: negative sample count in {list(num_samples)}")
    total = weights.sum()
    if total <= 0:
        raise ValueError("fedavg: total weight is zero")

    stacked = torch.stack([d.to(torch.float64) for d in deltas])
    return ((weights / total) @ stacked).to(deltas[0].dtype)


def coordinate_median(deltas: Sequence[torch.Tensor]) -> torch.Tensor:
    """Coordinate-wise median (unweighted; even counts average the two middle values)."""
    if len(deltas) == 0:
        raise ValueError("coordinate_median: no updates to aggregate")
    for i, d in enumerate(deltas):
        if not torch.isfinite(d).all():
            raise ValueError(f"coordinate_median: update {i} contains non-finite values (NaN/inf)")
    stacked = torch.stack([d.to(torch.float64) for d in deltas])
    return torch.quantile(stacked, 0.5, dim=0).to(deltas[0].dtype)


AGGREGATORS = ("fedavg", "median")


def aggregate(name: str, deltas: Sequence[torch.Tensor], num_samples: Sequence[int]) -> torch.Tensor:
    """Dispatch by config name. The median ignores sample counts (it is a robust statistic)."""
    if name == "fedavg":
        return fedavg(deltas, num_samples)
    if name == "median":
        return coordinate_median(deltas)
    raise ValueError(f"aggregator must be one of {AGGREGATORS}, got {name!r}")
