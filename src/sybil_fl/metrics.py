"""Attack-effect and detection metrics. (Clean accuracy is training.evaluate_accuracy.)

Detection ground truth (which clients are Sybils) is used here, after detection, and never
inside the detector. Undefined ratios are NaN rather than silently 0 or 1.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def attack_success_rate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, source: int, target: int, batch_size: int = 1000
) -> float:
    """Label-flipping ASR: fraction of source-class test samples the model predicts as target."""
    xs = x[y == source]
    if len(xs) == 0:
        raise ValueError(f"no test samples of source class {source}; ASR is undefined")
    model.eval()
    hits = 0
    for start in range(0, len(xs), batch_size):
        hits += (model(xs[start:start + batch_size]).argmax(dim=1) == target).sum().item()
    return hits / len(xs)


def confusion(flagged: np.ndarray, is_sybil: np.ndarray) -> dict[str, int]:
    """True/false positives/negatives of a flag vector against ground truth."""
    f, s = np.asarray(flagged, dtype=bool), np.asarray(is_sybil, dtype=bool)
    if f.shape != s.shape:
        raise ValueError(f"flagged has shape {f.shape} but is_sybil has shape {s.shape}")
    return {
        "tp": int((f & s).sum()), "fp": int((f & ~s).sum()),
        "fn": int((~f & s).sum()), "tn": int((~f & ~s).sum()),
    }


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def detection_scores(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """precision, recall, F1 = 2TP / (2TP + FP + FN), FPR = FP / (FP + TN); NaN when undefined.

    Summed counts over rounds give micro-averaged scores, which avoids undefined per-round values.
    """
    return {
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "fpr": _ratio(fp, fp + tn),
    }
