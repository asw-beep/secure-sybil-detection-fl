import math

import numpy as np
import pytest
import torch
from torch import nn

from sybil_fl.metrics import attack_success_rate, confusion, detection_scores


class _Fixed(nn.Module):
    """Stub classifier: each input is a single number, the class it 'predicts'."""

    def forward(self, x):
        return nn.functional.one_hot(x.view(-1).long(), num_classes=10).float()


def test_asr_matches_hand_calculation():
    y = torch.tensor([7, 7, 7, 7, 1, 2])
    preds = torch.tensor([1, 1, 7, 3, 1, 1]).float().view(-1, 1)
    # source-class (7) samples: 4; predicted as target (1): 2 -> 0.5
    assert attack_success_rate(_Fixed(), preds, y, source=7, target=1) == 0.5


def test_asr_ignores_non_source_samples():
    y = torch.tensor([7, 7, 3, 3, 3])
    preds = torch.tensor([7, 7, 1, 1, 1]).float().view(-1, 1)
    assert attack_success_rate(_Fixed(), preds, y, source=7, target=1) == 0.0


def test_asr_is_batch_size_independent():
    y = torch.tensor([7] * 10)
    preds = torch.tensor([1, 7] * 5).float().view(-1, 1)
    assert attack_success_rate(_Fixed(), preds, y, 7, 1, batch_size=3) == 0.5


def test_asr_without_source_samples_raises():
    y = torch.tensor([1, 2, 3])
    with pytest.raises(ValueError, match="source class"):
        attack_success_rate(_Fixed(), y.float().view(-1, 1), y, source=7, target=1)


# ---------- detection metrics ----------


def test_confusion_counts():
    flagged = np.array([True, True, False, False, True])
    sybil = np.array([True, False, True, False, True])
    assert confusion(flagged, sybil) == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}


def test_scores_hand_case():
    s = detection_scores(tp=2, fp=1, fn=1, tn=6)
    assert s["precision"] == 2 / 3 and s["recall"] == 2 / 3 and s["fpr"] == 1 / 7
    assert math.isclose(s["f1"], 2 / 3)


def test_scores_no_predicted_positives():
    s = detection_scores(tp=0, fp=0, fn=4, tn=10)
    assert math.isnan(s["precision"])  # undefined, not silently 0 or 1
    assert s["recall"] == 0.0 and s["f1"] == 0.0 and s["fpr"] == 0.0


def test_scores_no_true_positives_in_ground_truth():
    s = detection_scores(tp=0, fp=2, fn=0, tn=8)  # clean run
    assert math.isnan(s["recall"]) and s["precision"] == 0.0 and s["fpr"] == 0.2


def test_scores_everything_empty_is_nan_not_an_error():
    s = detection_scores(0, 0, 0, 0)
    assert all(math.isnan(v) for v in s.values())
