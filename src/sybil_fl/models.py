"""Small CNN for MNIST and helpers to move between a model and its flat parameter vector.

Federated updates are handled as flat vectors: delta = flat(local) - flat(global).
The model has no BatchNorm, so its trainable parameters are its entire state
(docs/DECISIONS.md D5) and nothing else needs aggregating.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class SmallCNN(nn.Module):
    """conv(1→16,5) → pool → conv(16→32,5) → pool → fc(512→64) → fc(64→10). d = 46,730."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),   # 28 → 24
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 24 → 12
            nn.Conv2d(16, 32, kernel_size=5),  # 12 → 8
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 8 → 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(seed: int) -> SmallCNN:
    """Create a SmallCNN with reproducible initial weights, without touching global RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SmallCNN()


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """All parameters as one detached 1-D float tensor (a copy, safe to modify)."""
    return parameters_to_vector(model.parameters()).detach().clone()


def set_flat_params(model: nn.Module, vec: torch.Tensor) -> None:
    """Load a flat parameter vector into the model, in place."""
    expected = num_params(model)
    if vec.ndim != 1 or vec.numel() != expected:
        raise ValueError(f"expected a 1-D vector of length {expected}, got shape {tuple(vec.shape)}")
    if not torch.isfinite(vec).all():
        raise ValueError("parameter vector contains non-finite values (NaN/inf)")
    with torch.no_grad():
        vector_to_parameters(vec.to(next(model.parameters()).dtype), model.parameters())
