"""Plain SGD training and evaluation on an index subset of an in-memory dataset.

Used by the central sanity check (E0) now and by each FL client later. Minibatch
order comes from an explicit seeded torch.Generator, so a run is reproducible
and independent of global RNG state.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def train_epochs(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    seed: int,
) -> float:
    """Train `model` in place on x[indices], y[indices]. Returns the mean loss of the last epoch."""
    idx = torch.as_tensor(np.asarray(indices), dtype=torch.long)
    if idx.numel() == 0:
        raise ValueError("cannot train on an empty index set")
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    last_epoch_loss = float("nan")
    for _ in range(epochs):
        order = idx[torch.randperm(idx.numel(), generator=gen)]
        total, seen = 0.0, 0
        for start in range(0, order.numel(), batch_size):
            batch = order[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            opt.step()
            total += loss.item() * batch.numel()
            seen += batch.numel()
        last_epoch_loss = total / seen
    return last_epoch_loss


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1000) -> float:
    """Fraction of samples classified correctly."""
    model.eval()
    correct = 0
    for start in range(0, len(y), batch_size):
        preds = model(x[start:start + batch_size]).argmax(dim=1)
        correct += (preds == y[start:start + batch_size]).sum().item()
    return correct / len(y)
