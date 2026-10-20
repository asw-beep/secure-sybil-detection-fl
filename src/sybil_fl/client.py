"""FL client: train a private copy of the global model on the client's own indices.

The returned delta is the client's original update, flat(local) - flat(global). It is
what aggregation uses; any normalised copy a detector makes later must stay separate.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import TrainConfig
from .models import get_flat_params
from .training import train_epochs


@dataclass
class ClientUpdate:
    client_id: int
    delta: torch.Tensor  # original update, 1-D, length d
    num_samples: int     # FedAvg weight
    train_loss: float    # mean loss of the last local epoch


def local_update(
    global_model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    client_id: int,
    train_cfg: TrainConfig,
    local_epochs: int,
    seed: int,
) -> ClientUpdate:
    """Train a copy of global_model on x[indices]; the global model itself is left unchanged.

    A fresh SGD optimizer is created each call, so momentum does not carry over between rounds.
    """
    local = copy.deepcopy(global_model)
    loss = train_epochs(
        local, x, y, indices, epochs=local_epochs,
        batch_size=train_cfg.batch_size, lr=train_cfg.lr, momentum=train_cfg.momentum, seed=seed,
    )
    delta = get_flat_params(local) - get_flat_params(global_model)
    return ClientUpdate(client_id=client_id, delta=delta, num_samples=len(indices), train_loss=loss)
