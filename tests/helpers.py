"""Shared test data. Imported directly by test modules (pytest puts tests/ on sys.path)."""

import torch


def synthetic_digits(n_per_class=60, seed=0):
    """Learnable fake 'digits': class c lights a 4x4 block at a class-specific spot, plus noise."""
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in range(10):
        x = 0.3 * torch.randn(n_per_class, 1, 28, 28, generator=g)
        r, col = 2 + (c // 5) * 12, 2 + (c % 5) * 5
        x[:, :, r:r + 4, col:col + 4] += 3.0
        xs.append(x)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)
