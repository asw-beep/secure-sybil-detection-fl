"""Sybil attacks: label flipping, with Sybil data from an attacker pool (D9) or their own shards,
plus evasion perturbations of the Sybils' submitted updates (E6, D10).

Which clients are Sybils is ground truth for evaluation and logging only. It decides
which data and labels a client trains on here, and must never be passed to a detector.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .config import Config
from .data import Partition
from .utils import derive_seed


@dataclass
class AttackPlan:
    sybil_ids: np.ndarray            # ground truth (sorted positions in client_indices): evaluation and logging only
    sybil_labels: torch.Tensor       # training labels the Sybil clients train on
    flipped_per_sybil: list[int]     # labels actually changed, per Sybil, in sybil_ids order
    client_indices: list[np.ndarray]  # every client's training indices, honest and Sybil, in client-id order


def select_sybils(num_clients: int, num_sybils: int, seed: int) -> np.ndarray:
    """Choose num_sybils distinct client ids uniformly at random; returned sorted."""
    if not 0 <= num_sybils <= num_clients:
        raise ValueError(f"num_sybils must be in [0, {num_clients}], got {num_sybils}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_clients, size=num_sybils, replace=False)).astype(np.int64)


def label_flip_labels(
    y: torch.Tensor, partition: Partition, sybil_ids: Sequence[int], source: int, target: int
) -> torch.Tensor:
    """Copy of y in which the Sybils' source-class samples are relabelled as target.

    Samples of honest clients, and the Sybils' other samples, keep their labels;
    y itself is not modified.
    """
    ids = [int(i) for i in sybil_ids]
    if len(set(ids)) != len(ids) or any(not 0 <= i < len(partition) for i in ids):
        raise ValueError(f"invalid sybil ids {ids} for {len(partition)} clients (must be distinct and in range)")
    poisoned = y.clone()
    for i in ids:
        idx = torch.as_tensor(partition[i], dtype=torch.long)
        poisoned[idx[y[idx] == source]] = target
    return poisoned


def prepare_attack(
    cfg: Config, y_train: torch.Tensor, partition: Partition, pool: np.ndarray | None = None
) -> AttackPlan | None:
    """Build the configured attack, or return None for a clean run.

    attacker_pool: num_sybils extra clients join the honest ones at random positions; each
    draws its own samples_per_sybil pool images (without replacement, overlap between Sybils
    allowed) and trains on them with source relabelled as target.
    own_shard: num_sybils honest-partition clients flip the source labels in their own shards.
    """
    a = cfg.attack
    if a.type == "none":
        return None
    if a.type != "label_flip":  # config validation should make this unreachable
        raise ValueError(f"unknown attack type {a.type!r}")

    if a.sybil_data == "own_shard":
        sybils = select_sybils(len(partition), a.num_sybils, derive_seed(cfg.seed, "sybil_selection"))
        roster = list(partition)
    else:
        if pool is None or len(pool) < a.samples_per_sybil:
            raise ValueError(
                f"attacker_pool needs a pool of at least samples_per_sybil={a.samples_per_sybil} images, "
                f"got {0 if pool is None else len(pool)}"
            )
        total = len(partition) + a.num_sybils
        sybils = select_sybils(total, a.num_sybils, derive_seed(cfg.seed, "sybil_selection"))
        honest_shards = iter(partition)
        roster = []
        for i in range(total):
            if i in sybils:
                rng = np.random.default_rng(derive_seed(cfg.seed, f"sybil_{i}_pool_sample"))
                roster.append(np.sort(rng.choice(pool, size=a.samples_per_sybil, replace=False)).astype(np.int64))
            else:
                roster.append(next(honest_shards))

    poisoned = label_flip_labels(y_train, roster, sybils, a.source_class, a.target_class)
    flipped = [int((poisoned[roster[i]] != y_train[roster[i]]).sum()) for i in sybils]
    return AttackPlan(sybil_ids=sybils, sybil_labels=poisoned, flipped_per_sybil=flipped, client_indices=roster)


EVASIONS = ("none", "known_projection", "guessed_projection", "gaussian")


def evasion_perturbations(
    deltas: torch.Tensor, mode: str, budget: float, seed: int, M: torch.Tensor | None = None
) -> torch.Tensor:
    """Perturbations p_i (one per row of `deltas`) with ||p_i|| = budget * ||delta_i|| exactly.

    gaussian: isotropic random direction.
    known_projection / guessed_projection: the minimum-norm preimage under M of a random k-dim
    direction, p = M^T (M M^T)^-1 v, i.e. a change confined to M's row space. With M = the
    server's secret R_t (key compromised) the projection magnifies it by about sqrt(d/k), so a
    small change in the full update scrambles the projected one. With M = the attacker's own
    guess, it is unrelated to R_t and behaves like random noise. Each Sybil draws its own v.
    The attacker adds p_i to its update before submitting it; the server aggregates the sum.
    """
    if mode not in EVASIONS or mode == "none":
        raise ValueError(f"evasion mode must be one of {EVASIONS[1:]}, got {mode!r}")
    if budget < 0:
        raise ValueError(f"evasion budget must be >= 0, got {budget}")
    n, d = deltas.shape
    g = torch.Generator().manual_seed(seed)
    if mode == "gaussian":
        directions = torch.randn(n, d, generator=g, dtype=torch.float64)
    else:
        if M is None:
            raise ValueError(f"evasion mode {mode!r} needs a projection matrix M")
        M64 = M.to(torch.float64)
        v = torch.randn(n, M.shape[0], generator=g, dtype=torch.float64)
        w = torch.linalg.solve(M64 @ M64.T, v.T)   # k x n
        directions = (M64.T @ w).T                 # n x d, rows in M's row space
    unit = directions / directions.norm(dim=1, keepdim=True)
    scale = budget * deltas.to(torch.float64).norm(dim=1, keepdim=True)
    return (unit * scale).to(deltas.dtype)


def attacker_guess_key(seed: int, round_t: int) -> bytes:
    """The guessing attacker's own 32-byte key for round t (its randomness, unrelated to K)."""
    return hashlib.sha256(f"sybil-fl|attacker-guess|seed={seed}|round={round_t}".encode("ascii")).digest()
