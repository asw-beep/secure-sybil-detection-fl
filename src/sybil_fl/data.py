"""Dataset loading and client partitioning.

Clients hold *index arrays* into one in-memory dataset tensor. That makes the
key correctness property easy to check: every training sample is assigned to
exactly one client (see validate_partition).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import DataConfig

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

Partition = list[np.ndarray]


# ---------------------------------------------------------------- loading

def load_mnist(data_dir: str | Path, download: bool = True):
    """Return ((x_train, y_train), (x_test, y_test)) as normalised tensors.

    x: float32, shape (N, 1, 28, 28), normalised with the standard MNIST
    mean/std. y: int64, shape (N,).
    """
    from torchvision.datasets import MNIST  # imported lazily: tests without MNIST stay fast

    def _to_tensors(ds):
        x = ds.data.float().div_(255.0).sub_(MNIST_MEAN).div_(MNIST_STD).unsqueeze(1)
        y = ds.targets.long()
        return x, y

    root = str(data_dir)
    train = MNIST(root, train=True, download=download)
    test = MNIST(root, train=False, download=download)
    return _to_tensors(train), _to_tensors(test)


# ---------------------------------------------------------------- partitioning

def partition_iid(n_samples: int, num_clients: int, seed: int) -> Partition:
    """Shuffle all indices and split them into num_clients near-equal parts."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    return [part.astype(np.int64) for part in np.array_split(perm, num_clients)]


def partition_dirichlet(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 0,
    max_retries: int = 100,
) -> Partition:
    """Label-skewed (non-IID) split.

    For each class c, draw client proportions p_c ~ Dirichlet(alpha, ..., alpha)
    and split that class's samples across clients according to p_c. Small alpha
    concentrates each class on few clients (strong skew); large alpha tends to IID.

    The draw is repeated (up to max_retries) until every client has at least
    min_size samples; a ValueError is raised if that never happens.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if min_size * num_clients > n:
        raise ValueError(
            f"min_size={min_size} x num_clients={num_clients} exceeds the {n} available samples"
        )

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(max_retries):
        buckets: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(labels == c))
            props = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
            for client, chunk in enumerate(np.split(idx, cuts)):
                buckets[client].append(chunk)
        parts = [np.concatenate(b).astype(np.int64) for b in buckets]
        if min(len(p) for p in parts) >= min_size:
            return parts

    raise ValueError(
        f"could not satisfy min_size={min_size} in {max_retries} Dirichlet draws "
        f"(alpha={alpha}, num_clients={num_clients}); lower min_client_samples or raise alpha"
    )


def validate_partition(parts: Partition, n_samples: int) -> None:
    """Raise ValueError unless parts cover range(n_samples) exactly once."""
    flat = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    counts = np.bincount(flat, minlength=n_samples) if len(flat) else np.zeros(n_samples, int)
    if len(counts) > n_samples:
        raise ValueError(f"partition contains indices >= n_samples={n_samples}")
    dup = np.flatnonzero(counts > 1)
    if len(dup):
        raise ValueError(f"{len(dup)} sample(s) assigned more than once, e.g. index {dup[0]}")
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        raise ValueError(f"{len(missing)} sample(s) not assigned to any client, e.g. index {missing[0]}")


def make_partition(labels: np.ndarray, cfg: DataConfig, seed: int) -> Partition:
    """Build the partition the config asks for, and check it before returning."""
    labels = np.asarray(labels)
    if cfg.partition == "iid":
        parts = partition_iid(len(labels), cfg.num_clients, seed)
    elif cfg.partition == "dirichlet":
        parts = partition_dirichlet(
            labels,
            cfg.num_clients,
            cfg.dirichlet_alpha,
            seed,
            min_size=cfg.min_client_samples,
            max_retries=cfg.max_partition_retries,
        )
    else:  # config validation should make this unreachable
        raise ValueError(f"unknown partition {cfg.partition!r}")
    validate_partition(parts, len(labels))
    return parts


# ---------------------------------------------------------------- attacker pool (DECISIONS D9)

def reserve_attacker_pool(
    labels: np.ndarray, size: int, source_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Withhold `size` random source-class samples for the attacker.

    Returns (pool, honest): sorted index arrays that together cover range(len(labels)) once.
    """
    labels = np.asarray(labels)
    candidates = np.flatnonzero(labels == source_class)
    if size > len(candidates):
        raise ValueError(
            f"attacker_pool_size={size} exceeds the {len(candidates)} samples of class {source_class}"
        )
    rng = np.random.default_rng(seed)
    pool = np.sort(rng.choice(candidates, size=size, replace=False)).astype(np.int64)
    honest = np.setdiff1d(np.arange(len(labels)), pool).astype(np.int64)
    return pool, honest


def make_federated_split(
    labels: np.ndarray, cfg: DataConfig, pool_class: int, partition_seed: int, pool_seed: int
) -> tuple[Partition, np.ndarray]:
    """Reserve the attacker pool, then partition the remaining samples among honest clients.

    Returns (parts, pool) as indices into labels. The pool is reserved in every run, clean
    ones included, so clean and attacked runs share identical honest clients. With
    attacker_pool_size = 0 the result equals make_partition(labels, cfg, partition_seed).
    """
    labels = np.asarray(labels)
    pool, honest = reserve_attacker_pool(labels, cfg.attacker_pool_size, pool_class, pool_seed)
    parts = [honest[p] for p in make_partition(labels[honest], cfg, partition_seed)]
    validate_partition(parts + [pool], len(labels))
    return parts, pool

