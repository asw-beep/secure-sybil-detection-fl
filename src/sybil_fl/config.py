"""Typed configuration loaded from YAML.

Every hyperparameter and threshold comes from here (project rule). Unknown
keys are rejected so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PARTITIONS = ("iid", "dirichlet")
ATTACKS = ("none", "label_flip")
SYBIL_DATA = ("attacker_pool", "own_shard")
PROJECTION_GENERATORS = ("shake256", "pcg64")  # must match projection.GENERATORS
EVASIONS = ("none", "known_projection", "guessed_projection", "gaussian")  # must match attacks.EVASIONS
AGGREGATORS = ("fedavg", "median")  # must match aggregators.AGGREGATORS
DETECTORS = ("none", "full", "projected")
FALLBACKS = ("skip",)
NUM_CLASSES = 10  # MNIST


@dataclass
class DataConfig:
    dataset: str = "mnist"
    data_dir: str = "data"
    num_clients: int = 20
    partition: str = "dirichlet"
    dirichlet_alpha: float = 0.5
    min_client_samples: int = 50
    max_partition_retries: int = 100
    attacker_pool_size: int = 1000  # source-class images withheld from honest clients in every run (D9)

    def validate(self) -> None:
        if self.dataset != "mnist":
            raise ValueError(f"dataset: only 'mnist' is supported so far, got {self.dataset!r}")
        if self.partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}, got {self.partition!r}")
        if self.num_clients < 1:
            raise ValueError(f"num_clients must be >= 1, got {self.num_clients}")
        if self.dirichlet_alpha <= 0:
            raise ValueError(f"dirichlet_alpha must be > 0, got {self.dirichlet_alpha}")
        if self.min_client_samples < 0:
            raise ValueError(f"min_client_samples must be >= 0, got {self.min_client_samples}")
        if self.max_partition_retries < 1:
            raise ValueError(f"max_partition_retries must be >= 1, got {self.max_partition_retries}")
        if self.attacker_pool_size < 0:
            raise ValueError(f"attacker_pool_size must be >= 0, got {self.attacker_pool_size}")


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 0.05
    momentum: float = 0.0
    central_epochs: int = 2

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if not 0 <= self.momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")
        if self.central_epochs < 1:
            raise ValueError(f"central_epochs must be >= 1, got {self.central_epochs}")


@dataclass
class FLConfig:
    rounds: int = 20
    local_epochs: int = 1

    def validate(self) -> None:
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")
        if self.local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {self.local_epochs}")


@dataclass
class AttackConfig:
    """source_class/target_class also define the ASR logged in clean runs (the natural confusion rate).

    sybil_data: 'attacker_pool' (D9) adds num_sybils extra clients, each training on its own
    samples_per_sybil images drawn from the attacker pool; 'own_shard' (Phase 3 baseline) turns
    num_sybils of the honest partition's clients into Sybils that flip their own labels.
    """

    type: str = "none"
    num_sybils: int = 0
    source_class: int = 7
    target_class: int = 1
    sybil_data: str = "attacker_pool"
    samples_per_sybil: int = 500
    evasion: str = "none"        # E6 (D10): none | known_projection | guessed_projection | gaussian
    evasion_budget: float = 0.0  # ||perturbation|| / ||update|| for each Sybil

    def validate(self) -> None:
        if self.type not in ATTACKS:
            raise ValueError(f"attack type must be one of {ATTACKS}, got {self.type!r}")
        if self.evasion not in EVASIONS:
            raise ValueError(f"evasion must be one of {EVASIONS}, got {self.evasion!r}")
        if self.evasion_budget < 0:
            raise ValueError(f"evasion_budget must be >= 0, got {self.evasion_budget}")
        if self.evasion != "none" and self.type == "none":
            raise ValueError("evasion needs an attack (attack type is 'none')")
        if self.sybil_data not in SYBIL_DATA:
            raise ValueError(f"sybil_data must be one of {SYBIL_DATA}, got {self.sybil_data!r}")
        if self.samples_per_sybil < 1:
            raise ValueError(f"samples_per_sybil must be >= 1, got {self.samples_per_sybil}")
        if self.type == "none" and self.num_sybils != 0:
            raise ValueError(f"num_sybils must be 0 when attack type is 'none', got {self.num_sybils}")
        if self.type != "none" and self.num_sybils < 1:
            raise ValueError(f"num_sybils must be >= 1 for attack type {self.type!r}, got {self.num_sybils}")
        for name in ("source_class", "target_class"):
            value = getattr(self, name)
            if not 0 <= value < NUM_CLASSES:
                raise ValueError(f"{name} must be in [0, {NUM_CLASSES}), got {value}")
        if self.source_class == self.target_class:
            raise ValueError(f"source_class and target_class must differ, both are {self.source_class}")


@dataclass
class ProjectionConfig:
    """Secret projection (D3): k rows; generator shake256 (cryptographic) or pcg64 (cost comparison)."""

    k: int = 64
    generator: str = "shake256"

    def validate(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.generator not in PROJECTION_GENERATORS:
            raise ValueError(f"generator must be one of {PROJECTION_GENERATORS}, got {self.generator!r}")


@dataclass
class DefenceConfig:
    """Server-side defence. detector: none | full (full-dimensional counterpart) | projected (D3).

    distance_threshold: average-linkage cut on cosine distance (1 - cosine similarity), in (0, 2).
    min_cluster_size: clusters with at least this many clients are flagged.
    fallback: what happens when every client is flagged; 'skip' keeps the global model unchanged.
    """

    aggregator: str = "fedavg"
    detector: str = "none"
    distance_threshold: float = 0.5
    min_cluster_size: int = 3
    fallback: str = "skip"

    def validate(self) -> None:
        if self.aggregator not in AGGREGATORS:
            raise ValueError(f"aggregator must be one of {AGGREGATORS}, got {self.aggregator!r}")
        if self.detector not in DETECTORS:
            raise ValueError(f"detector must be one of {DETECTORS}, got {self.detector!r}")
        if not 0 < self.distance_threshold < 2:
            raise ValueError(f"distance_threshold must be in (0, 2), got {self.distance_threshold}")
        if self.min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {self.min_cluster_size}")
        if self.fallback not in FALLBACKS:
            raise ValueError(f"fallback must be one of {FALLBACKS}, got {self.fallback!r}")


@dataclass
class EvaluationConfig:
    """Shadow evaluation: score the detector on every round's updates without affecting aggregation.

    shadow_ks: projected dimensions to score (the full-dimensional detector is always scored when
    shadow evaluation is on); shadow_thresholds: distance thresholds to score. Empty lists = off.
    """

    shadow_ks: list[int] = field(default_factory=list)
    shadow_thresholds: list[float] = field(default_factory=list)

    def validate(self) -> None:
        if any(int(k) < 1 for k in self.shadow_ks):
            raise ValueError(f"shadow_ks entries must be >= 1, got {self.shadow_ks}")
        if any(not 0 < t < 2 for t in self.shadow_thresholds):
            raise ValueError(f"shadow_thresholds entries must be in (0, 2), got {self.shadow_thresholds}")

    @property
    def enabled(self) -> bool:
        return bool(self.shadow_ks or self.shadow_thresholds)


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    defence: DefenceConfig = field(default_factory=DefenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        self.data.validate()
        self.train.validate()
        self.fl.validate()
        self.attack.validate()
        self.projection.validate()
        self.defence.validate()
        self.evaluation.validate()
        if self.attack.type == "none":
            return
        if self.attack.sybil_data == "own_shard" and self.attack.num_sybils > self.data.num_clients:
            raise ValueError(
                f"num_sybils={self.attack.num_sybils} exceeds num_clients={self.data.num_clients} "
                f"(own_shard Sybils are honest-partition clients)"
            )
        if self.attack.sybil_data == "attacker_pool" and self.attack.samples_per_sybil > self.data.attacker_pool_size:
            raise ValueError(
                f"samples_per_sybil={self.attack.samples_per_sybil} exceeds "
                f"attacker_pool_size={self.data.attacker_pool_size}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, stored alongside every experiment result."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        raw = dict(raw or {})
        sections = {
            "data": DataConfig, "train": TrainConfig, "fl": FLConfig,
            "attack": AttackConfig, "projection": ProjectionConfig,
            "defence": DefenceConfig, "evaluation": EvaluationConfig,
        }
        _reject_unknown(raw, {f.name for f in fields(cls)}, where="top level")
        kwargs: dict[str, Any] = {}
        if "seed" in raw:
            kwargs["seed"] = int(raw["seed"])
        for name, section_cls in sections.items():
            sub = dict(raw.get(name) or {})
            _reject_unknown(sub, {f.name for f in fields(section_cls)}, where=name)
            kwargs[name] = section_cls(**sub)
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg


def _reject_unknown(raw: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown config key(s) in {where}: {unknown}")


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.from_dict(raw)
