"""Typed, validated experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class Paths:
    labels: Path
    manifest: Path
    splits: Path
    profiles: Path
    artifacts: Path
    knowledge: Path
    references: Path


@dataclass(frozen=True)
class FeatureConfig:
    min_prevalence: int = 10
    max_families: int = 2000
    block_correlation: float = 0.98
    colocalization_k: int = 5
    contig_edge_bp: int = 200
    promoter_bp: int = 200
    near_mobile_cds: int = 3
    min_identity: float = 0.8
    min_coverage: float = 0.9


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 64
    query: int = 32
    dropout: float = 0.2
    context_layer: bool = True
    use_prior: bool = True
    interactions: bool = True
    drug_features: bool = True
    l1_edges: float = 1e-3
    l1_interactions: float = 1e-3
    l2_drug: float = 1e-3


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    patience: int = 8
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ensemble_seeds: int = 5
    threads: int = 8
    workers: int = 32


@dataclass(frozen=True)
class Config:
    paths: Paths
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)


def _build(cls: type[T], data: dict[str, Any] | None, where: str) -> T:
    data = data or {}
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Unknown {where} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> Config:
    """Load a YAML config; relative paths resolve against the config file's directory."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {"paths", "features", "model", "training"}
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    paths = {key: (path.parent / value).resolve() for key, value in (raw.get("paths") or {}).items()}
    return Config(
        paths=_build(Paths, paths, "paths"),
        features=_build(FeatureConfig, raw.get("features"), "features"),
        model=_build(ModelConfig, raw.get("model"), "model"),
        training=_build(TrainConfig, raw.get("training"), "training"),
    )
