from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from catalog_value.paths import config_dir, project_root


@dataclass(frozen=True)
class DataConfig:
    movielens_url: str
    min_user_ratings: int
    min_movie_ratings: int


@dataclass(frozen=True)
class RepresentationConfig:
    embedding_dim: int
    n_interests: int


@dataclass(frozen=True)
class CatalogValueConfig:
    tau: float
    n_eval_users: int


@dataclass(frozen=True)
class PhaseAConfig:
    catalog_size: int
    n_candidates: int
    n_annotate: int


@dataclass(frozen=True)
class TrainConfig:
    backbone: str
    batch_size: int
    epochs: int
    lr: float
    max_history: int
    n_holdout: int
    n_heads: int
    diversity_weight: float
    entropy_weight: float


@dataclass(frozen=True)
class PhaseBConfig:
    holdout_frac: float
    min_train_ratings: int
    epochs: int
    hidden: int
    lr: float
    n0: float
    n_posterior_samples: int


@dataclass(frozen=True)
class PhaseDConfig:
    n_shapley_catalogs: int
    shapley_catalog_size: int
    n_shapley_candidates: int
    greedy_size: int


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    representation: RepresentationConfig
    catalog_value: CatalogValueConfig
    phase_a: PhaseAConfig
    phase_b: PhaseBConfig
    phase_d: PhaseDConfig
    train: TrainConfig
    source: Path


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path) if path is not None else config_dir() / "phase_a.yaml"
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text())
    train_raw = raw.get("train", {})
    phase_b_raw = raw.get("phase_b", {})
    phase_d_raw = raw.get("phase_d", {})
    return Config(
        seed=int(raw["seed"]),
        data=DataConfig(**raw["data"]),
        representation=RepresentationConfig(**raw["representation"]),
        catalog_value=CatalogValueConfig(**raw["catalog_value"]),
        phase_a=PhaseAConfig(**raw["phase_a"]),
        phase_b=PhaseBConfig(
            holdout_frac=float(phase_b_raw.get("holdout_frac", 0.2)),
            min_train_ratings=int(phase_b_raw.get("min_train_ratings", 200)),
            epochs=int(phase_b_raw.get("epochs", 80)),
            hidden=int(phase_b_raw.get("hidden", 256)),
            lr=float(phase_b_raw.get("lr", 1e-3)),
            n0=float(phase_b_raw.get("n0", 400.0)),
            n_posterior_samples=int(phase_b_raw.get("n_posterior_samples", 16)),
        ),
        phase_d=PhaseDConfig(
            n_shapley_catalogs=int(phase_d_raw.get("n_shapley_catalogs", 80)),
            shapley_catalog_size=int(phase_d_raw.get("shapley_catalog_size", 60)),
            n_shapley_candidates=int(phase_d_raw.get("n_shapley_candidates", 40)),
            greedy_size=int(phase_d_raw.get("greedy_size", 40)),
        ),
        train=TrainConfig(
            backbone=str(train_raw.get("backbone", "taste_tokens")),
            batch_size=int(train_raw.get("batch_size", 256)),
            epochs=int(train_raw.get("epochs", 2)),
            lr=float(train_raw.get("lr", 1e-3)),
            max_history=int(train_raw.get("max_history", 64)),
            n_holdout=int(train_raw.get("n_holdout", 4)),
            n_heads=int(train_raw.get("n_heads", 4)),
            diversity_weight=float(train_raw.get("diversity_weight", 0.05)),
            entropy_weight=float(train_raw.get("entropy_weight", 0.01)),
        ),
        source=config_path,
    )
