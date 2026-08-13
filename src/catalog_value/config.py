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
class Config:
    seed: int
    data: DataConfig
    representation: RepresentationConfig
    catalog_value: CatalogValueConfig
    phase_a: PhaseAConfig
    source: Path


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path) if path is not None else config_dir() / "phase_a.yaml"
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text())
    return Config(
        seed=int(raw["seed"]),
        data=DataConfig(**raw["data"]),
        representation=RepresentationConfig(**raw["representation"]),
        catalog_value=CatalogValueConfig(**raw["catalog_value"]),
        phase_a=PhaseAConfig(**raw["phase_a"]),
        source=config_path,
    )
