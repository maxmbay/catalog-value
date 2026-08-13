"""A shared 2-D title map. PCA is a camera on the learned embeddings, not the model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

FloatArray = NDArray[np.float64]


def fit_atlas(z: FloatArray, *, seed: int = 0) -> tuple[FloatArray, PCA]:
    pca = PCA(n_components=2, random_state=seed)
    xy = pca.fit_transform(z)
    return xy.astype(np.float64), pca


def save_atlas(path: Path, xy: FloatArray, explained: FloatArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xy=xy, explained=np.asarray(explained, dtype=np.float64))


def load_atlas(path: Path) -> tuple[FloatArray, FloatArray]:
    payload = np.load(path)
    return payload["xy"], payload["explained"]


def short_title(title: str) -> str:
    if title.endswith(")") and "(" in title:
        return title.rsplit(" (", 1)[0]
    return title
