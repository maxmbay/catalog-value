from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

from catalog_value.models.types import TitleReps

FloatArray = NDArray[np.float64]


@dataclass
class CollaborativeFit:
    """Truncated SVD of mean-centered ratings plus item biases."""

    user_factors: FloatArray
    item_factors: FloatArray
    item_bias: FloatArray
    global_mean: float
    singular_values: FloatArray


def ratings_to_sparse(user_row: np.ndarray, movie_row: np.ndarray, rating: np.ndarray,
                      n_users: int, n_movies: int) -> csr_matrix:
    return coo_matrix(
        (rating.astype(np.float64), (user_row, movie_row)),
        shape=(n_users, n_movies),
        dtype=np.float64,
    ).tocsr()


def fit_svd(ratings: csr_matrix, dim: int) -> CollaborativeFit:
    """Collaborative backbone: residual SVD after global mean and item bias."""
    n_users, n_movies = ratings.shape
    nnz = ratings.nnz
    if nnz == 0:
        raise ValueError("Cannot fit SVD on an empty rating matrix")
    dim = min(dim, n_users - 1, n_movies - 1)
    if dim < 1:
        raise ValueError("Need at least 2 users and 2 movies to fit SVD")

    global_mean = float(ratings.data.mean())
    item_sum = np.asarray(ratings.sum(axis=0)).ravel()
    item_count = np.asarray(ratings.getnnz(axis=0)).ravel().astype(np.float64)
    item_count = np.maximum(item_count, 1.0)
    item_bias = item_sum / item_count - global_mean

    residuals = ratings.astype(np.float64).copy()
    residuals.data = residuals.data - global_mean - item_bias[residuals.indices]

    u, s, vt = svds(residuals, k=dim, which="LM")
    order = np.argsort(s)[::-1]
    u = np.ascontiguousarray(u[:, order])
    s = np.ascontiguousarray(s[order])
    vt = np.ascontiguousarray(vt[order, :])
    scale = np.sqrt(np.maximum(s, 0.0))
    user_factors = u * scale[None, :]
    item_factors = vt.T * scale[None, :]
    return CollaborativeFit(
        user_factors=user_factors.astype(np.float64),
        item_factors=item_factors.astype(np.float64),
        item_bias=item_bias.astype(np.float64),
        global_mean=global_mean,
        singular_values=s.astype(np.float64),
    )


def title_reps_from_fit(fit: CollaborativeFit) -> TitleReps:
    n = fit.item_factors.shape[0]
    return TitleReps(
        z=fit.item_factors,
        bias=fit.item_bias,
        movie_row=np.arange(n, dtype=np.int64),
    )


def save_fit(fit: CollaborativeFit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        user_factors=fit.user_factors,
        item_factors=fit.item_factors,
        item_bias=fit.item_bias,
        global_mean=np.array(fit.global_mean),
        singular_values=fit.singular_values,
    )


def load_fit(path: Path) -> CollaborativeFit:
    payload = np.load(path)
    return CollaborativeFit(
        user_factors=payload["user_factors"],
        item_factors=payload["item_factors"],
        item_bias=payload["item_bias"],
        global_mean=float(payload["global_mean"]),
        singular_values=payload["singular_values"],
    )
