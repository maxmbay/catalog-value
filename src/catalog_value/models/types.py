from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class AudienceStates:
    """Multi-interest user representation Z_u = {(π_uk, z_uk)}.

    Attributes
    ----------
    pi:
        Mixing weights, shape ``[n_users, n_interests]``, rows sum to 1.
    z:
        Taste embeddings, shape ``[n_users, n_interests, dim]``.
    user_row:
        Row index into the collaborative user factor matrix (for joins).
    """

    pi: FloatArray
    z: FloatArray
    user_row: IntArray

    def __post_init__(self) -> None:
        n_users, k = self.pi.shape
        if self.z.shape[:2] != (n_users, k):
            raise ValueError(f"pi {self.pi.shape} incompatible with z {self.z.shape}")
        if self.user_row.shape != (n_users,):
            raise ValueError("user_row must be [n_users]")


@dataclass
class TitleReps:
    """Point title representations."""

    z: FloatArray
    bias: FloatArray
    movie_row: IntArray

    def __post_init__(self) -> None:
        n = self.z.shape[0]
        if self.bias.shape != (n,) or self.movie_row.shape != (n,):
            raise ValueError("z, bias, movie_row must share leading dimension")


@dataclass
class TitlePosterior:
    """Isotropic Gaussian posterior over title embeddings: z_i ~ N(μ_i, σ_i² I)."""

    mu: FloatArray
    var: FloatArray
    bias: FloatArray
    movie_row: IntArray

    def __post_init__(self) -> None:
        n = self.mu.shape[0]
        if self.var.shape != (n,) or self.bias.shape != (n,) or self.movie_row.shape != (n,):
            raise ValueError("mu, var, bias, movie_row must share leading dimension")

    def mean_reps(self) -> TitleReps:
        return TitleReps(z=self.mu, bias=self.bias, movie_row=self.movie_row)

    def sample(self, rng: np.random.Generator) -> TitleReps:
        noise = rng.normal(size=self.mu.shape)
        z = self.mu + noise * np.sqrt(self.var)[:, None]
        return TitleReps(z=z, bias=self.bias, movie_row=self.movie_row)


@dataclass
class CatalogValueEstimate:
    """V(S) and optional per-user values. Later: posterior mean/variance."""

    mean: float
    per_user: FloatArray | None = None
    variance: float | None = None
