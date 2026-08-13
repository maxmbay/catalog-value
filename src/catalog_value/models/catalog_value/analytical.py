from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def logsumexp_with_zero(x: FloatArray) -> FloatArray:
    """Stable log(1 + sum(exp(x))) = logsumexp([0, x ...]) over the last axis."""
    zeros = np.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    stacked = np.concatenate([zeros, x], axis=-1)
    peak = stacked.max(axis=-1, keepdims=True)
    return np.squeeze(peak, axis=-1) + np.log(np.exp(stacked - peak).sum(axis=-1))


def coverage_g(affinities: FloatArray, tau: float) -> FloatArray:
    """Soft coverage: τ log(1 + Σ exp(a_i / τ)).

    ``affinities`` has titles on the last axis. As τ → 0 this approaches the
    max affinity (strong substitution); as τ → ∞ it approaches additive value.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    return tau * logsumexp_with_zero(affinities / tau)


def catalog_value(
    pi: FloatArray,
    affinities: FloatArray,
    tau: float,
) -> FloatArray:
    """V_u(S) = Σ_k π_uk g({a_uki : i ∈ S}).

    Parameters
    ----------
    pi:
        ``[n_users, n_interests]``
    affinities:
        ``[n_users, n_interests, n_titles]``
    """
    per_taste = coverage_g(affinities, tau)
    return (pi * per_taste).sum(axis=-1)
