"""Monte Carlo MCV under a title posterior."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from catalog_value.models.catalog_value.mcv import marginal_content_value
from catalog_value.models.types import AudienceStates, TitlePosterior

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def mcv_moments(
    audience: AudienceStates,
    posterior: TitlePosterior,
    catalog: IntArray,
    candidates: IntArray,
    tau: float,
    n_samples: int,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Return (mean, std) of MCV_i(S) over posterior draws of z."""
    rng = np.random.default_rng(seed)
    draws = np.zeros((n_samples, len(candidates)), dtype=np.float64)
    for s in range(n_samples):
        titles = posterior.sample(rng)
        draws[s] = marginal_content_value(audience, titles, catalog, candidates, tau)
    return draws.mean(axis=0), draws.std(axis=0)
