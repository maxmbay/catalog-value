"""Monte Carlo portfolio-adjusted content value: φ_i = E_S[MCV_i(S)]."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from catalog_value.models.catalog_value.mcv import marginal_content_value
from catalog_value.models.types import AudienceStates, TitleReps

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def portfolio_adjusted_value(
    audience: AudienceStates,
    titles: TitleReps,
    universe: IntArray,
    candidates: IntArray,
    *,
    n_catalogs: int,
    catalog_size: int,
    tau: float,
    seed: int,
) -> FloatArray:
    """Average MCV over random catalogs drawn from ``universe``."""
    rng = np.random.default_rng(seed)
    universe = np.asarray(universe, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    size = min(catalog_size, len(universe))
    acc = np.zeros(len(candidates), dtype=np.float64)
    for _ in range(n_catalogs):
        catalog = rng.choice(universe, size=size, replace=False)
        acc += marginal_content_value(audience, titles, catalog, candidates, tau)
    return acc / max(n_catalogs, 1)
