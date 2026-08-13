from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from catalog_value.models.catalog_value.mcv import marginal_content_value
from catalog_value.models.types import AudienceStates, TitleReps

IntArray = NDArray[np.int64]


def greedy_mcv(
    audience: AudienceStates,
    titles: TitleReps,
    candidates: IntArray,
    catalog_size: int,
    tau: float,
) -> IntArray:
    """Build a catalog by repeatedly adding the title with highest MCV."""
    remaining = [int(i) for i in candidates]
    selected: list[int] = []
    for _ in range(min(catalog_size, len(remaining))):
        mcv = marginal_content_value(
            audience,
            titles,
            catalog=np.asarray(selected, dtype=np.int64),
            candidates=np.asarray(remaining, dtype=np.int64),
            tau=tau,
        )
        best_pos = int(np.argmax(mcv))
        selected.append(remaining.pop(best_pos))
    return np.asarray(selected, dtype=np.int64)
