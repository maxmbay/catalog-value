from __future__ import annotations

import numpy as np

from catalog_value.models.catalog_value.pacv import portfolio_adjusted_value
from catalog_value.models.types import AudienceStates, TitleReps


def test_pacv_prefers_the_uncovered_direction_on_average() -> None:
    z_users = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    audience = AudienceStates(
        pi=np.array([[0.9, 0.1], [0.1, 0.9]]),
        z=z_users,
        user_row=np.arange(2, dtype=np.int64),
    )
    titles = TitleReps(
        z=np.array([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]], dtype=np.float64),
        bias=np.zeros(3),
        movie_row=np.arange(3, dtype=np.int64),
    )
    # Universe is mostly taste-0 titles, so the taste-1 title should win PACV.
    phi = portfolio_adjusted_value(
        audience,
        titles,
        universe=np.array([0, 1], dtype=np.int64),
        candidates=np.array([1, 2], dtype=np.int64),
        n_catalogs=12,
        catalog_size=1,
        tau=0.5,
        seed=0,
    )
    redundant, uncovered = phi
    assert uncovered > redundant
