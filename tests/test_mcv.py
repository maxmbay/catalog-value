from __future__ import annotations

import numpy as np

from catalog_value.models.catalog_value.mcv import (
    affinity,
    marginal_content_value,
    value_of_catalog,
)
from catalog_value.models.types import AudienceStates, TitleReps
from catalog_value.optimization.greedy import greedy_mcv


def _toy_model() -> tuple[AudienceStates, TitleReps]:
    # Two users, two tastes. Titles 0 and 1 serve taste 0; 2 and 3 serve taste 1.
    z_users = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    pi = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    audience = AudienceStates(pi=pi, z=z_users, user_row=np.arange(2, dtype=np.int64))
    titles = TitleReps(
        z=np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float64),
        bias=np.zeros(4, dtype=np.float64),
        movie_row=np.arange(4, dtype=np.int64),
    )
    return audience, titles


def test_item_already_in_catalog_has_zero_mcv() -> None:
    audience, titles = _toy_model()
    catalog = np.array([0, 2], dtype=np.int64)
    mcv = marginal_content_value(audience, titles, catalog, catalog, tau=0.5)
    np.testing.assert_allclose(mcv, 0.0)


def test_near_duplicate_has_lower_mcv_once_similar_title_is_present() -> None:
    audience, titles = _toy_model()
    tau = 0.4
    empty = np.array([], dtype=np.int64)
    mcv_empty = marginal_content_value(
        audience, titles, empty, np.array([0, 1], dtype=np.int64), tau=tau
    )
    mcv_after = marginal_content_value(
        audience,
        titles,
        catalog=np.array([0], dtype=np.int64),
        candidates=np.array([1], dtype=np.int64),
        tau=tau,
    )
    # Title 1 is a near-substitute for 0, so its MCV drops after 0 is acquired.
    assert mcv_after[0] < mcv_empty[1]


def test_uncovered_taste_has_higher_mcv_than_redundant_title() -> None:
    audience, titles = _toy_model()
    catalog = np.array([0], dtype=np.int64)
    mcv = marginal_content_value(
        audience, titles, catalog, np.array([1, 2], dtype=np.int64), tau=0.5
    )
    redundant, uncovered = mcv
    assert uncovered > redundant


def test_catalog_value_increases_when_adding_a_useful_title() -> None:
    audience, titles = _toy_model()
    v0 = value_of_catalog(audience, titles, np.array([0], dtype=np.int64), tau=0.5)
    v1 = value_of_catalog(audience, titles, np.array([0, 2], dtype=np.int64), tau=0.5)
    assert v1.mean > v0.mean


def test_affinity_shape() -> None:
    audience, titles = _toy_model()
    aff = affinity(audience, titles, np.array([0, 3], dtype=np.int64))
    assert aff.shape == (2, 2, 2)


def test_greedy_prefers_covering_both_tastes() -> None:
    audience, titles = _toy_model()
    selected = greedy_mcv(
        audience,
        titles,
        candidates=np.arange(4, dtype=np.int64),
        catalog_size=2,
        tau=0.5,
    )
    selected_set = set(int(i) for i in selected)
    covers_taste0 = bool(selected_set & {0, 1})
    covers_taste1 = bool(selected_set & {2, 3})
    assert covers_taste0 and covers_taste1
