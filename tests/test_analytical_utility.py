from __future__ import annotations

import numpy as np
import pytest

from catalog_value.models.catalog_value.analytical import (
    catalog_value,
    coverage_g,
    logsumexp_with_zero,
)


def test_logsumexp_with_zero_matches_naive_for_moderate_values() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3, 5))
    naive = np.log1p(np.exp(x).sum(axis=-1))
    np.testing.assert_allclose(logsumexp_with_zero(x), naive, rtol=1e-6, atol=1e-6)


def test_empty_catalog_has_zero_coverage() -> None:
    affinities = np.zeros((2, 3, 0))
    g = coverage_g(affinities, tau=0.5)
    np.testing.assert_allclose(g, 0.0)


def test_coverage_is_monotone_in_the_set() -> None:
    rng = np.random.default_rng(1)
    affinities = rng.normal(size=(6, 4))
    tau = 0.7
    v_small = coverage_g(affinities[:, :2], tau)
    v_large = coverage_g(affinities[:, :3], tau)
    assert np.all(v_large + 1e-12 >= v_small)


def test_coverage_is_submodular() -> None:
    rng = np.random.default_rng(2)
    tau = 0.5
    items = rng.normal(size=(8,))

    def g(idx: list[int]) -> float:
        if not idx:
            return float(coverage_g(np.zeros((1, 0)), tau)[0])
        return float(coverage_g(items[np.array(idx)][None, :], tau)[0])

    # Random nested sets A ⊆ B, item i ∉ B.
    for _ in range(20):
        perm = rng.permutation(8)
        a_size = int(rng.integers(0, 4))
        b_size = int(rng.integers(a_size, 7))
        a = perm[:a_size].tolist()
        b = perm[:b_size].tolist()
        i = int(perm[-1])
        delta_a = g(a + [i]) - g(a)
        delta_b = g(b + [i]) - g(b)
        assert delta_a + 1e-8 >= delta_b


def test_mixture_catalog_value_is_pi_weighted_sum() -> None:
    pi = np.array([[0.25, 0.75]], dtype=np.float64)
    affinities = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    tau = 1.0
    v = catalog_value(pi, affinities, tau)
    expected = 0.25 * coverage_g(affinities[:, 0, :], tau) + 0.75 * coverage_g(
        affinities[:, 1, :], tau
    )
    np.testing.assert_allclose(v, expected)


def test_tau_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tau"):
        coverage_g(np.zeros((1, 2)), tau=0.0)
