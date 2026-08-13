from __future__ import annotations

import numpy as np

from catalog_value.visualization.atlas import fit_atlas, short_title


def test_fit_atlas_is_two_dimensional_and_centered() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(40, 8))
    xy, pca = fit_atlas(z, seed=0)
    assert xy.shape == (40, 2)
    np.testing.assert_allclose(xy.mean(axis=0), 0.0, atol=1e-6)
    assert pca.explained_variance_ratio_.sum() > 0


def test_short_title_drops_year() -> None:
    assert short_title("Toy Story (1995)") == "Toy Story"
    assert short_title("Se7en") == "Se7en"
