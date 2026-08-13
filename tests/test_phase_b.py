from __future__ import annotations

import numpy as np
import torch

from catalog_value.data.genome import parse_year
from catalog_value.models.catalog_value.posterior_mcv import mcv_moments
from catalog_value.models.content.encoder import ContentEncoder
from catalog_value.models.content.posterior import residual_variance, shrink_posterior
from catalog_value.models.types import AudienceStates, TitlePosterior, TitleReps


def test_parse_year_from_movielens_title() -> None:
    assert parse_year("Toy Story (1995)") == 1995.0
    assert parse_year("Se7en") is None


def test_shrinkage_is_content_when_unobserved_and_collab_when_infinite() -> None:
    z_c = np.ones((3, 2))
    z_d = np.zeros((3, 2))
    bias_c = np.ones(3)
    bias_d = np.zeros(3)
    rows = np.arange(3, dtype=np.int64)
    cold = shrink_posterior(
        z_d, z_c, bias_d, bias_c, np.array([0, 0, 0]), rows, n0=100.0, content_var=0.25
    )
    np.testing.assert_allclose(cold.mu, z_c)
    np.testing.assert_allclose(cold.var, 0.25)
    hot = shrink_posterior(
        z_d,
        z_c,
        bias_d,
        bias_c,
        np.array([1e9, 1e9, 1e9]),
        rows,
        n0=100.0,
        content_var=0.25,
    )
    np.testing.assert_allclose(hot.mu, z_d, atol=1e-6)
    assert np.all(hot.var < 1e-4)


def test_holdout_mask_zeros_collaborative_precision() -> None:
    z_c = np.array([[1.0, 0.0], [1.0, 0.0]])
    z_d = np.array([[0.0, 1.0], [0.0, 1.0]])
    post = shrink_posterior(
        z_d,
        z_c,
        np.zeros(2),
        np.ones(2),
        np.array([5000, 5000]),
        np.arange(2, dtype=np.int64),
        n0=100.0,
        content_var=0.2,
        collab_mask=np.array([True, False]),
    )
    assert post.mu[1, 0] == 1.0
    assert post.mu[0, 0] < 0.5


def test_residual_variance_and_encoder_shapes() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(10, 4))
    hat = z + 0.1 * rng.normal(size=z.shape)
    var = residual_variance(z, hat, np.ones(10, dtype=bool))
    assert 0 < var < 1
    model = ContentEncoder(in_dim=7, out_dim=4, hidden=8)
    x = torch.randn(5, 7)
    z_hat, b_hat = model(x)
    assert z_hat.shape == (5, 4)
    assert b_hat.shape == (5,)


def test_mcv_moments_have_positive_spread_when_var_is_large() -> None:
    z_users = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    audience = AudienceStates(
        pi=np.array([[0.5, 0.5]]), z=z_users, user_row=np.array([0], dtype=np.int64)
    )
    titles = TitleReps(
        z=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        bias=np.zeros(2),
        movie_row=np.arange(2, dtype=np.int64),
    )
    posterior = TitlePosterior(
        mu=titles.z,
        var=np.full(2, 0.5),
        bias=titles.bias,
        movie_row=titles.movie_row,
    )
    mean, std = mcv_moments(
        audience,
        posterior,
        catalog=np.array([0], dtype=np.int64),
        candidates=np.array([1], dtype=np.int64),
        tau=0.5,
        n_samples=12,
        seed=0,
    )
    assert mean.shape == (1,)
    assert std.shape == (1,)
    assert np.all(std > 0)
