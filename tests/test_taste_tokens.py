from __future__ import annotations

import torch

from catalog_value.models.audience.taste_tokens import TasteTokenEncoder


def test_encode_history_shapes_and_pi_simplex() -> None:
    torch.manual_seed(0)
    model = TasteTokenEncoder(n_movies=10, dim=16, n_interests=4, n_heads=4)
    history = torch.tensor([[0, 1, 2, 10], [3, 4, 10, 10]])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    pi, z = model.encode_history(history, mask)
    assert pi.shape == (2, 4)
    assert z.shape == (2, 4, 16)
    torch.testing.assert_close(pi.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)


def test_pad_tokens_do_not_nan() -> None:
    torch.manual_seed(1)
    model = TasteTokenEncoder(n_movies=8, dim=8, n_interests=2, n_heads=2)
    history = torch.full((3, 5), 8, dtype=torch.long)
    history[:, 0] = torch.arange(3)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[:, 0] = True
    pi, z = model.encode_history(history, mask)
    assert torch.isfinite(pi).all()
    assert torch.isfinite(z).all()


def test_diversity_loss_is_zero_for_orthonormal_queries() -> None:
    model = TasteTokenEncoder(n_movies=4, dim=4, n_interests=4, n_heads=2)
    with torch.no_grad():
        model.taste_queries.copy_(torch.eye(4))
    loss = model.diversity_loss()
    assert float(loss.detach()) < 1e-6


def test_title_reps_drop_padding_row() -> None:
    model = TasteTokenEncoder(n_movies=7, dim=8, n_interests=2, n_heads=2)
    titles = model.title_reps()
    assert titles.z.shape == (7, 8)
    assert titles.bias.shape == (7,)
