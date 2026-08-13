from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from catalog_value.models.types import TitleReps


class TasteTokenEncoder(nn.Module):
    """K learned taste queries attend over a user's title set.

    This is the brief's "transformer with learned taste tokens": the user's
    history is a set (no positions), and K queries compete for mass so tastes
    can specialize instead of collapsing to one embedding.
    """

    def __init__(
        self,
        n_movies: int,
        dim: int,
        n_interests: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"embedding_dim {dim} must be divisible by n_heads {n_heads}")
        self.n_movies = n_movies
        self.dim = dim
        self.n_interests = n_interests
        self.pad_id = n_movies
        self.item_emb = nn.Embedding(n_movies + 1, dim, padding_idx=self.pad_id)
        self.item_bias = nn.Embedding(n_movies + 1, 1, padding_idx=self.pad_id)
        self.taste_queries = nn.Parameter(torch.empty(n_interests, dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.pi_head = nn.Linear(dim, 1)
        self.global_bias = nn.Parameter(torch.tensor(3.5))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.taste_queries)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.item_emb.weight[:-1], std=0.02)
        nn.init.zeros_(self.item_emb.weight[-1])

    def encode_history(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pi, z) with shapes [batch, K] and [batch, K, dim]."""
        batch = history.shape[0]
        item_h = self.item_emb(history)
        queries = self.taste_queries.unsqueeze(0).expand(batch, -1, -1)
        padding = ~history_mask
        all_pad = padding.all(dim=1)
        if all_pad.any():
            padding = padding.clone()
            padding[all_pad, 0] = False
        attended, _ = self.attn(
            queries,
            item_h,
            item_h,
            key_padding_mask=padding,
            need_weights=False,
        )
        z = self.norm(attended + queries)
        pi = torch.softmax(self.pi_head(z).squeeze(-1), dim=-1)
        return pi, z

    def predicted_affinity(
        self,
        pi: torch.Tensor,
        z: torch.Tensor,
        target_items: torch.Tensor,
    ) -> torch.Tensor:
        """Mixture affinity for targets: [batch, n_targets]."""
        z_i = self.item_emb(target_items)
        b_i = self.item_bias(target_items).squeeze(-1)
        dots = torch.einsum("bkd,btd->bkt", z, z_i)
        per_taste = dots + b_i.unsqueeze(1)
        return (pi.unsqueeze(-1) * per_taste).sum(dim=1) + self.global_bias

    def diversity_loss(self) -> torch.Tensor:
        """Penalize aligned taste queries so they cannot all copy one another."""
        q = F.normalize(self.taste_queries, dim=-1)
        gram = q @ q.T
        off = gram - torch.eye(self.n_interests, device=gram.device, dtype=gram.dtype)
        return off.square().mean()

    def entropy(self, pi: torch.Tensor) -> torch.Tensor:
        return -(pi * (pi.clamp_min(1e-8).log())).sum(dim=-1).mean()

    def title_reps(self) -> TitleReps:
        z = self.item_emb.weight[:-1].detach().float().cpu().numpy()
        bias = self.item_bias.weight[:-1, 0].detach().float().cpu().numpy()
        return TitleReps(
            z=z.astype("float64"),
            bias=bias.astype("float64"),
            movie_row=torch.arange(self.n_movies).numpy().astype("int64"),
        )
