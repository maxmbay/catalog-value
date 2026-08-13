"""Map title content features X_i to the collaborative embedding space."""

from __future__ import annotations

import torch
from torch import nn


class ContentEncoder(nn.Module):
    """MLP: genome tags + genres + year → (z, bias) in the taste-token space."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.z_head = nn.Linear(hidden, out_dim)
        self.bias_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.z_head(h), self.bias_head(h).squeeze(-1)
