from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn


class StatsMLPRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        last_dim = dims[-1]
        self.score_head = nn.Linear(last_dim, 1)
        self.dockq_head = nn.Linear(last_dim, 1)

    def forward(self, global_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(global_feat)
        return (
            self.score_head(hidden).squeeze(-1),
            self.dockq_head(hidden).squeeze(-1),
        )
