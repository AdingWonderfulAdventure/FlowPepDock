from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.numel() == 0:
        return src.new_zeros((dim_size, src.shape[-1]))
    out = src.new_zeros((dim_size, src.shape[-1]))
    count = src.new_zeros((dim_size, 1))
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones((src.shape[0], 1), device=src.device, dtype=src.dtype))
    return out / count.clamp_min(1.0)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.numel() == 0:
        return src.new_zeros((dim_size, src.shape[-1]))
    expanded_index = index.unsqueeze(-1).expand(-1, src.shape[-1])
    out = src.new_full((dim_size, src.shape[-1]), -1e9)
    out.scatter_reduce_(0, expanded_index, src, reduce="amax", include_self=True)
    return torch.where(out < -1e8, torch.zeros_like(out), out)


class EdgeMLP(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, src: torch.Tensor, dst: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([src, dst, edge_feat], dim=-1))


class GINELayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_mlp = EdgeMLP(hidden_dim, edge_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0 or x.numel() == 0:
            return x
        src, dst = edge_index
        messages = self.edge_mlp(x[src], x[dst], edge_feat)
        agg = scatter_mean(messages, dst, x.shape[0])
        updated = self.update(torch.cat([x, agg], dim=-1))
        return self.norm(x + updated)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros((num_graphs, x.shape[-1]))
        logits = self.proj(x).squeeze(-1)
        pooled = x.new_zeros((num_graphs, x.shape[-1]))
        for graph_id in range(num_graphs):
            mask = batch_index == graph_id
            if not torch.any(mask):
                continue
            weights = torch.softmax(logits[mask], dim=0)
            pooled[graph_id] = torch.sum(weights.unsqueeze(-1) * x[mask], dim=0)
        return pooled


class PoseCredIPGModel(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, global_dim: int, hidden_dim: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([GINELayer(hidden_dim, hidden_dim, dropout) for _ in range(2)])
        self.attn_pool = AttentionPooling(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3 + global_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.dockq_head = nn.Linear(hidden_dim, 1)

    def _encode_pose_repr(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        node_batch_index: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = global_feat.shape[0]
        x = self.node_proj(node_feat)
        e = self.edge_proj(edge_feat) if edge_feat.numel() > 0 else edge_feat.new_zeros((0, x.shape[-1]))
        for layer in self.layers:
            x = layer(x, edge_index, e)
        attn = self.attn_pool(x, node_batch_index, num_graphs)
        mean_pool = scatter_mean(x, node_batch_index, num_graphs) if x.numel() > 0 else global_feat.new_zeros((num_graphs, x.shape[-1]))
        max_pool = scatter_max(x, node_batch_index, num_graphs) if x.numel() > 0 else global_feat.new_zeros((num_graphs, x.shape[-1]))
        return self.readout(torch.cat([attn, mean_pool, max_pool, global_feat], dim=-1))

    def forward_score_only(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        node_batch_index: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        pose_repr = self._encode_pose_repr(node_feat, edge_index, edge_feat, node_batch_index, global_feat)
        return self.score_head(pose_repr).squeeze(-1)

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        node_batch_index: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pose_repr = self._encode_pose_repr(node_feat, edge_index, edge_feat, node_batch_index, global_feat)
        return (
            self.score_head(pose_repr).squeeze(-1),
            self.dockq_head(pose_repr).squeeze(-1),
        )
