from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

from ..config import PoseCredConfig


def _group_indices(group_ids: Iterable[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        out[str(group_id)].append(index)
    return out


def listwise_loss(scores: torch.Tensor, dockq: torch.Tensor, group_ids: Iterable[str], config: PoseCredConfig) -> torch.Tensor:
    groups = _group_indices(group_ids)
    loss = scores.new_tensor(0.0)
    counted = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        idx = torch.as_tensor(indices, dtype=torch.long, device=scores.device)
        target = torch.softmax(dockq[idx] / config.dockq_temperature, dim=0)
        pred = torch.log_softmax(scores[idx] / config.score_temperature, dim=0)
        loss = loss - torch.sum(target * pred)
        counted += 1
    return loss / max(counted, 1)


def pairwise_margin_loss(scores: torch.Tensor, dockq: torch.Tensor, group_ids: Iterable[str], config: PoseCredConfig) -> torch.Tensor:
    groups = _group_indices(group_ids)
    penalties: List[torch.Tensor] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        idx = torch.as_tensor(indices, dtype=torch.long, device=scores.device)
        score_group = scores[idx]
        dockq_group = dockq[idx]
        diff = dockq_group[:, None] - dockq_group[None, :]
        valid = diff > config.pair_margin_dockq
        if not torch.any(valid):
            continue
        score_gap = score_group[:, None] - score_group[None, :]
        penalties.append(F.relu(config.pair_margin_score - score_gap[valid]).mean())
    if not penalties:
        return scores.new_tensor(0.0)
    return torch.stack(penalties).mean()


def huber_dockq_loss(dockq_pred: torch.Tensor, dockq: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(dockq_pred, dockq, delta=0.1)


def total_loss(
    scores: torch.Tensor,
    dockq_pred: torch.Tensor,
    dockq: torch.Tensor,
    group_ids: Iterable[str],
    config: PoseCredConfig,
) -> Dict[str, torch.Tensor]:
    list_loss = listwise_loss(scores, dockq, group_ids, config)
    pair_loss = pairwise_margin_loss(scores, dockq, group_ids, config)
    dockq_loss = huber_dockq_loss(dockq_pred, dockq)
    w_list, w_pair, w_dockq = config.loss_weights
    total = w_list * list_loss + w_pair * pair_loss + w_dockq * dockq_loss
    return {
        "total": total,
        "listwise": list_loss,
        "pairwise": pair_loss,
        "dockq": dockq_loss,
    }
