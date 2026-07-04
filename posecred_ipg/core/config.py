from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple


BAD_HEAD_DEPRECATED_MESSAGE = (
    "PoseCred-IPG 的 bad head 已在当前主线代码中废弃并移除；"
    "请不要再传入 bad_loss_weight，也不要在 loss_weights 中启用第 4 个 bad-loss 权重。"
)


def canonical_loss_weights(loss_weights: Sequence[float]) -> Tuple[float, float, float]:
    values = tuple(float(value) for value in loss_weights)
    if len(values) == 3:
        return values  # type: ignore[return-value]
    if len(values) == 4:
        w_list, w_pair, w_dockq, w_bad = values
        if abs(w_bad) > 1e-12:
            raise ValueError(BAD_HEAD_DEPRECATED_MESSAGE)
        return (w_list, w_pair, w_dockq)
    raise ValueError(
        f"loss_weights 现在只支持 3 项 (listwise, pairwise, dockq)，收到 {len(values)} 项"
    )


@dataclass
class PoseCredConfig:
    candidate_distance_angstrom: float = 8.0
    node_limit: int = 32
    prune_strategy: str = "hybrid"
    max_neighbors: int = 12
    edge_distance_angstrom: float = 8.0
    topk_success_dockq_threshold: float = 0.49
    hidden_dim: int = 96
    num_message_passing_layers: int = 2
    dropout: float = 0.1
    dockq_temperature: float = 0.15
    score_temperature: float = 1.0
    pair_margin_dockq: float = 0.1
    pair_margin_score: float = 0.2
    loss_weights: Tuple[float, float, float] = (1.0, 0.5, 0.2)
    baseline_hidden_dims: Tuple[int, ...] = (128, 64)
    clash_penalty_weights: Tuple[float, float, float] = (1.0, 2.0, 1.0)
    physical_bad_thresholds: Tuple[float, float, float, float, float] = (8.0, 1.6, 0.7, 2.0, 6.0)
    receptor_chain_ids: Tuple[str, ...] = field(default_factory=tuple)
    peptide_chain_ids: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.loss_weights = canonical_loss_weights(self.loss_weights)
