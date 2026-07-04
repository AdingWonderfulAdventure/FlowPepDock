from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np


def _group_indices(group_ids: Sequence[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        out[str(group_id)].append(index)
    return out


def topk_success(scores: np.ndarray, dockq: np.ndarray, group_ids: Sequence[str], k: int, threshold: float = 0.49) -> float:
    groups = _group_indices(group_ids)
    hits = 0
    for indices in groups.values():
        ranked = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)[:k]
        hits += int(any(float(dockq[idx]) >= threshold for idx in ranked))
    return hits / max(len(groups), 1)


def mrr(scores: np.ndarray, dockq: np.ndarray, group_ids: Sequence[str], threshold: float = 0.49) -> float:
    groups = _group_indices(group_ids)
    values: List[float] = []
    for indices in groups.values():
        ranked = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)
        reciprocal = 0.0
        for rank, idx in enumerate(ranked, start=1):
            if float(dockq[idx]) >= threshold:
                reciprocal = 1.0 / rank
                break
        values.append(reciprocal)
    return float(np.mean(values)) if values else 0.0


def ndcg(scores: np.ndarray, dockq: np.ndarray, group_ids: Sequence[str], k: int | None = None) -> float:
    groups = _group_indices(group_ids)
    values: List[float] = []
    for indices in groups.values():
        ranked = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)
        ideal = sorted(indices, key=lambda idx: float(dockq[idx]), reverse=True)
        if k is not None:
            ranked = ranked[:k]
            ideal = ideal[:k]
        def _dcg(order: List[int]) -> float:
            return float(sum(((2.0 ** float(dockq[idx]) - 1.0) / np.log2(rank + 2.0)) for rank, idx in enumerate(order)))
        ideal_dcg = _dcg(ideal)
        values.append(_dcg(ranked) / max(ideal_dcg, 1e-8))
    return float(np.mean(values)) if values else 0.0


def spearman_like(scores: np.ndarray, dockq: np.ndarray) -> float:
    if len(scores) < 2:
        return 0.0
    score_rank = np.argsort(np.argsort(scores))
    dockq_rank = np.argsort(np.argsort(dockq))
    return float(np.corrcoef(score_rank, dockq_rank)[0, 1])


def global_topk_success(scores: np.ndarray, dockq: np.ndarray, peptide_ids: Sequence[str], k: int, threshold: float = 0.49) -> float:
    peptide_to_indices: Dict[str, List[int]] = defaultdict(list)
    for index, peptide_id in enumerate(peptide_ids):
        peptide_to_indices[str(peptide_id)].append(index)
    hits = 0
    for indices in peptide_to_indices.values():
        ranked = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)[:k]
        hits += int(any(float(dockq[idx]) >= threshold for idx in ranked))
    return hits / max(len(peptide_to_indices), 1)


def global_top_percent_enrichment(scores: np.ndarray, dockq: np.ndarray, peptide_ids: Sequence[str], fraction: float, threshold: float = 0.49) -> float:
    peptide_to_indices: Dict[str, List[int]] = defaultdict(list)
    for index, peptide_id in enumerate(peptide_ids):
        peptide_to_indices[str(peptide_id)].append(index)
    enrichments: List[float] = []
    for indices in peptide_to_indices.values():
        total_positive = sum(float(dockq[idx]) >= threshold for idx in indices)
        if total_positive == 0:
            continue
        ranked = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)
        cut = max(1, int(len(indices) * fraction))
        top_indices = ranked[:cut]
        observed = sum(float(dockq[idx]) >= threshold for idx in top_indices) / cut
        background = total_positive / len(indices)
        enrichments.append(observed / max(background, 1e-8))
    return float(np.mean(enrichments)) if enrichments else 0.0


def best_of_group_retrieval_success(scores: np.ndarray, dockq: np.ndarray, peptide_ids: Sequence[str], group_ids: Sequence[str], top_k: int) -> float:
    peptide_to_indices: Dict[str, List[int]] = defaultdict(list)
    for index, peptide_id in enumerate(peptide_ids):
        peptide_to_indices[str(peptide_id)].append(index)
    group_best: Dict[str, float] = {}
    for group_id in set(group_ids):
        idxs = [idx for idx, gid in enumerate(group_ids) if gid == group_id]
        group_best[str(group_id)] = max(float(dockq[idx]) for idx in idxs)
    hits = 0
    total = 0
    for indices in peptide_to_indices.values():
        ranked = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)[:top_k]
        retrieved_groups = {str(group_ids[idx]) for idx in ranked}
        best_group = None
        best_value = -1.0
        for idx in indices:
            gid = str(group_ids[idx])
            if group_best[gid] > best_value:
                best_value = group_best[gid]
                best_group = gid
        if best_group is not None:
            hits += int(best_group in retrieved_groups)
            total += 1
    return hits / max(total, 1)
