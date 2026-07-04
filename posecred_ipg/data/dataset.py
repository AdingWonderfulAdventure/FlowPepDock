from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from ..io import load_pose_record, load_pose_record_shard
from ..records import PoseRecord, PoseRecordRef, ShardedPoseRecordRef


class PoseRecordDataset(Dataset):
    def __init__(self, records: Sequence[PoseRecord | PoseRecordRef]) -> None:
        self.records = list(records)
        self._shard_cache: Dict[Path, List[PoseRecord]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PoseRecord:
        record = self.records[index]
        if isinstance(record, PoseRecordRef):
            loaded = load_pose_record(record.record_path)
            loaded.group_id = record.group_id
            loaded.peptide_id = record.peptide_id
            return loaded
        if isinstance(record, ShardedPoseRecordRef):
            if record.shard_path not in self._shard_cache:
                self._shard_cache[record.shard_path] = load_pose_record_shard(record.shard_path)
            loaded = self._shard_cache[record.shard_path][record.local_index]
            loaded.group_id = record.group_id
            loaded.peptide_id = record.peptide_id
            return loaded
        return record


class GroupBatchSampler(Sampler[List[int]]):
    def __init__(self, records: Sequence[PoseRecord | PoseRecordRef], groups_per_batch: int, poses_per_group: int, shuffle: bool = True) -> None:
        self.groups_per_batch = groups_per_batch
        self.poses_per_group = poses_per_group
        self.shuffle = shuffle
        self.group_to_indices: Dict[str, List[int]] = defaultdict(list)
        for index, record in enumerate(records):
            self.group_to_indices[record.group_id].append(index)
        self.group_ids = list(self.group_to_indices.keys())

    def __iter__(self) -> Iterator[List[int]]:
        group_ids = list(self.group_ids)
        if self.shuffle:
            random.shuffle(group_ids)
        batch: List[int] = []
        groups_in_batch = 0
        for group_id in group_ids:
            indices = list(self.group_to_indices[group_id])
            if self.shuffle:
                random.shuffle(indices)
            limit = self.poses_per_group
            if limit is None or limit <= 0 or limit >= len(indices):
                selected = indices
            else:
                selected = indices[:limit]
            batch.extend(selected)
            groups_in_batch += 1
            if groups_in_batch >= self.groups_per_batch:
                yield batch
                batch = []
                groups_in_batch = 0
        if batch:
            yield batch

    def __len__(self) -> int:
        return max(1, (len(self.group_ids) + max(1, self.groups_per_batch) - 1) // max(1, self.groups_per_batch))


def collate_pose_records(records: Sequence[PoseRecord]) -> Dict[str, torch.Tensor]:
    node_feats: List[torch.Tensor] = []
    edge_indices: List[torch.Tensor] = []
    edge_feats: List[torch.Tensor] = []
    node_batch_index: List[torch.Tensor] = []
    pose_batch_index: List[int] = []
    global_feats: List[torch.Tensor] = []
    dockqs: List[float] = []
    rmsds: List[float] = []
    bad_labels: List[int] = []
    pose_ids: List[str] = []
    complex_ids: List[str] = []
    group_ids: List[str] = []
    peptide_ids: List[str] = []
    receptor_ids: List[str] = []
    clash_penalty_feat: List[torch.Tensor] = []
    node_offset = 0
    for pose_index, record in enumerate(records):
        node_feat = torch.from_numpy(record.node_feat)
        edge_index = torch.from_numpy(record.edge_index)
        edge_feat = torch.from_numpy(record.edge_feat)
        node_feats.append(node_feat)
        if edge_index.numel() > 0:
            edge_indices.append(edge_index + node_offset)
            edge_feats.append(edge_feat)
        node_batch_index.append(torch.full((node_feat.shape[0],), pose_index, dtype=torch.long))
        pose_batch_index.append(pose_index)
        global_feats.append(torch.from_numpy(record.global_feat))
        dockqs.append(record.dockq)
        rmsds.append(record.rmsd)
        bad_labels.append(record.physical_bad_label)
        pose_ids.append(record.pose_id)
        complex_ids.append(record.complex_id)
        group_ids.append(record.group_id)
        peptide_ids.append(record.peptide_id)
        receptor_ids.append(record.receptor_id)
        clash_penalty_feat.append(
            torch.tensor(
                [
                    record.clash_summary["severe_clash_count"],
                    record.clash_summary["max_vdw_overlap"],
                    max(0.0, 2.0 - record.clash_summary["min_heavy_dist"]),
                ],
                dtype=torch.float32,
            )
        )
        node_offset += node_feat.shape[0]
    return {
        "node_feat": torch.cat(node_feats, dim=0) if node_feats else torch.zeros((0, 1), dtype=torch.float32),
        "edge_index": torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long),
        "edge_feat": torch.cat(edge_feats, dim=0) if edge_feats else torch.zeros((0, 6), dtype=torch.float32),
        "node_batch_index": torch.cat(node_batch_index, dim=0) if node_batch_index else torch.zeros((0,), dtype=torch.long),
        "global_feat": torch.stack(global_feats, dim=0),
        "dockq": torch.tensor(dockqs, dtype=torch.float32),
        "rmsd": torch.tensor(rmsds, dtype=torch.float32),
        "physical_bad_label": torch.tensor(bad_labels, dtype=torch.float32),
        "pose_id": pose_ids,
        "complex_id": complex_ids,
        "group_id": group_ids,
        "peptide_id": peptide_ids,
        "receptor_id": receptor_ids,
        "clash_penalty_feat": torch.stack(clash_penalty_feat, dim=0),
    }
