from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class PoseRecord:
    pose_id: str
    complex_id: str
    group_id: str
    receptor_id: str
    peptide_id: str
    dockq: float
    rmsd: float
    physical_bad_label: int
    global_feat: np.ndarray
    node_feat: np.ndarray
    edge_index: np.ndarray
    edge_feat: np.ndarray
    node_pair_index: np.ndarray
    clash_summary: Dict[str, float]

    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        return {
            "global_feat": self.global_feat.astype(np.float32),
            "node_feat": self.node_feat.astype(np.float32),
            "edge_index": self.edge_index.astype(np.int64),
            "edge_feat": self.edge_feat.astype(np.float32),
            "node_pair_index": self.node_pair_index.astype(np.int32),
            "dockq": np.asarray(self.dockq, dtype=np.float32),
            "rmsd": np.asarray(self.rmsd, dtype=np.float32),
            "physical_bad_label": np.asarray(self.physical_bad_label, dtype=np.int64),
        }


@dataclass
class PoseRecordRef:
    record_path: Path
    pose_id: str
    complex_id: str
    group_id: str
    receptor_id: str
    peptide_id: str
    dockq: float
    rmsd: float


@dataclass
class ShardedPoseRecordRef:
    shard_path: Path
    local_index: int
    pose_id: str
    complex_id: str
    group_id: str
    receptor_id: str
    peptide_id: str
    dockq: float
    rmsd: float


@dataclass
class PoseBatch:
    node_feat: np.ndarray
    edge_index: np.ndarray
    edge_feat: np.ndarray
    node_batch_index: np.ndarray
    pose_batch_index: np.ndarray
    global_feat: np.ndarray
    dockq: np.ndarray
    rmsd: np.ndarray
    physical_bad_label: np.ndarray
    group_id: List[str]
