from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from ..records import PoseRecord


def save_pose_record(record: PoseRecord, output_dir: Path, compressed: bool = True) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{record.pose_id}.npz"
    tmp_path = output_dir / f".{record.pose_id}.tmp.npz"
    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(
        tmp_path,
        pose_id=np.asarray(record.pose_id),
        complex_id=np.asarray(record.complex_id),
        group_id=np.asarray(record.group_id),
        receptor_id=np.asarray(record.receptor_id),
        peptide_id=np.asarray(record.peptide_id),
        dockq=np.asarray(record.dockq, dtype=np.float32),
        rmsd=np.asarray(record.rmsd, dtype=np.float32),
        physical_bad_label=np.asarray(record.physical_bad_label, dtype=np.int64),
        global_feat=record.global_feat.astype(np.float32),
        node_feat=record.node_feat.astype(np.float32),
        edge_index=record.edge_index.astype(np.int64),
        edge_feat=record.edge_feat.astype(np.float32),
        node_pair_index=record.node_pair_index.astype(np.int32),
        clash_summary_json=np.asarray(json.dumps(record.clash_summary)),
    )
    os.replace(tmp_path, output_path)
    return output_path


def load_pose_record(npz_path: Path) -> PoseRecord:
    data = np.load(npz_path, allow_pickle=False)
    edge_index = data["edge_index"].astype(np.int64)
    if edge_index.ndim == 1 and edge_index.size == 0:
        edge_index = edge_index.reshape(2, 0)
    elif edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"invalid edge_index shape {edge_index.shape} in {npz_path}")
    return PoseRecord(
        pose_id=str(data["pose_id"].item()),
        complex_id=str(data["complex_id"].item()),
        group_id=str(data["group_id"].item()),
        receptor_id=str(data["receptor_id"].item()),
        peptide_id=str(data["peptide_id"].item()),
        dockq=float(data["dockq"].item()),
        rmsd=float(data["rmsd"].item()),
        physical_bad_label=int(data["physical_bad_label"].item()),
        global_feat=data["global_feat"].astype(np.float32),
        node_feat=data["node_feat"].astype(np.float32),
        edge_index=edge_index,
        edge_feat=data["edge_feat"].astype(np.float32),
        node_pair_index=data["node_pair_index"].astype(np.int32),
        clash_summary=json.loads(str(data["clash_summary_json"].item())),
    )


def pose_record_to_payload(record: PoseRecord) -> Dict[str, object]:
    return {
        "pose_id": record.pose_id,
        "complex_id": record.complex_id,
        "group_id": record.group_id,
        "receptor_id": record.receptor_id,
        "peptide_id": record.peptide_id,
        "dockq": float(record.dockq),
        "rmsd": float(record.rmsd),
        "physical_bad_label": int(record.physical_bad_label),
        "global_feat": torch.from_numpy(record.global_feat.astype(np.float32)),
        "node_feat": torch.from_numpy(record.node_feat.astype(np.float32)),
        "edge_index": torch.from_numpy(record.edge_index.astype(np.int64)),
        "edge_feat": torch.from_numpy(record.edge_feat.astype(np.float32)),
        "node_pair_index": torch.from_numpy(record.node_pair_index.astype(np.int32)),
        "clash_summary": dict(record.clash_summary),
    }


def payload_to_pose_record(payload: Dict[str, object]) -> PoseRecord:
    edge_index = payload["edge_index"]
    if isinstance(edge_index, torch.Tensor):
        edge_index_np = edge_index.detach().cpu().numpy().astype(np.int64)
    else:
        edge_index_np = np.asarray(edge_index, dtype=np.int64)
    if edge_index_np.ndim == 1 and edge_index_np.size == 0:
        edge_index_np = edge_index_np.reshape(2, 0)
    elif edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
        raise ValueError(f"invalid edge_index shape {edge_index_np.shape}")
    return PoseRecord(
        pose_id=str(payload["pose_id"]),
        complex_id=str(payload["complex_id"]),
        group_id=str(payload["group_id"]),
        receptor_id=str(payload["receptor_id"]),
        peptide_id=str(payload["peptide_id"]),
        dockq=float(payload["dockq"]),
        rmsd=float(payload["rmsd"]),
        physical_bad_label=int(payload["physical_bad_label"]),
        global_feat=np.asarray(payload["global_feat"], dtype=np.float32),
        node_feat=np.asarray(payload["node_feat"], dtype=np.float32),
        edge_index=edge_index_np,
        edge_feat=np.asarray(payload["edge_feat"], dtype=np.float32),
        node_pair_index=np.asarray(payload["node_pair_index"], dtype=np.int32),
        clash_summary={key: float(value) for key, value in dict(payload["clash_summary"]).items()},
    )


def save_pose_record_shard(records: List[PoseRecord], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp"
    payload = {
        "format": "posecred_pt_shard_v1",
        "records": [pose_record_to_payload(record) for record in records],
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)
    return output_path


def load_pose_record_shard(shard_path: Path) -> List[PoseRecord]:
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "posecred_pt_shard_v1":
        raise ValueError(f"unsupported shard format in {shard_path}")
    return [payload_to_pose_record(item) for item in payload["records"]]


def write_manifest(record_paths: Iterable[Path], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for path in record_paths:
            handle.write(str(path) + "\n")


def load_manifest(manifest_path: Path) -> List[Path]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return [Path(line.strip()) for line in handle if line.strip()]
