from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import PoseCredConfig
from ..dataset import GroupBatchSampler, PoseRecordDataset, collate_pose_records
from ..features import build_pose_record_from_pair_pdbs
from ..io import save_pose_record, write_manifest
from ..paths import POSECRED_TRAIN_DOCKED_REL_TABLE, TMP_ROOT
from ..pdbio import ResidueData, load_all_residues
from ..runtime import maybe_wrap_data_parallel, resolve_device_and_gpu_ids
from ..train import build_model, evaluate, train_one_epoch


DEFAULT_POSE_TABLE = POSECRED_TRAIN_DOCKED_REL_TABLE
DEFAULT_GROUP_ID = "pepseq_643293ae695a"
DEFAULT_OUTPUT_DIR = TMP_ROOT / "posecred_ipg_overfit_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PoseCred-IPG overfit smoke test.")
    parser.add_argument("--pose_table", type=Path, default=DEFAULT_POSE_TABLE)
    parser.add_argument("--group_id", type=str, default=DEFAULT_GROUP_ID)
    parser.add_argument("--num_poses", type=int, default=32)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default="1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--model_name", type=str, default="posecred_ipg", choices=["posecred_ipg", "stats_mlp"])
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--clash_penalty_scale", type=float, default=0.0)
    parser.add_argument("--target_top1_success", type=float, default=1.0)
    parser.add_argument("--target_mrr", type=float, default=1.0)
    parser.add_argument("--target_spearman", type=float, default=0.95)
    parser.add_argument("--target_ndcg", type=float, default=0.99)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_group_rows(pose_table: Path, group_id: str, num_poses: int) -> pd.DataFrame:
    table = pd.read_csv(pose_table)
    group = table[table["group_id"] == group_id].copy()
    if len(group) < num_poses:
        raise ValueError(f"group {group_id} has only {len(group)} poses, need {num_poses}")
    group = group.sort_values("dockq", ascending=False).reset_index(drop=True)
    indices = []
    for rank in range(num_poses):
        index = min(len(group) - 1, round(rank * (len(group) - 1) / max(num_poses - 1, 1)))
        indices.append(index)
    sampled = group.iloc[sorted(set(indices))].copy().reset_index(drop=True)
    if len(sampled) != num_poses:
        sampled = group.iloc[:num_poses].copy().reset_index(drop=True)
    return sampled


def build_records_from_table(table: pd.DataFrame, output_dir: Path) -> Path:
    record_dir = output_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    config = PoseCredConfig()
    record_paths: List[Path] = []
    receptor_cache: Dict[str, List[ResidueData]] = {}
    for row in table.itertuples(index=False):
        row_dict = row._asdict()
        receptor_pdb = Path(str(row_dict["receptor_pdb"]))
        receptor_key = str(receptor_pdb)
        if receptor_key not in receptor_cache:
            receptor_cache[receptor_key] = load_all_residues(receptor_pdb)
        record = build_pose_record_from_pair_pdbs(
            receptor_pdb=receptor_pdb,
            peptide_pdb=Path(str(row_dict["pose_path"])),
            pose_id=f'{row_dict["complex_name"]}__{row_dict["pose_name"]}',
            complex_id=str(row_dict["complex_name"]),
            group_id=str(row_dict["group_id"]),
            receptor_id=str(row_dict["receptor_id"]),
            peptide_id=str(row_dict["group_id"]),
            dockq=float(row_dict["dockq"]),
            rmsd=float(row_dict.get("peptide_ca_rmsd", row_dict.get("complex_rmsd", 0.0))),
            config=config,
            receptor_residues=receptor_cache[receptor_key],
        )
        record_paths.append(save_pose_record(record, record_dir))
    manifest_path = record_dir / "manifest.txt"
    write_manifest(record_paths, manifest_path)
    return manifest_path


def run_training(
    manifest_path: Path,
    model_name: str,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    gpu_ids: List[int],
    clash_penalty_scale: float,
) -> List[Dict[str, float]]:
    from ..train import load_records_from_npz

    config = PoseCredConfig()
    records = load_records_from_npz(manifest_path)
    dataset = PoseRecordDataset(records)
    sampler = GroupBatchSampler(records, groups_per_batch=1, poses_per_group=len(records), shuffle=True)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_pose_records)
    example_batch = next(iter(loader))
    model = build_model(model_name, example_batch, config).to(device)
    model = maybe_wrap_data_parallel(model, gpu_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, loader, optimizer, config, device)
        eval_metrics = evaluate(model, loader, config, device, split_name="train_eval", clash_penalty_scale=clash_penalty_scale)
        history.append({"epoch": epoch, **train_metrics, **eval_metrics})
    return history


def validate_thresholds(history: List[Dict[str, float]], args: argparse.Namespace) -> Dict[str, float]:
    final_metrics = history[-1]
    thresholds = {
        "train_eval_top1_success": args.target_top1_success,
        "train_eval_mrr": args.target_mrr,
        "train_eval_spearman": args.target_spearman,
        "train_eval_ndcg": args.target_ndcg,
    }
    failures = {name: (final_metrics[name], threshold) for name, threshold in thresholds.items() if final_metrics[name] < threshold}
    if failures:
        raise RuntimeError(f"overfit smoke failed thresholds: {failures}")
    return {
        "top1_success": final_metrics["train_eval_top1_success"],
        "mrr": final_metrics["train_eval_mrr"],
        "spearman": final_metrics["train_eval_spearman"],
        "ndcg": final_metrics["train_eval_ndcg"],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device, gpu_ids = resolve_device_and_gpu_ids(args.device, args.gpu_ids)
    args.device = str(device)

    start_time = time.perf_counter()
    sampled = sample_group_rows(args.pose_table, args.group_id, args.num_poses)
    sample_csv = output_dir / "sampled_pose_table.csv"
    sampled.to_csv(sample_csv, index=False)

    record_start = time.perf_counter()
    manifest_path = build_records_from_table(sampled, output_dir)
    record_build_seconds = time.perf_counter() - record_start

    train_start = time.perf_counter()
    history = run_training(
        manifest_path=manifest_path,
        model_name=args.model_name,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=device,
        gpu_ids=gpu_ids,
        clash_penalty_scale=args.clash_penalty_scale,
    )
    train_seconds = time.perf_counter() - train_start
    total_seconds = time.perf_counter() - start_time

    final_metrics = validate_thresholds(history, args)
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "train_history.csv", index=False)

    summary = {
        "pose_table": str(args.pose_table),
        "group_id": args.group_id,
        "num_poses": args.num_poses,
        "model_name": args.model_name,
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "lr": args.lr,
        "clash_penalty_scale": args.clash_penalty_scale,
        "record_build_seconds": record_build_seconds,
        "train_seconds": train_seconds,
        "total_seconds": total_seconds,
        "thresholds": {
            "top1_success": args.target_top1_success,
            "mrr": args.target_mrr,
            "spearman": args.target_spearman,
            "ndcg": args.target_ndcg,
        },
        "final_metrics": final_metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
