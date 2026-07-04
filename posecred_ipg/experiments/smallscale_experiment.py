from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import PoseCredConfig
from ..features import build_pose_record_from_pair_pdbs
from ..io import save_pose_record, write_manifest
from ..paths import POSECRED_TRAIN_DOCKED_REL_TABLE, POSECRED_VAL_DOCKED_REL_TABLE
from ..pdbio import ResidueData, load_all_residues
from ..records import PoseRecord
from ..runtime import maybe_wrap_data_parallel, resolve_device_and_gpu_ids
from ..train import (
    apply_config_overrides,
    build_model,
    choose_metric,
    evaluate,
    make_loader,
    save_checkpoint,
    set_seed,
    snapshot_config,
    train_one_epoch,
)


DEFAULT_TRAIN_TABLE = POSECRED_TRAIN_DOCKED_REL_TABLE
DEFAULT_VAL_TABLE = POSECRED_VAL_DOCKED_REL_TABLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed small-scale PoseCred-IPG train/val experiment.")
    parser.add_argument("--train_pose_table", type=Path, default=DEFAULT_TRAIN_TABLE)
    parser.add_argument("--val_pose_table", type=Path, default=DEFAULT_VAL_TABLE)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--train_groups", type=int, default=64)
    parser.add_argument("--val_groups", type=int, default=16)
    parser.add_argument("--poses_per_group", type=int, default=10)
    parser.add_argument("--min_poses_per_group", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="posecred_ipg", choices=["posecred_ipg", "stats_mlp"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--groups_per_batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default="1")
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--main_metric", type=str, default="val_top1_success")
    parser.add_argument("--clash_penalty_scale", type=float, default=1.0)
    return parser.parse_args()


def _sample_groups(table: pd.DataFrame, num_groups: int, poses_per_group: int, min_poses_per_group: int, seed: int) -> pd.DataFrame:
    group_sizes = table.groupby("group_id").size()
    candidate_groups = group_sizes[group_sizes >= min_poses_per_group].index.to_list()
    if len(candidate_groups) < num_groups:
        raise ValueError(f"not enough groups: need {num_groups}, got {len(candidate_groups)}")
    rng = np.random.default_rng(seed)
    selected_groups = rng.choice(candidate_groups, size=num_groups, replace=False).tolist()
    rows = []
    for group_id in selected_groups:
        group = table[table["group_id"] == group_id].copy().sort_values("dockq", ascending=False).head(poses_per_group)
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def _save_subset(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def _build_records(
    records_dir: Path,
    table: pd.DataFrame,
    receptor_cache: Optional[Dict[str, List[ResidueData]]] = None,
    peptide_cache: Optional[Dict[str, List[ResidueData]]] = None,
) -> Path:
    config = PoseCredConfig()
    record_paths: List[Path] = []
    receptor_cache = {} if receptor_cache is None else receptor_cache
    peptide_cache = {} if peptide_cache is None else peptide_cache
    for row in table.itertuples(index=False):
        row_dict = row._asdict()
        receptor_pdb = Path(str(row_dict["receptor_pdb"]))
        receptor_key = str(receptor_pdb)
        peptide_pdb = Path(str(row_dict["pose_path"]))
        peptide_key = str(peptide_pdb)
        if receptor_key not in receptor_cache:
            receptor_cache[receptor_key] = load_all_residues(receptor_pdb)
        if peptide_key not in peptide_cache:
            peptide_cache[peptide_key] = load_all_residues(peptide_pdb)
        record = build_pose_record_from_pair_pdbs(
            receptor_pdb=receptor_pdb,
            peptide_pdb=peptide_pdb,
            pose_id=f'{row_dict["complex_name"]}__{row_dict["pose_name"]}',
            complex_id=str(row_dict["complex_name"]),
            group_id=str(row_dict["group_id"]),
            receptor_id=str(row_dict["receptor_id"]),
            peptide_id=str(row_dict["group_id"]),
            dockq=float(row_dict["dockq"]),
            rmsd=float(row_dict.get("peptide_ca_rmsd", row_dict.get("complex_rmsd", 0.0))),
            config=config,
            receptor_residues=receptor_cache[receptor_key],
            peptide_residues=peptide_cache[peptide_key],
        )
        record_paths.append(save_pose_record(record, records_dir))
    manifest_path = records_dir / "manifest.txt"
    write_manifest(record_paths, manifest_path)
    return manifest_path


def _load_records_from_manifest(manifest_path: Path) -> List[PoseRecord]:
    from ..train import load_records_from_npz

    return load_records_from_npz(manifest_path)


def _run_training(
    train_records: Sequence[PoseRecord],
    val_records: Sequence[PoseRecord],
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, object]:
    config = PoseCredConfig()
    config = apply_config_overrides(config, args)
    train_loader = make_loader(train_records, groups_per_batch=args.groups_per_batch, poses_per_group=args.poses_per_group, shuffle=True)
    val_loader = make_loader(val_records, groups_per_batch=args.groups_per_batch, poses_per_group=args.poses_per_group, shuffle=False)
    example_batch = next(iter(train_loader))
    device, gpu_ids = resolve_device_and_gpu_ids(args.device, args.gpu_ids)
    args.device = str(device)

    model = build_model(args.model_name, example_batch, config).to(device)
    model = maybe_wrap_data_parallel(model, gpu_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    best_metric = None
    best_checkpoint = None
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, config, device)
        val_metrics = evaluate(model, val_loader, config, device, split_name="val", clash_penalty_scale=args.clash_penalty_scale)
        train_eval_metrics = evaluate(
            model,
            train_loader,
            config,
            device,
            split_name="train_eval",
            clash_penalty_scale=args.clash_penalty_scale,
        )
        summary = {"epoch": epoch, "epoch_seconds": time.perf_counter() - epoch_start, **train_metrics, **train_eval_metrics, **val_metrics}
        summary["main_metric"] = choose_metric(summary, args.main_metric)
        history.append(summary)
        if best_metric is None or summary["main_metric"] > best_metric:
            best_metric = summary["main_metric"]
            best_checkpoint = save_checkpoint(output_dir, model, optimizer, epoch, summary, tag="best")
        print(summary)
    total_train_seconds = time.perf_counter() - train_start
    last_summary = history[-1]
    last_checkpoint = save_checkpoint(output_dir, model, optimizer, last_summary["epoch"], last_summary, tag="last")
    return {
        "history": history,
        "best_metric": best_metric,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else "",
        "last_checkpoint": str(last_checkpoint),
        "train_seconds": total_train_seconds,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_table = pd.read_csv(args.train_pose_table)
    val_table = pd.read_csv(args.val_pose_table)

    train_subset = _sample_groups(train_table, args.train_groups, args.poses_per_group, args.min_poses_per_group, seed=args.seed)
    val_subset = _sample_groups(val_table, args.val_groups, args.poses_per_group, args.min_poses_per_group, seed=args.seed + 1)

    _save_subset(train_subset, output_dir / "train_subset.csv")
    _save_subset(val_subset, output_dir / "val_subset.csv")

    record_start = time.perf_counter()
    train_manifest = _build_records(output_dir / "train_records", train_subset)
    val_manifest = _build_records(output_dir / "val_records", val_subset)
    record_build_seconds = time.perf_counter() - record_start

    train_records = _load_records_from_manifest(train_manifest)
    val_records = _load_records_from_manifest(val_manifest)
    snapshot_config(output_dir, args, apply_config_overrides(PoseCredConfig(), args), train_records, val_records)

    result = _run_training(train_records, val_records, args, output_dir)

    metrics_df = pd.DataFrame(result["history"])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    report = {
        "main_metric": args.main_metric,
        "best_metric": result["best_metric"],
        "best_checkpoint": result["best_checkpoint"],
        "last_checkpoint": result["last_checkpoint"],
        "record_build_seconds": record_build_seconds,
        "train_seconds": result["train_seconds"],
        "train_groups": args.train_groups,
        "val_groups": args.val_groups,
        "poses_per_group": args.poses_per_group,
        "model_name": args.model_name,
        "device": args.device,
    }
    with (output_dir / "experiment_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
