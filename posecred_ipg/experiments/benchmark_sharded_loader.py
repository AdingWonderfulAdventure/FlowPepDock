from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from ..config import PoseCredConfig
from ..train import (
    apply_config_overrides,
    build_model,
    evaluate,
    load_config_from_record_source,
    load_record_refs_from_index,
    load_sharded_record_refs_from_index,
    make_loader,
    set_seed,
)
from ..records import PoseRecordRef, ShardedPoseRecordRef
from ..runtime import load_model_state_dict_allowing_deprecated_heads, resolve_device_and_gpu_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark npz-index vs shard-index loading/evaluation.")
    parser.add_argument("--model_name", type=str, required=True, choices=["posecred_ipg", "stats_mlp"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--npz_records_index", type=Path, required=True)
    parser.add_argument("--shard_records_index", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--groups_per_batch", type=int, default=32)
    parser.add_argument("--poses_per_group", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default="1")
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--clash_penalty_scale", type=float, default=0.0)
    return parser.parse_args()


def iterate_loader(loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
    batch_count = 0
    pose_count = 0
    start = time.perf_counter()
    for batch in loader:
        pose_count += int(batch["dockq"].shape[0])
        batch_count += 1
        for key, value in batch.items():
            if torch.is_tensor(value):
                _ = value.to(device)
    seconds = time.perf_counter() - start
    return {
        "loader_seconds": seconds,
        "loader_batches": batch_count,
        "loader_poses": pose_count,
        "loader_poses_per_second": pose_count / max(seconds, 1e-8),
    }


def run_case(
    tag: str,
    records: Sequence[PoseRecordRef | ShardedPoseRecordRef],
    args: argparse.Namespace,
    config: PoseCredConfig,
    device: torch.device,
) -> Dict[str, float]:
    loader = make_loader(records, groups_per_batch=args.groups_per_batch, poses_per_group=args.poses_per_group, shuffle=False)
    loader_stats = iterate_loader(loader, device)
    eval_loader = make_loader(records, groups_per_batch=args.groups_per_batch, poses_per_group=args.poses_per_group, shuffle=False)
    example_batch = next(iter(eval_loader))
    model = build_model(args.model_name, example_batch, config).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_model_state_dict_allowing_deprecated_heads(model, payload["model_state_dict"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    metrics = evaluate(model, eval_loader, config, device, split_name="val", clash_penalty_scale=args.clash_penalty_scale)
    eval_seconds = time.perf_counter() - start
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    pose_count = len(records)
    return {
        "tag": tag,
        "num_records": pose_count,
        **loader_stats,
        "eval_seconds": eval_seconds,
        "eval_poses_per_second": pose_count / max(eval_seconds, 1e-8),
        "peak_memory_mb": peak_memory_mb,
        "val_top1_success": metrics["val_top1_success"],
        "val_global_top1_success": metrics["val_global_top1_success"],
        "val_top5_success": metrics["val_top5_success"],
        "val_ndcg": metrics["val_ndcg"],
        "val_mrr": metrics["val_mrr"],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device, _ = resolve_device_and_gpu_ids(args.device, args.gpu_ids)
    config = load_config_from_record_source(args)
    config = apply_config_overrides(config, args)
    npz_records = load_record_refs_from_index(args.npz_records_index)
    shard_records = load_sharded_record_refs_from_index(args.shard_records_index)
    rows = [
        run_case("npz_index", npz_records, args, config, device),
        run_case("shard_index", shard_records, args, config, device),
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
