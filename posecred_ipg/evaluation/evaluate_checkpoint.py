from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..config import PoseCredConfig
from ..default_paths import apply_default_shard_snapshot_args
from ..train import (
    apply_config_overrides,
    build_model,
    evaluate,
    load_config_from_record_source,
    prepare_datasets,
    set_seed,
    snapshot_config,
    make_loader,
)
from ..runtime import load_model_state_dict_allowing_deprecated_heads, resolve_device_and_gpu_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PoseCred-IPG checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True, choices=["posecred_ipg", "stats_mlp"])
    parser.add_argument("--train_records_manifest", type=Path, default=None)
    parser.add_argument("--val_records_manifest", type=Path, default=None)
    parser.add_argument("--train_records_index", type=Path, default=None)
    parser.add_argument("--val_records_index", type=Path, default=None)
    parser.add_argument("--train_records_shard_index", type=Path, default=None)
    parser.add_argument("--val_records_shard_index", type=Path, default=None)
    parser.add_argument("--use_default_shard_snapshot", action="store_true")
    parser.add_argument("--groups_per_batch", type=int, default=32)
    parser.add_argument("--poses_per_group", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--clash_penalty_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_default_shard_snapshot_args(args)
    set_seed(args.seed)
    config = load_config_from_record_source(args)
    config = apply_config_overrides(config, args)
    device, _ = resolve_device_and_gpu_ids(args.device, args.gpu_ids)
    train_records, val_records = prepare_datasets(args)
    train_loader = make_loader(train_records, groups_per_batch=args.groups_per_batch, poses_per_group=args.poses_per_group, shuffle=False)
    val_loader = make_loader(val_records, groups_per_batch=args.groups_per_batch, poses_per_group=args.poses_per_group, shuffle=False)
    example_batch = next(iter(train_loader))
    model = build_model(args.model_name, example_batch, config).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_model_state_dict_allowing_deprecated_heads(model, payload["model_state_dict"])
    train_metrics = evaluate(model, train_loader, config, device, split_name="train_eval", clash_penalty_scale=args.clash_penalty_scale)
    val_metrics = evaluate(model, val_loader, config, device, split_name="val", clash_penalty_scale=args.clash_penalty_scale)
    report = {
        "checkpoint": str(args.checkpoint),
        "model_name": args.model_name,
        "posecred_config": vars(config),
        **train_metrics,
        **val_metrics,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
