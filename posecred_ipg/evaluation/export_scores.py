from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import pandas as pd
import torch

from ..default_paths import apply_default_shard_snapshot_args
from ..evaluate_pooled_global import collect_predictions
from ..runtime import load_model_state_dict_allowing_deprecated_heads, resolve_device_and_gpu_ids
from ..train import apply_config_overrides, build_model, load_config_from_record_source, make_loader, prepare_datasets, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-pose scores for a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True, choices=["posecred_ipg", "stats_mlp"])
    parser.add_argument("--train_records_index", type=Path, default=None)
    parser.add_argument("--val_records_index", type=Path, default=None)
    parser.add_argument("--train_records_shard_index", type=Path, default=None)
    parser.add_argument("--val_records_shard_index", type=Path, default=None)
    parser.add_argument("--train_records_manifest", type=Path, default=None)
    parser.add_argument("--val_records_manifest", type=Path, default=None)
    parser.add_argument("--use_default_shard_snapshot", action="store_true")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--groups_per_batch", type=int, default=32)
    parser.add_argument("--poses_per_group", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default="1")
    parser.add_argument("--clash_penalty_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--full_model_forward", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_default_shard_snapshot_args(args)
    set_seed(args.seed)
    config = load_config_from_record_source(args)
    config = apply_config_overrides(config, args)
    device, _ = resolve_device_and_gpu_ids(args.device, args.gpu_ids)
    train_records, val_records = prepare_datasets(args)
    records = train_records if args.split == "train" else val_records
    loader = make_loader(
        records,
        groups_per_batch=args.groups_per_batch,
        poses_per_group=args.poses_per_group,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor,
    )
    loader_iter = iter(loader)
    example_batch = next(loader_iter)
    model = build_model(args.model_name, example_batch, config).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_model_state_dict_allowing_deprecated_heads(model, payload["model_state_dict"])
    df = collect_predictions(
        model=model,
        loader=itertools.chain([example_batch], loader_iter),
        model_name=args.model_name,
        clash_penalty_weights=config.clash_penalty_weights,
        clash_penalty_scale=args.clash_penalty_scale,
        device=device,
        use_score_only_fastpath=not args.full_model_forward,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(
        {
            "out_csv": str(args.out_csv),
            "num_rows": int(len(df)),
            "num_groups": int(df["group_id"].nunique()),
            "num_peptides": int(df["peptide_id"].nunique()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
