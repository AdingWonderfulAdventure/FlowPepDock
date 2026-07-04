from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from ..default_paths import apply_default_shard_snapshot_args
from ..metrics import best_of_group_retrieval_success, global_top_percent_enrichment, global_topk_success
from ..runtime import load_model_state_dict_allowing_deprecated_heads, resolve_device_and_gpu_ids
from ..train import apply_config_overrides, build_model, load_config_from_record_source, make_loader, prepare_datasets, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pooled/global ranking for a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True, choices=["posecred_ipg", "stats_mlp"])
    parser.add_argument("--train_records_index", type=Path, default=None)
    parser.add_argument("--val_records_index", type=Path, default=None)
    parser.add_argument("--train_records_shard_index", type=Path, default=None)
    parser.add_argument("--val_records_shard_index", type=Path, default=None)
    parser.add_argument("--train_records_manifest", type=Path, default=None)
    parser.add_argument("--val_records_manifest", type=Path, default=None)
    parser.add_argument("--use_default_shard_snapshot", action="store_true")
    parser.add_argument("--groups_per_batch", type=int, default=32)
    parser.add_argument("--poses_per_group", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--clash_penalty_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--threshold", type=float, default=0.49)
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--full_model_forward", action="store_true")
    return parser.parse_args()


def _compute_clash_penalty(clash_penalty_feat: torch.Tensor, weights: Sequence[float], scale: float) -> torch.Tensor:
    w = torch.tensor(weights, dtype=clash_penalty_feat.dtype, device=clash_penalty_feat.device)
    return float(scale) * torch.sum(clash_penalty_feat * w.unsqueeze(0), dim=-1)


def _forward_model(model: torch.nn.Module, batch: Dict[str, torch.Tensor], model_name: str):
    if model_name == "stats_mlp":
        return model(batch["global_feat"])
    return model(
        batch["node_feat"],
        batch["edge_index"],
        batch["edge_feat"],
        batch["node_batch_index"],
        batch["global_feat"],
    )


def _forward_score_only(model: torch.nn.Module, batch: Dict[str, torch.Tensor], model_name: str) -> torch.Tensor:
    if model_name == "stats_mlp":
        score, _ = model(batch["global_feat"])
        return score
    score_only = getattr(model, "forward_score_only", None)
    if callable(score_only):
        return score_only(
            batch["node_feat"],
            batch["edge_index"],
            batch["edge_feat"],
            batch["node_batch_index"],
            batch["global_feat"],
        )
    score, _ = _forward_model(model, batch, model_name)
    return score


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader,
    model_name: str,
    clash_penalty_weights: Sequence[float],
    clash_penalty_scale: float,
    device: torch.device,
    use_score_only_fastpath: bool = True,
    timing_rows: List[Dict[str, float | int | str]] | None = None,
) -> pd.DataFrame:
    model.eval()
    score_rows: List[float] = []
    dockq_rows: List[float] = []
    pose_rows: List[str] = []
    complex_rows: List[str] = []
    group_rows: List[str] = []
    peptide_rows: List[str] = []
    receptor_rows: List[str] = []
    for batch in loader:
        transfer_start = time.perf_counter()
        device_batch = {
            "node_feat": batch["node_feat"].to(device, non_blocking=True),
            "edge_index": batch["edge_index"].to(device, non_blocking=True),
            "edge_feat": batch["edge_feat"].to(device, non_blocking=True),
            "node_batch_index": batch["node_batch_index"].to(device, non_blocking=True),
            "global_feat": batch["global_feat"].to(device, non_blocking=True),
            "clash_penalty_feat": batch["clash_penalty_feat"].to(device, non_blocking=True),
        }
        if timing_rows is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        transfer_seconds = time.perf_counter() - transfer_start

        forward_start = time.perf_counter()
        if use_score_only_fastpath:
            score = _forward_score_only(model, device_batch, model_name)
        else:
            score, _, _ = _forward_model(model, device_batch, model_name)
        if timing_rows is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - forward_start

        penalty_start = time.perf_counter()
        final_score = score - _compute_clash_penalty(device_batch["clash_penalty_feat"], clash_penalty_weights, clash_penalty_scale)
        if timing_rows is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        penalty_seconds = time.perf_counter() - penalty_start

        collect_start = time.perf_counter()
        score_rows.extend(final_score.detach().cpu().tolist())
        dockq_rows.extend(batch["dockq"].tolist())
        pose_rows.extend([str(item) for item in batch.get("pose_id", [])])
        complex_rows.extend([str(item) for item in batch.get("complex_id", [])])
        group_rows.extend([str(item) for item in batch["group_id"]])
        peptide_rows.extend([str(item) for item in batch["peptide_id"]])
        receptor_rows.extend([str(item) for item in batch.get("receptor_id", [])])
        collect_seconds = time.perf_counter() - collect_start

        if timing_rows is not None:
            timing_rows.append(
                {
                    "num_poses": int(len(batch.get("group_id", []))),
                    "num_groups": int(len(set(str(item) for item in batch.get("group_id", [])))),
                    "transfer_seconds": float(transfer_seconds),
                    "forward_seconds": float(forward_seconds),
                    "penalty_seconds": float(penalty_seconds),
                    "collect_seconds": float(collect_seconds),
                }
            )
    return pd.DataFrame(
        {
            "pose_id": pose_rows,
            "complex_id": complex_rows,
            "score": score_rows,
            "dockq": dockq_rows,
            "group_id": group_rows,
            "peptide_id": peptide_rows,
            "receptor_id": receptor_rows,
        }
    )


def pooled_summary(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    peptide_ids = df["peptide_id"].tolist()
    group_ids = df["group_id"].tolist()
    scores = df["score"].to_numpy(dtype=np.float32)
    dockq = df["dockq"].to_numpy(dtype=np.float32)
    return {
        "num_rows": int(len(df)),
        "num_groups": int(df["group_id"].nunique()),
        "num_peptides": int(df["peptide_id"].nunique()),
        "peptides_with_multi_groups": int((df.groupby("peptide_id")["group_id"].nunique() > 1).sum()),
        "global_top1_success": global_topk_success(scores, dockq, peptide_ids, k=1, threshold=threshold),
        "global_top5_success": global_topk_success(scores, dockq, peptide_ids, k=5, threshold=threshold),
        "global_top10pct_enrichment": global_top_percent_enrichment(scores, dockq, peptide_ids, fraction=0.1, threshold=threshold),
        "best_of_group_retrieval_success": best_of_group_retrieval_success(scores, dockq, peptide_ids, group_ids, top_k=5),
    }


def peptide_level_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    group_best = df.groupby("group_id")["dockq"].max().to_dict()
    peptide_to_df = {peptide_id: sub_df.copy() for peptide_id, sub_df in df.groupby("peptide_id")}
    for peptide_id, sub_df in peptide_to_df.items():
        ranked = sub_df.sort_values("score", ascending=False).reset_index(drop=True)
        top1 = ranked.head(1)
        top5 = ranked.head(5)
        best_group = max(sub_df["group_id"].unique(), key=lambda gid: group_best[gid])
        rows.append(
            {
                "peptide_id": peptide_id,
                "num_groups": int(sub_df["group_id"].nunique()),
                "num_rows": int(len(sub_df)),
                "best_dockq": float(sub_df["dockq"].max()),
                "top1_dockq": float(top1["dockq"].max()),
                "top5_best_dockq": float(top5["dockq"].max()),
                "top1_hit": int((top1["dockq"] >= threshold).any()),
                "top5_hit": int((top5["dockq"] >= threshold).any()),
                "retrieved_best_group_top5": int(best_group in set(top5["group_id"].tolist())),
                "best_group": str(best_group),
            }
        )
    return pd.DataFrame(rows).sort_values(["num_groups", "best_dockq", "peptide_id"], ascending=[False, False, True]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    apply_default_shard_snapshot_args(args)
    set_seed(args.seed)
    config = load_config_from_record_source(args)
    config = apply_config_overrides(config, args)
    device, _ = resolve_device_and_gpu_ids(args.device, args.gpu_ids)
    _, val_records = prepare_datasets(args)
    val_loader = make_loader(
        val_records,
        groups_per_batch=args.groups_per_batch,
        poses_per_group=args.poses_per_group,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader_iter = iter(val_loader)
    example_batch = next(val_loader_iter)
    model = build_model(args.model_name, example_batch, config).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_model_state_dict_allowing_deprecated_heads(model, payload["model_state_dict"])

    df = collect_predictions(
        model=model,
        loader=itertools.chain([example_batch], val_loader_iter),
        model_name=args.model_name,
        clash_penalty_weights=config.clash_penalty_weights,
        clash_penalty_scale=args.clash_penalty_scale,
        device=device,
        use_score_only_fastpath=not args.full_model_forward,
    )
    all_summary = pooled_summary(df, threshold=args.threshold)
    multi_group_df = df[df["peptide_id"].isin(df.groupby("peptide_id").filter(lambda x: x["group_id"].nunique() > 1)["peptide_id"].unique())].copy()
    multi_summary = pooled_summary(multi_group_df, threshold=args.threshold) if not multi_group_df.empty else {}
    per_peptide = peptide_level_table(df, threshold=args.threshold)
    per_peptide_multi = per_peptide[per_peptide["num_groups"] > 1].reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "val_predictions.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    per_peptide.to_csv(args.out_dir / "per_peptide_summary.csv", index=False)
    per_peptide_multi.to_csv(args.out_dir / "per_peptide_multi_group_summary.csv", index=False)

    report = {
        "checkpoint": str(args.checkpoint),
        "model_name": args.model_name,
        "threshold": args.threshold,
        "all_val_peptides": all_summary,
        "multi_group_val_peptides": multi_summary,
    }
    (args.out_dir / "pooled_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
