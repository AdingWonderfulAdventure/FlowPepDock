from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import PoseCredConfig
from ..core.config import BAD_HEAD_DEPRECATED_MESSAGE
from ..default_paths import apply_default_shard_snapshot_args
from ..dataset import GroupBatchSampler, PoseRecordDataset, collate_pose_records
from ..io import load_manifest, load_pose_record
from ..losses import total_loss
from ..metrics import best_of_group_retrieval_success, global_top_percent_enrichment, global_topk_success, mrr, ndcg, spearman_like, topk_success
from ..models import PoseCredIPGModel, StatsMLPRanker
from ..paths import remap_legacy_repo_path
from ..records import PoseRecord, PoseRecordRef, ShardedPoseRecordRef
from ..runtime import (
    is_stats_model,
    load_model_state_dict_allowing_deprecated_heads,
    maybe_wrap_data_parallel,
    resolve_device_and_gpu_ids,
    unwrap_model,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_records_from_npz(manifest_path: Path) -> List[PoseRecord]:
    return [load_pose_record(path) for path in load_manifest(manifest_path)]


def load_record_refs_from_index(index_path: Path) -> List[PoseRecordRef]:
    df = pd.read_csv(index_path)
    return [
        PoseRecordRef(
            record_path=remap_legacy_repo_path(row.record_path),
            pose_id=str(row.pose_id),
            complex_id=str(row.complex_id),
            group_id=str(row.group_id),
            receptor_id=str(row.receptor_id),
            peptide_id=str(row.peptide_id),
            dockq=float(row.dockq),
            rmsd=float(row.rmsd),
        )
        for row in df.itertuples(index=False)
    ]


def load_sharded_record_refs_from_index(index_path: Path) -> List[ShardedPoseRecordRef]:
    df = pd.read_csv(index_path)
    return [
        ShardedPoseRecordRef(
            shard_path=remap_legacy_repo_path(row.shard_path),
            local_index=int(row.local_index),
            pose_id=str(row.pose_id),
            complex_id=str(row.complex_id),
            group_id=str(row.group_id),
            receptor_id=str(row.receptor_id),
            peptide_id=str(row.peptide_id),
            dockq=float(row.dockq),
            rmsd=float(row.rmsd),
        )
        for row in df.itertuples(index=False)
    ]


def load_config_from_record_source(args: argparse.Namespace) -> PoseCredConfig:
    candidate_paths: List[Path] = []
    for attr in [
        "train_records_index",
        "val_records_index",
        "train_records_manifest",
        "val_records_manifest",
        "records_manifest",
        "npz_records_index",
        "shard_records_index",
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            candidate_paths.append(remap_legacy_repo_path(value))
    for attr in ["train_records_shard_index", "val_records_shard_index"]:
        value = getattr(args, attr, None)
        if value is not None:
            candidate_paths.append(remap_legacy_repo_path(value))
    for source_path in candidate_paths:
        if source_path.name in {"records_index.csv", "shard_index.csv"}:
            config_path = source_path.parent.parent / "config_snapshot.json"
        elif source_path.name == "manifest.txt":
            config_path = source_path.parent.parent / "config_snapshot.json"
        else:
            continue
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            config_dict = payload.get("config") or payload.get("posecred_config")
            if config_dict:
                return PoseCredConfig(**config_dict)
    return PoseCredConfig()


def apply_config_overrides(config: PoseCredConfig, args: argparse.Namespace) -> PoseCredConfig:
    bad_loss_weight = getattr(args, "bad_loss_weight", None)
    if bad_loss_weight is not None:
        raise ValueError(BAD_HEAD_DEPRECATED_MESSAGE)
    return config


def split_records_by_group(records: Sequence[PoseRecord | PoseRecordRef], val_ratio: float, seed: int) -> Tuple[List[PoseRecord | PoseRecordRef], List[PoseRecord | PoseRecordRef]]:
    group_ids = sorted({record.group_id for record in records})
    if len(group_ids) < 2:
        return list(records), list(records)
    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids)
    val_count = max(1, int(len(group_ids) * val_ratio))
    val_groups = set(group_ids[:val_count])
    train_records = [record for record in records if record.group_id not in val_groups]
    val_records = [record for record in records if record.group_id in val_groups]
    if not train_records or not val_records:
        return list(records), list(records)
    return train_records, val_records


def _compute_clash_penalty(clash_penalty_feat: torch.Tensor, config: PoseCredConfig, scale: float = 1.0) -> torch.Tensor:
    weights = torch.tensor(config.clash_penalty_weights, dtype=clash_penalty_feat.dtype, device=clash_penalty_feat.device)
    return float(scale) * torch.sum(clash_penalty_feat * weights.unsqueeze(0), dim=-1)


def _forward_model(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if is_stats_model(model):
        return model(batch["global_feat"])
    return model(
        batch["node_feat"],
        batch["edge_index"],
        batch["edge_feat"],
        batch["node_batch_index"],
        batch["global_feat"],
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: PoseCredConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {"total": 0.0, "listwise": 0.0, "pairwise": 0.0, "dockq": 0.0}
    steps = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        score, dockq_pred = _forward_model(model, batch)
        loss_dict = total_loss(score, dockq_pred, batch["dockq"], batch["group_id"], config)
        loss_dict["total"].backward()
        optimizer.step()
        for key, value in loss_dict.items():
            totals[key] += float(value.detach().cpu())
        steps += 1
    return {f"train_{key}": value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    config: PoseCredConfig,
    device: torch.device,
    split_name: str,
    clash_penalty_scale: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    score_list: List[np.ndarray] = []
    dockq_list: List[np.ndarray] = []
    group_ids: List[str] = []
    peptide_ids: List[str] = []
    total_eval_loss = 0.0
    total_listwise = 0.0
    total_pairwise = 0.0
    total_dockq = 0.0
    steps = 0
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        score, dockq_pred = _forward_model(model, batch)
        loss_dict = total_loss(score, dockq_pred, batch["dockq"], batch["group_id"], config)
        total_eval_loss += float(loss_dict["total"].detach().cpu())
        total_listwise += float(loss_dict["listwise"].detach().cpu())
        total_pairwise += float(loss_dict["pairwise"].detach().cpu())
        total_dockq += float(loss_dict["dockq"].detach().cpu())
        final_score = score - _compute_clash_penalty(batch["clash_penalty_feat"], config, scale=clash_penalty_scale)
        score_list.append(final_score.detach().cpu().numpy())
        dockq_list.append(batch["dockq"].detach().cpu().numpy())
        group_ids.extend(batch["group_id"])
        peptide_ids.extend(batch["peptide_id"])
        steps += 1
    scores = np.concatenate(score_list, axis=0) if score_list else np.zeros((0,), dtype=np.float32)
    dockq = np.concatenate(dockq_list, axis=0) if dockq_list else np.zeros((0,), dtype=np.float32)
    prefix = f"{split_name}_"
    return {
        f"{prefix}total": total_eval_loss / max(steps, 1),
        f"{prefix}listwise": total_listwise / max(steps, 1),
        f"{prefix}pairwise": total_pairwise / max(steps, 1),
        f"{prefix}dockq": total_dockq / max(steps, 1),
        f"{prefix}top1_success": topk_success(scores, dockq, group_ids, k=1, threshold=config.topk_success_dockq_threshold),
        f"{prefix}top5_success": topk_success(scores, dockq, group_ids, k=5, threshold=config.topk_success_dockq_threshold),
        f"{prefix}ndcg": ndcg(scores, dockq, group_ids),
        f"{prefix}mrr": mrr(scores, dockq, group_ids, threshold=config.topk_success_dockq_threshold),
        f"{prefix}spearman": spearman_like(scores, dockq),
        f"{prefix}global_top1_success": global_topk_success(
            scores, dockq, peptide_ids, k=1, threshold=config.topk_success_dockq_threshold
        ),
        f"{prefix}global_top5_success": global_topk_success(
            scores, dockq, peptide_ids, k=5, threshold=config.topk_success_dockq_threshold
        ),
        f"{prefix}global_top10pct_enrichment": global_top_percent_enrichment(
            scores, dockq, peptide_ids, fraction=0.1, threshold=config.topk_success_dockq_threshold
        ),
        f"{prefix}best_of_group_retrieval_success": best_of_group_retrieval_success(scores, dockq, peptide_ids, group_ids, top_k=5),
    }


def build_model(model_name: str, example_batch: Dict[str, torch.Tensor], config: PoseCredConfig) -> torch.nn.Module:
    if model_name == "stats_mlp":
        return StatsMLPRanker(input_dim=example_batch["global_feat"].shape[-1], hidden_dims=config.baseline_hidden_dims)
    if model_name == "posecred_ipg":
        return PoseCredIPGModel(
            node_dim=example_batch["node_feat"].shape[-1],
            edge_dim=example_batch["edge_feat"].shape[-1],
            global_dim=example_batch["global_feat"].shape[-1],
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    raise ValueError(f"unknown model_name: {model_name}")


def make_loader(
    records: Sequence[PoseRecord | PoseRecordRef],
    groups_per_batch: int,
    poses_per_group: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    dataset = PoseRecordDataset(records)
    sampler = GroupBatchSampler(records, groups_per_batch=groups_per_batch, poses_per_group=poses_per_group, shuffle=shuffle)
    loader_kwargs = {
        "batch_sampler": sampler,
        "collate_fn": collate_pose_records,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(pin_memory),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    tag: str,
) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{tag}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    return path


def snapshot_config(output_dir: Path, args: argparse.Namespace, config: PoseCredConfig, train_records: Sequence[PoseRecord], val_records: Sequence[PoseRecord]) -> None:
    payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "posecred_config": asdict(config),
        "dataset_summary": {
            "train_records": len(train_records),
            "val_records": len(val_records),
            "train_groups": len({record.group_id for record in train_records}),
            "val_groups": len({record.group_id for record in val_records}),
        },
    }
    with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def choose_metric(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise ValueError(f"main metric {metric_name} not found in metrics: {sorted(metrics.keys())}")
    return float(metrics[metric_name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PoseCred-IPG v1.")
    parser.add_argument("--records_manifest", type=Path, default=None)
    parser.add_argument("--train_records_manifest", type=Path, default=None)
    parser.add_argument("--val_records_manifest", type=Path, default=None)
    parser.add_argument("--train_records_index", type=Path, default=None)
    parser.add_argument("--val_records_index", type=Path, default=None)
    parser.add_argument("--train_records_shard_index", type=Path, default=None)
    parser.add_argument("--val_records_shard_index", type=Path, default=None)
    parser.add_argument("--use_default_shard_snapshot", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="posecred_ipg", choices=["posecred_ipg", "stats_mlp"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init_checkpoint", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--groups_per_batch", type=int, default=16)
    parser.add_argument("--poses_per_group", type=int, default=8)
    parser.add_argument("--eval_poses_per_group", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default="1")
    parser.add_argument("--clash_penalty_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--main_metric", type=str, default="val_top1_success")
    parser.add_argument("--maximize_metric", action="store_true", default=True)
    parser.add_argument("--save_every", type=int, default=0)
    return parser.parse_args()


def prepare_datasets(args: argparse.Namespace) -> Tuple[List[PoseRecord | PoseRecordRef], List[PoseRecord | PoseRecordRef]]:
    if args.train_records_shard_index is not None and args.val_records_shard_index is not None:
        train_records = load_sharded_record_refs_from_index(args.train_records_shard_index)
        val_records = load_sharded_record_refs_from_index(args.val_records_shard_index)
        return train_records, val_records
    if args.train_records_index is not None and args.val_records_index is not None:
        train_records = load_record_refs_from_index(args.train_records_index)
        val_records = load_record_refs_from_index(args.val_records_index)
        return train_records, val_records
    if args.train_records_manifest is not None and args.val_records_manifest is not None:
        train_records = load_records_from_npz(args.train_records_manifest)
        val_records = load_records_from_npz(args.val_records_manifest)
        return train_records, val_records
    if args.records_manifest is None:
        raise ValueError("either --records_manifest or both --train_records_manifest/--val_records_manifest are required")
    records = load_records_from_npz(args.records_manifest)
    return split_records_by_group(records, val_ratio=args.val_ratio, seed=args.seed)


def main() -> None:
    args = parse_args()
    apply_default_shard_snapshot_args(args)
    set_seed(args.seed)
    output_dir = args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config_from_record_source(args)
    config = apply_config_overrides(config, args)
    device, gpu_ids = resolve_device_and_gpu_ids(args.device, args.gpu_ids)
    args.device = str(device)

    train_records, val_records = prepare_datasets(args)
    snapshot_config(output_dir, args, config, train_records, val_records)

    train_loader = make_loader(train_records, groups_per_batch=args.groups_per_batch, poses_per_group=args.poses_per_group, shuffle=True)
    eval_poses_per_group = args.eval_poses_per_group
    val_loader = make_loader(val_records, groups_per_batch=args.groups_per_batch, poses_per_group=eval_poses_per_group, shuffle=False)
    train_eval_loader = make_loader(train_records, groups_per_batch=args.groups_per_batch, poses_per_group=eval_poses_per_group, shuffle=False)

    example_batch = next(iter(train_loader))
    model = build_model(args.model_name, example_batch, config).to(device)
    model = maybe_wrap_data_parallel(model, gpu_ids)
    if args.init_checkpoint is not None:
        payload = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        load_model_state_dict_allowing_deprecated_heads(model, payload["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    best_metric = None
    best_checkpoint = None
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, config, device)
        val_metrics = evaluate(model, val_loader, config, device, split_name="val", clash_penalty_scale=args.clash_penalty_scale)
        train_eval_metrics = evaluate(model, train_eval_loader, config, device, split_name="train_eval", clash_penalty_scale=args.clash_penalty_scale)
        epoch_seconds = time.perf_counter() - epoch_start

        summary = {"epoch": epoch, "epoch_seconds": epoch_seconds, **train_metrics, **train_eval_metrics, **val_metrics}
        summary["main_metric"] = choose_metric(summary, args.main_metric)
        history.append(summary)
        print(summary, flush=True)
        pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)

        current_metric = float(summary["main_metric"])
        is_better = best_metric is None or current_metric > best_metric
        if is_better:
            best_metric = current_metric
            best_checkpoint = save_checkpoint(output_dir, model, optimizer, epoch, summary, tag="best")
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir, model, optimizer, epoch, summary, tag=f"epoch_{epoch}")

    total_seconds = time.perf_counter() - start_time
    last_summary = history[-1]
    last_checkpoint = save_checkpoint(output_dir, model, optimizer, last_summary["epoch"], last_summary, tag="last")

    pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)
    report = {
        "best_metric_name": args.main_metric,
        "best_metric_value": best_metric,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else "",
        "last_checkpoint": str(last_checkpoint),
        "epochs": args.epochs,
        "total_seconds": total_seconds,
    }
    with (output_dir / "train_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
