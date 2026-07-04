from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch

from ..config import PoseCredConfig
from ..core.paths import POSECRED_ROOT
from ..data.features import build_pose_record_from_pair_pdbs
from ..data.io import load_manifest, save_pose_record, write_manifest
from ..data.pdbio import ResidueData, load_all_residues
from ..default_paths import DEFAULT_SNAPSHOT_ROOT
from ..engine.train import build_model, make_loader, set_seed
from ..evaluation.evaluate_pooled_global import collect_predictions
from ..records import PoseRecord, PoseRecordRef
from ..runtime import load_model_state_dict_allowing_deprecated_heads, resolve_device_and_gpu_ids


DEFAULT_CHECKPOINT = POSECRED_ROOT / "final_exports" / "graph_main_best.pt"
DEFAULT_CONFIG_SNAPSHOT = DEFAULT_SNAPSHOT_ROOT / "config_snapshot.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score cross-docking pose tables with PoseCred-IPG.")
    parser.add_argument("--cross_pose_table", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config_snapshot", type=Path, default=DEFAULT_CONFIG_SNAPSHOT)
    parser.add_argument("--records_manifest", type=Path, default=None)
    parser.add_argument("--records_index", type=Path, default=None)
    parser.add_argument("--save_records_dir", type=Path, default=None)
    parser.add_argument("--save_uncompressed_npz", action="store_true")
    parser.add_argument("--groups_per_batch", type=int, default=0)
    parser.add_argument("--poses_per_group", type=int, default=0)
    parser.add_argument("--build_workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--clash_penalty_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--full_model_forward", action="store_true")
    parser.add_argument("--limit_rows", type=int, default=0)
    parser.add_argument("--enable_batch_timing", action="store_true")
    return parser.parse_args()


def _load_config_snapshot(config_snapshot: Path) -> PoseCredConfig:
    payload = json.loads(config_snapshot.read_text(encoding="utf-8"))
    config_dict = payload.get("config") or payload.get("posecred_config") or payload
    return PoseCredConfig(**config_dict)


def _normalize_cross_pose_table(table: pd.DataFrame) -> pd.DataFrame:
    required = {"complex_name", "group_id", "pose_name", "pose_path", "receptor_pdb"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"cross_pose_table missing required columns: {missing}")

    out = table.copy()
    out["row_order"] = list(range(len(out)))
    out["pose_id"] = out["complex_name"].astype(str) + "__" + out["pose_name"].astype(str)
    out["complex_id"] = out["complex_name"].astype(str)
    out["source_group_id"] = out["group_id"].astype(str)
    out["group_id"] = out["complex_name"].astype(str)
    out["peptide_id"] = out["source_group_id"]
    if "receptor_id" not in out.columns:
        out["receptor_id"] = out["complex_name"].astype(str).map(
            lambda value: value.split("-R_", 1)[1] if "-R_" in value else value
        )
    out["receptor_id"] = out["receptor_id"].astype(str)
    out["pose_name"] = out["pose_name"].astype(str)
    out["pose_path"] = out["pose_path"].astype(str)
    out["receptor_pdb"] = out["receptor_pdb"].astype(str)
    if "dockq" not in out.columns:
        out["dockq"] = 0.0
    if "complex_rmsd" in out.columns:
        out["rmsd"] = out["complex_rmsd"].fillna(0.0).astype(float)
    elif "peptide_ca_rmsd" in out.columns:
        out["rmsd"] = out["peptide_ca_rmsd"].fillna(0.0).astype(float)
    else:
        out["rmsd"] = 0.0
    out["dockq"] = out["dockq"].fillna(0.0).astype(float)
    return out


def _build_record_refs_from_manifest(manifest_path: Path, meta_df: pd.DataFrame) -> List[PoseRecordRef]:
    meta_map = meta_df.set_index("pose_id").to_dict("index")
    refs: List[PoseRecordRef] = []
    for record_path in load_manifest(manifest_path):
        pose_id = record_path.stem
        if pose_id not in meta_map:
            raise ValueError(f"pose_id {pose_id} from manifest not found in cross_pose_table")
        meta = meta_map[pose_id]
        refs.append(
            PoseRecordRef(
                record_path=record_path,
                pose_id=pose_id,
                complex_id=str(meta["complex_id"]),
                group_id=str(meta["group_id"]),
                receptor_id=str(meta["receptor_id"]),
                peptide_id=str(meta["peptide_id"]),
                dockq=float(meta["dockq"]),
                rmsd=float(meta["rmsd"]),
            )
        )
    return refs


def _build_record_refs_from_index(index_path: Path, meta_df: pd.DataFrame) -> List[PoseRecordRef]:
    index_df = pd.read_csv(index_path)
    if "pose_id" not in index_df.columns or "record_path" not in index_df.columns:
        raise ValueError("records_index 缺少 pose_id / record_path 列")
    meta_pose_ids = set(meta_df["pose_id"].tolist())
    filtered = index_df[index_df["pose_id"].astype(str).isin(meta_pose_ids)].copy()
    meta_map = meta_df.set_index("pose_id").to_dict("index")
    refs: List[PoseRecordRef] = []
    for row in filtered.itertuples(index=False):
        meta = meta_map[str(row.pose_id)]
        refs.append(
            PoseRecordRef(
                record_path=Path(str(row.record_path)),
                pose_id=str(row.pose_id),
                complex_id=str(meta["complex_id"]),
                group_id=str(meta["group_id"]),
                receptor_id=str(meta["receptor_id"]),
                peptide_id=str(meta["peptide_id"]),
                dockq=float(meta["dockq"]),
                rmsd=float(meta["rmsd"]),
            )
        )
    return refs


def _build_single_record_from_row(
    row_dict: Dict[str, object],
    config: PoseCredConfig,
    receptor_residues: Sequence[ResidueData],
    peptide_residues: Sequence[ResidueData],
) -> PoseRecord:
    return build_pose_record_from_pair_pdbs(
        receptor_pdb=Path(str(row_dict["receptor_pdb"])),
        peptide_pdb=Path(str(row_dict["pose_path"])),
        pose_id=str(row_dict["pose_id"]),
        complex_id=str(row_dict["complex_id"]),
        group_id=str(row_dict["group_id"]),
        receptor_id=str(row_dict["receptor_id"]),
        peptide_id=str(row_dict["peptide_id"]),
        dockq=float(row_dict["dockq"]),
        rmsd=float(row_dict["rmsd"]),
        config=config,
        receptor_residues=receptor_residues,
        peptide_residues=peptide_residues,
    )


def _build_records_for_receptor_group(
    row_dicts: Sequence[Dict[str, object]],
    config_payload: Dict[str, object],
    save_records_dir: str | None,
    save_compressed: bool,
) -> List[Dict[str, object]]:
    config = PoseCredConfig(**config_payload)
    receptor_cache: Dict[str, List[ResidueData]] = {}
    peptide_cache: Dict[str, List[ResidueData]] = {}
    results: List[Dict[str, object]] = []
    for row_dict in row_dicts:
        receptor_pdb = Path(str(row_dict["receptor_pdb"]))
        peptide_pdb = Path(str(row_dict["pose_path"]))
        receptor_key = str(receptor_pdb)
        peptide_key = str(peptide_pdb)
        if receptor_key not in receptor_cache:
            receptor_cache[receptor_key] = load_all_residues(receptor_pdb)
        if peptide_key not in peptide_cache:
            peptide_cache[peptide_key] = load_all_residues(peptide_pdb)
        record = _build_single_record_from_row(
            row_dict=row_dict,
            config=config,
            receptor_residues=receptor_cache[receptor_key],
            peptide_residues=peptide_cache[peptide_key],
        )
        record_path = None
        if save_records_dir is not None:
            record_path = save_pose_record(record, Path(save_records_dir), compressed=save_compressed).as_posix()
        results.append(
            {
                "row_order": int(row_dict["row_order"]),
                "record": record,
                "record_path": record_path,
            }
        )
    return results


def _build_records_from_table(
    meta_df: pd.DataFrame,
    config: PoseCredConfig,
    save_records_dir: Path | None,
    save_compressed: bool,
    build_workers: int,
) -> tuple[List[PoseRecord], Dict[str, float]]:
    receptor_cache: Dict[str, List[ResidueData]] = {}
    peptide_cache: Dict[str, List[ResidueData]] = {}
    records: List[PoseRecord] = []
    record_paths: List[Path] = []
    build_start = time.perf_counter()

    build_workers = max(1, int(build_workers))
    if build_workers <= 1:
        for row in meta_df.itertuples(index=False):
            row_dict = row._asdict()
            receptor_pdb = Path(str(row_dict["receptor_pdb"]))
            peptide_pdb = Path(str(row_dict["pose_path"]))
            receptor_key = str(receptor_pdb)
            peptide_key = str(peptide_pdb)
            if receptor_key not in receptor_cache:
                receptor_cache[receptor_key] = load_all_residues(receptor_pdb)
            if peptide_key not in peptide_cache:
                peptide_cache[peptide_key] = load_all_residues(peptide_pdb)
            record = _build_single_record_from_row(
                row_dict=row_dict,
                config=config,
                receptor_residues=receptor_cache[receptor_key],
                peptide_residues=peptide_cache[peptide_key],
            )
            records.append(record)
            if save_records_dir is not None:
                record_paths.append(save_pose_record(record, save_records_dir, compressed=save_compressed))
    else:
        grouped_rows = [
            [row._asdict() for row in group.itertuples(index=False)]
            for _receptor_pdb, group in meta_df.groupby("receptor_pdb", sort=False)
        ]
        config_payload = config.__dict__.copy()
        save_records_dir_str = save_records_dir.as_posix() if save_records_dir is not None else None
        built_items: List[Dict[str, object]] = []
        with ProcessPoolExecutor(max_workers=build_workers) as executor:
            futures = [
                executor.submit(
                    _build_records_for_receptor_group,
                    row_dicts,
                    config_payload,
                    save_records_dir_str,
                    save_compressed,
                )
                for row_dicts in grouped_rows
            ]
            for future in futures:
                built_items.extend(future.result())
        built_items.sort(key=lambda item: int(item["row_order"]))
        records = [item["record"] for item in built_items]
        if save_records_dir is not None:
            record_paths = [Path(str(item["record_path"])) for item in built_items]

    build_seconds = time.perf_counter() - build_start
    save_seconds = 0.0
    if save_records_dir is not None:
        save_start = time.perf_counter()
        manifest_path = save_records_dir / "manifest.txt"
        write_manifest(record_paths, manifest_path)
        records_index = meta_df[
            ["pose_id", "complex_id", "group_id", "receptor_id", "peptide_id", "dockq", "rmsd"]
        ].copy()
        records_index["record_path"] = [path.as_posix() for path in record_paths]
        records_index.to_csv(save_records_dir / "records_index.csv", index=False)
        save_seconds = time.perf_counter() - save_start
    return records, {"build_seconds": build_seconds, "save_seconds": save_seconds}


def _score_records(
    records: Sequence[PoseRecord | PoseRecordRef],
    checkpoint: Path,
    config: PoseCredConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    score_loader_prepare_start = time.perf_counter()
    resolved_groups_per_batch = int(args.groups_per_batch)
    if resolved_groups_per_batch <= 0:
        resolved_groups_per_batch = max(1, len({record.group_id for record in records}))
    loader = make_loader(
        records,
        groups_per_batch=resolved_groups_per_batch,
        poses_per_group=args.poses_per_group,
        shuffle=False,
        num_workers=args.num_workers if records and isinstance(records[0], PoseRecordRef) else 0,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0 and records and isinstance(records[0], PoseRecordRef),
        prefetch_factor=args.prefetch_factor,
    )
    loader_iter = iter(loader)
    example_batch = next(loader_iter)
    loader_prepare_seconds = time.perf_counter() - score_loader_prepare_start

    model_load_start = time.perf_counter()
    model = build_model("posecred_ipg", example_batch, config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    load_model_state_dict_allowing_deprecated_heads(model, payload["model_state_dict"])
    model_load_seconds = time.perf_counter() - model_load_start
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch_timing_rows: List[Dict[str, float | int | str]] | None = [] if args.enable_batch_timing else None
    score_start = time.perf_counter()
    pred_df = collect_predictions(
        model=model,
        loader=itertools.chain([example_batch], loader_iter),
        model_name="posecred_ipg",
        clash_penalty_weights=config.clash_penalty_weights,
        clash_penalty_scale=args.clash_penalty_scale,
        device=device,
        use_score_only_fastpath=not args.full_model_forward,
        timing_rows=batch_timing_rows,
    )
    score_seconds = time.perf_counter() - score_start
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    timing_summary: Dict[str, float | int] = {
        "score_seconds": score_seconds,
        "loader_prepare_seconds": loader_prepare_seconds,
        "model_load_seconds": model_load_seconds,
        "max_gpu_mem_mb": peak_memory_mb,
        "resolved_groups_per_batch": resolved_groups_per_batch,
    }
    if batch_timing_rows is not None:
        transfer_seconds = sum(float(row["transfer_seconds"]) for row in batch_timing_rows)
        forward_seconds = sum(float(row["forward_seconds"]) for row in batch_timing_rows)
        penalty_seconds = sum(float(row["penalty_seconds"]) for row in batch_timing_rows)
        collect_seconds = sum(float(row["collect_seconds"]) for row in batch_timing_rows)
        timing_summary.update(
            {
                "timed_batches": int(len(batch_timing_rows)),
                "transfer_seconds_sum": transfer_seconds,
                "forward_seconds_sum": forward_seconds,
                "penalty_seconds_sum": penalty_seconds,
                "collect_seconds_sum": collect_seconds,
            }
        )
    return pred_df, {
        **timing_summary,
        "batch_timing_rows": batch_timing_rows if batch_timing_rows is not None else [],
    }


def _finalize_score_table(meta_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    score_df = meta_df.merge(pred_df[["pose_id", "score"]], on="pose_id", how="left", validate="one_to_one")
    if score_df["score"].isna().any():
        missing_pose_ids = score_df.loc[score_df["score"].isna(), "pose_id"].tolist()[:10]
        raise ValueError(f"missing scores for pose_ids: {missing_pose_ids}")
    score_df = score_df[
        [
            "pose_id",
            "complex_id",
            "group_id",
            "source_group_id",
            "peptide_id",
            "receptor_id",
            "pose_name",
            "pose_path",
            "receptor_pdb",
            "score",
        ]
    ].copy()
    score_df["rank_in_group"] = score_df.groupby("group_id")["score"].rank(method="first", ascending=False).astype(int)
    score_df["rank_in_peptide"] = score_df.groupby("peptide_id")["score"].rank(method="first", ascending=False).astype(int)
    score_df = score_df.sort_values(["peptide_id", "score"], ascending=[True, False]).reset_index(drop=True)
    return score_df


def _write_outputs(score_df: pd.DataFrame, out_dir: Path, report: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(out_dir / "per_pose_scores.csv", index=False)

    group_best = (
        score_df.sort_values(["group_id", "score"], ascending=[True, False])
        .groupby("group_id", as_index=False)
        .first()
        .sort_values(["peptide_id", "score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    group_best["group_rank_in_peptide"] = group_best.groupby("peptide_id")["score"].rank(method="first", ascending=False).astype(int)
    group_best.to_csv(out_dir / "group_best_scores.csv", index=False)

    peptide_summary = (
        group_best.groupby("peptide_id")
        .agg(
            num_receptors=("group_id", "nunique"),
            best_score=("score", "max"),
            best_group=("group_id", "first"),
        )
        .reset_index()
        .sort_values("best_score", ascending=False)
        .reset_index(drop=True)
    )
    peptide_summary.to_csv(out_dir / "peptide_summary.csv", index=False)

    (out_dir / "scoring_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.records_manifest is not None and args.records_index is not None:
        raise ValueError("--records_manifest 和 --records_index 只能二选一")

    set_seed(args.seed)
    out_dir = args.out_dir.resolve()
    cross_pose_table = args.cross_pose_table.resolve()
    table_df = pd.read_csv(cross_pose_table)
    if args.limit_rows > 0:
        table_df = table_df.head(args.limit_rows).copy()
    meta_df = _normalize_cross_pose_table(table_df)
    device, _ = resolve_device_and_gpu_ids(args.device, args.gpu_ids)

    report: Dict[str, object] = {
        "cross_pose_table": str(cross_pose_table),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "groups_per_batch": int(args.groups_per_batch),
        "num_pose_rows": int(len(meta_df)),
        "num_groups": int(meta_df["group_id"].nunique()),
        "num_peptides": int(meta_df["peptide_id"].nunique()),
        "full_model_forward": bool(args.full_model_forward),
        "poses_per_group": int(args.poses_per_group),
    }

    total_start = time.perf_counter()
    if args.records_index is not None:
        records = _build_record_refs_from_index(args.records_index.resolve(), meta_df)
        config = _load_config_snapshot(args.config_snapshot.resolve())
        report["input_mode"] = "existing_npz_index"
        report["records_index"] = str(args.records_index.resolve())
    elif args.records_manifest is not None:
        records = _build_record_refs_from_manifest(args.records_manifest.resolve(), meta_df)
        config = _load_config_snapshot(args.config_snapshot.resolve())
        report["input_mode"] = "existing_npz_manifest"
        report["records_manifest"] = str(args.records_manifest.resolve())
    else:
        config = _load_config_snapshot(args.config_snapshot.resolve())
        save_records_dir = args.save_records_dir.resolve() if args.save_records_dir is not None else None
        records, build_stats = _build_records_from_table(
            meta_df=meta_df,
            config=config,
            save_records_dir=save_records_dir,
            save_compressed=not args.save_uncompressed_npz,
            build_workers=args.build_workers,
        )
        report["input_mode"] = "inmemory_build"
        report.update(build_stats)
        report["build_workers"] = max(1, int(args.build_workers))
        if save_records_dir is not None:
            report["saved_records_dir"] = str(save_records_dir)
            report["npz_compressed"] = not args.save_uncompressed_npz

    pred_df, score_stats = _score_records(records, args.checkpoint.resolve(), config, args, device)
    score_df = _finalize_score_table(meta_df, pred_df)
    report.update(score_stats)
    batch_timing_rows = report.pop("batch_timing_rows", [])
    report["elapsed_sec"] = round(time.perf_counter() - total_start, 2)

    _write_outputs(score_df, out_dir, report)
    if batch_timing_rows:
        pd.DataFrame(batch_timing_rows).to_csv(out_dir / "batch_timing.csv", index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
