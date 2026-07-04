from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.io import load_pose_record, save_pose_record_shard
from ..records import PoseRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 PoseCred-IPG 续训用 mixed hard-negative shard 数据。")
    parser.add_argument("--strict_selection_manifest_all", type=Path, required=True)
    parser.add_argument("--strict_cross_pose_table_all", type=Path, required=True)
    parser.add_argument("--strict_records_index", type=Path, required=True)
    parser.add_argument("--strict_per_pose_scores", type=Path, required=True)
    parser.add_argument("--strict_geometry_csv", type=Path, required=True)
    parser.add_argument("--positive_eval_root", type=Path, required=True)
    parser.add_argument("--default_train_shard_index", type=Path, required=True)
    parser.add_argument("--config_snapshot", type=Path, required=True)
    parser.add_argument("--variant", choices=["v1_conservative", "v2_aggressive"], required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--shard_size", type=int, default=512)
    return parser.parse_args()


def load_positive_eval_table(positive_eval_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for sample_root in sorted(path for path in positive_eval_root.glob("sample_*") if path.is_dir()):
        csv_path = sample_root / "positive_receptor_eval_all_poses_real.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"缺少正例真实评测 CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        required = {"complex_name", "pose", "dockq", "complex_rmsd"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{csv_path} 缺少列: {missing}")
        df = df.copy()
        df["sample_id"] = sample_root.name
        df["pose_name"] = df["pose"].astype(str).map(lambda value: value.replace(".pdb", ""))
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged[
        ["sample_id", "complex_name", "pose_name", "dockq", "complex_rmsd"]
    ].copy()
    merged = merged.rename(columns={"complex_rmsd": "real_rmsd"})
    return merged


def build_source_group_split(selection_df: pd.DataFrame, seed: int, train_fraction: float, val_fraction: float, test_fraction: float) -> pd.DataFrame:
    total_fraction = train_fraction + val_fraction + test_fraction
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"split 比例之和必须为 1.0，当前是 {total_fraction}")

    unique_source_groups = sorted(selection_df["source_group_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_source_groups)

    total_count = len(unique_source_groups)
    val_count = int(round(total_count * val_fraction))
    test_count = int(round(total_count * test_fraction))
    train_count = total_count - val_count - test_count
    if train_count <= 0 or val_count <= 0 or test_count <= 0:
        raise ValueError(
            f"split 后 train/val/test 至少都要有 1 个 source_group，当前 train={train_count}, val={val_count}, test={test_count}"
        )

    train_groups = unique_source_groups[:train_count]
    val_groups = unique_source_groups[train_count : train_count + val_count]
    test_groups = unique_source_groups[train_count + val_count :]

    rows: List[Dict[str, object]] = []
    for split_name, source_groups in [("train", train_groups), ("val", val_groups), ("test", test_groups)]:
        for rank, source_group_id in enumerate(source_groups, start=1):
            rows.append(
                {
                    "source_group_id": source_group_id,
                    "split": split_name,
                    "split_rank": rank,
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "split_rank"]).reset_index(drop=True)


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} 缺少列: {missing}")


def choose_v1_negative(complex_best_df: pd.DataFrame) -> List[Tuple[str, str]]:
    if complex_best_df.empty:
        return []
    top = complex_best_df.sort_values("score", ascending=False).iloc[0]
    return [("top_score", str(top["complex_id"]))]


def _pick_distinct_complex(
    complex_best_df: pd.DataFrame,
    used_complex_ids: set[str],
    condition: pd.Series,
) -> str | None:
    filtered = complex_best_df.loc[condition].sort_values("score", ascending=False)
    for row in filtered.itertuples(index=False):
        complex_id = str(row.complex_id)
        if complex_id not in used_complex_ids:
            return complex_id
    fallback = complex_best_df.sort_values("score", ascending=False)
    for row in fallback.itertuples(index=False):
        complex_id = str(row.complex_id)
        if complex_id not in used_complex_ids:
            return complex_id
    return None


def choose_v2_negatives(complex_best_df: pd.DataFrame) -> List[Tuple[str, str]]:
    if complex_best_df.empty:
        return []

    used_complex_ids: set[str] = set()
    selections: List[Tuple[str, str]] = []

    overall_complex = _pick_distinct_complex(
        complex_best_df=complex_best_df,
        used_complex_ids=used_complex_ids,
        condition=pd.Series([True] * len(complex_best_df), index=complex_best_df.index),
    )
    if overall_complex is not None:
        selections.append(("top_score", overall_complex))
        used_complex_ids.add(overall_complex)

    nonbad_condition = (
        complex_best_df["physical_bad_label"].astype(int).eq(0)
        & complex_best_df["max_vdw_overlap"].astype(float).lt(0.35)
        & complex_best_df["min_heavy_dist"].astype(float).gt(1.75)
    )
    nonbad_complex = _pick_distinct_complex(complex_best_df, used_complex_ids, nonbad_condition)
    if nonbad_complex is not None:
        selections.append(("top_nonbad", nonbad_complex))
        used_complex_ids.add(nonbad_complex)

    bad_condition = (
        complex_best_df["physical_bad_label"].astype(int).eq(1)
        | complex_best_df["max_vdw_overlap"].astype(float).ge(0.35)
        | complex_best_df["min_heavy_dist"].astype(float).le(1.75)
    )
    bad_complex = _pick_distinct_complex(complex_best_df, used_complex_ids, bad_condition)
    if bad_complex is not None:
        selections.append(("top_bad", bad_complex))
        used_complex_ids.add(bad_complex)

    deduped: List[Tuple[str, str]] = []
    seen_complex_ids: set[str] = set()
    for bucket_name, complex_id in selections:
        if complex_id in seen_complex_ids:
            continue
        deduped.append((bucket_name, complex_id))
        seen_complex_ids.add(complex_id)
    return deduped


def save_records_to_shards(records: Sequence[PoseRecord], out_dir: Path, shard_size: int) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_rows: List[Dict[str, object]] = []
    if not records:
        empty_df = pd.DataFrame(
            columns=["shard_path", "local_index", "pose_id", "complex_id", "group_id", "receptor_id", "peptide_id", "dockq", "rmsd"]
        )
        empty_df.to_csv(out_dir / "shard_index.csv", index=False)
        return empty_df

    num_shards = math.ceil(len(records) / max(1, shard_size))
    for shard_id in range(num_shards):
        shard_records = list(records[shard_id * shard_size : (shard_id + 1) * shard_size])
        shard_path = out_dir / f"shard_{shard_id:05d}.pt"
        save_pose_record_shard(shard_records, shard_path)
        for local_index, record in enumerate(shard_records):
            shard_rows.append(
                {
                    "shard_path": str(shard_path),
                    "local_index": local_index,
                    "pose_id": record.pose_id,
                    "complex_id": record.complex_id,
                    "group_id": record.group_id,
                    "receptor_id": record.receptor_id,
                    "peptide_id": record.peptide_id,
                    "dockq": float(record.dockq),
                    "rmsd": float(record.rmsd),
                }
            )
    shard_index_df = pd.DataFrame(shard_rows)
    shard_index_df.to_csv(out_dir / "shard_index.csv", index=False)
    return shard_index_df


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selection_df = pd.read_csv(args.strict_selection_manifest_all)
    cross_pose_table_df = pd.read_csv(args.strict_cross_pose_table_all)
    records_index_df = pd.read_csv(args.strict_records_index)
    pose_scores_df = pd.read_csv(args.strict_per_pose_scores)
    geometry_df = pd.read_csv(args.strict_geometry_csv)
    positive_eval_df = load_positive_eval_table(args.positive_eval_root.resolve())
    base_train_shard_df = pd.read_csv(args.default_train_shard_index)

    ensure_required_columns(
        selection_df,
        {
            "bootstrap_sample_id",
            "bootstrap_slot",
            "group_id",
            "source_group_id",
            "complex_name",
            "pair_label",
            "receptor_id",
        },
        "strict selection manifest",
    )
    ensure_required_columns(
        cross_pose_table_df,
        {"bootstrap_sample_id", "bootstrap_slot", "source_group_id", "complex_name"},
        "strict cross pose table",
    )
    ensure_required_columns(
        pose_scores_df,
        {"sample_id", "pose_id", "complex_id", "source_group_id", "receptor_id", "pose_name", "score"},
        "strict per_pose_scores",
    )
    ensure_required_columns(
        records_index_df,
        {"sample_id", "pose_id", "record_path"},
        "strict records index",
    )
    ensure_required_columns(
        geometry_df,
        {"sample_id", "pose_id", "physical_bad_label", "min_heavy_dist", "max_vdw_overlap"},
        "strict geometry csv",
    )

    split_df = build_source_group_split(
        selection_df=selection_df,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    split_map = dict(zip(split_df["source_group_id"].astype(str), split_df["split"].astype(str)))

    selection_df = selection_df.copy()
    selection_df["sample_id"] = selection_df["bootstrap_sample_id"].astype(str)
    selection_df["bootstrap_group_id"] = selection_df["group_id"].astype(str)
    selection_df["source_group_id"] = selection_df["source_group_id"].astype(str)
    selection_df["split"] = selection_df["source_group_id"].map(split_map)

    pose_scores_df = pose_scores_df.copy()
    pose_scores_df["sample_id"] = pose_scores_df["sample_id"].astype(str)
    pose_scores_df["bootstrap_group_id"] = pose_scores_df["source_group_id"].astype(str)
    pose_scores_df = pose_scores_df.rename(columns={"source_group_id": "bootstrap_group_id_original"})

    slot_meta_df = selection_df[
        [
            "sample_id",
            "bootstrap_sample_id",
            "bootstrap_slot",
            "bootstrap_group_id",
            "source_group_id",
            "split",
            "complex_name",
            "pair_label",
        ]
    ].drop_duplicates(["sample_id", "bootstrap_group_id", "complex_name"])

    pose_scores_df = pose_scores_df.merge(
        slot_meta_df,
        left_on=["sample_id", "bootstrap_group_id_original", "complex_id"],
        right_on=["sample_id", "bootstrap_group_id", "complex_name"],
        how="left",
        validate="many_to_one",
    )
    if pose_scores_df["pair_label"].isna().any():
        missing_rows = int(pose_scores_df["pair_label"].isna().sum())
        raise ValueError(f"pose_scores 与 selection manifest 对齐失败，缺少 {missing_rows} 行标签")

    pose_scores_df = pose_scores_df.merge(
        records_index_df[["sample_id", "pose_id", "record_path"]],
        on=["sample_id", "pose_id"],
        how="left",
        validate="one_to_one",
    )
    pose_scores_df = pose_scores_df.merge(
        geometry_df[["sample_id", "pose_id", "physical_bad_label", "min_heavy_dist", "max_vdw_overlap", "mean_vdw_overlap", "severe_clash_count", "steric_overcrowding"]],
        on=["sample_id", "pose_id"],
        how="left",
        validate="one_to_one",
    )
    pose_scores_df = pose_scores_df.merge(
        positive_eval_df[["sample_id", "complex_name", "pose_name", "dockq", "real_rmsd"]],
        left_on=["sample_id", "complex_id", "pose_name"],
        right_on=["sample_id", "complex_name", "pose_name"],
        how="left",
    )
    pose_scores_df = pose_scores_df.rename(columns={"dockq": "real_dockq"})
    pose_scores_df["is_positive_complex"] = pose_scores_df["pair_label"].astype(str).eq("cocrystal").astype(int)
    if pose_scores_df["record_path"].isna().any():
        missing_rows = int(pose_scores_df["record_path"].isna().sum())
        raise ValueError(f"pose_scores 缺少 record_path，数量={missing_rows}")

    record_cache: Dict[str, PoseRecord] = {}

    def load_cached_record(record_path_str: str) -> PoseRecord:
        if record_path_str not in record_cache:
            record_cache[record_path_str] = load_pose_record(Path(record_path_str))
        return copy.deepcopy(record_cache[record_path_str])

    split_records: Dict[str, List[PoseRecord]] = {"train": [], "val": [], "test": []}
    group_rows: List[Dict[str, object]] = []
    pose_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []

    slot_groups = selection_df[
        ["sample_id", "bootstrap_sample_id", "bootstrap_slot", "bootstrap_group_id", "source_group_id", "split"]
    ].drop_duplicates()
    slot_groups = slot_groups.sort_values(["sample_id", "bootstrap_slot"]).reset_index(drop=True)

    for slot in slot_groups.itertuples(index=False):
        slot_pose_df = pose_scores_df.loc[
            pose_scores_df["sample_id"].astype(str).eq(str(slot.sample_id))
            & pose_scores_df["bootstrap_group_id_original"].astype(str).eq(str(slot.bootstrap_group_id))
        ].copy()
        if slot_pose_df.empty:
            skipped_rows.append(
                {
                    "sample_id": slot.sample_id,
                    "bootstrap_slot": int(slot.bootstrap_slot),
                    "bootstrap_group_id": slot.bootstrap_group_id,
                    "source_group_id": slot.source_group_id,
                    "split": slot.split,
                    "reason": "empty_slot_pose_df",
                }
            )
            continue

        positive_complex_rows = slot_pose_df.loc[slot_pose_df["is_positive_complex"].astype(int).eq(1)].copy()
        positive_complex_ids = positive_complex_rows["complex_id"].astype(str).unique().tolist()
        if len(positive_complex_ids) != 1:
            skipped_rows.append(
                {
                    "sample_id": slot.sample_id,
                    "bootstrap_slot": int(slot.bootstrap_slot),
                    "bootstrap_group_id": slot.bootstrap_group_id,
                    "source_group_id": slot.source_group_id,
                    "split": slot.split,
                    "reason": f"positive_complex_count={len(positive_complex_ids)}",
                }
            )
            continue

        positive_complex_id = positive_complex_ids[0]
        positive_pose_df = positive_complex_rows.copy()
        if positive_pose_df["real_dockq"].isna().any() or positive_pose_df["real_rmsd"].isna().any():
            skipped_rows.append(
                {
                    "sample_id": slot.sample_id,
                    "bootstrap_slot": int(slot.bootstrap_slot),
                    "bootstrap_group_id": slot.bootstrap_group_id,
                    "source_group_id": slot.source_group_id,
                    "split": slot.split,
                    "reason": "missing_positive_real_labels",
                }
            )
            continue

        negative_pose_df = slot_pose_df.loc[slot_pose_df["complex_id"].astype(str).ne(positive_complex_id)].copy()
        complex_best_df = (
            negative_pose_df.sort_values(["complex_id", "score"], ascending=[True, False])
            .groupby("complex_id", as_index=False)
            .first()
            .reset_index(drop=True)
        )

        if args.variant == "v1_conservative":
            negative_choices = choose_v1_negative(complex_best_df)
        else:
            negative_choices = choose_v2_negatives(complex_best_df)

        if not negative_choices:
            skipped_rows.append(
                {
                    "sample_id": slot.sample_id,
                    "bootstrap_slot": int(slot.bootstrap_slot),
                    "bootstrap_group_id": slot.bootstrap_group_id,
                    "source_group_id": slot.source_group_id,
                    "split": slot.split,
                    "reason": "no_negative_choices",
                }
            )
            continue

        for variant_rank, (bucket_name, negative_complex_id) in enumerate(negative_choices, start=1):
            selected_negative_df = negative_pose_df.loc[
                negative_pose_df["complex_id"].astype(str).eq(str(negative_complex_id))
            ].copy()
            if selected_negative_df.empty:
                skipped_rows.append(
                    {
                        "sample_id": slot.sample_id,
                        "bootstrap_slot": int(slot.bootstrap_slot),
                        "bootstrap_group_id": slot.bootstrap_group_id,
                        "source_group_id": slot.source_group_id,
                        "split": slot.split,
                        "reason": f"empty_negative_df:{negative_complex_id}",
                    }
                )
                continue

            mixed_group_id = (
                f"{args.variant}__{slot.split}__{slot.sample_id}__slot{int(slot.bootstrap_slot):02d}"
                f"__src_{slot.source_group_id}__neg_{selected_negative_df['receptor_id'].iloc[0]}__{bucket_name}"
            )

            mixed_pose_df = pd.concat([positive_pose_df, selected_negative_df], ignore_index=True)
            negative_best_row = (
                selected_negative_df.sort_values("score", ascending=False).iloc[0]
            )
            positive_best_dockq = float(positive_pose_df["real_dockq"].max())

            group_rows.append(
                {
                    "variant": args.variant,
                    "split": slot.split,
                    "source_group_id": slot.source_group_id,
                    "sample_id": slot.sample_id,
                    "bootstrap_sample_id": slot.bootstrap_sample_id,
                    "bootstrap_slot": int(slot.bootstrap_slot),
                    "bootstrap_group_id": slot.bootstrap_group_id,
                    "group_id": mixed_group_id,
                    "variant_rank": variant_rank,
                    "negative_bucket": bucket_name,
                    "positive_complex_id": positive_complex_id,
                    "negative_complex_id": negative_complex_id,
                    "negative_receptor_id": str(selected_negative_df["receptor_id"].iloc[0]),
                    "num_positive_poses": int(len(positive_pose_df)),
                    "num_negative_poses": int(len(selected_negative_df)),
                    "max_positive_real_dockq": positive_best_dockq,
                    "negative_top_score": float(negative_best_row["score"]),
                    "negative_top_bad_label": int(negative_best_row["physical_bad_label"]),
                    "negative_top_min_heavy_dist": float(negative_best_row["min_heavy_dist"]),
                    "negative_top_max_vdw_overlap": float(negative_best_row["max_vdw_overlap"]),
                }
            )

            for pose_row in mixed_pose_df.itertuples(index=False):
                is_positive = int(str(pose_row.complex_id) == str(positive_complex_id))
                record = load_cached_record(str(pose_row.record_path))
                new_pose_id = f"{mixed_group_id}__{pose_row.pose_id}"
                record.pose_id = new_pose_id
                record.group_id = mixed_group_id
                record.peptide_id = mixed_group_id
                record.dockq = float(pose_row.real_dockq) if is_positive else 0.0
                record.rmsd = float(pose_row.real_rmsd) if is_positive else 0.0
                split_records[str(slot.split)].append(record)

                pose_rows.append(
                    {
                        "variant": args.variant,
                        "split": slot.split,
                        "group_id": mixed_group_id,
                        "source_group_id": slot.source_group_id,
                        "sample_id": slot.sample_id,
                        "bootstrap_slot": int(slot.bootstrap_slot),
                        "negative_bucket": bucket_name,
                        "role": "positive" if is_positive else "negative",
                        "original_pose_id": pose_row.pose_id,
                        "new_pose_id": new_pose_id,
                        "complex_id": pose_row.complex_id,
                        "receptor_id": pose_row.receptor_id,
                        "score": float(pose_row.score),
                        "label_dockq": float(record.dockq),
                        "label_rmsd": float(record.rmsd),
                        "physical_bad_label": int(pose_row.physical_bad_label),
                        "min_heavy_dist": float(pose_row.min_heavy_dist),
                        "max_vdw_overlap": float(pose_row.max_vdw_overlap),
                        "record_path": str(pose_row.record_path),
                    }
                )

    shard_indices: Dict[str, pd.DataFrame] = {}
    for split_name in ["train", "val", "test"]:
        shard_indices[split_name] = save_records_to_shards(
            records=split_records[split_name],
            out_dir=out_dir / f"mixed_{split_name}_shards",
            shard_size=args.shard_size,
        )

    combined_train_root = out_dir / "combined_train"
    combined_train_root.mkdir(parents=True, exist_ok=True)
    combined_train_df = pd.concat([base_train_shard_df, shard_indices["train"]], ignore_index=True)
    combined_train_df.to_csv(combined_train_root / "shard_index.csv", index=False)

    config_payload = json.loads(args.config_snapshot.read_text(encoding="utf-8"))
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    split_df.to_csv(out_dir / "source_group_split.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out_dir / "mixed_group_manifest.csv", index=False)
    pd.DataFrame(pose_rows).to_csv(out_dir / "mixed_pose_manifest.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(out_dir / "mixed_skipped_slots.csv", index=False)

    for split_name in ["train", "val", "test"]:
        split_source_groups = set(split_df.loc[split_df["split"].eq(split_name), "source_group_id"].astype(str))
        selection_subset = selection_df.loc[selection_df["source_group_id"].astype(str).isin(split_source_groups)].copy()
        cross_subset = cross_pose_table_df.loc[cross_pose_table_df["source_group_id"].astype(str).isin(split_source_groups)].copy()
        selection_subset.to_csv(out_dir / f"strict_{split_name}_selection_manifest.csv", index=False)
        cross_subset.to_csv(out_dir / f"strict_{split_name}_cross_pose_table.csv", index=False)

    split_summary_rows: List[Dict[str, object]] = []
    for split_name in ["train", "val", "test"]:
        split_source_groups = set(split_df.loc[split_df["split"].eq(split_name), "source_group_id"].astype(str))
        slot_subset = slot_groups.loc[slot_groups["source_group_id"].astype(str).isin(split_source_groups)]
        split_summary_rows.append(
            {
                "split": split_name,
                "source_groups": int(len(split_source_groups)),
                "bootstrap_slots": int(len(slot_subset)),
                "mixed_groups": int(sum(row["split"] == split_name for row in group_rows)),
                "mixed_poses": int(sum(row["split"] == split_name for row in pose_rows)),
                "mixed_shard_rows": int(len(shard_indices[split_name])),
            }
        )
    split_summary_df = pd.DataFrame(split_summary_rows).sort_values("split").reset_index(drop=True)
    split_summary_df.to_csv(out_dir / "split_summary.csv", index=False)

    overlap_rows: List[Dict[str, object]] = []
    split_to_source_groups = {
        split_name: set(split_df.loc[split_df["split"].eq(split_name), "source_group_id"].astype(str))
        for split_name in ["train", "val", "test"]
    }
    for left_split in ["train", "val", "test"]:
        for right_split in ["train", "val", "test"]:
            overlap_rows.append(
                {
                    "left_split": left_split,
                    "right_split": right_split,
                    "source_group_overlap": int(len(split_to_source_groups[left_split] & split_to_source_groups[right_split])),
                }
            )
    pd.DataFrame(overlap_rows).to_csv(out_dir / "source_group_overlap_matrix.csv", index=False)

    receptor_overlap_rows: List[Dict[str, object]] = []
    split_to_receptor_ids = {
        split_name: set(
            selection_df.loc[selection_df["split"].eq(split_name), "receptor_id"].astype(str)
        )
        for split_name in ["train", "val", "test"]
    }
    for left_split in ["train", "val", "test"]:
        for right_split in ["train", "val", "test"]:
            receptor_overlap_rows.append(
                {
                    "left_split": left_split,
                    "right_split": right_split,
                    "receptor_overlap": int(len(split_to_receptor_ids[left_split] & split_to_receptor_ids[right_split])),
                }
            )
    pd.DataFrame(receptor_overlap_rows).to_csv(out_dir / "receptor_overlap_matrix.csv", index=False)

    report = {
        "variant": args.variant,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "num_strict_selection_rows": int(len(selection_df)),
        "num_strict_pose_score_rows": int(len(pose_scores_df)),
        "num_positive_eval_rows": int(len(positive_eval_df)),
        "num_cached_records": int(len(record_cache)),
        "split_summary": split_summary_rows,
        "num_skipped_slots": int(len(skipped_rows)),
        "leakage_control": {
            "split_unit": "source_group_id",
            "reason": "bootstrap sample 之间存在 source_group_id 重复，不能按 sample 直接切 train/val/test",
            "residual_risk": "negative receptor 来自同一 strict536 受体池；即使按 source_group_id 切分，split 之间仍会共享 receptor_id，所以 strict536 内部 holdout 只能算半干净验证，最终主结论必须看 rec70 外测",
            "receptor_overlap_matrix_csv": "receptor_overlap_matrix.csv",
        },
        "dataset_summary": {
            "mixed_train_rows": int(len(shard_indices["train"])),
            "mixed_val_rows": int(len(shard_indices["val"])),
            "mixed_test_rows": int(len(shard_indices["test"])),
            "combined_train_rows": int(len(combined_train_df)),
        },
        "posecred_config": config_payload.get("config") or config_payload.get("posecred_config") or config_payload,
    }
    (out_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
