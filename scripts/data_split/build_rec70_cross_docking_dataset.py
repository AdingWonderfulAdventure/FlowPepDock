#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按当前 20x50 口径构建 rec70 test cross-docking 数据集，并支持保留旧项目做增量扩容。"
    )
    parser.add_argument(
        "--usable_groups_csv",
        type=Path,
        default=Path("data/diagnostics/diag_posecred_ipg_rec70_testgroups_eval_20260324/test_groups_usable.csv"),
    )
    parser.add_argument(
        "--pose_table_csv",
        type=Path,
        default=Path("data/diagnostics/diag_posecred_ipg_rec70_testgroups_eval_20260324/test_docked_pose_table.csv"),
    )
    parser.add_argument(
        "--existing_selected_peptides_csv",
        type=Path,
        default=Path("data/cross_docking_rec70_test20x50_20260324/selected_peptides.csv"),
    )
    parser.add_argument(
        "--existing_selection_manifest_csv",
        type=Path,
        default=Path("data/cross_docking_rec70_test20x50_20260324/selection_manifest_20peptides_x50receptors.csv"),
    )
    parser.add_argument(
        "--processed_root",
        type=Path,
        default=Path("data/rebuild_isolated/rebuild_20251221_163301/processed"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/cross_docking_rec70_test100x100_20260408"),
    )
    parser.add_argument("--target_peptides", type=int, default=100)
    parser.add_argument("--receptors_per_peptide", type=int, default=100)
    parser.add_argument("--max_cocrystal_receptors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260408)
    return parser.parse_args()


def _rename_group_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "group_id" not in out.columns and "source_group_id" in out.columns:
        out = out.rename(columns={"source_group_id": "group_id"})
    return out


def _evenly_sample_sorted(values: Sequence[str], limit: int) -> List[str]:
    ordered = sorted({str(value) for value in values})
    if len(ordered) <= limit:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, num=limit)
    indices = sorted({int(round(pos)) for pos in positions})
    while len(indices) < limit:
        for candidate in range(len(ordered)):
            if candidate not in indices:
                indices.append(candidate)
            if len(indices) == limit:
                break
    return [ordered[idx] for idx in sorted(indices[:limit])]


def _sample_without_replacement(rng: np.random.Generator, values: Sequence[str], count: int) -> List[str]:
    ordered = [str(value) for value in values]
    if count < 0:
        raise ValueError("count 不能为负")
    if count == 0:
        return []
    if len(ordered) < count:
        raise ValueError(f"可选对象不足：需要 {count}，只有 {len(ordered)}")
    chosen = rng.choice(np.array(ordered, dtype=object), size=count, replace=False)
    return [str(item) for item in chosen.tolist()]


def _build_selected_groups(
    usable_df: pd.DataFrame,
    existing_selected_df: pd.DataFrame,
    target_peptides: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_groups = existing_selected_df["group_id"].astype(str).tolist()
    if len(existing_groups) > target_peptides:
        raise ValueError("existing selected groups 数超过 target_peptides")

    usable_df = usable_df.copy()
    usable_df["group_id"] = usable_df["group_id"].astype(str)
    remaining_df = usable_df[~usable_df["group_id"].isin(existing_groups)].reset_index(drop=True)
    need = target_peptides - len(existing_groups)
    if need < 0:
        raise ValueError("target_peptides 小于已有 group 数")
    if len(remaining_df) < need:
        raise ValueError(f"usable group 不足：还需要 {need}，仅剩 {len(remaining_df)}")

    if need == 0:
        additional_df = remaining_df.iloc[[]].copy()
    else:
        sampled_indices = rng.choice(np.arange(len(remaining_df)), size=need, replace=False)
        additional_df = remaining_df.iloc[sampled_indices].copy()
    existing_selected_df = existing_selected_df.copy()
    existing_selected_df["group_id"] = existing_selected_df["group_id"].astype(str)

    full_selected_df = pd.concat([existing_selected_df, additional_df], ignore_index=True)
    return full_selected_df, additional_df


def _group_existing_manifest(existing_manifest_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    grouped: Dict[str, pd.DataFrame] = {}
    for group_id, sub_df in existing_manifest_df.groupby("group_id", sort=False):
        grouped[str(group_id)] = sub_df.sort_values("selection_rank").reset_index(drop=True)
    return grouped


def _build_manifest_rows(
    selected_groups_df: pd.DataFrame,
    existing_manifest_map: Dict[str, pd.DataFrame],
    positive_map: Dict[str, List[str]],
    all_receptors: List[str],
    processed_root: Path,
    receptors_per_peptide: int,
    max_cocrystal_receptors: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_rows: List[Dict[str, object]] = []
    delta_rows: List[Dict[str, object]] = []

    for selected_row in selected_groups_df.itertuples(index=False):
        group_id = str(selected_row.group_id)
        peptide_seq = str(selected_row.peptide_seq)
        all_cocrystal_receptors = positive_map.get(group_id, [])
        if not all_cocrystal_receptors:
            raise ValueError(f"group_id={group_id} 没有任何共晶受体，无法构建 provider")

        selected_cocrystals = _evenly_sample_sorted(all_cocrystal_receptors, max_cocrystal_receptors)
        provider_pdb = selected_cocrystals[0]
        peptide_pdb = (processed_root / provider_pdb / "peptide.pdb").as_posix()
        total_cocrystal = len(all_cocrystal_receptors)
        selected_cocrystal_str = ";".join(selected_cocrystals)

        if group_id in existing_manifest_map:
            preserved_df = existing_manifest_map[group_id]
            preserved_receptors = preserved_df["receptor_id"].astype(str).tolist()
            extra_needed = receptors_per_peptide - len(preserved_receptors)
            if extra_needed < 0:
                raise ValueError(
                    f"group_id={group_id} 旧项目已有 {len(preserved_receptors)} 个受体，超过目标 {receptors_per_peptide}"
                )
            excluded = set(preserved_receptors)
            excluded.update(all_cocrystal_receptors)
            negative_pool = [rid for rid in all_receptors if rid not in excluded]
            extra_receptors = _sample_without_replacement(rng, negative_pool, extra_needed)
            full_receptors = preserved_receptors + extra_receptors
            delta_receptors = extra_receptors
        else:
            negative_pool = [rid for rid in all_receptors if rid not in set(all_cocrystal_receptors)]
            negative_count = receptors_per_peptide - len(selected_cocrystals)
            sampled_negatives = _sample_without_replacement(rng, negative_pool, negative_count)
            full_receptors = selected_cocrystals + sampled_negatives
            delta_receptors = full_receptors

        if len(full_receptors) != receptors_per_peptide:
            raise ValueError(f"group_id={group_id} 受体数错误：{len(full_receptors)} != {receptors_per_peptide}")

        def _build_row(receptor_id: str, rank: int) -> Dict[str, object]:
            receptor_pdb = (processed_root / receptor_id / "receptor.pdb").as_posix()
            return {
                "group_id": group_id,
                "peptide_seq": peptide_seq,
                "provider_pdb": provider_pdb,
                "provider_receptor_id": provider_pdb,
                "peptide_pdb": peptide_pdb,
                "receptor_id": receptor_id,
                "receptor_pdb": receptor_pdb,
                "pair_label": "cocrystal" if receptor_id in selected_cocrystals else "non_cocrystal_test",
                "is_provider_receptor": bool(receptor_id == provider_pdb),
                "selection_rank": rank,
                "total_cocrystal_receptors_for_group": total_cocrystal,
                "selected_cocrystal_receptors": selected_cocrystal_str,
                "complex_name": f"L_{provider_pdb}-R_{receptor_id}",
            }

        for rank, receptor_id in enumerate(full_receptors, start=1):
            full_rows.append(_build_row(receptor_id, rank))

        if group_id in existing_manifest_map:
            base_rank = len(full_receptors) - len(delta_receptors)
            for offset, receptor_id in enumerate(delta_receptors, start=1):
                delta_rows.append(_build_row(receptor_id, base_rank + offset))
        else:
            for rank, receptor_id in enumerate(delta_receptors, start=1):
                delta_rows.append(_build_row(receptor_id, rank))

    full_df = pd.DataFrame(full_rows)
    delta_df = pd.DataFrame(delta_rows)
    expected_delta_groups = set(selected_groups_df["group_id"].astype(str).tolist())
    if set(delta_df["group_id"].astype(str).tolist()) != expected_delta_groups:
        raise ValueError("delta manifest 的 group 覆盖不完整")
    return full_df, delta_df


def _to_flow_input(manifest_df: pd.DataFrame) -> pd.DataFrame:
    return manifest_df[["group_id", "complex_name", "receptor_pdb", "peptide_pdb"]].copy()


def _validate_paths(manifest_df: pd.DataFrame) -> Dict[str, int]:
    receptor_missing = 0
    peptide_missing = 0
    for row in manifest_df.itertuples(index=False):
        if not Path(str(row.receptor_pdb)).exists():
            receptor_missing += 1
        if not Path(str(row.peptide_pdb)).exists():
            peptide_missing += 1
    return {
        "missing_receptor_pdb_rows": receptor_missing,
        "missing_peptide_pdb_rows": peptide_missing,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    usable_df = _rename_group_column(pd.read_csv(args.usable_groups_csv))
    existing_selected_df = pd.read_csv(args.existing_selected_peptides_csv)
    existing_manifest_df = pd.read_csv(args.existing_selection_manifest_csv)
    pose_df = pd.read_csv(args.pose_table_csv)

    existing_selected_df["group_id"] = existing_selected_df["group_id"].astype(str)
    existing_manifest_df["group_id"] = existing_manifest_df["group_id"].astype(str)
    usable_df["group_id"] = usable_df["group_id"].astype(str)
    pose_df["group_id"] = pose_df["group_id"].astype(str)
    pose_df["receptor_id"] = pose_df["receptor_id"].astype(str)

    positive_pairs_df = pose_df[pose_df["is_positive"] == 1][["group_id", "receptor_id"]].drop_duplicates()
    positive_map = {
        str(group_id): sorted(sub_df["receptor_id"].astype(str).tolist())
        for group_id, sub_df in positive_pairs_df.groupby("group_id", sort=False)
    }
    all_receptors = sorted(pose_df["receptor_id"].astype(str).unique().tolist())

    full_selected_df, additional_selected_df = _build_selected_groups(
        usable_df=usable_df,
        existing_selected_df=existing_selected_df,
        target_peptides=args.target_peptides,
        rng=rng,
    )
    existing_manifest_map = _group_existing_manifest(existing_manifest_df)

    full_manifest_df, delta_manifest_df = _build_manifest_rows(
        selected_groups_df=full_selected_df,
        existing_manifest_map=existing_manifest_map,
        positive_map=positive_map,
        all_receptors=all_receptors,
        processed_root=args.processed_root,
        receptors_per_peptide=args.receptors_per_peptide,
        max_cocrystal_receptors=args.max_cocrystal_receptors,
        rng=rng,
    )

    full_flow_input_df = _to_flow_input(full_manifest_df)
    delta_flow_input_df = _to_flow_input(delta_manifest_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_selected_df.to_csv(args.out_dir / "selected_peptides_100groups.csv", index=False)
    additional_selected_df.to_csv(args.out_dir / "selected_peptides_increment_new80groups.csv", index=False)
    full_manifest_df.to_csv(args.out_dir / "selection_manifest_100peptides_x100receptors.csv", index=False)
    delta_manifest_df.to_csv(args.out_dir / "selection_manifest_increment_from20x50_to100x100.csv", index=False)
    full_flow_input_df.to_csv(args.out_dir / "flow_input_100peptides_x100receptors.csv", index=False)
    delta_flow_input_df.to_csv(args.out_dir / "flow_input_increment_from20x50_to100x100.csv", index=False)

    summary = {
        "seed": int(args.seed),
        "target_peptides": int(args.target_peptides),
        "receptors_per_peptide": int(args.receptors_per_peptide),
        "max_cocrystal_receptors": int(args.max_cocrystal_receptors),
        "existing_base_selected_groups": int(existing_selected_df["group_id"].nunique()),
        "newly_added_groups": int(additional_selected_df["group_id"].nunique()),
        "full_selected_groups": int(full_selected_df["group_id"].nunique()),
        "full_complexes": int(len(full_manifest_df)),
        "delta_complexes": int(len(delta_manifest_df)),
        "full_cocrystal_rows": int((full_manifest_df["pair_label"] == "cocrystal").sum()),
        "delta_cocrystal_rows": int((delta_manifest_df["pair_label"] == "cocrystal").sum()),
        "full_path_check": _validate_paths(full_manifest_df),
        "delta_path_check": _validate_paths(delta_manifest_df),
        "source_files": {
            "usable_groups_csv": str(args.usable_groups_csv),
            "pose_table_csv": str(args.pose_table_csv),
            "existing_selected_peptides_csv": str(args.existing_selected_peptides_csv),
            "existing_selection_manifest_csv": str(args.existing_selection_manifest_csv),
        },
    }
    (args.out_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
