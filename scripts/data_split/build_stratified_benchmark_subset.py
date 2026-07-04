#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 clean test(536) 构建可复现的小规模 benchmark 子集。

目标：
1. 只使用与模型结果无关的输入属性分层，避免 cherry-pick 质疑；
2. 固定随机种子与稳定哈希排序，保证结果可复现；
3. 同时导出 Flow 推理输入表、clean 表和选择清单，方便后续 AF2M / AF3 跑批。

当前默认分层字段：
- peptide 长度分箱：5-7 / 8-10 / 11-13 / 14-20
- receptor crop 残基数四分位分箱
- receptor 链数分箱：1 / 2 / 3+
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


DEFAULT_PEPTIDE_BINS = [4, 7, 10, 13, 20]
DEFAULT_PEPTIDE_LABELS = ["pep_5_7", "pep_8_10", "pep_11_13", "pep_14_20"]


@dataclass(frozen=True)
class PdbStats:
    residue_count: int
    chain_count: int
    atom_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建已知口袋任务的可复现分层 benchmark 小集")
    parser.add_argument(
        "--src_csv",
        type=Path,
        default=Path("data/runtime_tables/flow_infer_test536_rel.csv"),
        help="Flow 推理输入表，至少包含 complex_name/receptor_pdb/peptide_pdb",
    )
    parser.add_argument(
        "--clean_csv",
        type=Path,
        default=Path("data/processed_test30/pt_available.fully_clean.csv"),
        help="clean test 主表，用于同步导出小集 clean CSV",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/benchmark_subsets/test536_known_pocket_stratified_128_seed20260325"),
        help="输出目录",
    )
    parser.add_argument("--sample_size", type=int, default=128, help="小集条目数")
    parser.add_argument("--seed", type=int, default=20260325, help="稳定选择种子")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "complex_name" not in df.columns:
        raise ValueError(f"CSV 缺少 complex_name 列：{path}")
    df = df.copy()
    df["complex_name"] = df["complex_name"].astype(str).str.strip().str.lower()
    df = df[df["complex_name"] != ""].reset_index(drop=True)
    return df


def parse_pdb_stats(path: Path) -> PdbStats:
    residues = set()
    chains = set()
    atom_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip()
            if resname == "HOH":
                continue
            atom_count += 1
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            insertion_code = line[26].strip()
            residues.add((chain_id, resseq, insertion_code, resname))
            chains.add(chain_id)
    if not residues:
        raise ValueError(f"PDB 未解析到有效残基：{path}")
    return PdbStats(residue_count=len(residues), chain_count=len(chains), atom_count=atom_count)


def chain_bin(chain_count: int) -> str:
    if chain_count <= 1:
        return "chain_1"
    if chain_count == 2:
        return "chain_2"
    return "chain_3p"


def build_feature_table(src_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"complex_name", "receptor_pdb", "peptide_pdb"}
    missing = required_cols - set(src_df.columns)
    if missing:
        raise ValueError(f"源 CSV 缺少必要列：{sorted(missing)}")

    rows: List[Dict[str, object]] = []
    for row in src_df.itertuples(index=False):
        receptor_path = Path(row.receptor_pdb)
        peptide_path = Path(row.peptide_pdb)
        receptor_stats = parse_pdb_stats(receptor_path)
        peptide_stats = parse_pdb_stats(peptide_path)
        rows.append(
            {
                "complex_name": row.complex_name,
                "receptor_pdb": row.receptor_pdb,
                "peptide_pdb": row.peptide_pdb,
                "receptor_residue_count": receptor_stats.residue_count,
                "receptor_chain_count": receptor_stats.chain_count,
                "receptor_atom_count": receptor_stats.atom_count,
                "peptide_residue_count": peptide_stats.residue_count,
                "peptide_chain_count": peptide_stats.chain_count,
                "peptide_atom_count": peptide_stats.atom_count,
            }
        )

    feat_df = pd.DataFrame(rows)
    feat_df["peptide_len_bin"] = pd.cut(
        feat_df["peptide_residue_count"],
        bins=DEFAULT_PEPTIDE_BINS,
        labels=DEFAULT_PEPTIDE_LABELS,
        include_lowest=True,
    ).astype(str)

    quantiles = feat_df["receptor_residue_count"].quantile([0.25, 0.5, 0.75]).tolist()
    edges: List[float] = [float(feat_df["receptor_residue_count"].min()) - 1.0]
    for value in quantiles:
        if value > edges[-1]:
            edges.append(float(value))
    edges.append(float(feat_df["receptor_residue_count"].max()))
    receptor_labels = [f"rec_q{i + 1}" for i in range(len(edges) - 1)]
    feat_df["receptor_size_bin"] = pd.cut(
        feat_df["receptor_residue_count"],
        bins=edges,
        labels=receptor_labels,
        include_lowest=True,
    ).astype(str)
    feat_df["receptor_chain_bin"] = feat_df["receptor_chain_count"].map(chain_bin)
    feat_df["stratum"] = (
        feat_df["peptide_len_bin"]
        + "|"
        + feat_df["receptor_size_bin"]
        + "|"
        + feat_df["receptor_chain_bin"]
    )
    return feat_df


def stable_rank(name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def allocate_counts(stratum_sizes: pd.Series, sample_size: int) -> Dict[str, int]:
    nonempty = stratum_sizes[stratum_sizes > 0].sort_index()
    if sample_size <= 0:
        raise ValueError("sample_size 必须 > 0")
    if sample_size > int(nonempty.sum()):
        raise ValueError(f"sample_size={sample_size} 超过可用总量 {int(nonempty.sum())}")

    strata = list(nonempty.index)
    if sample_size < len(strata):
        ranked = sorted(strata, key=lambda key: (-int(nonempty[key]), key))
        chosen = ranked[:sample_size]
        return {key: (1 if key in chosen else 0) for key in strata}

    allocation = {key: 1 for key in strata}
    remaining = sample_size - len(strata)
    weights = nonempty / float(nonempty.sum())
    base_extra = {key: int(math.floor(float(weights[key]) * remaining)) for key in strata}
    for key, extra in base_extra.items():
        allocation[key] += extra

    used = sum(allocation.values())
    remainders = sorted(
        ((float(weights[key]) * remaining - base_extra[key], key) for key in strata),
        key=lambda item: (-item[0], item[1]),
    )
    for _, key in remainders[: sample_size - used]:
        allocation[key] += 1

    changed = True
    while changed:
        changed = False
        overflow = 0
        for key in strata:
            cap = int(nonempty[key])
            if allocation[key] > cap:
                overflow += allocation[key] - cap
                allocation[key] = cap
                changed = True
        if overflow <= 0:
            continue

        spare_keys = [key for key in strata if allocation[key] < int(nonempty[key])]
        if not spare_keys:
            continue
        spare_keys = sorted(
            spare_keys,
            key=lambda key: (-(int(nonempty[key]) - allocation[key]), key),
        )
        for key in spare_keys:
            if overflow <= 0:
                break
            gap = int(nonempty[key]) - allocation[key]
            if gap <= 0:
                continue
            add = min(gap, overflow)
            allocation[key] += add
            overflow -= add
            changed = True

    total = sum(allocation.values())
    if total != sample_size:
        raise RuntimeError(f"分层配额分配失败：expect={sample_size}, got={total}")
    return allocation


def select_subset(feature_df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    allocation = allocate_counts(feature_df["stratum"].value_counts(), sample_size=sample_size)
    ranked_df = feature_df.copy()
    ranked_df["stable_rank"] = ranked_df["complex_name"].map(lambda name: stable_rank(name, seed))
    ranked_df = ranked_df.sort_values(["stratum", "stable_rank", "complex_name"]).reset_index(drop=True)

    picked_parts: List[pd.DataFrame] = []
    for stratum, quota in sorted(allocation.items()):
        if quota <= 0:
            continue
        part = ranked_df[ranked_df["stratum"] == stratum].head(quota)
        if len(part) != quota:
            raise RuntimeError(f"stratum={stratum} quota={quota} 但实际只取到 {len(part)}")
        picked_parts.append(part)

    subset_df = pd.concat(picked_parts, ignore_index=True)
    subset_df = subset_df.sort_values("complex_name").reset_index(drop=True)
    if len(subset_df) != sample_size:
        raise RuntimeError(f"抽样结果条目数不对：expect={sample_size}, got={len(subset_df)}")
    if subset_df["complex_name"].duplicated().any():
        dup_names = subset_df.loc[subset_df["complex_name"].duplicated(), "complex_name"].tolist()
        raise RuntimeError(f"抽样结果出现重复 complex_name：{dup_names[:10]}")
    return subset_df


def summarize_distribution(df: pd.DataFrame, prefix: str) -> Dict[str, object]:
    return {
        f"{prefix}_count": int(len(df)),
        f"{prefix}_peptide_length": {
            "min": int(df["peptide_residue_count"].min()),
            "median": float(df["peptide_residue_count"].median()),
            "max": int(df["peptide_residue_count"].max()),
        },
        f"{prefix}_receptor_residue_count": {
            "min": int(df["receptor_residue_count"].min()),
            "median": float(df["receptor_residue_count"].median()),
            "max": int(df["receptor_residue_count"].max()),
        },
        f"{prefix}_peptide_len_bin": df["peptide_len_bin"].value_counts().sort_index().to_dict(),
        f"{prefix}_receptor_size_bin": df["receptor_size_bin"].value_counts().sort_index().to_dict(),
        f"{prefix}_receptor_chain_bin": df["receptor_chain_bin"].value_counts().sort_index().to_dict(),
        f"{prefix}_stratum_count": int(df["stratum"].nunique()),
    }


def write_csv(path: Path, df: pd.DataFrame, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, list(columns)].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_readme(path: Path, sample_size: int, seed: int) -> None:
    lines = [
        "# test536 已知口袋 benchmark 小集",
        "",
        "- 来源：`data/runtime_tables/flow_infer_test536_rel.csv`（536 条）",
        f"- 目标条目数：`{sample_size}`",
        f"- 固定种子：`{seed}`",
        "- 选择原则：只使用输入属性分层，不使用任何模型指标、RMSD、DockQ 或人工挑样结果。",
        "- 分层字段：`peptide 长度分箱` × `receptor crop 残基数四分位分箱` × `receptor 链数分箱`。",
        "- 层内选择：按 `sha256(seed:complex_name)` 的稳定哈希排序取前若干条，保证可复现。",
        "",
        "## 文件说明",
        "",
        "- `flow_infer_subset_rel.csv`：可直接给 Flow / 其他对接基线跑批的相对路径输入表。",
        "- `clean_subset.csv`：与 clean test 主表对齐的小集清单。",
        "- `selection_manifest.csv`：每个样本的分层属性和稳定排序值。",
        "- `selection_summary.json`：全集与小集的分布汇总。",
        "",
        "## 使用建议",
        "",
        "- 论文正文中把它表述为 `pre-registered stratified subset` 或 `representative stratified subset`。",
        "- 主对比若资源允许仍建议保留至少一个基线的全量结果；该小集更适合 AF3 或昂贵基线。",
        "- 若后续要重建小集，请直接重跑脚本：`scripts/data_split/build_stratified_benchmark_subset.py`。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    src_df = read_csv(args.src_csv)
    clean_df = read_csv(args.clean_csv)
    feature_df = build_feature_table(src_df)
    subset_df = select_subset(feature_df, sample_size=args.sample_size, seed=args.seed)

    subset_names = set(subset_df["complex_name"].tolist())
    flow_subset = src_df[src_df["complex_name"].isin(subset_names)].sort_values("complex_name").reset_index(drop=True)
    clean_subset = clean_df[clean_df["complex_name"].isin(subset_names)].sort_values("complex_name").reset_index(drop=True)

    if len(flow_subset) != args.sample_size:
        raise RuntimeError(f"flow 子集条目数不对：expect={args.sample_size}, got={len(flow_subset)}")
    if len(clean_subset) != args.sample_size:
        raise RuntimeError(f"clean 子集条目数不对：expect={args.sample_size}, got={len(clean_subset)}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "flow_infer_subset_rel.csv", flow_subset, ["complex_name", "receptor_pdb", "peptide_pdb"])
    write_csv(out_dir / "clean_subset.csv", clean_subset, clean_subset.columns.tolist())
    write_csv(
        out_dir / "selection_manifest.csv",
        subset_df.sort_values(["stratum", "stable_rank", "complex_name"]).reset_index(drop=True),
        [
            "complex_name",
            "receptor_pdb",
            "peptide_pdb",
            "peptide_residue_count",
            "receptor_residue_count",
            "receptor_chain_count",
            "peptide_len_bin",
            "receptor_size_bin",
            "receptor_chain_bin",
            "stratum",
            "stable_rank",
        ],
    )

    summary = {
        "source_csv": str(args.src_csv),
        "clean_csv": str(args.clean_csv),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "selection_rule": "stratified_by_input_features_only",
        "stratification_fields": [
            "peptide_len_bin",
            "receptor_size_bin",
            "receptor_chain_bin",
        ],
        "full_summary": summarize_distribution(feature_df, prefix="full"),
        "subset_summary": summarize_distribution(subset_df, prefix="subset"),
    }
    (out_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(out_dir / "README.md", sample_size=args.sample_size, seed=args.seed)

    print(
        f"[ok] src={args.src_csv} total={len(src_df)} sample_size={args.sample_size} "
        f"out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
