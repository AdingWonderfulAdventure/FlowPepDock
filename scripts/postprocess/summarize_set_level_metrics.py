#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 docking 全候选 pose 的集合级指标。")
    parser.add_argument("--input_csv", type=Path, required=True, help="all_poses 结果表")
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument("--out_dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--ks", type=str, default="1,5,10", help="覆盖统计的 k 列表")
    parser.add_argument("--dockq_threshold", type=float, default=0.49, help="DockQ 成功阈值")
    parser.add_argument("--rmsd_threshold", type=float, default=2.0, help="complex RMSD 成功阈值")
    return parser.parse_args()


def pose_order_key(pose_name: str) -> int:
    match = re.search(r"(?:pose|rank)(\d+)$", str(pose_name).replace(".pdb", ""))
    if match is None:
        return 10**9
    return int(match.group(1))


def build_per_complex(df: pd.DataFrame, ks: List[int], dockq_threshold: float, rmsd_threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for complex_name, sub_df in df.groupby("complex_name", sort=True):
        sub_df = sub_df.copy()
        sub_df["pose_order"] = sub_df["pose"].astype(str).map(pose_order_key)
        sub_df = sub_df.sort_values(["pose_order", "pose"], ascending=[True, True]).reset_index(drop=True)

        top1 = sub_df.iloc[0]
        oracle_dockq = sub_df.sort_values("dockq", ascending=False).iloc[0]
        oracle_rmsd = sub_df.sort_values("complex_rmsd", ascending=True).iloc[0]

        row: Dict[str, object] = {
            "complex_name": complex_name,
            "num_poses": int(len(sub_df)),
            "top1_pose": str(top1["pose"]),
            "top1_dockq": float(top1["dockq"]),
            "top1_complex_rmsd": float(top1["complex_rmsd"]),
            "top1_peptide_ca_rmsd": float(top1["peptide_ca_rmsd"]),
            "oracle_dockq_pose": str(oracle_dockq["pose"]),
            "oracle_dockq": float(oracle_dockq["dockq"]),
            "oracle_dockq_complex_rmsd": float(oracle_dockq["complex_rmsd"]),
            "oracle_rmsd_pose": str(oracle_rmsd["pose"]),
            "oracle_rmsd": float(oracle_rmsd["complex_rmsd"]),
            "oracle_rmsd_dockq": float(oracle_rmsd["dockq"]),
        }
        for k in ks:
            topk = sub_df.head(k)
            row[f"hit@{k}_dockq"] = int((topk["dockq"] >= dockq_threshold).any())
            row[f"hit@{k}_rmsd"] = int((topk["complex_rmsd"] <= rmsd_threshold).any())
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(model_name: str, per_complex: pd.DataFrame, ks: List[int], dockq_threshold: float, rmsd_threshold: float) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "model_name": model_name,
        "num_complexes": int(len(per_complex)),
        "dockq_threshold": float(dockq_threshold),
        "rmsd_threshold": float(rmsd_threshold),
        "top1_success_dockq": float((per_complex["top1_dockq"] >= dockq_threshold).mean()),
        "oracle_success_dockq": float((per_complex["oracle_dockq"] >= dockq_threshold).mean()),
        "top1_success_rmsd": float((per_complex["top1_complex_rmsd"] <= rmsd_threshold).mean()),
        "oracle_success_rmsd": float((per_complex["oracle_rmsd"] <= rmsd_threshold).mean()),
        "oracle_dockq_mean": float(per_complex["oracle_dockq"].mean()),
        "oracle_dockq_median": float(per_complex["oracle_dockq"].median()),
        "oracle_rmsd_mean": float(per_complex["oracle_rmsd"].mean()),
        "oracle_rmsd_median": float(per_complex["oracle_rmsd"].median()),
    }
    for k in ks:
        summary[f"hit@{k}_dockq"] = float(per_complex[f"hit@{k}_dockq"].mean())
        summary[f"hit@{k}_rmsd"] = float(per_complex[f"hit@{k}_rmsd"].mean())
    return summary


def write_summary_md(summary: Dict[str, object], out_path: Path, ks: List[int]) -> None:
    lines = [
        f"# {summary['model_name']} strict-536 集合级指标汇总",
        "",
        f"- 复合物数：`{summary['num_complexes']}`",
        f"- DockQ 成功阈值：`{summary['dockq_threshold']}`",
        f"- complex RMSD 成功阈值：`{summary['rmsd_threshold']} Å`",
        "",
        "## DockQ 成功覆盖",
        "",
        f"- `Top-1`：`{summary['top1_success_dockq']:.4f}`",
    ]
    for k in ks:
        lines.append(f"- `Hit@{k}`：`{summary[f'hit@{k}_dockq']:.4f}`")
    lines.extend(
        [
            f"- `Oracle`：`{summary['oracle_success_dockq']:.4f}`",
            "",
            "## Complex RMSD 成功覆盖",
            "",
            f"- `Top-1`：`{summary['top1_success_rmsd']:.4f}`",
        ]
    )
    for k in ks:
        lines.append(f"- `Hit@{k}`：`{summary[f'hit@{k}_rmsd']:.4f}`")
    lines.extend(
        [
            f"- `Oracle`：`{summary['oracle_success_rmsd']:.4f}`",
            "",
            "## Oracle 统计",
            "",
            f"- `Oracle DockQ` 均值 / 中位数：`{summary['oracle_dockq_mean']:.4f}` / `{summary['oracle_dockq_median']:.4f}`",
            f"- `Oracle complex RMSD` 均值 / 中位数：`{summary['oracle_rmsd_mean']:.4f}` / `{summary['oracle_rmsd_median']:.4f}`",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ks = [int(token.strip()) for token in args.ks.split(",") if token.strip()]
    if not ks:
        raise ValueError("ks 不能为空")

    df = pd.read_csv(args.input_csv)
    required_columns = {"complex_name", "pose", "complex_rmsd", "peptide_ca_rmsd", "dockq"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"输入 CSV 缺少列: {missing_columns}")

    for column in ["complex_rmsd", "peptide_ca_rmsd", "dockq"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["complex_name", "pose", "complex_rmsd", "peptide_ca_rmsd", "dockq"]).copy()
    if df.empty:
        raise ValueError("输入 CSV 在数值清洗后为空")

    per_complex = build_per_complex(df, ks=ks, dockq_threshold=args.dockq_threshold, rmsd_threshold=args.rmsd_threshold)
    summary = build_summary(
        model_name=args.model_name,
        per_complex=per_complex,
        ks=ks,
        dockq_threshold=args.dockq_threshold,
        rmsd_threshold=args.rmsd_threshold,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_complex.to_csv(args.out_dir / "per_complex_set_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(args.out_dir / "set_metrics_summary.csv", index=False)
    (args.out_dir / "set_metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary_md(summary, args.out_dir / "set_metrics_summary.md", ks=ks)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
