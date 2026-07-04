#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总 scripts/eval_rmsd_from_preds.py 输出的 metrics.csv。

输出：
- summary.json
- summary.md
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def _safe_float(val):
    return None if pd.isna(val) else float(val)


def _load_expected_complexes(input_csv: Path):
    df = pd.read_csv(input_csv)
    if "complex_name" in df.columns:
        names = df["complex_name"].astype(str).str.strip().str.lower()
    elif "pdb_id" in df.columns:
        names = df["pdb_id"].astype(str).str.strip().str.lower()
    else:
        raise ValueError(f"输入 CSV 缺少 complex_name / pdb_id 列：{input_csv}")
    return [name for name in names.tolist() if name]


def build_summary(metrics_csv: Path, input_csv: Optional[Path] = None):
    df = pd.read_csv(metrics_csv)
    for col in ["complex_rmsd", "peptide_ca_rmsd", "dockq", "fnat", "irmsd", "lrmsd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    expected_names = None
    skipped_names = []
    expected_total = None
    if input_csv is not None:
        expected_names = _load_expected_complexes(input_csv)
        expected_total = len(expected_names)
        observed = set(df["complex_name"].astype(str).str.strip().str.lower().tolist())
        skipped_names = [name for name in expected_names if name not in observed]

    summary = {
        "expected_total": expected_total,
        "evaluated_total": int(len(df)),
        "skipped_total": int(len(skipped_names)) if expected_names is not None else None,
        "skipped_complexes": skipped_names if expected_names is not None else [],
        "success_peptide_ca_rmsd_le_2A": int((df["peptide_ca_rmsd"] <= 2.0).sum()),
        "success_peptide_ca_rmsd_le_4A": int((df["peptide_ca_rmsd"] <= 4.0).sum()),
        "success_complex_rmsd_le_2A": int((df["complex_rmsd"] <= 2.0).sum()),
        "success_complex_rmsd_le_4A": int((df["complex_rmsd"] <= 4.0).sum()),
        "success_rate_peptide_ca_rmsd_le_2A": float((df["peptide_ca_rmsd"] <= 2.0).mean()),
        "success_rate_peptide_ca_rmsd_le_4A": float((df["peptide_ca_rmsd"] <= 4.0).mean()),
        "success_rate_complex_rmsd_le_2A": float((df["complex_rmsd"] <= 2.0).mean()),
        "success_rate_complex_rmsd_le_4A": float((df["complex_rmsd"] <= 4.0).mean()),
        "median_peptide_ca_rmsd": _safe_float(df["peptide_ca_rmsd"].median()),
        "mean_peptide_ca_rmsd": _safe_float(df["peptide_ca_rmsd"].mean()),
        "median_complex_rmsd": _safe_float(df["complex_rmsd"].median()),
        "mean_complex_rmsd": _safe_float(df["complex_rmsd"].mean()),
        "median_dockq": _safe_float(df["dockq"].median()) if "dockq" in df.columns else None,
        "mean_dockq": _safe_float(df["dockq"].mean()) if "dockq" in df.columns else None,
        "median_fnat": _safe_float(df["fnat"].median()) if "fnat" in df.columns else None,
        "mean_fnat": _safe_float(df["fnat"].mean()) if "fnat" in df.columns else None,
        "median_irmsd": _safe_float(df["irmsd"].median()) if "irmsd" in df.columns else None,
        "mean_irmsd": _safe_float(df["irmsd"].mean()) if "irmsd" in df.columns else None,
        "median_lrmsd": _safe_float(df["lrmsd"].median()) if "lrmsd" in df.columns else None,
        "mean_lrmsd": _safe_float(df["lrmsd"].mean()) if "lrmsd" in df.columns else None,
        "best5_by_dockq": df.sort_values("dockq", ascending=False)[
            ["complex_name", "pose", "dockq", "peptide_ca_rmsd", "complex_rmsd"]
        ].head(5).to_dict(orient="records") if "dockq" in df.columns else [],
        "worst5_by_peptide_ca_rmsd": df.sort_values("peptide_ca_rmsd", ascending=False)[
            ["complex_name", "pose", "peptide_ca_rmsd", "dockq", "complex_rmsd"]
        ].head(5).to_dict(orient="records"),
    }
    return summary


def write_summary_md(summary: dict, title: str, metrics_csv: Path, out_md: Path):
    expected_total = summary.get("expected_total")
    skipped_total = summary.get("skipped_total")
    skipped_complexes = summary.get("skipped_complexes") or []
    lines = [
        f"# {title}",
        "",
        f"- 逐项目评测文件：`{metrics_csv}`",
    ]
    if expected_total is not None:
        lines.extend(
            [
                f"- 期望复合物数：`{expected_total}`",
                f"- 成功评测数：`{summary['evaluated_total']}`",
                f"- 跳过数：`{skipped_total}`",
                f"- 跳过项目：`{', '.join(skipped_complexes) if skipped_complexes else '无'}`",
            ]
        )
    else:
        lines.append(f"- 成功评测数：`{summary['evaluated_total']}`")
    lines.extend(
        [
            "",
            "## 主指标",
            "",
            f"- `peptide_ca_rmsd <= 2Å`：`{summary['success_peptide_ca_rmsd_le_2A']}` / `{summary['evaluated_total']}` (`{summary['success_rate_peptide_ca_rmsd_le_2A']:.4f}`)",
            f"- `peptide_ca_rmsd <= 4Å`：`{summary['success_peptide_ca_rmsd_le_4A']}` / `{summary['evaluated_total']}` (`{summary['success_rate_peptide_ca_rmsd_le_4A']:.4f}`)",
            f"- `complex_rmsd <= 2Å`：`{summary['success_complex_rmsd_le_2A']}` / `{summary['evaluated_total']}` (`{summary['success_rate_complex_rmsd_le_2A']:.4f}`)",
            f"- `complex_rmsd <= 4Å`：`{summary['success_complex_rmsd_le_4A']}` / `{summary['evaluated_total']}` (`{summary['success_rate_complex_rmsd_le_4A']:.4f}`)",
            f"- `peptide_ca_rmsd` 中位数 / 均值：`{summary['median_peptide_ca_rmsd']:.4f}` / `{summary['mean_peptide_ca_rmsd']:.4f}`",
            f"- `complex_rmsd` 中位数 / 均值：`{summary['median_complex_rmsd']:.4f}` / `{summary['mean_complex_rmsd']:.4f}`",
        ]
    )
    if summary.get("median_dockq") is not None:
        lines.extend(
            [
                f"- `DockQ` 中位数 / 均值：`{summary['median_dockq']:.4f}` / `{summary['mean_dockq']:.4f}`",
                f"- `Fnat` 中位数 / 均值：`{summary['median_fnat']:.4f}` / `{summary['mean_fnat']:.4f}`",
                f"- `iRMSD` 中位数 / 均值：`{summary['median_irmsd']:.4f}` / `{summary['mean_irmsd']:.4f}`",
                f"- `LRMSD` 中位数 / 均值：`{summary['median_lrmsd']:.4f}` / `{summary['mean_lrmsd']:.4f}`",
            ]
        )
    lines.extend(["", "## DockQ 最优 5 个项目", ""])
    for row in summary.get("best5_by_dockq", []):
        lines.append(
            f"- `{row['complex_name']}` | `{row['pose']}` | DockQ=`{row['dockq']:.4f}` | "
            f"pepCA=`{row['peptide_ca_rmsd']:.4f}` | complex=`{row['complex_rmsd']:.4f}`"
        )
    lines.extend(["", "## peptide CA RMSD 最差 5 个项目", ""])
    for row in summary.get("worst5_by_peptide_ca_rmsd", []):
        lines.append(
            f"- `{row['complex_name']}` | `{row['pose']}` | pepCA=`{row['peptide_ca_rmsd']:.4f}` | "
            f"DockQ=`{row['dockq']:.4f}` | complex=`{row['complex_rmsd']:.4f}`"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True, help="metrics_rmsd_dockq.csv")
    ap.add_argument("--input_csv", default=None, help="原始输入 CSV（可选，用于识别 skipped 样本）")
    ap.add_argument("--output_json", required=True, help="summary.json")
    ap.add_argument("--output_md", required=True, help="summary.md")
    ap.add_argument("--title", default="评测汇总", help="Markdown 标题")
    args = ap.parse_args()

    metrics_csv = Path(args.metrics_csv)
    input_csv = Path(args.input_csv) if args.input_csv else None
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(metrics_csv, input_csv)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary_md(summary, args.title, metrics_csv, output_md)
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
