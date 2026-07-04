from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 PoseCred-IPG hard-negative 续训的默认 val / strict / rec70 结果。")
    parser.add_argument("--eval_root", type=Path, required=True, help="评估根目录，内部按模型名分子目录。")
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_md", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dataframe_to_markdown_fallback(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(column) for column in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    lines = [header, separator]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    model_roots = sorted(path for path in eval_root.iterdir() if path.is_dir())
    if not model_roots:
        raise ValueError(f"{eval_root} 下没有模型评估目录")

    rows: List[Dict[str, object]] = []
    for model_root in model_roots:
        model_name = model_root.name
        default_val = load_json(model_root / "default_val.json")
        strict_test = load_json(model_root / "strict_test" / "benchmark_metrics_summary.json")
        rec70 = load_json(model_root / "rec70" / "benchmark_metrics_summary.json")

        row = {
            "model": model_name,
            "default_val_top1_success": default_val.get("val_top1_success"),
            "default_val_top5_success": default_val.get("val_top5_success"),
            "default_val_mrr": default_val.get("val_mrr"),
            "default_val_ndcg": default_val.get("val_ndcg"),
            "default_val_global_top1_success": default_val.get("val_global_top1_success"),
            "strict_receptor_top1_hit": ((strict_test.get("receptor_level") or {}).get("top1_hit")),
            "strict_receptor_top5_hit": ((strict_test.get("receptor_level") or {}).get("top5_hit")),
            "strict_receptor_mrr": ((strict_test.get("receptor_level") or {}).get("mrr")),
            "strict_pose_top1_hit": ((strict_test.get("pose_level") or {}).get("top1_hit")),
            "strict_pose_top10_hit": ((strict_test.get("pose_level") or {}).get("top10_hit")),
            "strict_pose_mrr": ((strict_test.get("pose_level") or {}).get("mrr")),
            "rec70_receptor_top1_hit": ((rec70.get("receptor_level") or {}).get("top1_hit")),
            "rec70_receptor_top5_hit": ((rec70.get("receptor_level") or {}).get("top5_hit")),
            "rec70_receptor_mrr": ((rec70.get("receptor_level") or {}).get("mrr")),
            "rec70_pose_top1_hit": ((rec70.get("pose_level") or {}).get("top1_hit")),
            "rec70_pose_top10_hit": ((rec70.get("pose_level") or {}).get("top10_hit")),
            "rec70_pose_mrr": ((rec70.get("pose_level") or {}).get("mrr")),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_csv, index=False)

    md_lines = ["# PoseCred-IPG Hard-Negative Resume Summary", ""]
    try:
        md_lines.append(summary_df.to_markdown(index=False))
    except ImportError:
        md_lines.append(dataframe_to_markdown_fallback(summary_df))
    md_lines.append("")
    md_lines.append("## 说明")
    md_lines.append("")
    md_lines.append("- `default_val_*` 来自原始 PoseCred-IPG 默认验证集。")
    md_lines.append("- `strict_*` 来自 strict536 的 source_group 去重 test split cross benchmark。")
    md_lines.append("- `rec70_*` 来自 rec70 全量外部 cross benchmark。")
    args.out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
