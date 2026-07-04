#!/usr/bin/env python
"""
快速用 ESM 嵌入替换 onehot 嵌入，生成完整的 features_esm.pt。

用法示例：
PYTHONPATH=$(pwd) python scripts/feature_tools/replace_onehot_with_esm.py \
  --csv data/csv_backup/train_1000.csv \
  --out_root data/processed \
  --suffix_out features_esm.pt

要求：每个条目目录下已有
  - features_onehot.pt  （完整图、包含几何）
  - features_esm.pt     （完整图、包含 ESM 尾部嵌入）
脚本会读取二者，保留几何和前缀特征，替换尾部嵌入（rec: 最后104维 → 1280维；pep: 最后104维 → 1280维），
写出新的 features_esm.pt（或自定义后缀）。
"""

import argparse
import csv
import os
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description="用 ESM 嵌入替换 onehot，生成完整 features_esm.pt")
    p.add_argument("--csv", required=True, help="含 complex_name 的 CSV；默认在 out_root/<name> 查找 pt")
    p.add_argument("--out_root", default="data/processed", help="pdb 目录根路径")
    p.add_argument("--onehot_name", default="features_onehot.pt", help="onehot pt 文件名")
    p.add_argument("--esm_name", default="features_esm.pt", help="已有 esm pt 文件名（用于取尾部嵌入）")
    p.add_argument("--suffix_out", default="features_esm.pt", help="输出文件名")
    return p.parse_args()


def replace_tail(onehot, esm, tail_onehot=104, tail_esm=1280):
    def _merge(x_onehot, x_esm):
        # 前缀长度 = 全长 - tail_onehot
        prefix = x_onehot[..., :-tail_onehot]
        tail = x_esm[..., -tail_esm:]
        return torch.cat([prefix, tail], dim=-1)

    onehot["receptor"].x = _merge(onehot["receptor"].x, esm["receptor"].x)
    onehot["peptide"].x = _merge(onehot["peptide"].x, esm["peptide"].x)
    return onehot


def main():
    args = parse_args()
    with open(args.csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("complex_name")]

    for row in rows:
        name = row["complex_name"].strip().lower()
        base = Path(args.out_root) / name
        onehot_path = base / args.onehot_name
        esm_path = base / args.esm_name
        out_path = base / args.suffix_out
        if not onehot_path.exists() or not esm_path.exists():
            print(f"[SKIP] {name}: missing {onehot_path} or {esm_path}")
            continue
        onehot = torch.load(onehot_path, map_location="cpu")
        esm = torch.load(esm_path, map_location="cpu")
        merged = replace_tail(onehot, esm)
        torch.save(merged, out_path)
        print(f"[OK] {name} -> {out_path}")


if __name__ == "__main__":
    main()
