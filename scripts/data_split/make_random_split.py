#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从一个“路径齐全”的CSV里随机抽取 train/val 子集（互斥），用于中等规模 sanity。

输入CSV：至少包含 complex_name；建议同时包含 receptor_pdb/peptide_pdb（方便后续 prepare_training_data.py 直接用）。
输出CSV：保持原行不变（只是抽样），并保证 train/val 不重叠。

示例：
python -u "scripts/data_split/make_random_split.py" \
  --src_csv "data/csv_new/step06_train_onehot_no_test_available.csv" \
  --train_out "data/csv_new/mid_train500_from_step06_available_seed42.csv" \
  --val_out "data/csv_new/mid_val100_from_step06_available_seed43.csv" \
  --train_n 500 \
  --val_n 100 \
  --seed_train 42 \
  --seed_val 43
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple


def _read_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "complex_name" not in fieldnames:
            raise ValueError(f"CSV缺少列 complex_name：{path}")
        rows = []
        for row in reader:
            name = (row.get("complex_name") or "").strip().lower()
            if not name:
                continue
            row = dict(row)
            row["complex_name"] = name
            rows.append(row)
    return fieldnames, rows


def _sample_unique(rows: List[Dict[str, str]], n: int, seed: int) -> List[Dict[str, str]]:
    if n <= 0:
        return []
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    picked = [rows[i] for i in indices[:n]]
    return picked


def _write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    p = argparse.ArgumentParser(description="随机抽取 train/val 子集（互斥）")
    p.add_argument("--src_csv", required=True, help="源CSV（至少 complex_name）")
    p.add_argument("--train_out", required=True, help="输出 train CSV 路径")
    p.add_argument("--val_out", required=True, help="输出 val CSV 路径")
    p.add_argument("--train_n", type=int, required=True, help="train 条目数")
    p.add_argument("--val_n", type=int, required=True, help="val 条目数")
    p.add_argument("--seed_train", type=int, default=42, help="train 抽样随机种子")
    p.add_argument("--seed_val", type=int, default=43, help="val 抽样随机种子（在剩余集合上抽）")
    args = p.parse_args()

    src = Path(args.src_csv)
    fieldnames, rows = _read_rows(src)
    if len(rows) < args.train_n + args.val_n:
        raise ValueError(f"源CSV行数不足：rows={len(rows)} need={args.train_n + args.val_n}")

    train = _sample_unique(rows, args.train_n, args.seed_train)
    used = {r["complex_name"] for r in train}
    remain = [r for r in rows if r["complex_name"] not in used]
    val = _sample_unique(remain, args.val_n, args.seed_val)

    # 最终防重
    train_names = {r["complex_name"] for r in train}
    val_names = {r["complex_name"] for r in val}
    overlap = train_names & val_names
    if overlap:
        raise RuntimeError(f"train/val 抽样出现重叠：{sorted(list(overlap))[:10]}")

    _write_rows(Path(args.train_out), fieldnames, train)
    _write_rows(Path(args.val_out), fieldnames, val)
    print(f"[ok] src={src} rows={len(rows)} train={len(train)} val={len(val)}")


if __name__ == "__main__":
    main()

