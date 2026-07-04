#!/usr/bin/env python
"""
从一个“带链信息”的 CSV（例如 candidate_chain_le10.csv）里筛出：
只保留当前 `data/processed/<id>/` 已存在的条目，生成一个更小的 CSV 便于重跑拆分/修复。

为什么需要这个脚本？
- 你可能有 1w+ 候选条目，但 `data/processed/` 里只落了其中一部分；
- 你要“覆盖修复 data/processed”，就应该只重跑这部分，避免无意扩充数据集或占用资源。

输入 CSV（至少包含其一）：
- complex_name
- pdb_id
- PDB编号(pdb_id)

用法示例：
PYTHONPATH=$(pwd) python scripts/data_qc/filter_csv_by_existing_processed.py \
  --csv_in data/csv_backup/candidate_chain_le10.csv \
  --processed_root data/processed \
  --csv_out data/csv_new/processed_rebuild_from_candidate_chain_le10.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional


def parse_args():
    p = argparse.ArgumentParser(description="筛选 CSV：仅保留 data/processed 已存在的条目")
    p.add_argument("--csv_in", required=True, help="输入 CSV（带链信息/或至少有 complex_name/pdb_id）")
    p.add_argument("--processed_root", default="data/processed", help="processed 目录根")
    p.add_argument("--csv_out", required=True, help="输出 CSV")
    return p.parse_args()


def _pick_id(row: Dict[str, str]) -> Optional[str]:
    for key in ("complex_name", "pdb_id", "PDB编号(pdb_id)", "PDB编号"):
        v = (row.get(key) or "").strip()
        if v:
            return v
    return None


def main():
    args = parse_args()
    processed_root = Path(args.processed_root)
    if not processed_root.exists():
        raise SystemExit(f"processed_root 不存在：{processed_root}")
    existing = {p.name.lower() for p in processed_root.iterdir() if p.is_dir()}

    csv_in = Path(args.csv_in)
    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    total = 0
    with csv_in.open("r", newline="") as fi, csv_out.open("w", newline="") as fo:
        reader = csv.DictReader(fi)
        if not reader.fieldnames:
            raise SystemExit(f"CSV 无表头：{csv_in}")
        writer = csv.DictWriter(fo, fieldnames=list(reader.fieldnames))
        writer.writeheader()
        for row in reader:
            total += 1
            pid = _pick_id(row)
            if not pid:
                skipped += 1
                continue
            if pid.lower() in existing:
                writer.writerow(row)
                kept += 1
            else:
                skipped += 1

    print(f"done. total={total} kept={kept} skipped={skipped} -> {csv_out}")


if __name__ == "__main__":
    main()

