#!/usr/bin/env python3
"""
基于现有 overlap 审计结果，生成不同口径的 clean test CSV。

输入：
- 原始 test CSV（至少包含 complex_name,pdb_dir）
- overlap audit 目录（包含 overlap_complex_name.csv / overlap_pair_key.csv /
  overlap_peptide_seq.csv / overlap_receptor_seq.csv）

输出：
- strict exact clean：仅移除 complex_name / exact pair 泄露
- unseen peptide clean：移除 peptide_seq 泄露
- fully clean：移除任一 overlap（complex/pair/peptide/receptor）

示例：
PYTHONPATH=$(pwd) python scripts/data_split/build_clean_test_csvs_from_audit.py \
  --test_csv data/processed_test30/pt_available.csv \
  --audit_dir data/diagnostics/diag_docking_leakage_audit_current \
  --out_dir data/processed_test30
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean test CSVs from leakage audit outputs.")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--audit_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def read_test_overlap_set(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {str(row["test_complex_name"]).lower() for row in reader}


def load_rows(test_csv: Path) -> List[Dict[str, str]]:
    with test_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_rows(rows: List[Dict[str, str]], blocked: Set[str]) -> List[Dict[str, str]]:
    return [row for row in rows if str(row["complex_name"]).lower() not in blocked]


def main() -> None:
    args = parse_args()
    test_csv = Path(args.test_csv)
    audit_dir = Path(args.audit_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(test_csv)
    fieldnames = list(rows[0].keys()) if rows else ["complex_name", "pdb_dir"]

    overlap_complex = read_test_overlap_set(audit_dir / "overlap_complex_name.csv")
    overlap_pair = read_test_overlap_set(audit_dir / "overlap_pair_key.csv")
    overlap_peptide = read_test_overlap_set(audit_dir / "overlap_peptide_seq.csv")
    overlap_receptor = read_test_overlap_set(audit_dir / "overlap_receptor_seq.csv")

    strict_exact = overlap_complex | overlap_pair
    fully_clean = overlap_complex | overlap_pair | overlap_peptide | overlap_receptor

    outputs = {
        "pt_available.strict_exact_clean.csv": filter_rows(rows, strict_exact),
        "pt_available.unseen_peptide_clean.csv": filter_rows(rows, overlap_peptide),
        "pt_available.fully_clean.csv": filter_rows(rows, fully_clean),
    }

    for filename, out_rows in outputs.items():
        write_rows(out_dir / filename, out_rows, fieldnames)

    summary = {
        "source_test_csv": str(test_csv.resolve()),
        "source_audit_dir": str(audit_dir.resolve()),
        "input_rows": len(rows),
        "strict_exact_removed": len(strict_exact),
        "strict_exact_kept": len(outputs["pt_available.strict_exact_clean.csv"]),
        "unseen_peptide_removed": len(overlap_peptide),
        "unseen_peptide_kept": len(outputs["pt_available.unseen_peptide_clean.csv"]),
        "fully_clean_removed": len(fully_clean),
        "fully_clean_kept": len(outputs["pt_available.fully_clean.csv"]),
    }
    (out_dir / "pt_available.clean_splits.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
