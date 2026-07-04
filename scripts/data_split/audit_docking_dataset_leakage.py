#!/usr/bin/env python3
"""
审计 docking 数据集之间是否存在泄露。

默认对比：
- dev = train.csv + val.csv
- test = pt_available.csv

检查层级：
1. complex_name 完全重叠
2. peptide_seq 完全重叠
3. receptor_seq 完全重叠
4. (receptor_seq, peptide_seq) pair 完全重叠

输入 CSV 只要求包含：
- complex_name
- pdb_dir

脚本会自动在 `pdb_dir` 下查找：
- receptor.pdb 或 receptor_crop.pdb
- peptide.pdb

示例：
PYTHONPATH=$(pwd) python scripts/data_split/audit_docking_dataset_leakage.py \
  --dev_csv data/rebuild_isolated/rebuild_20251221_163301/11_split_train_val_9to1/train.csv \
  --dev_csv data/rebuild_isolated/rebuild_20251221_163301/11_split_train_val_9to1/val.csv \
  --test_csv data/processed_test30/pt_available.csv \
  --out_dir data/diagnostics/diag_docking_leakage_audit_current
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "SEC": "U", "PYL": "O",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit leakage between docking dev and test datasets.")
    ap.add_argument("--dev_csv", action="append", required=True, help="开发集 CSV，可重复传入多次")
    ap.add_argument("--test_csv", required=True, help="测试集 CSV")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    return ap.parse_args()


def _candidate_pdb_paths(pdb_dir: Path) -> Tuple[Path, Path]:
    receptor_candidates = [pdb_dir / "receptor.pdb", pdb_dir / "receptor_crop.pdb"]
    peptide_candidates = [pdb_dir / "peptide.pdb"]
    receptor = next((p for p in receptor_candidates if p.exists()), receptor_candidates[0])
    peptide = next((p for p in peptide_candidates if p.exists()), peptide_candidates[0])
    return receptor, peptide


def _sequence_by_chain_from_pdb(pdb_path: Path) -> Dict[str, str]:
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")
    chain_seen: Dict[str, set] = defaultdict(set)
    chain_order: List[str] = []
    chain_letters: Dict[str, List[str]] = defaultdict(list)
    with pdb_path.open("r", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or len(line) < 27:
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            chain = line[21].strip() or "_"
            resi = line[22:26].strip()
            icode = line[26].strip() or "_"
            key = (resi, icode)
            if key in chain_seen[chain]:
                continue
            chain_seen[chain].add(key)
            if chain not in chain_order:
                chain_order.append(chain)
            resn = line[17:20].strip().upper()
            chain_letters[chain].append(AA3_TO_1.get(resn, "X"))
    return {chain: "".join(chain_letters[chain]) for chain in chain_order}


def _canonical_chain_join(chain_map: Dict[str, str]) -> str:
    items = [(chain, seq) for chain, seq in chain_map.items() if seq]
    items.sort(key=lambda x: x[0])
    return "|".join(seq for _, seq in items)


def load_rows(csv_paths: Iterable[str], split_name: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pdb_dir = Path(str(row["pdb_dir"])).resolve()
                receptor_pdb, peptide_pdb = _candidate_pdb_paths(pdb_dir)
                pep_seq = _canonical_chain_join(_sequence_by_chain_from_pdb(peptide_pdb))
                rec_seq = _canonical_chain_join(_sequence_by_chain_from_pdb(receptor_pdb))
                rows.append(
                    {
                        "split": split_name,
                        "source_csv": str(Path(csv_path).resolve()),
                        "complex_name": str(row["complex_name"]),
                        "pdb_dir": str(pdb_dir),
                        "receptor_pdb": str(receptor_pdb.resolve()),
                        "peptide_pdb": str(peptide_pdb.resolve()),
                        "peptide_seq": pep_seq,
                        "receptor_seq": rec_seq,
                        "pair_key": f"{rec_seq}__{pep_seq}",
                    }
                )
    return rows


def _build_index(rows: List[Dict[str, str]], key: str) -> Dict[str, List[Dict[str, str]]]:
    index: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        index[row[key]].append(row)
    return index


def _collect_overlap(
    dev_rows: List[Dict[str, str]],
    test_rows: List[Dict[str, str]],
    key: str,
) -> List[Dict[str, str]]:
    dev_index = _build_index(dev_rows, key)
    test_index = _build_index(test_rows, key)
    overlaps = sorted(set(dev_index.keys()) & set(test_index.keys()))
    out: List[Dict[str, str]] = []
    for item in overlaps:
        for d in dev_index[item]:
            for t in test_index[item]:
                out.append(
                    {
                        "overlap_type": key,
                        "overlap_value": item,
                        "dev_complex_name": d["complex_name"],
                        "dev_source_csv": d["source_csv"],
                        "dev_pdb_dir": d["pdb_dir"],
                        "test_complex_name": t["complex_name"],
                        "test_source_csv": t["source_csv"],
                        "test_pdb_dir": t["pdb_dir"],
                    }
                )
    return out


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "overlap_type",
                    "overlap_value",
                    "dev_complex_name",
                    "dev_source_csv",
                    "dev_pdb_dir",
                    "test_complex_name",
                    "test_source_csv",
                    "test_pdb_dir",
                ]
            )
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_rows = load_rows(args.dev_csv, "dev")
    test_rows = load_rows([args.test_csv], "test")

    overlap_specs = {
        "complex_name": _collect_overlap(dev_rows, test_rows, "complex_name"),
        "peptide_seq": _collect_overlap(dev_rows, test_rows, "peptide_seq"),
        "receptor_seq": _collect_overlap(dev_rows, test_rows, "receptor_seq"),
        "pair_key": _collect_overlap(dev_rows, test_rows, "pair_key"),
    }

    for name, rows in overlap_specs.items():
        _write_csv(out_dir / f"overlap_{name}.csv", rows)

    summary = {
        "dev_rows": len(dev_rows),
        "test_rows": len(test_rows),
        "unique_dev_complex_name": len({r["complex_name"] for r in dev_rows}),
        "unique_test_complex_name": len({r["complex_name"] for r in test_rows}),
        "unique_dev_peptide_seq": len({r["peptide_seq"] for r in dev_rows}),
        "unique_test_peptide_seq": len({r["peptide_seq"] for r in test_rows}),
        "unique_dev_receptor_seq": len({r["receptor_seq"] for r in dev_rows}),
        "unique_test_receptor_seq": len({r["receptor_seq"] for r in test_rows}),
        "overlap_complex_name": len({r["overlap_value"] for r in overlap_specs["complex_name"]}),
        "overlap_peptide_seq": len({r["overlap_value"] for r in overlap_specs["peptide_seq"]}),
        "overlap_receptor_seq": len({r["overlap_value"] for r in overlap_specs["receptor_seq"]}),
        "overlap_pair_key": len({r["overlap_value"] for r in overlap_specs["pair_key"]}),
        "notes": {
            "complex_name": "最严格的样本 ID 重叠",
            "peptide_seq": "测试肽序列是否在开发集中出现过",
            "receptor_seq": "测试受体序列是否在开发集中出现过",
            "pair_key": "测试的精确受体序列-肽序列配对是否在开发集中出现过",
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
