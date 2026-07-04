#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按肽中心裁剪受体（默认 30Å），保留整残基，不劈残基。
输入 CSV 需包含列：complex_name,receptor_pdb,peptide_pdb
输出：裁剪后 PDB（receptor_crop.pdb, peptide.pdb）及新 CSV。

示例：
PYTHONPATH=$(pwd) python scripts/data_prep/crop_receptor_around_peptide.py \
  --csv path/to/input.csv \
  --out_dir data/processed_test30 \
  --out_csv path/to/output.csv \
  --radius 30
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import MDAnalysis as mda


def crop_one(rec_path: Path, pep_path: Path, out_dir: Path, radius: float):
    rec_u = mda.Universe(str(rec_path))
    pep_u = mda.Universe(str(pep_path))
    center = pep_u.atoms.positions.mean(axis=0)
    rec_atoms = rec_u.select_atoms("protein")
    if len(rec_atoms) == 0:
        sel = rec_u.atoms
    else:
        dist = np.linalg.norm(rec_atoms.positions - center, axis=1)
        mask = dist <= radius
        if mask.any():
            res_idx = np.unique(rec_atoms.atoms[mask].resindices)
            sel = rec_atoms.atoms[np.isin(rec_atoms.resindices, res_idx)]
        else:
            sel = rec_atoms
    out_dir.mkdir(parents=True, exist_ok=True)
    sel.write(str(out_dir / "receptor_crop.pdb"))
    pep_u.atoms.write(str(out_dir / "peptide.pdb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="输入 CSV，含 complex_name,receptor_pdb,peptide_pdb")
    ap.add_argument("--out_dir", required=True, help="裁剪后 PDB 的输出根目录")
    ap.add_argument("--out_csv", required=True, help="写出的新 CSV 路径")
    ap.add_argument("--radius", type=float, default=30.0, help="裁剪半径(Å)，默认 30")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rows = []
    skipped = []
    for _, r in df.iterrows():
        name = str(r.get("complex_name", "")).strip()
        rec = Path(str(r.get("receptor_pdb", ""))).expanduser()
        pep = Path(str(r.get("peptide_pdb", ""))).expanduser()
        if not name or not rec.is_file() or not pep.is_file():
            skipped.append((name, rec, pep))
            continue
        out_base = Path(args.out_dir) / name.lower()
        try:
            crop_one(rec, pep, out_base, args.radius)
            rows.append(
                {
                    "complex_name": name.lower(),
                    "receptor_pdb": str(out_base / "receptor_crop.pdb"),
                    "peptide_pdb": str(out_base / "peptide.pdb"),
                }
            )
        except Exception as e:
            skipped.append((name, rec, pep, e))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {len(rows)} rows -> {out_csv}")
    if skipped:
        print(f"[WARN] skipped {len(skipped)} entries")
        for s in skipped[:10]:
            print("  skipped:", s)


if __name__ == "__main__":
    main()
