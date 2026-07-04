#!/usr/bin/env python
"""
把 `data/processed/` 里“不符合训练/ESM 拓扑规范”的样本目录移入 `data/processed/error/`，避免后续被误用。

判定规则（可调，但默认够硬）：
- receptor.pdb / peptide.pdb 必须存在
- receptor/peptide PDB 里不能有任何 HETATM（非蛋白组分必须在 PDB 阶段就被清掉）
- features_onehot.pt 必须存在且可加载（可选维度校验）

注意：
- 默认 dry-run：只打印将要移动的列表，不做任何文件移动
- 真要动盘请加 `--apply`

用法示例：
PYTHONPATH=$(pwd) python scripts/data_qc/quarantine_bad_processed.py \
  --root data/processed \
  --apply
"""

import argparse
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

import torch


def parse_args():
    p = argparse.ArgumentParser(description="把 processed 里的坏样本移入 error/")
    p.add_argument("--root", default="data/processed", help="processed 根目录")
    p.add_argument("--error_subdir", default="error", help="坏样本子目录名（默认 error）")
    p.add_argument("--pt_name", default="features_onehot.pt", help="需要检查存在的 PT 文件名")
    p.add_argument("--rec_dim", type=int, default=114, help="期望 receptor.x dim（10+104=114）")
    p.add_argument("--pep_dim", type=int, default=105, help="期望 pep.x dim（1+104=105）")
    p.add_argument("--apply", action="store_true", help="实际执行移动（否则 dry-run）")
    return p.parse_args()


def _count_hetatm(pdb_path: Path) -> int:
    n = 0
    with pdb_path.open("r", errors="ignore") as f:
        for line in f:
            if line.startswith("HETATM"):
                n += 1
    return n


def _check_pt(pt_path: Path, rec_dim: int, pep_dim: int) -> Tuple[bool, Optional[str]]:
    try:
        data = torch.load(pt_path, map_location="cpu")
        if "receptor" not in getattr(data, "node_types", []):
            return False, "pt missing receptor"
        pep_key = "pep" if "pep" in data.node_types else ("peptide" if "peptide" in data.node_types else None)
        if pep_key is None:
            return False, "pt missing pep/peptide"
        if data["receptor"].x.shape[-1] != rec_dim:
            return False, f"receptor.x dim mismatch: {data['receptor'].x.shape[-1]} != {rec_dim}"
        if data[pep_key].x.shape[-1] != pep_dim:
            return False, f"pep.x dim mismatch: {data[pep_key].x.shape[-1]} != {pep_dim}"
        return True, None
    except Exception as e:  # noqa: B902
        return False, f"pt load failed: {e}"


def _unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    ts = time.strftime("%Y%m%d_%H%M%S")
    i = 0
    while True:
        cand = dest.parent / f"{dest.name}_{ts}_{i}"
        if not cand.exists():
            return cand
        i += 1


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root 不存在：{root}")
    error_root = root / args.error_subdir
    error_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name != args.error_subdir])
    bad = []
    for d in sample_dirs:
        rec_pdb = d / "receptor.pdb"
        pep_pdb = d / "peptide.pdb"
        pt = d / args.pt_name
        reasons = []
        if not rec_pdb.exists() or not pep_pdb.exists():
            reasons.append("missing_pdb")
        else:
            if _count_hetatm(rec_pdb) > 0:
                reasons.append("receptor_has_hetatm")
            if _count_hetatm(pep_pdb) > 0:
                reasons.append("peptide_has_hetatm")
        if not pt.exists():
            reasons.append("missing_pt")
        else:
            ok, reason = _check_pt(pt, args.rec_dim, args.pep_dim)
            if not ok:
                reasons.append(reason or "bad_pt")
        if reasons:
            bad.append((d, ";".join(reasons)))

    print(f"scan done. total={len(sample_dirs)} bad={len(bad)} apply={int(args.apply)}")
    if not bad:
        return

    for d, reason in bad:
        dest = _unique_dest(error_root / d.name)
        print(f"[BAD] {d.name}: {reason} -> {dest}")
        if args.apply:
            shutil.move(str(d), str(dest))


if __name__ == "__main__":
    main()

