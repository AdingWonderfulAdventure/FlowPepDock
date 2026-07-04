#!/usr/bin/env python
"""
自动从一堆PDB里挑受体链和肽链，生成 prepare_training_data.py 可用的CSV。

规则：
- 肽链：残基数在[min_pep, max_pep]范围内的最短链（默认3~30）。
- 受体链：残基数>=50且>=肽链长度3倍的最长链，且不能和肽链相同。
- 找不到符合条件的就跳过该PDB。

输出的CSV列：complex_name, complex_pdb, receptor_chains, peptide_chains, receptor_pdb, peptide_pdb。
其中receptor_pdb/peptide_pdb留空，后续prepare_training_data.py会按链拆分。

用法示例：
    python scripts/data_prep/auto_chain_picker.py \
      --pdb_dir /path/to/pdbs \
      --output_csv scripts/auto_chains.csv
"""

import argparse
import csv
import os
from typing import List, Tuple

from Bio.PDB import PDBParser
import numpy as np


def chain_len(chain) -> int:
    """统计一条链的标准残基数（忽略水/杂原子）。"""
    return sum(1 for res in chain if res.id[0] == " ")

def _ca_coords(chain) -> np.ndarray:
    coords = []
    for res in chain:
        if res.id[0] != " ":
            continue
        if "CA" not in res:
            continue
        coords.append(res["CA"].get_coord())
    if not coords:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _min_ca_dist(chain_a, chain_b) -> float:
    a = _ca_coords(chain_a)
    b = _ca_coords(chain_b)
    if a.size == 0 or b.size == 0:
        return float("inf")
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
    return float(np.sqrt(d2.min()))


def pick_chains(structure, min_pep: int, max_pep: int) -> Tuple[str, str]:
    """从结构中选出受体链ID和肽链ID，不合规则返回(None, None)。"""
    chains: List[Tuple[str, int, object]] = []
    for chain in structure:
        cid = chain.id.strip()
        if not cid:
            continue
        l = chain_len(chain)
        if l <= 0:
            continue
        chains.append((cid, l, chain))

    if not chains:
        return None, None

    # 先挑受体：最长的“蛋白链”（>=50），避免多个肽链时先选错肽导致后续受体条件错误。
    receptor_candidates = [(cid, l, ch) for cid, l, ch in chains if l >= 50]
    if not receptor_candidates:
        return None, None
    rec_id, rec_len, rec_chain = max(receptor_candidates, key=lambda x: x[1])

    # 再挑肽：长度在[min_pep,max_pep]，且受体长度需 >= 3*pep_len；若候选多个，选“离受体最近”的那条（解决重复配体/多拷贝肽）。
    peptide_candidates = [(cid, l, ch) for cid, l, ch in chains if min_pep <= l <= max_pep and cid != rec_id and rec_len >= 3 * l]
    if not peptide_candidates:
        return None, None
    # 先按距离，再按长度（更短优先），最后按链ID稳定排序
    peptide_candidates.sort(key=lambda x: (_min_ca_dist(rec_chain, x[2]), x[1], x[0]))
    pep_id, pep_len, _ = peptide_candidates[0]

    receptors = [
        (cid, l)
        for cid, l, _ in chains
        if cid != pep_id and cid == rec_id
    ]
    if not receptors:
        return None, None
    rec_id, _ = max(receptors, key=lambda x: x[1])
    return rec_id, pep_id


def main():
    parser = argparse.ArgumentParser(description="自动选择受体/肽链并生成CSV")
    parser.add_argument("--pdb_dir", required=True, help="包含PDB文件的目录")
    parser.add_argument("--output_csv", required=True, help="输出CSV路径")
    parser.add_argument("--min_pep", type=int, default=3, help="肽链最小残基数")
    parser.add_argument("--max_pep", type=int, default=30, help="肽链最大残基数")
    args = parser.parse_args()

    parser_bio = PDBParser(QUIET=True)
    rows = []

    for fname in os.listdir(args.pdb_dir):
        if not fname.lower().endswith(".pdb"):
            continue
        path = os.path.join(args.pdb_dir, fname)
        try:
            structure = parser_bio.get_structure("x", path)[0]
        except Exception as exc:
            print(f"[WARN] 跳过 {fname}: 解析失败 {exc}")
            continue

        rec_id, pep_id = pick_chains(structure, args.min_pep, args.max_pep)
        if rec_id is None or pep_id is None:
            continue

        rows.append(
            {
                "complex_name": os.path.splitext(fname)[0],
                "complex_pdb": path,
                "receptor_chains": rec_id,
                "peptide_chains": pep_id,
                "receptor_pdb": "",
                "peptide_pdb": "",
            }
        )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "complex_name",
                "complex_pdb",
                "receptor_chains",
                "peptide_chains",
                "receptor_pdb",
                "peptide_pdb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"收集到 {len(rows)} 条，写入 {args.output_csv}")


if __name__ == "__main__":
    main()
