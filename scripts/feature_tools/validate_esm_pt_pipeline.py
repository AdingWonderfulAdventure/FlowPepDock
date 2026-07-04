#!/usr/bin/env python
"""
校验 onehot→ESM PT 全流程是否对齐：
- features_onehot.pt / esm_embedding.pt / features_esm.pt 是否存在
- 节点数是否一致（onehot vs esm_embedding vs features_esm）
- ESM embedding 维度是否为 1280
- onehot 与 features_esm 的前缀是否一致（只换尾巴）

输出 report_csv（逐条结果）+ summary（汇总）。
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="校验 onehot→ESM PT 流程是否对齐")
    p.add_argument("--csv", required=True, help="含 complex_name 的 CSV；可选 pdb_dir / receptor_pdb")
    p.add_argument("--out_root", default="data/processed", help="默认样本目录根：<out_root>/<complex_name>/")
    p.add_argument("--onehot_name", default="features_onehot.pt", help="onehot PT 文件名")
    p.add_argument("--esm_emb_name", default="esm_embedding.pt", help="ESM embedding 文件名")
    p.add_argument("--esm_name", default="features_esm.pt", help="ESM PT 文件名")
    p.add_argument("--tail_onehot", type=int, default=104, help="onehot 尾部维度")
    p.add_argument("--tail_esm", type=int, default=1280, help="ESM 尾部维度")
    p.add_argument("--report_csv", required=True, help="逐条校验结果输出 CSV")
    p.add_argument("--summary", required=True, help="汇总输出文本")
    return p.parse_args()


def _resolve_base(row: Dict[str, str], out_root: str) -> Path:
    name = (row.get("complex_name") or "").strip().lower()
    if not name:
        raise ValueError("missing complex_name")
    pdb_dir = (row.get("pdb_dir") or "").strip()
    if pdb_dir:
        return Path(pdb_dir)
    rec_pdb = (row.get("receptor_pdb") or "").strip()
    if rec_pdb:
        return Path(rec_pdb).resolve().parent
    return Path(out_root) / name


def _split_feat(x: torch.Tensor, tail_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape[1] < tail_dim:
        raise ValueError(f"feature dim {x.shape[1]} < tail_dim {tail_dim}")
    return x[:, :-tail_dim], x[:, -tail_dim:]


def _check_pair(onehot_path: Path, emb_path: Path, esm_path: Path, tail_onehot: int, tail_esm: int):
    reasons: List[str] = []
    prefix_rec_equal = ""
    prefix_pep_equal = ""
    rec_nodes = ""
    pep_nodes = ""
    rec_emb_len = ""
    pep_emb_len = ""

    if not onehot_path.exists():
        reasons.append("missing_onehot")
    if not emb_path.exists():
        reasons.append("missing_esm_embedding")
    if not esm_path.exists():
        reasons.append("missing_features_esm")
    if reasons:
        return reasons, prefix_rec_equal, prefix_pep_equal, rec_nodes, pep_nodes, rec_emb_len, pep_emb_len

    onehot = torch.load(onehot_path, map_location="cpu")
    emb = torch.load(emb_path, map_location="cpu")
    esm = torch.load(esm_path, map_location="cpu")

    pep_key = "pep" if "pep" in onehot.node_types else ("peptide" if "peptide" in onehot.node_types else None)
    if pep_key is None or "receptor" not in onehot.node_types:
        reasons.append("missing_node_type")
        return reasons, prefix_rec_equal, prefix_pep_equal, rec_nodes, pep_nodes, rec_emb_len, pep_emb_len

    rec_nodes = onehot["receptor"].x.shape[0]
    pep_nodes = onehot[pep_key].x.shape[0]

    rec_emb = emb.get("rec_emb")
    pep_emb = emb.get("pep_emb")
    if rec_emb is None or pep_emb is None:
        reasons.append("missing_rec_or_pep_emb")
        return reasons, prefix_rec_equal, prefix_pep_equal, rec_nodes, pep_nodes, rec_emb_len, pep_emb_len

    rec_emb_len = rec_emb.shape[0]
    pep_emb_len = pep_emb.shape[0]

    if rec_emb.shape[1] != tail_esm or pep_emb.shape[1] != tail_esm:
        reasons.append("emb_dim_mismatch")
    if rec_emb_len != rec_nodes:
        reasons.append("rec_len_mismatch")
    if pep_emb_len != pep_nodes:
        reasons.append("pep_len_mismatch")

    rec_x_esm = esm["receptor"].x
    pep_x_esm = esm[pep_key].x
    if rec_x_esm.shape[0] != rec_nodes:
        reasons.append("rec_nodes_mismatch_esm")
    if pep_x_esm.shape[0] != pep_nodes:
        reasons.append("pep_nodes_mismatch_esm")

    try:
        rec_prefix_onehot, _ = _split_feat(onehot["receptor"].x, tail_onehot)
        pep_prefix_onehot, _ = _split_feat(onehot[pep_key].x, tail_onehot)
        rec_prefix_esm, _ = _split_feat(rec_x_esm, tail_esm)
        pep_prefix_esm, _ = _split_feat(pep_x_esm, tail_esm)
        prefix_rec_equal = str(bool(torch.allclose(rec_prefix_onehot, rec_prefix_esm, atol=1e-6)))
        prefix_pep_equal = str(bool(torch.allclose(pep_prefix_onehot, pep_prefix_esm, atol=1e-6)))
        if prefix_rec_equal != "True":
            reasons.append("prefix_mismatch_rec")
        if prefix_pep_equal != "True":
            reasons.append("prefix_mismatch_pep")
    except Exception as e:
        reasons.append(f"prefix_check_error:{str(e).replace(' ', '_')}")

    return reasons, prefix_rec_equal, prefix_pep_equal, rec_nodes, pep_nodes, rec_emb_len, pep_emb_len


def main() -> None:
    args = parse_args()
    with open(args.csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("complex_name") or "").strip()]

    stats = Counter()
    out_path = Path(args.report_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "complex_name",
                "status",
                "reasons",
                "prefix_rec_equal",
                "prefix_pep_equal",
                "rec_nodes",
                "pep_nodes",
                "rec_emb_len",
                "pep_emb_len",
            ],
        )
        w.writeheader()
        for row in rows:
            name = row["complex_name"].strip().lower()
            base = _resolve_base(row, args.out_root)
            onehot_path = base / args.onehot_name
            emb_path = base / args.esm_emb_name
            esm_path = base / args.esm_name
            try:
                reasons, pre_rec, pre_pep, rec_nodes, pep_nodes, rec_emb_len, pep_emb_len = _check_pair(
                    onehot_path,
                    emb_path,
                    esm_path,
                    args.tail_onehot,
                    args.tail_esm,
                )
            except Exception as e:
                reasons = [f"error:{str(e).replace(' ', '_')[:120]}"]
                pre_rec = ""
                pre_pep = ""
                rec_nodes = ""
                pep_nodes = ""
                rec_emb_len = ""
                pep_emb_len = ""

            status = "ok" if not reasons else "fail"
            stats[status] += 1
            if reasons:
                for r in reasons:
                    stats[f"reason:{r}"] += 1
            w.writerow(
                {
                    "complex_name": name,
                    "status": status,
                    "reasons": ";".join(reasons),
                    "prefix_rec_equal": pre_rec,
                    "prefix_pep_equal": pre_pep,
                    "rec_nodes": rec_nodes,
                    "pep_nodes": pep_nodes,
                    "rec_emb_len": rec_emb_len,
                    "pep_emb_len": pep_emb_len,
                }
            )

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        f.write(f"total={len(rows)}\n")
        f.write(f"ok={stats.get('ok', 0)}\n")
        f.write(f"fail={stats.get('fail', 0)}\n")
        for k, v in sorted(stats.items()):
            if k.startswith("reason:"):
                f.write(f"{k.replace('reason:', '')}={v}\n")

    print(f"[validate_esm_pt] total={len(rows)} ok={stats.get('ok', 0)} fail={stats.get('fail', 0)}")


if __name__ == "__main__":
    main()
