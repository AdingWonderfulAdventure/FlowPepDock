#!/usr/bin/env python
"""
查看/巡检 HeteroData .pt 的工具。

功能：
- 单文件模式：打印顶层字段或完整结构。
- 批量巡检模式：遍历 root 目录下的 features_{embedding}.pt，检查缺失/零字节/加载失败/NaN/维度错误。

批量默认期望：
- 受体 onehot 版 x 维度 = 114（1 索引 + 9 几何 + 104 onehot）。
- 肽 onehot 版 x 维度 = 105（1 索引 + 104 onehot）。
 - 受体 esm 版 x 维度 = 1290（1 索引 + 9 几何 + 1280 ESM）。
 - 肽 esm 版 x 维度 = 1281（1 索引 + 1280 ESM）。
可通过 --receptor_dim/--peptide_dim 覆盖。
"""
import argparse
import os
from pprint import pprint
from typing import Dict, List, Tuple, Optional

import torch

ONEHOT_REC_DIM = 114
ONEHOT_PEP_DIM = 105
ESM_REC_DIM = 10 + 1280
ESM_PEP_DIM = 1 + 1280


def _print_single(pt_path: str, only_keys: bool) -> None:
    data = torch.load(pt_path, map_location="cpu")
    if only_keys:
        print("顶层字段:")
        if hasattr(data, "keys"):
            for k in data.keys():
                print("-", k)
        else:
            print(list(vars(data).keys()))
        return

    print(f"文件: {pt_path}")
    if hasattr(data, "name"):
        print("name:", data.name)
    if hasattr(data, "num_nodes"):
        print("num_nodes:", data.num_nodes)
    if hasattr(data, "node_types"):
        print("\n节点/边类型:")
        for ntype in data.node_types:
            x = data[ntype].x if "x" in data[ntype] else None
            pos = data[ntype].pos if "pos" in data[ntype] else None
            print(
                f"  [{ntype}] x: {tuple(x.shape) if x is not None else None}, "
                f"pos: {tuple(pos.shape) if pos is not None else None}"
            )
        for etype in data.edge_types:
            ei = data[etype].edge_index if "edge_index" in data[etype] else None
            es = data[etype].edge_attr if "edge_attr" in data[etype] else data[etype].get("edge_s", None)
            print(
                f"  edge {etype} edge_index: {tuple(ei.shape) if ei is not None else None}, "
                f"edge_attr: {tuple(es.shape) if es is not None else None}"
            )

    print("\n完整字段：")
    pprint(data)


def _check_tensor(name: str, t: torch.Tensor, expect_dim: Optional[int] = None) -> Tuple[bool, str]:
    if t is None:
        return False, f"{name} missing"
    if not torch.isfinite(t).all():
        return False, f"{name} has NaN/Inf"
    if expect_dim is not None and t.shape[-1] != expect_dim:
        return False, f"{name} last dim {t.shape[-1]} != {expect_dim}"
    return True, ""


def _check_entry(
    entry: str,
    root: str,
    embedding: str,
    receptor_dim: int,
    peptide_dim: int,
    receptor_x_dim: Optional[int] = None,
    pep_atom_dim: int = 19,
) -> Tuple[str, str]:
    pt_path = os.path.join(root, entry, f"features_{embedding}.pt")
    if not os.path.isfile(pt_path):
        return entry, "missing"
    if os.path.getsize(pt_path) == 0:
        return entry, "zero"
    try:
        data: Dict = torch.load(pt_path, map_location="cpu")
    except Exception as e:  # noqa: BLE001
        return entry, f"load_fail:{e}"

    try:
        rec = data["receptor"]
        pep = data["pep"]
        pep_a = data["pep_a"]
    except Exception as e:  # noqa: BLE001
        return entry, f"bad_key:missing node {e}"

    ok, err = _check_tensor("receptor.pos", rec.get("pos", None), None)
    if not ok:
        return entry, f"non_finite:{err}"
    ok, err = _check_tensor(
        "receptor.x",
        rec.get("x", None),
        receptor_x_dim,
    )
    if not ok:
        return entry, ("non_finite:" + err) if ("NaN" in err or "Inf" in err) else f"bad_shape:{err}"
    ok, err = _check_tensor("receptor.tips", rec.get("tips", None), None)
    if not ok:
        return entry, f"non_finite:{err}"
    ok, err = _check_tensor("receptor.node_v", rec.get("node_v", None), None)
    if not ok:
        return entry, f"non_finite:{err}"

    ok, err = _check_tensor("pep.x", pep.get("x", None), peptide_dim)
    if not ok:
        return entry, ("non_finite:" + err) if ("NaN" in err or "Inf" in err) else f"bad_shape:{err}"

    ok, err = _check_tensor("pep_a.pos", pep_a.get("pos", None), None)
    if not ok:
        return entry, f"non_finite:{err}"
    ok, err = _check_tensor("pep_a.orig_pos", pep_a.get("orig_pos", None), None)
    if not ok:
        return entry, f"non_finite:{err}"
    ok, err = _check_tensor("pep_a.x", pep_a.get("x", None), pep_atom_dim)
    if not ok:
        return entry, ("non_finite:" + err) if ("NaN" in err or "Inf" in err) else f"bad_shape:{err}"

    try:
        rec_edge = data["receptor", "rec_contact", "receptor"]
        pep_edge = data["pep_a", "to", "pep_a"]
    except Exception as e:  # noqa: BLE001
        return entry, f"bad_key:missing edge {e}"

    ei = rec_edge.get("edge_index", None)
    if ei is None or ei.shape[0] != 2:
        return entry, "bad_shape:rec edge_index"
    ok, err = _check_tensor("rec edge_s", rec_edge.get("edge_s", None), None)
    if not ok:
        return entry, f"non_finite:{err}"
    ok, err = _check_tensor("rec edge_v", rec_edge.get("edge_v", None), None)
    if not ok:
        return entry, f"non_finite:{err}"

    ei = pep_edge.get("edge_index", None)
    if ei is None or ei.shape[0] != 2:
        return entry, "bad_shape:pep edge_index"
    return entry, "ok"


def _inspect_root(
    root: str,
    embedding: str,
    receptor_dim: int,
    peptide_dim: int,
    num_workers: int = 1,
    maxtasksperchild: Optional[int] = None,
    output_csv: Optional[str] = None,
) -> None:
    missing_file: List[str] = []
    zero_size: List[str] = []
    load_fail: List[Tuple[str, str]] = []
    bad_key: List[Tuple[str, str]] = []
    bad_shape: List[Tuple[str, str]] = []
    non_finite: List[Tuple[str, str]] = []

    entries = sorted(os.listdir(root))
    if num_workers > 1:
        import multiprocessing as mp
        from functools import partial

        worker = partial(
            _check_entry,
            root=root,
            embedding=embedding,
            receptor_dim=receptor_dim,
            peptide_dim=peptide_dim,
            receptor_x_dim=receptor_dim,
        )
        with mp.get_context("spawn").Pool(processes=num_workers, maxtasksperchild=maxtasksperchild) as pool:
            results = pool.imap_unordered(worker, entries, chunksize=8)
            for entry, status in results:
                if status == "ok":
                    continue
                if status == "missing":
                    missing_file.append(entry)
                elif status == "zero":
                    zero_size.append(entry)
                elif status.startswith("load_fail:"):
                    load_fail.append((entry, status[len("load_fail:") :]))
                elif status.startswith("bad_key:"):
                    bad_key.append((entry, status[len("bad_key:") :]))
                elif status.startswith("bad_shape:"):
                    bad_shape.append((entry, status[len("bad_shape:") :]))
                elif status.startswith("non_finite:"):
                    non_finite.append((entry, status[len("non_finite:") :]))
    else:
        for entry in entries:
            entry, status = _check_entry(
                entry,
                root,
                embedding,
                receptor_dim,
                peptide_dim,
                receptor_x_dim=receptor_dim,
            )
            if status == "ok":
                continue
            if status == "missing":
                missing_file.append(entry)
            elif status == "zero":
                zero_size.append(entry)
            elif status.startswith("load_fail:"):
                load_fail.append((entry, status[len("load_fail:") :]))
            elif status.startswith("bad_key:"):
                bad_key.append((entry, status[len("bad_key:") :]))
            elif status.startswith("bad_shape:"):
                bad_shape.append((entry, status[len("bad_shape:") :]))
            elif status.startswith("non_finite:"):
                non_finite.append((entry, status[len("non_finite:") :]))

    def _print(title: str, items: List[Tuple[str, str]]):
        if not items:
            return
        print(f"{title}: {len(items)}")
        for pid, err in items[:20]:
            print("  ", pid, "->", err)

    print(f"巡检目录: {root}, embedding={embedding}")
    print(f"缺失文件: {len(missing_file)}, 零字节: {len(zero_size)}, 加载失败: {len(load_fail)}")
    _print("缺字段/键", bad_key)
    _print("形状/维度异常", bad_shape)
    _print("包含 NaN/Inf", non_finite)
    if missing_file:
        print("示例缺失:", missing_file[:20])
    if zero_size:
        print("示例零字节:", zero_size[:20])
    if load_fail:
        print("示例加载失败:", load_fail[:5])

    if output_csv:
        import csv

        out_rows = []
        for e in missing_file:
            out_rows.append({"complex_name": e, "reason": "missing"})
        for e in zero_size:
            out_rows.append({"complex_name": e, "reason": "zero"})
        for e, r in load_fail:
            out_rows.append({"complex_name": e, "reason": f"load_fail:{r}"})
        for e, r in bad_key:
            out_rows.append({"complex_name": e, "reason": f"bad_key:{r}"})
        for e, r in bad_shape:
            out_rows.append({"complex_name": e, "reason": f"bad_shape:{r}"})
        for e, r in non_finite:
            out_rows.append({"complex_name": e, "reason": f"non_finite:{r}"})
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["complex_name", "reason"])
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"已写入问题列表到 {output_csv} (共 {len(out_rows)} 条)")


def main():
    ap = argparse.ArgumentParser(description="查看或巡检 .pt 结构")
    ap.add_argument("pt", nargs="?", help="单文件路径，或省略配合 --root 批量巡检")
    ap.add_argument("--keys", action="store_true", help="仅打印顶层字段（单文件模式）")
    ap.add_argument("--root", help="批量巡检的根目录，子目录名视为 pdbid")
    ap.add_argument("--embedding", choices=["onehot", "esm"], default="onehot", help="批量巡检时选择要检查的 pt 名称")
    ap.add_argument("--check_all", action="store_true", help="启用批量巡检（需配合 --root）")
    ap.add_argument("--receptor_dim", type=int, default=None, help="期望受体 x 最后维度（默认随 embedding 自动设置）")
    ap.add_argument("--peptide_dim", type=int, default=None, help="期望肽 x 最后维度（默认随 embedding 自动设置）")
    ap.add_argument("--num_workers", type=int, default=1, help="批量巡检并行进程数")
    ap.add_argument("--maxtasksperchild", type=int, default=None, help="为防内存泄漏可设定进程复用次数上限")
    ap.add_argument("--output_csv", type=str, default=None, help="将发现的问题写入 CSV（仅批量模式）")
    args = ap.parse_args()

    if args.check_all:
        if not args.root:
            raise SystemExit("--check_all 需要提供 --root")
        if args.receptor_dim is None:
            args.receptor_dim = ONEHOT_REC_DIM if args.embedding == "onehot" else ESM_REC_DIM
        if args.peptide_dim is None:
            args.peptide_dim = ONEHOT_PEP_DIM if args.embedding == "onehot" else ESM_PEP_DIM
        _inspect_root(
            args.root,
            args.embedding,
            args.receptor_dim,
            args.peptide_dim,
            num_workers=args.num_workers,
            maxtasksperchild=args.maxtasksperchild,
            output_csv=args.output_csv,
        )
        return

    if not args.pt:
        ap.error("单文件模式需要提供 pt 路径，或使用 --root --check_all 批量巡检。")
    _print_single(args.pt, args.keys)


if __name__ == "__main__":
    main()
