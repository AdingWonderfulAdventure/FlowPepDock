#!/usr/bin/env python
"""
巡检 data/processed 下的样本，检查 PDB 序列、原子完整性与 PT 文件的存在/维度。

输出 CSV（默认 /tmp/processed_inspect.csv），字段示例：
complex_name,rec_len,pep_len,rec_chains,pep_chains,rec_seq_empty,pep_seq_empty,pt_onehot_exists,pt_esm_exists

用法：
PYTHONPATH=$(pwd) python scripts/data_qc/inspect_processed.py \
  --root data/processed \
  --output /tmp/processed_inspect.csv \
  --receptor_dim 114 --peptide_dim 105 \
  --esm_receptor_dim 1290 --esm_peptide_dim 1281
"""

import argparse
import ast
import csv
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from Bio.Data import IUPACData
from Bio.PDB import PDBParser, Polypeptide


ONEHOT_REC_DIM = 114
ONEHOT_PEP_DIM = 105
ESM_REC_DIM = 10 + 1280
ESM_PEP_DIM = 1 + 1280


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_three_list() -> List[str]:
    path = _repo_root() / "dataset" / "peptide_feature.py"
    text = path.read_text(encoding="utf-8", errors="replace")
    anchor = "three2idx = {k:v for v, k in enumerate("
    idx = text.find(anchor)
    if idx < 0:
        raise RuntimeError(f"[inspect_processed] 找不到 three2idx 定义：{path}")
    start = text.find("[", idx)
    if start < 0:
        raise RuntimeError(f"[inspect_processed] 找不到 three2idx 列表起始 '['：{path}")
    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise RuntimeError(f"[inspect_processed] three2idx 列表括号不闭合：{path}")
    codes = ast.literal_eval(text[start:end])
    if not isinstance(codes, list) or not all(isinstance(x, str) for x in codes):
        raise RuntimeError(f"[inspect_processed] three2idx 列表解析失败：{path}")
    return codes


def _extract_bracket_block(text: str, start: int) -> Tuple[str, int]:
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
    raise RuntimeError("[inspect_processed] 列表括号不闭合")


def _load_atomname2idx() -> List[Dict[str, int]]:
    path = _repo_root() / "dataset" / "peptide_feature.py"
    text = path.read_text(encoding="utf-8", errors="replace")
    anchor = "atomname2idx="
    idx = text.find(anchor)
    if idx < 0:
        raise RuntimeError(f"[inspect_processed] 找不到 atomname2idx 定义：{path}")
    start = text.find("[", idx)
    if start < 0:
        raise RuntimeError(f"[inspect_processed] 找不到 atomname2idx 列表起始 '['：{path}")
    block, _ = _extract_bracket_block(text, start)

    atom_lists: List[List[str]] = []
    pos = 0
    while True:
        enum_idx = block.find("enumerate([", pos)
        if enum_idx < 0:
            break
        list_start = block.find("[", enum_idx)
        if list_start < 0:
            break
        list_text, next_pos = _extract_bracket_block(block, list_start)
        items = ast.literal_eval(list_text)
        if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
            raise RuntimeError(f"[inspect_processed] atomname2idx 列表项解析失败：{path}")
        atom_lists.append(items)
        pos = next_pos

    if not atom_lists:
        raise RuntimeError(f"[inspect_processed] atomname2idx 未解析到任何列表：{path}")

    return [{atom: i for i, atom in enumerate(items)} for items in atom_lists]


THREE_LIST = _load_three_list()
ATOMNAME2IDX = _load_atomname2idx()
if len(THREE_LIST) != len(ATOMNAME2IDX):
    raise RuntimeError(
        f"[inspect_processed] three2idx/atomname2idx 长度不一致：{len(THREE_LIST)} vs {len(ATOMNAME2IDX)}"
    )
RESNAME_TO_ATOMS = {
    res: set(ATOMNAME2IDX[i].keys())
    for i, res in enumerate(THREE_LIST)
}

BACKBONE_ATOMS = {"N", "CA", "C", "O"}
OPTIONAL_ATOMS = {"OXT", "X"}


def parse_args():
    p = argparse.ArgumentParser(description="巡检 processed 目录的 PDB/PT")
    p.add_argument("--root", default="data/processed", help="处理后数据根目录")
    p.add_argument("--output", default="/tmp/processed_inspect.csv", help="输出 CSV")
    p.add_argument("--receptor_dim", type=int, default=None, help="onehot 期望 rec.x 最后维度（可选，不检查则留空）")
    p.add_argument("--peptide_dim", type=int, default=None, help="onehot 期望 pep.x 最后维度（可选，不检查则留空）")
    p.add_argument("--esm_receptor_dim", type=int, default=None, help="esm 期望 rec.x 最后维度（可选，不检查则留空）")
    p.add_argument("--esm_peptide_dim", type=int, default=None, help="esm 期望 pep.x 最后维度（可选，不检查则留空）")
    p.add_argument("--skip_pt", action="store_true", help="仅检查 PDB，不加载 PT")
    p.add_argument("--max_min_pep_rec_atom_dist", type=float, default=8.0, help="肽-受体最近原子距离阈值（Å），默认 8.0")
    p.add_argument("--num_workers", type=int, default=8, help="并行进程数")
    p.add_argument("--maxtasksperchild", type=int, default=64, help="每个子进程处理的样本数（避免内存积累）")
    return p.parse_args()


def fasta_from_pdb(pdb_path: Path) -> Dict[str, str]:
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("p", str(pdb_path))
    except Exception:
        # 遇到乱序残基编号等异常时返回空，后续标记为空序列便于排查
        return {}
    seqs = {}
    three_to_one = IUPACData.protein_letters_3to1
    for chain in struct.get_chains():
        ppb = Polypeptide.PPBuilder()
        chain_seqs = [str(pp.get_sequence()) for pp in ppb.build_peptides(chain)]
        if chain_seqs:
            seqs[chain.id] = "".join(chain_seqs)
        else:
            # fallback：按残基三字码映射，不认识记 X
            residues = list(chain.get_residues())
            letters = [three_to_one.get(res.get_resname().upper().strip(), "X") for res in residues]
            if letters:
                seqs[chain.id] = "".join(letters)
    return seqs


def check_pt(pt_path: Path, rec_dim: int, pep_dim: int) -> Tuple[bool, bool]:
    """返回 (exists, ok)"""
    if not pt_path.exists():
        return False, False
    try:
        data = torch.load(pt_path, map_location="cpu")
        rec_ok = pep_ok = True
        pep_key = None
        if hasattr(data, "node_types"):
            pep_key = "pep" if "pep" in data.node_types else ("peptide" if "peptide" in data.node_types else None)
        if rec_dim is not None:
            rec_ok = hasattr(data, "node_types") and ("receptor" in data.node_types) and hasattr(data["receptor"], "x") and data["receptor"].x.shape[-1] == rec_dim
        if pep_dim is not None:
            if pep_key is None:
                pep_ok = False
            else:
                pep_ok = hasattr(data, "node_types") and (pep_key in data.node_types) and hasattr(data[pep_key], "x") and data[pep_key].x.shape[-1] == pep_dim
        return True, bool(rec_ok and pep_ok)
    except Exception:
        return True, False


def count_hetatm_lines(pdb_path: Path) -> int:
    if not pdb_path.exists():
        return 0
    n = 0
    with pdb_path.open("r", errors="ignore") as f:
        for line in f:
            if line.startswith("HETATM"):
                n += 1
    return n


def _parse_residue_atoms(pdb_path: Path) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    residues: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    if not pdb_path.exists():
        return residues
    with pdb_path.open("r", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            altloc = (line[16] if len(line) > 16 else " ").strip()
            if altloc not in {"", "A"}:
                continue
            atom = (line[12:16] if len(line) >= 16 else "").strip()
            if not atom:
                continue
            if atom.startswith("H"):
                continue
            resname = (line[17:20] if len(line) >= 20 else "").strip().upper()
            chain = (line[21] if len(line) > 21 else " ").strip() or " "
            resid = (line[22:26] if len(line) >= 26 else "").strip()
            icode = (line[26] if len(line) > 26 else " ").strip()
            key = (chain, resid, icode)
            if key not in residues:
                residues[key] = {"resname": resname, "atoms": set()}
            residues[key]["atoms"].add(atom)
    return residues


def _load_atom_coords(pdb_path: Path) -> np.ndarray:
    coords = []
    if not pdb_path.exists():
        return np.zeros((0, 3), dtype=float)
    with pdb_path.open("r", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            altloc = (line[16] if len(line) > 16 else " ").strip()
            if altloc not in {"", "A"}:
                continue
            atom = (line[12:16] if len(line) >= 16 else "").strip()
            if not atom or atom.startswith("H"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue
            coords.append((x, y, z))
    if not coords:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _min_interatomic_distance(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> float:
    if a.size == 0 or b.size == 0:
        return float("inf")
    min_d2 = float("inf")
    for i in range(0, len(a), chunk):
        aa = a[i : i + chunk]
        diff = aa[:, None, :] - b[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        local_min = float(d2.min())
        if local_min < min_d2:
            min_d2 = local_min
            if min_d2 == 0.0:
                break
    return float(np.sqrt(min_d2))


def _min_pep_rec_atom_dist(pep_pdb: Path, rec_pdb: Path) -> float:
    pep_coords = _load_atom_coords(pep_pdb)
    rec_coords = _load_atom_coords(rec_pdb)
    return _min_interatomic_distance(pep_coords, rec_coords)


def validate_pdb_pair(
    rec_pdb: Path,
    pep_pdb: Path,
    max_min_pep_rec_atom_dist: float = 8.0,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not rec_pdb.exists():
        reasons.append("missing_receptor_pdb")
        return False, reasons
    if not pep_pdb.exists():
        reasons.append("missing_peptide_pdb")
        return False, reasons
    if count_hetatm_lines(rec_pdb) != 0:
        reasons.append("rec_hetatm")
    if count_hetatm_lines(pep_pdb) != 0:
        reasons.append("pep_hetatm")
    _, rec_bb_missing = _check_receptor_backbone(rec_pdb)
    if rec_bb_missing != 0:
        reasons.append("rec_backbone_missing")
    _, pep_bb_missing, pep_sc_missing, _ = _check_peptide_completeness(pep_pdb)
    if pep_bb_missing != 0:
        reasons.append("pep_backbone_missing")
    if pep_sc_missing != 0:
        reasons.append("pep_sidechain_missing")
    pep_contiguous, _, _ = _check_peptide_contiguity(pep_pdb)
    if pep_contiguous != 1:
        reasons.append("pep_non_contiguous")
    min_dist = _min_pep_rec_atom_dist(pep_pdb, rec_pdb)
    if min_dist > max_min_pep_rec_atom_dist:
        reasons.append(f"min_pep_rec_atom_dist_gt{max_min_pep_rec_atom_dist:g}")
    return (len(reasons) == 0), reasons


def _check_receptor_backbone(pdb_path: Path) -> Tuple[int, int]:
    residues = _parse_residue_atoms(pdb_path)
    total = len(residues)
    missing = 0
    for info in residues.values():
        atoms = info["atoms"]
        if not BACKBONE_ATOMS.issubset(atoms):
            missing += 1
    return total, missing


def _expected_peptide_atoms(resname: str) -> Tuple[bool, set]:
    if resname not in RESNAME_TO_ATOMS:
        return False, set()
    expected = set(RESNAME_TO_ATOMS[resname])
    expected -= OPTIONAL_ATOMS
    return True, expected


def _check_peptide_completeness(pdb_path: Path) -> Tuple[int, int, int, int]:
    residues = _parse_residue_atoms(pdb_path)
    total = len(residues)
    backbone_missing = 0
    sidechain_missing = 0
    unknown_res = 0
    for info in residues.values():
        resname = info["resname"]
        atoms = info["atoms"]
        if not BACKBONE_ATOMS.issubset(atoms):
            backbone_missing += 1
        ok, expected = _expected_peptide_atoms(resname)
        if not ok:
            unknown_res += 1
            sidechain_missing += 1
            continue
        sidechain_expected = expected - BACKBONE_ATOMS
        if not sidechain_expected.issubset(atoms):
            sidechain_missing += 1
    return total, backbone_missing, sidechain_missing, unknown_res


def _resid_sort_key(resid_str: str, icode: str) -> Tuple[int, int]:
    try:
        resid = int(resid_str)
    except Exception:
        resid = -1
    icode = (icode or "").strip()
    if not icode:
        icode_rank = 0
    else:
        icode_rank = ord(icode[0])
    return resid, icode_rank


def _check_peptide_contiguity(pdb_path: Path) -> Tuple[int, int, int]:
    residues = _parse_residue_atoms(pdb_path)
    chains: Dict[str, List[Tuple[str, str]]] = {}
    for (chain, resid, icode) in residues.keys():
        chains.setdefault(chain, []).append((resid, icode))

    total_gaps = 0
    fragments = 0
    chains_with_res = 0
    for res_list in chains.values():
        if not res_list:
            continue
        chains_with_res += 1
        res_list = sorted(res_list, key=lambda x: _resid_sort_key(x[0], x[1]))
        gap = 0
        prev_resid = None
        for resid_str, _icode in res_list:
            try:
                resid = int(resid_str)
            except Exception:
                resid = None
            if resid is None:
                continue
            if prev_resid is None:
                prev_resid = resid
                continue
            if resid - prev_resid > 1:
                gap += 1
            if resid >= prev_resid:
                prev_resid = resid
        total_gaps += gap
        fragments += gap + 1

    contiguous = 1 if (chains_with_res == 1 and total_gaps == 0) else 0
    return contiguous, total_gaps, fragments


def _process(sample_dir: Path, rec_dim: int, pep_dim: int, esm_rec_dim: int, esm_pep_dim: int) -> Dict:
    name = sample_dir.name
    rec_pdb = sample_dir / "receptor.pdb"
    pep_pdb = sample_dir / "peptide.pdb"
    rec_hetatm = count_hetatm_lines(rec_pdb)
    pep_hetatm = count_hetatm_lines(pep_pdb)
    rec_total, rec_bb_missing = _check_receptor_backbone(rec_pdb)
    pep_total, pep_bb_missing, pep_sc_missing, pep_unknown = _check_peptide_completeness(pep_pdb)
    pep_contiguous, pep_gap_count, pep_fragment_count = _check_peptide_contiguity(pep_pdb)
    min_pep_rec_atom_dist = _min_pep_rec_atom_dist(pep_pdb, rec_pdb)
    rec_seqs = fasta_from_pdb(rec_pdb) if rec_pdb.exists() else {}
    pep_seqs = fasta_from_pdb(pep_pdb) if pep_pdb.exists() else {}
    rec_len = sum(len(s) for s in rec_seqs.values())
    pep_len = sum(len(s) for s in pep_seqs.values())
    onehot_exists = onehot_ok = False
    esm_exists = esm_ok = False
    if rec_dim is not None or pep_dim is not None or True:  # keep signature; actual skip handled in caller
        onehot_path = sample_dir / "features_onehot.pt"
        esm_path = sample_dir / "features_esm.pt"
        # 是否跳过由外层控制
        onehot_exists = onehot_ok = None
        esm_exists = esm_ok = None
    return {
        "complex_name": name,
        "rec_len": rec_len,
        "pep_len": pep_len,
        "rec_res_total": rec_total,
        "rec_backbone_missing": rec_bb_missing,
        "pep_res_total": pep_total,
        "pep_backbone_missing": pep_bb_missing,
        "pep_sidechain_missing": pep_sc_missing,
        "pep_unknown_res": pep_unknown,
        "pep_contiguous": pep_contiguous,
        "pep_gap_count": pep_gap_count,
        "pep_fragment_count": pep_fragment_count,
        "min_pep_rec_atom_dist": min_pep_rec_atom_dist,
        "rec_chains": len(rec_seqs),
        "pep_chains": len(pep_seqs),
        "rec_seq_empty": int(rec_len == 0),
        "pep_seq_empty": int(pep_len == 0),
        "rec_hetatm_lines": rec_hetatm,
        "pep_hetatm_lines": pep_hetatm,
        "pt_onehot_exists": onehot_exists,
        "pt_onehot_ok": onehot_ok,
        "pt_esm_exists": esm_exists,
        "pt_esm_ok": esm_ok,
    }


def main():
    args = parse_args()
    root = Path(args.root)
    out = Path(args.output)
    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    rec_dim = args.receptor_dim if args.receptor_dim is not None else ONEHOT_REC_DIM
    pep_dim = args.peptide_dim if args.peptide_dim is not None else ONEHOT_PEP_DIM
    esm_rec_dim = args.esm_receptor_dim if args.esm_receptor_dim is not None else ESM_REC_DIM
    esm_pep_dim = args.esm_peptide_dim if args.esm_peptide_dim is not None else ESM_PEP_DIM

    worker = partial(
        _process,
        rec_dim=rec_dim,
        pep_dim=pep_dim,
        esm_rec_dim=esm_rec_dim,
        esm_pep_dim=esm_pep_dim,
    )
    if args.num_workers <= 1:
        rows = [worker(d) for d in sample_dirs]
    else:
        with mp.Pool(processes=args.num_workers, maxtasksperchild=args.maxtasksperchild) as pool:
            rows = list(pool.imap_unordered(worker, sample_dirs, chunksize=8))

    # 如果不跳过 PT，再补填 PT 检查结果
    if not args.skip_pt:
        for row in rows:
            sample_dir = root / row["complex_name"]
            onehot_path = sample_dir / "features_onehot.pt"
            esm_path = sample_dir / "features_esm.pt"
            onehot_exists, onehot_ok = check_pt(onehot_path, rec_dim, pep_dim)
            esm_exists, esm_ok = check_pt(esm_path, esm_rec_dim, esm_pep_dim)
            row["pt_onehot_exists"] = int(onehot_exists)
            row["pt_onehot_ok"] = int(onehot_ok)
            row["pt_esm_exists"] = int(esm_exists)
            row["pt_esm_ok"] = int(esm_ok)
    else:
        for row in rows:
            row["pt_onehot_exists"] = ""
            row["pt_onehot_ok"] = ""
            row["pt_esm_exists"] = ""
            row["pt_esm_ok"] = ""

    # 计算总检验结果
    for row in rows:
        reasons = []
        if row["rec_hetatm_lines"] != 0:
            reasons.append("rec_hetatm")
        if row["pep_hetatm_lines"] != 0:
            reasons.append("pep_hetatm")
        if row["rec_backbone_missing"] != 0:
            reasons.append("rec_backbone_missing")
        if row["pep_backbone_missing"] != 0:
            reasons.append("pep_backbone_missing")
        if row["pep_sidechain_missing"] != 0:
            reasons.append("pep_sidechain_missing")
        if row["pep_contiguous"] != 1:
            reasons.append("pep_non_contiguous")
        if row["pt_onehot_ok"] != 1:
            reasons.append("pt_onehot_bad")
        min_dist = row.get("min_pep_rec_atom_dist")
        if min_dist is None or float(min_dist) > args.max_min_pep_rec_atom_dist:
            reasons.append(f"min_pep_rec_atom_dist_gt{args.max_min_pep_rec_atom_dist:g}")
        row["check_pass"] = 1 if not reasons else 0
        row["check_reason"] = ";".join(reasons)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fo:
        fieldnames = [
            "complex_name",
            "rec_len",
            "pep_len",
            "rec_res_total",
            "rec_backbone_missing",
            "pep_res_total",
            "pep_backbone_missing",
            "pep_sidechain_missing",
            "pep_unknown_res",
            "pep_contiguous",
            "pep_gap_count",
            "pep_fragment_count",
            "min_pep_rec_atom_dist",
            "rec_chains",
            "pep_chains",
            "rec_seq_empty",
            "pep_seq_empty",
            "rec_hetatm_lines",
            "pep_hetatm_lines",
            "pt_onehot_exists",
            "pt_onehot_ok",
            "pt_esm_exists",
            "pt_esm_ok",
            "check_pass",
            "check_reason",
        ]
        w = csv.DictWriter(fo, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
