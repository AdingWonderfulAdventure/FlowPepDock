#!/usr/bin/env python
"""
把 CSV 里的 CIF 复合物按链拆成受体/肽 PDB，以肽为中心对受体做固定半径裁剪，并生成带路径的新 CSV 便于后续 prepare_training_data.py 处理，支持多进程。

输入 CSV 需包含：
    complex_name, complex_pdb, receptor_chains, peptide_chains
链 ID 可用逗号/分号分隔，如 "A,B" 或 "A;B"。

参数说明（常用）：
- --csv: 输入 CSV 路径（至少含 complex_name, receptor_chains, peptide_chains；或已含 receptor_pdb/peptide_pdb）。
- --out_dir: 拆分输出根目录，按 pdbid 建子目录存 receptor.pdb/peptide.pdb。
- --output_csv: 写出的带 receptor_pdb/peptide_pdb 路径的 CSV。
- --cif_dir: CIF 所在目录（默认按 pdbid.cif 在此目录查找）。
- --contact_threshold: 受体链与肽最近原子距离阈值（Å），超出则剔除该链，全部超阈值则保留最近一条。
- --receptor_crop_radius: 受体口袋裁剪半径（Å），仅保留距肽 <= 该半径的残基，<=0 不裁剪。
- --num_workers: 并行进程数，>1 开启多进程。
- 过滤规则：
  - 输出的 `receptor.pdb/peptide.pdb` **只保留蛋白/肽聚合物残基（entity_poly.type 含 peptide 的 entity，且 group_PDB=ATOM）**：
    - 所有非蛋白组分（配体/辅因子/溶剂/离子等）都会被剔除，不会进入后续 PT/ESM。
    - “蛋白组分但不在 onehot 词表里的三字码”会被保留在 PDB 中，并记录到输出 CSV 的 `receptor_unsupported_res/peptide_unsupported_res` 便于审计；下游 PT onehot 会统一落到第 104 维兜底桶。

示例运行：
    # 输入（candidate.csv）示例:
    # pdb_id,receptor_chains,peptide_chains
    # 7r6q,A,B
    #
    # 命令:
    # PYTHONPATH=$(pwd) python scripts/data_prep/split_cif_chains.py \
    #   --csv candidate.csv \
    #   --out_dir data/processed \
    #   --output_csv data/processed/split_paths.csv \
    #   --cif_dir data/raw_cif \
    #   --contact_threshold 8.0 \
    #   --receptor_crop_radius 30.0 \
    #   --num_workers 8
    #
    # 输出 CSV（完整字段）示例：
    # PDB编号(pdb_id),...,receptor_pdb,peptide_pdb
    # 6ZVF,...,data/processed/6zvf/receptor.pdb,data/processed/6zvf/peptide.pdb
    #
    # 生成 pt 时可另存一份精简版 CSV（仅保留必要列）：
    # complex_name,receptor_pdb,peptide_pdb
    # 6ZVF,data/processed/6zvf/receptor.pdb,data/processed/6zvf/peptide.pdb
    #
    # 输出:
    # data/processed/<pdbid>/receptor.pdb
    # data/processed/<pdbid>/peptide.pdb
    # data/processed/split_paths.csv （receptor_pdb/peptide_pdb 路径已填好）
"""

import argparse
import ast
import csv
import os
import string
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Bio.PDB import MMCIFParser, PDBIO, Select
import numpy as np

def _load_supported_three_letter_codes() -> set:
    """
    读取 `dataset/peptide_feature.py:three2idx` 里的三字码词表，避免脚本内再维护一份“支持残基列表”导致漂移。
    这里用静态解析（不 import），因为 peptide_feature.py 引用 RDKit/MDAnalysis/torch，启动多进程时会很重。
    """
    path = Path(__file__).resolve().parents[2] / "dataset" / "peptide_feature.py"
    text = path.read_text(encoding="utf-8", errors="replace")
    anchor = "three2idx = {k:v for v, k in enumerate("
    idx = text.find(anchor)
    if idx < 0:
        raise RuntimeError(f"[split_cif_chains] 找不到 three2idx 定义：{path}")
    start = text.find("[", idx)
    if start < 0:
        raise RuntimeError(f"[split_cif_chains] 找不到 three2idx 列表起始 '['：{path}")
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
        raise RuntimeError(f"[split_cif_chains] three2idx 列表括号不闭合：{path}")
    codes = ast.literal_eval(text[start:end])
    if not isinstance(codes, list) or not all(isinstance(x, str) for x in codes):
        raise RuntimeError(f"[split_cif_chains] three2idx 列表解析失败：{path}")
    return set(codes)


SUPPORTED_THREE = _load_supported_three_letter_codes()

CHAIN_ID_ALPHABET = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)


class ChainSelect(Select):
    def __init__(
        self,
        chains: List[str],
        keep_residues: Optional[Dict[str, set]] = None,
        polymer_residues: Optional[Dict[str, set]] = None,
    ):
        self.chains = set(chains)
        self.keep_residues = keep_residues or {}
        self.polymer_residues = polymer_residues or {}

    def accept_chain(self, chain):
        return chain.id.strip() in self.chains

    def accept_residue(self, residue):
        cid = residue.get_parent().id.strip()
        if cid not in self.chains:
            return 0
        polymer_ids = self.polymer_residues.get(cid)
        if polymer_ids is not None and residue.id not in polymer_ids:
            return 0
        keep_ids = self.keep_residues.get(cid)
        if keep_ids is None:
            return 1
        return 1 if residue.id in keep_ids else 0

    def accept_atom(self, atom):
        # altloc 只保留 A 构象；空白 altloc 视为主构象
        alt = (atom.get_altloc() or "").strip()
        return 1 if alt in {"", "A"} else 0


def parse_chains(s: str) -> List[str]:
    return [c.strip() for c in s.replace(";", ",").split(",") if c.strip()]


def resolve_chains(raw_chains: List[str], chain_map: Dict[str, object], orig_to_new: Dict[str, str]) -> List[str]:
    """把原始链ID映射到重命名后的链ID；若无法映射则回退到“同名单字符”。"""
    resolved = []
    for cid in raw_chains:
        cid = cid.strip()
        if not cid:
            continue
        mapped = None
        if cid in chain_map:
            mapped = cid
        elif cid in orig_to_new:
            mapped = orig_to_new[cid]
        elif len(cid) == 1 and cid in chain_map:
            mapped = cid
        if mapped and mapped not in resolved:
            resolved.append(mapped)
    return resolved


def _chain_len(chain) -> int:
    return sum(1 for res in chain.get_residues() if res.id[0] == " ")


def _pick_best_peptide_chain(
    chain_map: Dict[str, object],
    rec_chains: List[str],
    pep_chains: List[str],
    contact_threshold: float,
    polymer_residues: Dict[str, Optional[set]],
) -> List[str]:
    """
    当同一个复合物里存在多个“完全相同的肽/配体拷贝”时，CSV 里给的 peptide_chains 可能挑到了一个“离受体很远”的拷贝。
    这里做一次纠偏：在长度相同的候选里，选与受体最近的那条（优先落在 contact_threshold 内）。
    """
    if not rec_chains or not pep_chains:
        return pep_chains
    pep0 = chain_map[pep_chains[0]]
    target_len = _chain_len(pep0)
    if target_len <= 0:
        return pep_chains

    # 只在“当前 pep 明显离受体很远”时触发（避免把本来正确的链换掉）
    current_min = min(
        _min_contact(chain_map[rc], chain_map[pc], polymer_residues.get(rc), polymer_residues.get(pc))
        for rc in rec_chains
        for pc in pep_chains
    )
    if not (current_min > contact_threshold):
        return pep_chains

    candidates = []
    for cid, ch in chain_map.items():
        if cid in rec_chains:
            continue
        if _chain_len(ch) != target_len:
            continue
        d = min(_min_contact(chain_map[rc], ch, polymer_residues.get(rc), polymer_residues.get(cid)) for rc in rec_chains)
        candidates.append((d, cid))
    if not candidates:
        return pep_chains
    candidates.sort()
    best_d, best_cid = candidates[0]
    if best_d < current_min:
        print(f"[INFO] override peptide_chain {pep_chains} -> {[best_cid]} (min_contact {current_min:.2f} -> {best_d:.2f})")
        return [best_cid]
    return pep_chains


def _min_contact(chain_a, chain_b, residue_ids_a: Optional[set] = None, residue_ids_b: Optional[set] = None) -> float:
    """计算两条链最小原子距离，返回 inf 表示其中一条无原子。"""
    atoms_a = list(_iter_atoms_from_residue_ids(chain_a, residue_ids_a))
    atoms_b = list(_iter_atoms_from_residue_ids(chain_b, residue_ids_b))
    if not atoms_a or not atoms_b:
        return float("inf")
    coords_a = np.array([a.get_coord() for a in atoms_a])
    coords_b = np.array([b.get_coord() for b in atoms_b])
    dists = np.linalg.norm(coords_a[:, None, :] - coords_b[None, :, :], axis=-1)
    return float(dists.min())

def _iter_atoms_from_residue_ids(chain, residue_ids: Optional[set]):
    if residue_ids is None:
        yield from chain.get_atoms()
        return
    for res in chain.get_residues():
        if res.id not in residue_ids:
            continue
        yield from res.get_atoms()


def _residues_within(chain, target_coords: np.ndarray, radius: float, residue_ids: Optional[set]) -> set:
    """返回在给定半径内的残基id集合；若无原子则空。"""
    atoms = list(_iter_atoms_from_residue_ids(chain, residue_ids))
    if not atoms or target_coords.size == 0:
        return set()
    coords = np.array([a.get_coord() for a in atoms])
    dists = np.linalg.norm(coords[:, None, :] - target_coords[None, :, :], axis=-1)
    close_atom_idx = np.where(dists.min(axis=1) <= radius)[0]
    keep_res = set()
    atom_list = atoms
    for idx in close_atom_idx:
        keep_res.add(atom_list[idx].get_parent().id)
    return keep_res


def _residues_within_center(chain, center: np.ndarray, radius: float, residue_ids: Optional[set]) -> set:
    """以给定中心为基准，返回在半径内的残基id集合。"""
    atoms = list(_iter_atoms_from_residue_ids(chain, residue_ids))
    if not atoms or center.size == 0:
        return set()
    coords = np.array([a.get_coord() for a in atoms])
    dists = np.linalg.norm(coords - center[None, :], axis=-1)
    keep_res = set()
    atom_list = atoms
    for idx, d in enumerate(dists):
        if d <= radius:
            keep_res.add(atom_list[idx].get_parent().id)
    return keep_res


def _mmcif_get_list(d: dict, key: str) -> Optional[list]:
    v = d.get(key)
    if v is None:
        return None
    return v if isinstance(v, list) else [v]


def _normalize_ins_code(x: str) -> str:
    x = (x or "").strip()
    return " " if (not x or x in {".", "?"}) else x


def _as_int_or_none(x: str) -> Optional[int]:
    x = (x or "").strip()
    if not x or x in {".", "?"}:
        return None
    try:
        return int(x)
    except ValueError:
        return None


def _extract_polymer_residue_ids(parser: MMCIFParser, orig_to_new: Dict[str, str]) -> Dict[str, set]:
    """
    从 mmCIF 表里抽取“多肽聚合物(蛋白/肽)残基集合”，用于在写 PDB 时**强制剔除所有非蛋白组分**。
    规则：只接受 entity_poly.type 含 'peptide' 的 entity；再用 _atom_site.group_PDB == 'ATOM' 过滤掉 HETATM。
    """
    d = getattr(parser, "_mmcif_dict", None) or {}
    entity_ids = _mmcif_get_list(d, "_entity_poly.entity_id") or []
    entity_types = _mmcif_get_list(d, "_entity_poly.type") or []
    polymer_entity = set()
    for eid, etype in zip(entity_ids, entity_types):
        if not eid:
            continue
        t = (etype or "").lower()
        if "peptide" in t:
            polymer_entity.add(str(eid))

    if not polymer_entity:
        return {new: None for new in orig_to_new.values()}

    ent_col = "_atom_site.label_entity_id" if "_atom_site.label_entity_id" in d else None
    if not ent_col:
        return {new: None for new in orig_to_new.values()}

    group_col = "_atom_site.group_PDB" if "_atom_site.group_PDB" in d else None
    chain_col = "_atom_site.auth_asym_id" if "_atom_site.auth_asym_id" in d else "_atom_site.label_asym_id"
    seq_col = "_atom_site.auth_seq_id" if "_atom_site.auth_seq_id" in d else "_atom_site.label_seq_id"
    ins_col = "_atom_site.pdbx_PDB_ins_code" if "_atom_site.pdbx_PDB_ins_code" in d else None

    ent = _mmcif_get_list(d, ent_col) or []
    group = _mmcif_get_list(d, group_col) if group_col else None
    chain = _mmcif_get_list(d, chain_col) or []
    seq = _mmcif_get_list(d, seq_col) or []
    ins = _mmcif_get_list(d, ins_col) if ins_col else None

    polymer_ids_by_orig: Dict[str, set] = {}
    n = min(len(ent), len(chain), len(seq))
    for i in range(n):
        if str(ent[i]) not in polymer_entity:
            continue
        if group is not None and (group[i] or "").upper() != "ATOM":
            continue
        chain_id = (chain[i] or "").strip()
        seq_id = _as_int_or_none(seq[i])
        if not chain_id or seq_id is None:
            continue
        icode = _normalize_ins_code(ins[i]) if ins is not None and i < len(ins) else " "
        polymer_ids_by_orig.setdefault(chain_id, set()).add((" ", seq_id, icode))

    polymer_ids_by_new: Dict[str, set] = {}
    for orig, new in orig_to_new.items():
        polymer_ids_by_new[new] = polymer_ids_by_orig.get(orig, set())
    return polymer_ids_by_new


def split_one(record: dict, out_dir: Path, cif_dir: Path, parser: MMCIFParser, contact_threshold: float, crop_radius: float) -> tuple[str, str]:
    # 兼容不同列名
    def pick(keys):
        for k in keys:
            if k in record and record[k]:
                return record[k]
        return ""

    pdb_id = pick(["complex_name", "pdb_id", "PDB编号(pdb_id)", "PDB编号"]) or ""
    name = pdb_id or "complex"
    cif_path_str = pick(["complex_pdb", "cif_path"])
    if cif_path_str:
        cif_path = Path(cif_path_str)
    else:
        cif_path = cif_dir / f"{pdb_id.lower()}.cif"

    # 调试输出当前处理的 pdb
    print(f"[INFO] processing {pdb_id} from {cif_path}")

    rec_chain_str = pick(["receptor_chains", "受体链ID(receptor_chain)", "受体链ID"])
    pep_chain_str = pick(["peptide_chains", "肽链ID(peptide_chain)", "肽链ID"])

    rec_chains = parse_chains(rec_chain_str)
    pep_chains = parse_chains(pep_chain_str)
    structure = parser.get_structure(name, cif_path)
    # 只保留第一个model，避免多构象全部写出；同时把 CIF 的多字符链ID映射为 PDB 单字符链ID（并保留映射以便解析 CSV）
    model = structure[0]
    orig_to_new: Dict[str, str] = {}
    used = set()
    for chain in list(model.get_chains()):
        orig = (chain.id or "").strip() or "X"
        if len(orig) == 1 and orig in CHAIN_ID_ALPHABET and orig not in used:
            new = orig
        else:
            pool = [c for c in CHAIN_ID_ALPHABET if c not in used]
            new = pool[0] if pool else "X"
        used.add(new)
        orig_to_new[orig] = new
        chain.id = new

    chain_map = {chain.id: chain for chain in model.get_chains()}
    polymer_residues = _extract_polymer_residue_ids(parser, orig_to_new)
    # mmCIF 缺字段/解析失败时兜底：只保留 ATOM 残基（res.id[0] == ' '），仍能剔除 HETATM
    for cid, ch in chain_map.items():
        if polymer_residues.get(cid) is None:
            polymer_residues[cid] = {res.id for res in ch.get_residues() if res.id[0] == " "}
    # 解析链ID映射（CSV 里一般是 CIF 的原始链ID）
    rec_chains = resolve_chains(rec_chains, chain_map, orig_to_new)
    pep_chains = resolve_chains(pep_chains, chain_map, orig_to_new)
    if not pep_chains:
        raise ValueError(f"{name} 找不到肽链，请检查链ID")
    if sum(len(polymer_residues.get(c, set()) or set()) for c in pep_chains) == 0:
        raise ValueError(f"{name} 肽链未检测到任何蛋白/肽聚合物残基（可能链ID指到了非蛋白实体）")

    # 纠偏：多拷贝肽/重复配体时，CSV 可能指到了“离受体很远”的那条
    if rec_chains:
        pep_chains = _pick_best_peptide_chain(chain_map, rec_chains, pep_chains, contact_threshold, polymer_residues)

    # 按接触距离过滤受体链，阈值内保留，若全部超阈值则保留最近的一条
    pep_objs = [chain_map[c] for c in pep_chains]
    selected_rec: List[str] = []
    if rec_chains:
        contact_list: List[Tuple[float, str]] = []
        for rc in rec_chains:
            chain_rc = chain_map[rc]
            min_d = min(_min_contact(chain_rc, pc, polymer_residues.get(rc), polymer_residues.get(pc.id)) for pc in pep_objs)
            contact_list.append((min_d, rc))
        contact_list.sort()
        selected_rec = [cid for d, cid in contact_list if d <= contact_threshold]
        if not selected_rec:
            # 全部超阈值，保留最近的一条
            selected_rec = [contact_list[0][1]]
    rec_chains = selected_rec or rec_chains
    if rec_chains and sum(len(polymer_residues.get(c, set()) or set()) for c in rec_chains) == 0:
        raise ValueError(f"{name} 受体链未检测到任何蛋白/肽聚合物残基（可能链ID指到了非蛋白实体）")

    # 可选：对保留的受体链做口袋裁剪（以肽原子中心为零点），只保留距中心 <= crop_radius 的残基
    keep_residues: Dict[str, set] = {}
    if crop_radius > 0:
        pep_atoms = [a for pc in pep_objs for a in _iter_atoms_from_residue_ids(pc, polymer_residues.get(pc.id))]
        pep_coords = np.array([a.get_coord() for a in pep_atoms]) if pep_atoms else np.zeros((0, 3))
        center = pep_coords.mean(axis=0) if pep_coords.size > 0 else np.zeros(3)
        for rc in rec_chains:
            chain_rc = chain_map[rc]
            close_res = _residues_within_center(chain_rc, center, crop_radius, polymer_residues.get(rc))
            # 若裁剪后为空，则保留原链全部残基避免写出空文件
            keep_residues[rc] = close_res or None

    # 写出 PDB，并记录“不在 onehot 词表里的蛋白残基三字码”（仍保留在 PDB 中；下游会用 onehot 第104维兜底）
    io = PDBIO()
    io.set_structure(structure[0])
    subdir = out_dir / name.lower()
    subdir.mkdir(parents=True, exist_ok=True)
    rec_pdb = subdir / "receptor.pdb"
    pep_pdb = subdir / "peptide.pdb"

    rec_keep_res = keep_residues.copy()  # 仅用于裁剪半径；对残基类型不过滤
    rec_unsupported = set()
    pep_unsupported = set()
    _collect_unsupported_res(rec_chains, chain_map, polymer_residues, rec_unsupported)
    _collect_unsupported_res(pep_chains, chain_map, polymer_residues, pep_unsupported)

    with rec_pdb.open("w") as f:
        io.save(
            f,
            ChainSelect(rec_chains, rec_keep_res, polymer_residues=polymer_residues),
            write_end=True,
            preserve_atom_numbering=True,
        )
    with pep_pdb.open("w") as f:
        io.save(
            f,
            ChainSelect(pep_chains, {c: None for c in pep_chains}, polymer_residues=polymer_residues),
            write_end=True,
            preserve_atom_numbering=True,
        )
    return str(rec_pdb), str(pep_pdb), rec_unsupported, pep_unsupported

def _collect_unsupported_res(
    chain_ids: List[str],
    chain_map: Dict[str, object],
    polymer_residues: Dict[str, set],
    unsupported: set,
) -> None:
    """
    记录“蛋白/肽聚合物残基里，不在 onehot 词表(104维) 的三字码”。
    注意：这里只做审计记录，不负责过滤；过滤（剔除非蛋白）在写 PDB 时由 `polymer_residues` 白名单完成。
    """
    for cid in chain_ids:
        chain = chain_map[cid]
        polymer_ids = polymer_residues.get(cid)
        for res in chain.get_residues():
            resname = res.get_resname().strip()
            if polymer_ids is not None and res.id not in polymer_ids:
                continue
            if resname not in SUPPORTED_THREE:
                unsupported.add(resname)


def _process_row(
    record: dict,
    out_dir: str,
    cif_dir: str,
    contact_threshold: float,
    crop_radius: float,
) -> Tuple[dict, Optional[str]]:
    """工作进程包装，返回更新后的行和可选错误信息。"""
    try:
        parser = MMCIFParser(QUIET=True)
        rec_pdb, pep_pdb, rec_unsupported, pep_unsupported = split_one(
            record,
            Path(out_dir),
            Path(cif_dir),
            parser,
            contact_threshold,
            crop_radius,
        )
        # 只回传必要字段，避免把大序列串复制回父进程
        slim = {
            "complex_name": record.get("complex_name") or record.get("pdb_id") or record.get("PDB编号(pdb_id)"),
            "receptor_pdb": rec_pdb,
            "peptide_pdb": pep_pdb,
            "receptor_unsupported_res": ";".join(sorted(rec_unsupported)) if rec_unsupported else "",
            "peptide_unsupported_res": ";".join(sorted(pep_unsupported)) if pep_unsupported else "",
        }
        return slim, None
    except Exception as e:  # noqa: B902
        return record, str(e)


def _process_row_star(args):
    return _process_row(*args)


def main():
    ap = argparse.ArgumentParser(description="按链拆分 CIF，输出受体/肽 PDB 及新 CSV")
    ap.add_argument("--csv", required=True, help="包含 pdb_id 和 受体/肽链ID 的 CSV")
    ap.add_argument("--out_dir", required=True, help="拆分后 PDB 的输出目录")
    ap.add_argument("--output_csv", required=True, help="写出含 receptor_pdb/peptide_pdb 的新 CSV")
    ap.add_argument("--cif_dir", default="", help="CIF 所在目录（默认与下载时相同，文件名按pdbid.cif）")
    ap.add_argument("--contact_threshold", type=float, default=8.0, help="受体链与肽最近原子距离阈值，超出则剔除（默认8Å，若全超阈值则保留最近一条）")
    ap.add_argument("--receptor_crop_radius", type=float, default=30.0, help="裁剪受体口袋半径，单位Å；以肽为中心，仅保留距肽 <= 该半径的残基，<=0 则不裁剪（默认30Å）")
    ap.add_argument("--num_workers", type=int, default=1, help="并行进程数（>1 启用多进程）")
    ap.add_argument("--maxtasksperchild", type=int, default=1, help="每个子进程处理的任务数上限，完成后重启以释放内存（默认1，建议大规模处理保持1）")
    ap.add_argument("--fail_csv", default=None, help="失败记录 CSV：complex_name,reason")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = Path(args.cif_dir) if args.cif_dir else None

    fail_rows = []
    with open(args.csv) as fin, open(args.output_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        # 添加新列
        for col in ["receptor_pdb", "peptide_pdb", "receptor_unsupported_res", "peptide_unsupported_res"]:
            if col not in fieldnames:
                fieldnames.append(col)
        # 如果没有 complex_name 列，但有 pdb_id 列，添加以便写出瘦身结果
        if "complex_name" not in fieldnames:
            fieldnames.append("complex_name")
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        # 流式读取/写出，避免把所有结果一次性堆在内存
        def gen_work():
            for row in reader:
                row = {k.strip(): v for k, v in row.items()}
                yield (
                    row,
                    str(out_dir),
                    str(cif_dir or Path(row.get("complex_pdb") or ".").parent),
                    args.contact_threshold,
                    args.receptor_crop_radius,
                )

        if args.num_workers > 1:
            with Pool(processes=args.num_workers, maxtasksperchild=args.maxtasksperchild) as pool:
                for row, err in pool.imap_unordered(_process_row_star, gen_work(), chunksize=1):
                    if err:
                        bad_name = row.get("complex_name") or row.get("pdb_id") or row.get("PDB编号(pdb_id)") or row.get("PDB编号")
                        print(f"[WARN] {bad_name}: {err}")
                        fail_rows.append({"complex_name": (bad_name or "").strip().lower(), "reason": err})
                        continue
                    writer.writerow(row)
                    print(f"{row.get('complex_name') or row.get('pdb_id')} -> {row['receptor_pdb']}, {row['peptide_pdb']}")
        else:
            for work_item in gen_work():
                row, err = _process_row(*work_item)
                if err:
                    bad_name = row.get("complex_name") or row.get("pdb_id") or row.get("PDB编号(pdb_id)") or row.get("PDB编号")
                    print(f"[WARN] {bad_name}: {err}")
                    fail_rows.append({"complex_name": (bad_name or "").strip().lower(), "reason": err})
                    continue
                writer.writerow(row)
                print(f"{row.get('complex_name') or row.get('pdb_id')} -> {row['receptor_pdb']}, {row['peptide_pdb']}")

    if args.fail_csv:
        Path(args.fail_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.fail_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["complex_name", "reason"])
            writer.writeheader()
            writer.writerows(fail_rows)


if __name__ == "__main__":
    main()
