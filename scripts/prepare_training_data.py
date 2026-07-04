#!/usr/bin/env python
##########################################################################
# File Name: prepare_training_data.py
# Description: 读取精简 CSV，根据受体/肽 PDB 构建图特征（onehot/esm），生成可训练的 HeteroData .pt，并同步保存受体/肽文件便于后续训练/推理。
#
# 输入 CSV 示例（精简版，仅必要列）:
# complex_name,receptor_pdb,peptide_pdb
# 7r6q,data/processed/7r6q/receptor.pdb,data/processed/7r6q/peptide.pdb
#
# onehot 生成示例:
# PYTHONPATH=$(pwd) python scripts/prepare_training_data.py \
#   --csv data/processed/split_paths_for_pt.csv \
#   --output_dir data/processed \
#   --cache_dir preprocess_cache/tmp_cache \
#   --clean_cache \
#   --embedding onehot
# 输出: data/processed/<pdbid>/features_onehot.pt
#
# esm 生成示例（耗时/耗显存）:
# PYTHONPATH=$(pwd) python scripts/prepare_training_data.py \
#   --csv data/processed/split_paths_for_pt.csv \
#   --output_dir data/processed \
#   --cache_dir preprocess_cache/tmp_cache \
#   --clean_cache \
#   --embedding esm
# 输出: data/processed/<pdbid>/features_esm.pt
#
# 生成的 pt 关键字段说明（HeteroData）:
# - receptor.x: 残基节点特征，onehot(22) + 几何标量；pos/tips/node_v: 坐标与方向
# - pep.x: 肽残基节点特征，onehot(104) 或 ESM
# - pep_a.*: 肽原子级坐标/映射/掩码/边
# - 边: (receptor, rec_contact, receptor).edge_index/edge_s/edge_v；(pep_a, to, pep_a).edge_index 等
# - 辅助: name, success, original_center, peptide_inits, partials, noh_mda
#########################################################################

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch_geometric.data import HeteroData
from Bio import PDB

# 允许直接 `python scripts/prepare_training_data.py ...` 运行（无需手动 PYTHONPATH=$(pwd)）。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.inference_utils import InferenceDataset
try:
    from scripts.data_qc.inspect_processed import validate_pdb_pair
except ImportError:  # pragma: no cover - 兼容旧路径
    from scripts.inspect_processed import validate_pdb_pair


@dataclass
class ComplexRecord:
    name: str
    pdb_path: str
    receptor_chains: List[str]
    peptide_chains: List[str]
    receptor_pdb_path: Optional[str] = None
    peptide_pdb_path: Optional[str] = None


def parse_chain_ids(chain_str: str) -> List[str]:
    return [c.strip() for c in chain_str.replace(";", ",").replace(":", ",").split(",") if c.strip()]


def read_metadata(csv_path: str) -> List[ComplexRecord]:
    records = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"complex_name", "receptor_pdb", "peptide_pdb"}
        if not required.issubset(fieldnames):
            raise ValueError("CSV缺少列: complex_name, receptor_pdb, peptide_pdb")
        supported = required
        unknown = fieldnames - supported
        if unknown:
            print(f"[WARN] CSV包含未使用的列，将被忽略: {unknown}")
        for row in reader:
            complex_pdb = (row.get("complex_pdb") or "").strip()
            receptor_pdb = (row.get("receptor_pdb") or "").strip()
            peptide_pdb = (row.get("peptide_pdb") or "").strip()

            if not complex_pdb and not (receptor_pdb and peptide_pdb):
                raise ValueError(
                    f"{row.get('complex_name')} 必须要么提供complex_pdb，要么提供receptor_pdb+peptide_pdb"
                )

            receptor_chains = parse_chain_ids(row.get("receptor_chains", ""))
            peptide_chains = parse_chain_ids(row.get("peptide_chains", ""))
            if complex_pdb and (not receptor_chains or not peptide_chains):
                raise ValueError(
                    f"{row.get('complex_name')} 提供了complex_pdb，但缺少receptor_chains/peptide_chains"
                )

            records.append(
                ComplexRecord(
                    name=row["complex_name"],
                    pdb_path=complex_pdb,
                    receptor_chains=receptor_chains,
                    peptide_chains=peptide_chains,
                    receptor_pdb_path=receptor_pdb or None,
                    peptide_pdb_path=peptide_pdb or None,
                )
            )
    return records


class ChainSelect(PDB.Select):
    def __init__(self, chain_ids: List[str]):
        self.chain_ids = set(chain_ids)

    def accept_chain(self, chain):
        return chain.id.strip() in self.chain_ids


def extract_chains(pdb_path: str, chains: List[str], out_path: str) -> None:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(out_path, ChainSelect(chains))


def split_complex(record: ComplexRecord, workspace: str) -> Tuple[str, str]:
    os.makedirs(workspace, exist_ok=True)
    receptor_pdb = os.path.join(workspace, f"{record.name}_receptor.pdb")
    peptide_pdb = os.path.join(workspace, f"{record.name}_peptide.pdb")
    extract_chains(record.pdb_path, record.receptor_chains, receptor_pdb)
    extract_chains(record.pdb_path, record.peptide_chains, peptide_pdb)
    return receptor_pdb, peptide_pdb


def build_graph_from_files(
    complex_name: str,
    receptor_pdb: str,
    peptide_pdb: str,
    tmp_dir: str,
    use_lm_protein: bool,
    use_lm_peptide: bool,
    embedding: str = "onehot",
) -> HeteroData:
    """利用InferenceDataset复用正式推理流程，生成HeteroData。

    生成的HeteroData结构等同于推理时的输入，关键字段包括：
        - data['receptor']: 残基级节点（x、pos、tips、node_sigma_emb）
        - data['pep']: 肽残基级节点
        - data['pep_a']: 肽原子级节点（atom2res_index / atom2atomid_index / mask_edges_backbone 等）
        - data['receptor','receptor'] 和 data['pep','pep'] 等边索引/特征
        - 其它属性：初始构象、语言模型嵌入等
    训练脚本会在此基础上注入噪声和flow标签。
    """
    os.makedirs(os.path.join(tmp_dir, complex_name), exist_ok=True)
    dataset = InferenceDataset(
        output_dir=tmp_dir,
        complex_name_list=[complex_name],
        protein_description_list=[receptor_pdb],
        peptide_description_list=[peptide_pdb],
        lm_embeddings=use_lm_protein,
        lm_embeddings_pep=use_lm_peptide,
        embedding_mode=embedding,
        use_native_peptide_pose=True,
    )
    data = dataset[0]
    data.name = complex_name
    # 为了兼容训练脚本，强制保存name，后续FlowMatchingTransform会用到
    return data


def save_graph(data, output_dir: str, receptor_pdb: Optional[str] = None, peptide_pdb: Optional[str] = None, embedding: str = "onehot") -> str:
    """
    把 HeteroData 存到以 pdbid 为子目录的 dataset 结构，顺便把受体/肽 PDB 放进去便于管理。
    输出结构：<output_dir>/<name>/features_<embedding>.pt [+ receptor.pdb/peptide.pdb]
    """
    sample_dir = Path(output_dir) / data.name.lower()
    sample_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_dir / f"features_{embedding}.pt"
    torch.save(data, out_path)

    # 方便后续检查，同步拷贝当前使用的受体/肽 PDB
    if receptor_pdb and os.path.exists(receptor_pdb):
        dest = sample_dir / "receptor.pdb"
        if os.path.abspath(receptor_pdb) != os.path.abspath(dest):
            shutil.copy2(receptor_pdb, dest)
    if peptide_pdb and os.path.exists(peptide_pdb):
        dest = sample_dir / "peptide.pdb"
        if os.path.abspath(peptide_pdb) != os.path.abspath(dest):
            shutil.copy2(peptide_pdb, dest)
    return str(out_path)


def process_record(record: ComplexRecord, args) -> None:
    workspace = os.path.join(args.cache_dir, record.name)
    tmp_dir = os.path.join(workspace, "graphs")
    os.makedirs(tmp_dir, exist_ok=True)
    if record.receptor_pdb_path and record.peptide_pdb_path:
        receptor_pdb = record.receptor_pdb_path
        peptide_pdb = record.peptide_pdb_path
    else:
        receptor_pdb, peptide_pdb = split_complex(record, workspace)
    if not args.skip_validation:
        ok, reasons = validate_pdb_pair(
            Path(receptor_pdb),
            Path(peptide_pdb),
            max_min_pep_rec_atom_dist=args.max_min_pep_rec_atom_dist,
        )
        if not ok:
            raise ValueError(f"validation_failed:{';'.join(reasons)}")
    data = build_graph_from_files(
        record.name,
        receptor_pdb,
        peptide_pdb,
        tmp_dir,
        args.embedding == "esm",
        args.embedding == "esm",
        embedding=args.embedding,
    )
    save_path = save_graph(data, args.output_dir, receptor_pdb=receptor_pdb, peptide_pdb=peptide_pdb, embedding=args.embedding)
    if args.clean_cache:
        shutil.rmtree(workspace, ignore_errors=True)
    print(f"Saved {record.name} -> {save_path}")

    return save_path


def report_paths(output_dir: str, embedding: str, name: str):
    sample_dir = Path(output_dir) / name.lower()
    pt_path = sample_dir / f"features_{embedding}.pt"
    return str(sample_dir), str(pt_path)


def process_record_safe(record: ComplexRecord, args) -> dict:
    try:
        process_record(record, args)
        pdb_dir, pt_path = report_paths(args.output_dir, args.embedding, record.name)
        return {
            "complex_name": record.name.lower(),
            "ok": True,
            "pdb_dir": pdb_dir,
            "pt_path": pt_path,
            "error": "",
        }
    except Exception as exc:
        pdb_dir, pt_path = report_paths(args.output_dir, args.embedding, record.name)
        print(f"[ERROR] {record.name}: {exc}")
        return {
            "complex_name": record.name.lower(),
            "ok": False,
            "pdb_dir": pdb_dir,
            "pt_path": pt_path,
            "error": str(exc),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="从复合物PDB生成训练用.pt")
    parser.add_argument("--csv", required=True, help="包含复合物路径/链信息的CSV")
    parser.add_argument("--output_dir", required=True, help="保存.pt的目录")
    parser.add_argument("--cache_dir", default="preprocess_cache", help="拆分/临时文件目录")
    parser.add_argument("--embedding", choices=["onehot", "esm"], default="onehot", help="节点序列嵌入方式：onehot 或 esm")
    parser.add_argument(
        "--peptide_pose",
        choices=["native"],
        default="native",
        help="肽坐标策略固定为 native：直接使用输入 peptide.pdb。序列生成器入口已废弃。",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="跳过PDB质量校验（默认会按inspect_processed口径校验）",
    )
    parser.add_argument(
        "--max_min_pep_rec_atom_dist",
        type=float,
        default=8.0,
        help="肽-受体最小原子距离阈值（默认8Å，用于PDB质量校验）",
    )
    parser.add_argument("--clean_cache", action="store_true", help="保存pt后删除缓存")
    parser.add_argument("--num_workers", type=int, default=1, help="并行进程数，>1 时开启多进程处理（默认1）")
    parser.add_argument(
        "--report_csv",
        default=None,
        help="写出处理结果报告CSV（默认写到 <output_dir>/prepare_report.csv）",
    )
    parser.add_argument(
        "--available_csv",
        default=None,
        help='写出可用样本清单CSV（默认写到 <output_dir>/pt_available.csv）。若目标文件已存在，会先备份到 report_csv 同目录的 "pt_available.backup.csv"。',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    records = read_metadata(args.csv)

    results = []
    if args.num_workers > 1:
        from functools import partial
        from multiprocessing import Pool
        worker = partial(process_record_safe, args=args)
        with Pool(processes=args.num_workers, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(worker, records, chunksize=1):
                results.append(r)
    else:
        for record in records:
            results.append(process_record_safe(record, args))

    report_csv = args.report_csv or str(Path(args.output_dir) / "prepare_report.csv")
    with open(report_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["complex_name", "ok", "pdb_dir", "pt_path", "error"]
        )
        writer.writeheader()
        writer.writerows(results)

    available_csv = args.available_csv or str(Path(args.output_dir) / "pt_available.csv")
    if os.path.exists(available_csv):
        backup_csv = str(Path(report_csv).parent / "pt_available.backup.csv")
        try:
            shutil.copyfile(available_csv, backup_csv)
        except Exception as exc:
            print(f"[WARN] 备份 pt_available 失败：src={available_csv} dst={backup_csv} err={exc}")
    with open(available_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["complex_name", "pdb_dir"])
        writer.writeheader()
        for r in results:
            if r["ok"]:
                writer.writerow({"complex_name": r["complex_name"], "pdb_dir": r["pdb_dir"]})

    ok_n = sum(1 for r in results if r["ok"])
    print(f"[report] ok={ok_n}/{len(results)} report_csv={report_csv} available_csv={available_csv}")


if __name__ == "__main__":
    main()
