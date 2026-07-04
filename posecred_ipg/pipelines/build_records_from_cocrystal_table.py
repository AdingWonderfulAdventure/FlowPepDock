from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..config import PoseCredConfig
from ..features import build_pose_record_from_pair_pdbs
from ..io import save_pose_record, write_manifest
from ..pdbio import ResidueData, load_all_residues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PoseCred-IPG records from cocrystal-positive-only pose tables."
    )
    parser.add_argument("--pose_table", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--rmsd_col", type=str, default="peptide_ca_rmsd")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.pose_table)
    required_columns = ["complex_name", "group_id", "receptor_id", "pose_name", "pose_path", "receptor_pdb", "dockq", "peptide_seq"]
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        raise ValueError(f"pose_table missing required columns: {missing}")

    table = table.copy()
    if "positive_type" in table.columns:
        table = table[table["positive_type"].astype(str).isin(["docked", "native"])]
    if "pair_type" in table.columns:
        table = table[table["pair_type"].astype(str) == "native"]
    table = table[pd.notna(table["dockq"])]
    if args.rmsd_col in table.columns:
        table = table[pd.notna(table[args.rmsd_col])]
    elif "complex_rmsd" in table.columns:
        table = table[pd.notna(table["complex_rmsd"])]
    table = table.reset_index(drop=True)
    if args.limit > 0:
        table = table.iloc[: args.limit].copy()

    config = PoseCredConfig()
    record_paths: List[Path] = []
    receptor_cache: Dict[str, List[ResidueData]] = {}
    peptide_cache: Dict[str, List[ResidueData]] = {}
    for row in table.itertuples(index=False):
        row_dict = row._asdict()
        rmsd_value = float(row_dict.get(args.rmsd_col, row_dict.get("complex_rmsd", 0.0)))
        receptor_pdb = Path(str(row_dict["receptor_pdb"]))
        peptide_pdb = Path(str(row_dict["pose_path"]))
        receptor_key = str(receptor_pdb)
        peptide_key = str(peptide_pdb)
        if receptor_key not in receptor_cache:
            receptor_cache[receptor_key] = load_all_residues(receptor_pdb)
        if peptide_key not in peptide_cache:
            peptide_cache[peptide_key] = load_all_residues(peptide_pdb)
        pose_record = build_pose_record_from_pair_pdbs(
            receptor_pdb=receptor_pdb,
            peptide_pdb=peptide_pdb,
            pose_id=f'{row_dict["complex_name"]}__{row_dict["pose_name"]}',
            complex_id=str(row_dict["complex_name"]),
            group_id=str(row_dict["group_id"]),
            receptor_id=str(row_dict["receptor_id"]),
            peptide_id=str(row_dict["group_id"]),
            dockq=float(row_dict["dockq"]),
            rmsd=rmsd_value,
            config=config,
            receptor_residues=receptor_cache[receptor_key],
            peptide_residues=peptide_cache[peptide_key],
        )
        record_paths.append(save_pose_record(pose_record, args.output_dir))
    write_manifest(record_paths, args.manifest_path)


if __name__ == "__main__":
    main()
