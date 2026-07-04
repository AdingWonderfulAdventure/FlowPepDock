from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import PoseCredConfig
from ..features import build_pose_record
from ..io import save_pose_record, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PoseCred-IPG pose records from a CSV table.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--pdb_col", type=str, default="pdb_path")
    parser.add_argument("--pose_id_col", type=str, default="pose_id")
    parser.add_argument("--complex_id_col", type=str, default="complex_id")
    parser.add_argument("--group_id_col", type=str, default="group_id")
    parser.add_argument("--receptor_id_col", type=str, default="receptor_id")
    parser.add_argument("--peptide_id_col", type=str, default="peptide_id")
    parser.add_argument("--dockq_col", type=str, default="DockQ")
    parser.add_argument("--rmsd_col", type=str, default="RMSD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.input_csv)
    config = PoseCredConfig()
    output_paths = []
    for row in table.itertuples(index=False):
        row_dict = row._asdict()
        pdb_path = Path(row_dict[args.pdb_col])
        pose_record = build_pose_record(
            pdb_path=pdb_path,
            pose_id=str(row_dict[args.pose_id_col]),
            complex_id=str(row_dict[args.complex_id_col]),
            group_id=str(row_dict[args.group_id_col]),
            receptor_id=str(row_dict[args.receptor_id_col]),
            peptide_id=str(row_dict[args.peptide_id_col]),
            dockq=float(row_dict[args.dockq_col]),
            rmsd=float(row_dict[args.rmsd_col]),
            config=config,
        )
        output_paths.append(save_pose_record(pose_record, args.output_dir))
    write_manifest(output_paths, args.manifest_path)


if __name__ == "__main__":
    main()
