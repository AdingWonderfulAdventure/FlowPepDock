from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from ..build_record_snapshot import _filter_pose_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build index CSVs for an existing PoseCred-IPG record snapshot.")
    parser.add_argument("--snapshot_dir", type=Path, required=True)
    parser.add_argument("--variant_name", type=str, required=True)
    return parser.parse_args()


def _record_path(split_root: Path, row: pd.Series) -> Path:
    pose_id = f'{row["complex_name"]}__{row["pose_name"]}'
    return split_root / "records" / f"{pose_id}.npz"


def main() -> None:
    args = parse_args()
    snapshot_dir = args.snapshot_dir
    variant_root = snapshot_dir / args.variant_name
    meta = json.loads((snapshot_dir / "snapshot_meta.json").read_text(encoding="utf-8"))
    rmsd_col = meta["rmsd_col"]
    source_tables: Dict[str, str] = meta["source_tables"]

    for split, table_path_str in source_tables.items():
        split_root = variant_root / split
        table = _filter_pose_table(pd.read_csv(table_path_str))
        rows = []
        for row in table.itertuples(index=False):
            row_dict = row._asdict()
            rmsd_value = float(row_dict.get(rmsd_col, row_dict.get("complex_rmsd", 0.0)))
            record_path = _record_path(split_root, row_dict)
            rows.append(
                {
                    "record_path": str(record_path),
                    "pose_id": f'{row_dict["complex_name"]}__{row_dict["pose_name"]}',
                    "complex_id": str(row_dict["complex_name"]),
                    "group_id": str(row_dict["complex_name"]),
                    "source_group_id": str(row_dict["group_id"]),
                    "receptor_id": str(row_dict["receptor_id"]),
                    "peptide_id": str(row_dict.get("peptide_seq", row_dict["group_id"])),
                    "dockq": float(row_dict["dockq"]),
                    "rmsd": rmsd_value,
                }
            )
        index_df = pd.DataFrame(rows)
        index_df = index_df[pd.notna(index_df["dockq"]) & pd.notna(index_df["rmsd"])].reset_index(drop=True)
        out_path = split_root / "records_index.csv"
        index_df.to_csv(out_path, index=False)
        print(f"{split}\t{len(index_df)}\t{out_path}")


if __name__ == "__main__":
    main()
