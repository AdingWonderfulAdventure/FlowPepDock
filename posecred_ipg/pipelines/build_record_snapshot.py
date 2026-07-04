from __future__ import annotations

import argparse
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from ..config import PoseCredConfig
from ..features import build_pose_record_from_pair_pdbs
from ..io import save_pose_record, write_manifest
from ..paths import POSECRED_ROOT, POSECRED_TRAIN_DOCKED_REL_TABLE, POSECRED_VAL_DOCKED_REL_TABLE
from ..pdbio import ResidueData, load_all_residues


DEFAULT_TRAIN_TABLE = POSECRED_TRAIN_DOCKED_REL_TABLE
DEFAULT_VAL_TABLE = POSECRED_VAL_DOCKED_REL_TABLE
DEFAULT_SNAPSHOT_ROOT = POSECRED_ROOT / "record_snapshots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build versioned PoseCred-IPG record snapshots from cocrystal-positive-only tables."
    )
    parser.add_argument("--snapshot_root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--snapshot_name", type=str, default="cocrystal_positive_only_20260320_v1")
    parser.add_argument("--train_pose_table", type=Path, default=DEFAULT_TRAIN_TABLE)
    parser.add_argument("--val_pose_table", type=Path, default=DEFAULT_VAL_TABLE)
    parser.add_argument("--splits", type=str, default="train,val")
    parser.add_argument("--node_limits", type=str, default="32,128")
    parser.add_argument("--prune_strategy", type=str, default="hybrid")
    parser.add_argument("--rmsd_col", type=str, default="peptide_ca_rmsd")
    parser.add_argument("--limit_per_split", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    return parser.parse_args()


def _filter_pose_table(table: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "complex_name",
        "group_id",
        "receptor_id",
        "pose_name",
        "pose_path",
        "receptor_pdb",
        "dockq",
        "peptide_seq",
    ]
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        raise ValueError(f"pose_table missing required columns: {missing}")
    filtered = table.copy()
    if "positive_type" in filtered.columns:
        filtered = filtered[filtered["positive_type"].astype(str).isin(["docked", "native"])]
    if "pair_type" in filtered.columns:
        filtered = filtered[filtered["pair_type"].astype(str) == "native"]
    filtered = filtered[pd.notna(filtered["dockq"])]
    if "peptide_ca_rmsd" in filtered.columns:
        filtered = filtered[pd.notna(filtered["peptide_ca_rmsd"])]
    elif "complex_rmsd" in filtered.columns:
        filtered = filtered[pd.notna(filtered["complex_rmsd"])]
    return filtered.reset_index(drop=True)


def _build_records(
    table: pd.DataFrame,
    output_dir: Path,
    config: PoseCredConfig,
    rmsd_col: str,
    receptor_cache: Dict[str, List[ResidueData]],
    peptide_cache: Dict[str, List[ResidueData]],
) -> Tuple[Path, int]:
    record_paths: List[Path] = []
    for row in table.itertuples(index=False):
        row_dict = row._asdict()
        receptor_pdb = Path(str(row_dict["receptor_pdb"]))
        peptide_pdb = Path(str(row_dict["pose_path"]))
        receptor_key = str(receptor_pdb)
        peptide_key = str(peptide_pdb)
        if receptor_key not in receptor_cache:
            receptor_cache[receptor_key] = load_all_residues(receptor_pdb)
        if peptide_key not in peptide_cache:
            peptide_cache[peptide_key] = load_all_residues(peptide_pdb)
        rmsd_value = float(row_dict.get(rmsd_col, row_dict.get("complex_rmsd", 0.0)))
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
        record_paths.append(save_pose_record(pose_record, output_dir / "records"))
    manifest_path = output_dir / "manifest.txt"
    write_manifest(record_paths, manifest_path)
    return manifest_path, len(record_paths)


def _record_output_path(output_dir: Path, complex_name: str, pose_name: str) -> Path:
    return output_dir / "records" / f"{complex_name}__{pose_name}.npz"


def _build_single_record(
    row_dict: Dict[str, object],
    output_dir: Path,
    config_payload: Dict[str, object],
    rmsd_col: str,
    skip_existing: bool,
) -> str:
    output_path = _record_output_path(output_dir, str(row_dict["complex_name"]), str(row_dict["pose_name"]))
    if skip_existing and output_path.exists():
        return str(output_path)

    receptor_pdb = Path(str(row_dict["receptor_pdb"]))
    peptide_pdb = Path(str(row_dict["pose_path"]))
    rmsd_value = float(row_dict.get(rmsd_col, row_dict.get("complex_rmsd", 0.0)))
    config = PoseCredConfig(**config_payload)
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
    )
    return str(save_pose_record(pose_record, output_dir / "records"))


def _build_records_parallel(
    table: pd.DataFrame,
    output_dir: Path,
    config: PoseCredConfig,
    rmsd_col: str,
    skip_existing: bool,
    num_workers: int,
) -> Tuple[Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row._asdict() for row in table.itertuples(index=False)]
    config_payload = asdict(config)
    record_paths: List[Path] = []

    if num_workers <= 1:
        receptor_cache: Dict[str, List[ResidueData]] = {}
        peptide_cache: Dict[str, List[ResidueData]] = {}
        manifest_path, record_count = _build_records(
            table=table,
            output_dir=output_dir,
            config=config,
            rmsd_col=rmsd_col,
            receptor_cache=receptor_cache,
            peptide_cache=peptide_cache,
        )
        if skip_existing:
            # Single-process fallback should still resume from already-built records.
            record_paths = sorted((output_dir / "records").glob("*.npz"))
            write_manifest(record_paths, manifest_path)
            return manifest_path, len(record_paths)
        return manifest_path, record_count

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _build_single_record,
                row_dict,
                output_dir,
                config_payload,
                rmsd_col,
                skip_existing,
            )
            for row_dict in rows
        ]
        for future in futures:
            record_paths.append(Path(future.result()))

    manifest_path = output_dir / "manifest.txt"
    if skip_existing:
        record_paths = sorted((output_dir / "records").glob("*.npz"))
    else:
        record_paths = sorted(record_paths)
    write_manifest(record_paths, manifest_path)
    return manifest_path, len(record_paths)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _resolve_split_tables(args: argparse.Namespace) -> Dict[str, Path]:
    tables = {"train": args.train_pose_table, "val": args.val_pose_table}
    requested = [token.strip() for token in args.splits.split(",") if token.strip()]
    unknown = [split for split in requested if split not in tables]
    if unknown:
        raise ValueError(f"Unknown splits requested: {unknown}")
    return {split: tables[split] for split in requested}


def main() -> None:
    args = parse_args()
    snapshot_root = args.snapshot_root / args.snapshot_name
    snapshot_root.mkdir(parents=True, exist_ok=True)

    split_tables = _resolve_split_tables(args)
    node_limits = [int(token.strip()) for token in args.node_limits.split(",") if token.strip()]
    build_rows: List[Dict[str, object]] = []

    snapshot_meta = {
        "snapshot_name": args.snapshot_name,
        "snapshot_root": str(snapshot_root),
        "splits": list(split_tables.keys()),
        "node_limits": node_limits,
        "prune_strategy": args.prune_strategy,
        "rmsd_col": args.rmsd_col,
        "limit_per_split": args.limit_per_split,
        "source_tables": {name: str(path) for name, path in split_tables.items()},
    }
    _write_json(snapshot_root / "snapshot_meta.json", snapshot_meta)

    for node_limit in node_limits:
        config = PoseCredConfig(node_limit=node_limit, prune_strategy=args.prune_strategy)
        variant_name = f"hybrid_npair{node_limit}" if args.prune_strategy == "hybrid" else f"{args.prune_strategy}_npair{node_limit}"
        variant_root = snapshot_root / variant_name
        _write_json(
            variant_root / "config_snapshot.json",
            {
                "snapshot_name": args.snapshot_name,
                "variant_name": variant_name,
                "config": asdict(config),
            },
        )
        for split, table_path in split_tables.items():
            split_root = variant_root / split
            manifest_path = split_root / "manifest.txt"
            build_meta_path = split_root / "build_meta.json"
            if args.skip_existing and manifest_path.exists() and build_meta_path.exists():
                existing_count = sum(1 for _ in manifest_path.open("r", encoding="utf-8") if _.strip())
                build_rows.append(
                    {
                        "variant_name": variant_name,
                        "split": split,
                        "record_count": existing_count,
                        "build_seconds": 0.0,
                        "status": "skipped_existing",
                        "manifest_path": str(manifest_path),
                    }
                )
                continue

            table = _filter_pose_table(pd.read_csv(table_path))
            if args.limit_per_split > 0:
                table = table.iloc[: args.limit_per_split].copy()

            split_root.mkdir(parents=True, exist_ok=True)
            build_start = time.perf_counter()
            manifest_path, record_count = _build_records_parallel(
                table=table,
                output_dir=split_root,
                config=config,
                rmsd_col=args.rmsd_col,
                skip_existing=args.skip_existing,
                num_workers=args.num_workers,
            )
            build_seconds = time.perf_counter() - build_start
            build_meta = {
                "snapshot_name": args.snapshot_name,
                "variant_name": variant_name,
                "split": split,
                "source_table": str(table_path),
                "record_count": record_count,
                "group_count": int(table["group_id"].nunique()),
                "mean_dockq": float(table["dockq"].mean()),
                "mean_rmsd": float(
                    table[args.rmsd_col].mean()
                    if args.rmsd_col in table.columns
                    else table.get("complex_rmsd", pd.Series([0.0])).mean()
                ),
                "build_seconds": build_seconds,
                "seconds_per_pose": build_seconds / max(record_count, 1),
                "manifest_path": str(manifest_path),
                "num_workers": args.num_workers,
                "config": asdict(config),
            }
            _write_json(split_root / "build_meta.json", build_meta)
            build_rows.append(
                {
                    "variant_name": variant_name,
                    "split": split,
                    "record_count": record_count,
                    "build_seconds": build_seconds,
                    "status": "built",
                    "manifest_path": str(manifest_path),
                }
            )

    build_df = pd.DataFrame(build_rows)
    build_df.to_csv(snapshot_root / "build_summary.csv", index=False)
    print(build_df.to_string(index=False))


if __name__ == "__main__":
    main()
