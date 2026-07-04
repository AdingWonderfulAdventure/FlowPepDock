from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import PoseCredConfig
from ..features import (
    _build_edges,
    _candidate_pairs,
    _global_features,
    _node_features,
    _prune_candidates,
)
from ..pdbio import ResidueData, load_all_residues
from ..smallscale_experiment import DEFAULT_TRAIN_TABLE, _sample_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile PoseCred-IPG record build stages.")
    parser.add_argument("--pose_table", type=Path, default=DEFAULT_TRAIN_TABLE)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_groups", type=int, default=16)
    parser.add_argument("--poses_per_group", type=int, default=10)
    parser.add_argument("--min_poses_per_group", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--node_limit", type=int, default=128)
    parser.add_argument("--prune_strategy", type=str, default="hybrid")
    parser.add_argument("--save_profile_records", action="store_true")
    return parser.parse_args()


def _summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else 0.0,
        "std": float(array.std()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
        "p50": float(np.percentile(array, 50)) if array.size else 0.0,
        "p90": float(np.percentile(array, 90)) if array.size else 0.0,
    }


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config = PoseCredConfig(node_limit=args.node_limit, prune_strategy=args.prune_strategy)

    table = pd.read_csv(args.pose_table)
    subset = _sample_groups(
        table,
        num_groups=args.num_groups,
        poses_per_group=args.poses_per_group,
        min_poses_per_group=args.min_poses_per_group,
        seed=args.seed,
    )
    subset.to_csv(out_dir / "profile_subset.csv", index=False)

    receptor_cache: Dict[str, List[ResidueData]] = {}
    peptide_cache: Dict[str, List[ResidueData]] = {}

    stage_times: Dict[str, List[float]] = {
        "load_receptor_seconds": [],
        "load_peptide_seconds": [],
        "candidate_seconds": [],
        "prune_seconds": [],
        "global_seconds": [],
        "node_seconds": [],
        "edge_seconds": [],
        "total_seconds": [],
    }
    shape_rows: List[Dict[str, float]] = []

    for row in subset.itertuples(index=False):
        row_dict = row._asdict()
        receptor_pdb = Path(str(row_dict["receptor_pdb"]))
        peptide_pdb = Path(str(row_dict["pose_path"]))
        receptor_key = str(receptor_pdb)
        peptide_key = str(peptide_pdb)

        t0 = time.perf_counter()
        if receptor_key not in receptor_cache:
            receptor_cache[receptor_key] = load_all_residues(receptor_pdb)
        t1 = time.perf_counter()
        if peptide_key not in peptide_cache:
            peptide_cache[peptide_key] = load_all_residues(peptide_pdb)
        t2 = time.perf_counter()

        receptor_residues = receptor_cache[receptor_key]
        peptide_residues = peptide_cache[peptide_key]

        candidate_start = time.perf_counter()
        candidates = _candidate_pairs(receptor_residues, peptide_residues, config.candidate_distance_angstrom)
        candidate_end = time.perf_counter()

        prune_start = time.perf_counter()
        pruned = _prune_candidates(
            candidates,
            peptide_count=len(peptide_residues),
            node_limit=config.node_limit,
            strategy=config.prune_strategy,
        )
        prune_end = time.perf_counter()

        global_start = time.perf_counter()
        global_feat, clash_summary, physical_bad_label = _global_features(receptor_residues, peptide_residues, pruned)
        global_end = time.perf_counter()

        node_start = time.perf_counter()
        node_features = []
        node_pair_index = []
        for candidate in pruned:
            receptor_idx = int(candidate["receptor_index"])
            peptide_idx = int(candidate["peptide_index"])
            node_features.append(
                _node_features(
                    receptor_residues[receptor_idx],
                    peptide_residues[peptide_idx],
                    peptide_position=peptide_idx,
                    peptide_length=len(peptide_residues),
                    candidate=candidate,
                )
            )
            node_pair_index.append((receptor_idx, peptide_idx))
        node_end = time.perf_counter()

        edge_start = time.perf_counter()
        edge_index, edge_feat = _build_edges(pruned, config.max_neighbors)
        edge_end = time.perf_counter()

        stage_times["load_receptor_seconds"].append(t1 - t0)
        stage_times["load_peptide_seconds"].append(t2 - t1)
        stage_times["candidate_seconds"].append(candidate_end - candidate_start)
        stage_times["prune_seconds"].append(prune_end - prune_start)
        stage_times["global_seconds"].append(global_end - global_start)
        stage_times["node_seconds"].append(node_end - node_start)
        stage_times["edge_seconds"].append(edge_end - edge_start)
        stage_times["total_seconds"].append(edge_end - t0)

        shape_rows.append(
            {
                "pose_id": f'{row_dict["complex_name"]}__{row_dict["pose_name"]}',
                "candidate_count": len(candidates),
                "pruned_count": len(pruned),
                "edge_count": int(edge_index.shape[1]),
                "peptide_len": len(peptide_residues),
                "severe_clash_count": float(clash_summary["severe_clash_count"]),
                "physical_bad_label": int(physical_bad_label),
                "global_feat_dim": int(global_feat.shape[0]),
                "node_feat_dim": int(node_features[0].shape[0]) if node_features else 0,
                "edge_feat_dim": int(edge_feat.shape[1]) if edge_feat.size else 0,
            }
        )

    summary = {name: _summarize(values) for name, values in stage_times.items()}
    summary["config"] = {
        "node_limit": config.node_limit,
        "prune_strategy": config.prune_strategy,
        "candidate_distance_angstrom": config.candidate_distance_angstrom,
        "max_neighbors": config.max_neighbors,
    }
    summary["subset"] = {
        "pose_count": int(len(subset)),
        "group_count": int(subset["group_id"].nunique()),
        "seed": args.seed,
    }

    pd.DataFrame(shape_rows).to_csv(out_dir / "profile_shapes.csv", index=False)
    with (out_dir / "profile_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
