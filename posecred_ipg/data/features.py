from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from ..config import PoseCredConfig
from ..constants import AA_TO_INDEX, AROMATIC, HYDROPHOBIC, NEGATIVE, POLAR, POSITIVE
from ..pdbio import ResidueData, load_all_residues, load_residues, split_chains_by_size
from ..records import PoseRecord


def _one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    out[index] = 1.0
    return out


def _safe_unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.zeros_like(vector)
    return vector / norm


def _residue_flags(resname: str) -> np.ndarray:
    return np.asarray(
        [
            float(resname in HYDROPHOBIC),
            float(resname in POLAR),
            float(resname in POSITIVE),
            float(resname in NEGATIVE),
            float(resname in AROMATIC),
        ],
        dtype=np.float32,
    )


def _pairwise_min_distance(res_a: ResidueData, res_b: ResidueData) -> Tuple[float, float, float]:
    dist = cdist(res_a.atom_coords, res_b.atom_coords, metric="euclidean")
    min_index = np.unravel_index(np.argmin(dist), dist.shape)
    vdw_a = float(res_a.atom_vdw_radii[min_index[0]])
    vdw_b = float(res_b.atom_vdw_radii[min_index[1]])
    min_dist = float(dist[min_index])
    overlap = max(0.0, vdw_a + vdw_b - min_dist)
    return min_dist, float(overlap), float(dist.mean())


def _residue_extent(residue: ResidueData) -> float:
    return float(residue.extent)


def _charge_code(resname_a: str, resname_b: str) -> np.ndarray:
    pos_a = resname_a in POSITIVE
    neg_a = resname_a in NEGATIVE
    pos_b = resname_b in POSITIVE
    neg_b = resname_b in NEGATIVE
    attractive = float((pos_a and neg_b) or (neg_a and pos_b))
    repulsive = float((pos_a and pos_b) or (neg_a and neg_b))
    return np.asarray([attractive, repulsive], dtype=np.float32)


def _candidate_pairs(
    receptor_residues: Sequence[ResidueData],
    peptide_residues: Sequence[ResidueData],
    cutoff: float,
) -> List[Dict[str, float]]:
    candidates: List[Dict[str, float]] = []
    if not receptor_residues or not peptide_residues:
        return candidates

    receptor_centroids = np.stack([residue.centroid for residue in receptor_residues], axis=0).astype(np.float32)
    peptide_centroids = np.stack([residue.centroid for residue in peptide_residues], axis=0).astype(np.float32)
    receptor_extents = np.asarray([_residue_extent(residue) for residue in receptor_residues], dtype=np.float32)
    peptide_extents = np.asarray([_residue_extent(residue) for residue in peptide_residues], dtype=np.float32)

    centroid_delta = receptor_centroids[:, None, :] - peptide_centroids[None, :, :]
    centroid_dist = np.linalg.norm(centroid_delta, axis=-1)
    lower_bound = centroid_dist - receptor_extents[:, None] - peptide_extents[None, :]
    candidate_mask = lower_bound < cutoff
    receptor_indices, peptide_indices = np.nonzero(candidate_mask)

    for receptor_index, peptide_index in zip(receptor_indices.tolist(), peptide_indices.tolist()):
        receptor_residue = receptor_residues[receptor_index]
        peptide_residue = peptide_residues[peptide_index]
        min_dist, max_overlap, mean_dist = _pairwise_min_distance(receptor_residue, peptide_residue)
        if min_dist >= cutoff:
            continue
        pair_center = 0.5 * (receptor_residue.centroid + peptide_residue.centroid)
        local_density = 1.0 / max(min_dist, 1.0)
        clash_flag = float(max_overlap > 0.0 or min_dist < 1.8)
        candidates.append(
            {
                "receptor_index": receptor_index,
                "peptide_index": peptide_index,
                "min_dist": min_dist,
                "max_overlap": max_overlap,
                "mean_dist": mean_dist,
                "center_x": float(pair_center[0]),
                "center_y": float(pair_center[1]),
                "center_z": float(pair_center[2]),
                "density": local_density,
                "clash_flag": clash_flag,
            }
        )
    return candidates


def _prune_candidates(
    candidates: List[Dict[str, float]],
    peptide_count: int,
    node_limit: int,
    strategy: str,
) -> List[Dict[str, float]]:
    if len(candidates) <= node_limit:
        return candidates

    if strategy == "distance_only":
        return sorted(candidates, key=lambda item: (item["min_dist"], -item["density"]))[:node_limit]

    if strategy == "distance_clash":
        selected: List[Dict[str, float]] = []
        seen_keys = set()
        by_distance = sorted(candidates, key=lambda item: item["min_dist"])
        for item in by_distance[: min(node_limit * 3 // 4, len(by_distance))]:
            key = (item["receptor_index"], item["peptide_index"])
            selected.append(item)
            seen_keys.add(key)
        clash_candidates = [item for item in candidates if item["clash_flag"] > 0]
        clash_candidates.sort(key=lambda item: (item["min_dist"], -item["max_overlap"]))
        for item in clash_candidates:
            if len(selected) >= node_limit:
                break
            key = (item["receptor_index"], item["peptide_index"])
            if key not in seen_keys:
                selected.append(item)
                seen_keys.add(key)
        if len(selected) < node_limit:
            remaining = [item for item in candidates if (item["receptor_index"], item["peptide_index"]) not in seen_keys]
            remaining.sort(key=lambda item: (item["min_dist"], -item["density"]))
            selected.extend(remaining[: node_limit - len(selected)])
        return selected[:node_limit]

    if strategy != "hybrid":
        raise ValueError(f"unknown prune strategy: {strategy}")

    selected: List[Dict[str, float]] = []
    seen_keys = set()

    by_distance = sorted(candidates, key=lambda item: item["min_dist"])
    for item in by_distance[: min(node_limit // 2, len(by_distance))]:
        key = (item["receptor_index"], item["peptide_index"])
        selected.append(item)
        seen_keys.add(key)

    clash_candidates = [item for item in candidates if item["clash_flag"] > 0]
    clash_candidates.sort(key=lambda item: (item["min_dist"], -item["max_overlap"]))
    for item in clash_candidates:
        if len(selected) >= min(node_limit, len(by_distance) + len(clash_candidates)):
            break
        key = (item["receptor_index"], item["peptide_index"])
        if key not in seen_keys:
            selected.append(item)
            seen_keys.add(key)

    covered_peptide = {item["peptide_index"] for item in selected}
    if len(covered_peptide) < peptide_count:
        best_by_peptide: Dict[int, Dict[str, float]] = {}
        for item in candidates:
            peptide_index = int(item["peptide_index"])
            best = best_by_peptide.get(peptide_index)
            if best is None or item["min_dist"] < best["min_dist"]:
                best_by_peptide[peptide_index] = item
        for peptide_index in range(peptide_count):
            item = best_by_peptide.get(peptide_index)
            if item is None:
                continue
            key = (item["receptor_index"], item["peptide_index"])
            if key not in seen_keys:
                selected.append(item)
                seen_keys.add(key)
                covered_peptide.add(peptide_index)
            if len(selected) >= node_limit:
                break

    if len(selected) < node_limit:
        remaining = [item for item in candidates if (item["receptor_index"], item["peptide_index"]) not in seen_keys]
        remaining.sort(key=lambda item: (item["min_dist"], -item["density"]))
        selected.extend(remaining[: node_limit - len(selected)])

    return selected[:node_limit]


def _node_features(
    receptor: ResidueData,
    peptide: ResidueData,
    peptide_position: int,
    peptide_length: int,
    candidate: Dict[str, float],
) -> np.ndarray:
    receptor_aa = _one_hot(AA_TO_INDEX.get(receptor.resname, AA_TO_INDEX["UNK"]), len(AA_TO_INDEX))
    peptide_aa = _one_hot(AA_TO_INDEX.get(peptide.resname, AA_TO_INDEX["UNK"]), len(AA_TO_INDEX))
    receptor_flags = _residue_flags(receptor.resname)
    peptide_flags = _residue_flags(peptide.resname)
    ca_vector = peptide.ca_coord - receptor.ca_coord
    cb_vector = peptide.cb_coord - receptor.cb_coord
    centroid_vector = peptide.centroid - receptor.centroid
    geometry = np.asarray(
        [
            np.linalg.norm(ca_vector),
            np.linalg.norm(cb_vector),
            candidate["min_dist"],
            candidate["mean_dist"],
            candidate["max_overlap"],
            np.dot(_safe_unit(ca_vector), _safe_unit(cb_vector)),
            peptide_position / max(peptide_length - 1, 1),
        ],
        dtype=np.float32,
    )
    chemistry = np.concatenate(
        [
            _charge_code(receptor.resname, peptide.resname),
            np.asarray(
                [
                    float(receptor.resname in HYDROPHOBIC and peptide.resname in HYDROPHOBIC),
                    float(receptor.resname in AROMATIC and peptide.resname in AROMATIC),
                    float(candidate["density"]),
                ],
                dtype=np.float32,
            ),
        ]
    )
    clash = np.asarray(
        [
            float(candidate["clash_flag"]),
            float(candidate["min_dist"] < 1.8),
            candidate["max_overlap"],
        ],
        dtype=np.float32,
    )
    direction = centroid_vector.astype(np.float32) / max(np.linalg.norm(centroid_vector), 1.0)
    return np.concatenate([receptor_aa, peptide_aa, receptor_flags, peptide_flags, geometry, chemistry, clash, direction]).astype(
        np.float32
    )


def _edge_features(candidate_a: Dict[str, float], candidate_b: Dict[str, float]) -> Tuple[bool, np.ndarray]:
    same_receptor = int(candidate_a["receptor_index"] == candidate_b["receptor_index"])
    same_peptide = int(candidate_a["peptide_index"] == candidate_b["peptide_index"])
    center_a = np.asarray([candidate_a["center_x"], candidate_a["center_y"], candidate_a["center_z"]], dtype=np.float32)
    center_b = np.asarray([candidate_b["center_x"], candidate_b["center_y"], candidate_b["center_z"]], dtype=np.float32)
    distance = float(np.linalg.norm(center_a - center_b))
    if not (same_receptor or same_peptide or distance < 8.0):
        return False, np.zeros(6, dtype=np.float32)
    feature = np.asarray(
        [
            float(same_receptor),
            float(same_peptide),
            float(distance < 8.0),
            distance,
            float(abs(candidate_a["peptide_index"] - candidate_b["peptide_index"])),
            float(abs(candidate_a["receptor_index"] - candidate_b["receptor_index"])),
        ],
        dtype=np.float32,
    )
    return True, feature


def _build_edges(candidates: Sequence[Dict[str, float]], max_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    num_candidates = len(candidates)
    if num_candidates == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 6), dtype=np.float32)

    receptor_index = np.asarray([int(item["receptor_index"]) for item in candidates], dtype=np.int32)
    peptide_index = np.asarray([int(item["peptide_index"]) for item in candidates], dtype=np.int32)
    centers = np.asarray(
        [[item["center_x"], item["center_y"], item["center_z"]] for item in candidates],
        dtype=np.float32,
    )

    center_delta = centers[:, None, :] - centers[None, :, :]
    distance = np.linalg.norm(center_delta, axis=-1).astype(np.float32)
    same_receptor = receptor_index[:, None] == receptor_index[None, :]
    same_peptide = peptide_index[:, None] == peptide_index[None, :]
    spatial_neighbor = distance < 8.0
    keep_mask = (same_receptor | same_peptide | spatial_neighbor) & (~np.eye(num_candidates, dtype=bool))

    edge_pairs: List[Tuple[int, int]] = []
    edge_features: List[np.ndarray] = []
    for src in range(num_candidates):
        neighbor_idx = np.flatnonzero(keep_mask[src])
        if neighbor_idx.size == 0:
            continue
        if neighbor_idx.size > max_neighbors:
            order = np.argsort(distance[src, neighbor_idx], kind="stable")[:max_neighbors]
            neighbor_idx = neighbor_idx[order]
        for dst in neighbor_idx.tolist():
            feat = np.asarray(
                [
                    float(same_receptor[src, dst]),
                    float(same_peptide[src, dst]),
                    float(spatial_neighbor[src, dst]),
                    float(distance[src, dst]),
                    float(abs(int(peptide_index[src]) - int(peptide_index[dst]))),
                    float(abs(int(receptor_index[src]) - int(receptor_index[dst]))),
                ],
                dtype=np.float32,
            )
            edge_pairs.append((src, dst))
            edge_features.append(feat)

    if not edge_pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 6), dtype=np.float32)
    edge_index = np.asarray(edge_pairs, dtype=np.int64).T
    edge_feat = np.stack(edge_features, axis=0).astype(np.float32)
    return edge_index, edge_feat


def _global_features(
    receptor_residues: Sequence[ResidueData],
    peptide_residues: Sequence[ResidueData],
    candidates: Sequence[Dict[str, float]],
) -> Tuple[np.ndarray, Dict[str, float], int]:
    severe_clash_count = float(sum(item["min_dist"] < 1.6 for item in candidates))
    mild_clash_count = float(sum(item["max_overlap"] > 0.0 for item in candidates))
    bb_bb_clash_count = 0.0
    sc_bb_clash_count = severe_clash_count
    min_heavy_dist = float(min((item["min_dist"] for item in candidates), default=99.0))
    max_overlap = float(max((item["max_overlap"] for item in candidates), default=0.0))
    mean_overlap = float(np.mean([item["max_overlap"] for item in candidates])) if candidates else 0.0
    steric_overcrowding = float(sum(item["density"] > 0.5 for item in candidates))
    hydrophobic_ratio = float(
        np.mean(
            [
                receptor_residues[int(item["receptor_index"])].resname in HYDROPHOBIC
                and peptide_residues[int(item["peptide_index"])].resname in HYDROPHOBIC
                for item in candidates
            ]
        )
    ) if candidates else 0.0
    charge_ratio = float(
        np.mean(
            [
                _charge_code(
                    receptor_residues[int(item["receptor_index"])].resname,
                    peptide_residues[int(item["peptide_index"])].resname,
                )[0]
                for item in candidates
            ]
        )
    ) if candidates else 0.0
    global_feat = np.asarray(
        [
            float(len(peptide_residues)),
            float(len(candidates)),
            hydrophobic_ratio,
            charge_ratio,
            severe_clash_count,
            mild_clash_count,
            bb_bb_clash_count,
            sc_bb_clash_count,
            min_heavy_dist,
            max_overlap,
            mean_overlap,
            steric_overcrowding,
        ],
        dtype=np.float32,
    )
    clash_summary = {
        "severe_clash_count": severe_clash_count,
        "mild_clash_count": mild_clash_count,
        "bb_bb_clash_count": bb_bb_clash_count,
        "sc_bb_clash_count": sc_bb_clash_count,
        "sc_sc_clash_count": 0.0,
        "min_heavy_dist": min_heavy_dist,
        "max_vdw_overlap": max_overlap,
        "mean_vdw_overlap": mean_overlap,
        "steric_overcrowding": steric_overcrowding,
    }
    bad_label = int(
        severe_clash_count >= 8.0 or min_heavy_dist <= 1.6 or max_overlap >= 0.7 or bb_bb_clash_count >= 2.0 or steric_overcrowding >= 6.0
    )
    return global_feat, clash_summary, bad_label


def _build_pose_record_from_residue_lists(
    receptor_residues: Sequence[ResidueData],
    peptide_residues: Sequence[ResidueData],
    pose_id: str,
    complex_id: str,
    group_id: str,
    receptor_id: str,
    peptide_id: str,
    dockq: float,
    rmsd: float,
    config: PoseCredConfig,
) -> PoseRecord:
    candidates = _candidate_pairs(receptor_residues, peptide_residues, config.candidate_distance_angstrom)
    candidates = _prune_candidates(
        candidates,
        peptide_count=len(peptide_residues),
        node_limit=config.node_limit,
        strategy=config.prune_strategy,
    )
    global_feat, clash_summary, physical_bad_label = _global_features(receptor_residues, peptide_residues, candidates)

    node_features: List[np.ndarray] = []
    node_pair_index: List[Tuple[int, int]] = []
    for candidate in candidates:
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
    if node_features:
        node_feat = np.stack(node_features, axis=0).astype(np.float32)
        pair_index = np.asarray(node_pair_index, dtype=np.int32)
    else:
        node_feat = np.zeros((0, len(AA_TO_INDEX) * 2 + 5 * 2 + 7 + 5 + 3 + 3), dtype=np.float32)
        pair_index = np.zeros((0, 2), dtype=np.int32)
    edge_index, edge_feat = _build_edges(candidates, config.max_neighbors)
    return PoseRecord(
        pose_id=pose_id,
        complex_id=complex_id,
        group_id=group_id,
        receptor_id=receptor_id,
        peptide_id=peptide_id,
        dockq=float(dockq),
        rmsd=float(rmsd),
        physical_bad_label=physical_bad_label,
        global_feat=global_feat,
        node_feat=node_feat,
        edge_index=edge_index,
        edge_feat=edge_feat,
        node_pair_index=pair_index,
        clash_summary=clash_summary,
    )


def build_pose_record(
    pdb_path: Path,
    pose_id: str,
    complex_id: str,
    group_id: str,
    receptor_id: str,
    peptide_id: str,
    dockq: float,
    rmsd: float,
    config: PoseCredConfig,
) -> PoseRecord:
    receptor_chains = config.receptor_chain_ids
    peptide_chains = config.peptide_chain_ids
    if not receptor_chains or not peptide_chains:
        receptor_chains, peptide_chains = split_chains_by_size(pdb_path)
    receptor_residues = load_residues(pdb_path, receptor_chains)
    peptide_residues = load_residues(pdb_path, peptide_chains)
    return _build_pose_record_from_residue_lists(
        receptor_residues=receptor_residues,
        peptide_residues=peptide_residues,
        pose_id=pose_id,
        complex_id=complex_id,
        group_id=group_id,
        receptor_id=receptor_id,
        peptide_id=peptide_id,
        dockq=dockq,
        rmsd=rmsd,
        config=config,
    )


def build_pose_record_from_pair_pdbs(
    receptor_pdb: Path,
    peptide_pdb: Path,
    pose_id: str,
    complex_id: str,
    group_id: str,
    receptor_id: str,
    peptide_id: str,
    dockq: float,
    rmsd: float,
    config: PoseCredConfig,
    receptor_residues: Optional[Sequence[ResidueData]] = None,
    peptide_residues: Optional[Sequence[ResidueData]] = None,
) -> PoseRecord:
    loaded_receptor_residues = list(receptor_residues) if receptor_residues is not None else load_all_residues(receptor_pdb)
    loaded_peptide_residues = list(peptide_residues) if peptide_residues is not None else load_all_residues(peptide_pdb)
    return _build_pose_record_from_residue_lists(
        receptor_residues=loaded_receptor_residues,
        peptide_residues=loaded_peptide_residues,
        pose_id=pose_id,
        complex_id=complex_id,
        group_id=group_id,
        receptor_id=receptor_id,
        peptide_id=peptide_id,
        dockq=dockq,
        rmsd=rmsd,
        config=config,
    )
