from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from Bio.PDB import PDBParser

from ..constants import AA_VDW_RADII


@dataclass
class ResidueData:
    chain_id: str
    residue_number: int
    insertion_code: str
    resname: str
    atom_names: List[str]
    atom_coords: np.ndarray
    atom_elements: List[str]
    atom_vdw_radii: np.ndarray
    ca_coord: np.ndarray
    cb_coord: np.ndarray
    centroid: np.ndarray
    extent: float


def _infer_element(atom_name: str) -> str:
    atom_name = atom_name.strip()
    if not atom_name:
        return "C"
    return atom_name[0].upper()


def load_residues(pdb_path: Path, chain_ids: Iterable[str]) -> List[ResidueData]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pose", str(pdb_path))
    chain_ids = set(chain_ids)
    residues: List[ResidueData] = []
    for model in structure:
        for chain in model:
            if chain_ids and chain.id not in chain_ids:
                continue
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                atom_names: List[str] = []
                atom_coords: List[np.ndarray] = []
                atom_elements: List[str] = []
                for atom in residue:
                    if atom.element == "H" or atom.get_name().startswith("H"):
                        continue
                    atom_names.append(atom.get_name())
                    atom_coords.append(np.asarray(atom.coord, dtype=np.float32))
                    atom_elements.append(atom.element.strip().upper() or _infer_element(atom.get_name()))
                if not atom_coords:
                    continue
                coords = np.stack(atom_coords, axis=0)
                atom_vdw_radii = np.asarray(
                    [AA_VDW_RADII.get(element, 1.7) for element in atom_elements],
                    dtype=np.float32,
                )
                ca_coord = coords[0]
                cb_coord = coords[0]
                for name, coord in zip(atom_names, coords):
                    if name == "CA":
                        ca_coord = coord
                    if name == "CB":
                        cb_coord = coord
                centroid = coords.mean(axis=0).astype(np.float32)
                extent = float(np.linalg.norm(coords - centroid[None, :], axis=1).max())
                residues.append(
                    ResidueData(
                        chain_id=chain.id,
                        residue_number=int(residue.id[1]),
                        insertion_code=str(residue.id[2]).strip(),
                        resname=residue.resname.strip().upper(),
                        atom_names=atom_names,
                        atom_coords=coords,
                        atom_elements=atom_elements,
                        atom_vdw_radii=atom_vdw_radii,
                        ca_coord=np.asarray(ca_coord, dtype=np.float32),
                        cb_coord=np.asarray(cb_coord, dtype=np.float32),
                        centroid=centroid,
                        extent=extent,
                    )
                )
        break
    return residues


def load_all_residues(pdb_path: Path) -> List[ResidueData]:
    return load_residues(pdb_path, chain_ids=())


def split_chains_by_size(pdb_path: Path) -> Tuple[List[str], List[str]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pose", str(pdb_path))
    chain_lengths: Dict[str, int] = {}
    for model in structure:
        for chain in model:
            count = sum(1 for residue in chain if residue.id[0] == " ")
            if count > 0:
                chain_lengths[chain.id] = count
        break
    if len(chain_lengths) < 2:
        raise ValueError(f"expected at least two chains in {pdb_path}")
    sorted_chains = sorted(chain_lengths.items(), key=lambda item: item[1], reverse=True)
    receptor = [sorted_chains[0][0]]
    peptide = [sorted_chains[-1][0]]
    return receptor, peptide
