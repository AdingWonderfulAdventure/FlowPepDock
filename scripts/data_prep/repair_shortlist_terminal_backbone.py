#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


BACKBONE = ("N", "CA", "C", "O")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared_root",
        default="data/rcsb_novel_hot_peptide_shortlist_prepared_20260427",
    )
    parser.add_argument(
        "--output_subdir",
        default="terminal_backbone_repaired",
    )
    return parser.parse_args()


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("zero-length vector")
    return vector / norm


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_pdb_atoms(path: Path) -> list[dict]:
    atoms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        atoms.append(
            {
                "raw": line,
                "name": line[12:16],
                "name_s": line[12:16].strip(),
                "altloc": line[16],
                "resname": line[17:20],
                "chain": line[21],
                "resseq": int(line[22:26]),
                "icode": line[26],
                "coord": np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=np.float64,
                ),
                "occ": line[54:60],
                "bfactor": line[60:66],
                "element": line[76:78].strip() if len(line) >= 78 else "",
                "serial": int(line[6:11]),
            }
        )
    return atoms


def format_atom(serial: int, atom: dict) -> str:
    element = atom["element"] or atom["name_s"][0]
    x, y, z = atom["coord"]
    return (
        f"ATOM  {serial:5d} {atom['name']:<4} {atom['resname']:>3} "
        f"{atom['chain']}{atom['resseq']:4d}{atom['icode']}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{atom['occ']}{atom['bfactor']}"
        f"          {element:>2}  "
    )


def group_residues(atoms: list[dict]) -> list[dict]:
    residues = []
    for atom in atoms:
        key = (atom["chain"], atom["resseq"], atom["icode"], atom["resname"])
        if not residues or residues[-1]["key"] != key:
            residues.append({"key": key, "atoms": []})
        residues[-1]["atoms"].append(atom)
    return residues


def atom_by_name(residue: dict, name: str) -> dict:
    for atom in residue["atoms"]:
        if atom["name_s"] == name:
            return atom
    raise KeyError(name)


def clone_atom(template: dict, name: str, coord: np.ndarray, element: str) -> dict:
    atom = dict(template)
    atom["name_s"] = name
    atom["name"] = f" {name:<3}"[:4]
    atom["coord"] = coord.astype(np.float64)
    atom["element"] = element
    atom["serial"] = 0
    return atom


def residue_missing(residue: dict) -> list[str]:
    names = {atom["name_s"] for atom in residue["atoms"]}
    return [name for name in BACKBONE if name not in names]


def repair_terminal_backbone(atoms: list[dict]) -> tuple[list[dict], list[str]]:
    residues = group_residues(atoms)
    if len(residues) < 2:
        return atoms, ["too_few_residues"]

    actions = []
    first = residues[0]
    if residue_missing(first) == ["N"]:
        ca = atom_by_name(first, "CA")
        c_atom = atom_by_name(first, "C")
        next_n = atom_by_name(residues[1], "N")
        direction = -(unit(c_atom["coord"] - ca["coord"]) + unit(next_n["coord"] - ca["coord"]))
        coord = ca["coord"] + 1.458 * unit(direction)
        first["atoms"].insert(0, clone_atom(ca, "N", coord, "N"))
        actions.append("add_nterm_n")

    last = residues[-1]
    missing_last = residue_missing(last)
    if missing_last == ["C", "O"]:
        prev_c = atom_by_name(residues[-2], "C")
        n_atom = atom_by_name(last, "N")
        ca = atom_by_name(last, "CA")
        direction = -(unit(n_atom["coord"] - ca["coord"]) + unit(prev_c["coord"] - ca["coord"]))
        c_coord = ca["coord"] + 1.525 * unit(direction)
        c_atom = clone_atom(ca, "C", c_coord, "C")
        last["atoms"].append(c_atom)
        direction = -(unit(ca["coord"] - c_coord) + unit(n_atom["coord"] - c_coord))
        o_coord = c_coord + 1.231 * unit(direction)
        last["atoms"].append(clone_atom(c_atom, "O", o_coord, "O"))
        actions.append("add_cterm_c_o")

    remaining = []
    for residue in residues:
        missing = residue_missing(residue)
        if missing:
            chain, resseq, icode, resname = residue["key"]
            remaining.append(f"{chain}:{resname}{resseq}{icode.strip()}:{','.join(missing)}")
    if remaining:
        return atoms, [*actions, f"remaining_missing={';'.join(remaining)}"]

    flattened = []
    order = {name: index for index, name in enumerate(BACKBONE)}
    for residue in residues:
        flattened.extend(
            sorted(
                residue["atoms"],
                key=lambda atom: (order.get(atom["name_s"], 99), atom["serial"]),
            )
        )
    return flattened, actions


def write_pdb(path: Path, atoms: list[dict]) -> None:
    lines = []
    for serial, atom in enumerate(atoms, start=1):
        lines.append(format_atom(serial, atom))
    last = atoms[-1]
    lines.append(
        f"TER   {len(atoms) + 1:5d}      {last['resname']:>3} "
        f"{last['chain']}{last['resseq']:4d}{last['icode']}"
    )
    lines.append("END")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prepared_root = Path(args.prepared_root)
    manifest_dir = prepared_root / "manifests"
    source_csv = manifest_dir / "flow_input_rel.csv"
    rows = read_csv(source_csv)
    repair_root = prepared_root / args.output_subdir
    report_rows = []
    patched_rows = []

    for row in rows:
        case_name = row["complex_name"]
        peptide_path = Path(row["peptide_pdb"])
        atoms = parse_pdb_atoms(peptide_path)
        repaired_atoms, actions = repair_terminal_backbone(atoms)
        if actions and not any(action.startswith("remaining_missing=") for action in actions):
            repaired_path = repair_root / case_name / "peptide.pdb"
            write_pdb(repaired_path, repaired_atoms)
            patched_row = dict(row)
            patched_row["peptide_pdb"] = str(repaired_path)
            patched_rows.append(patched_row)
            report_rows.append(
                {
                    "complex_name": case_name,
                    "source_peptide_pdb": str(peptide_path),
                    "repaired_peptide_pdb": str(repaired_path),
                    "actions": ";".join(actions),
                    "status": "repaired",
                }
            )
        elif actions:
            patched_rows.append(dict(row))
            report_rows.append(
                {
                    "complex_name": case_name,
                    "source_peptide_pdb": str(peptide_path),
                    "repaired_peptide_pdb": "",
                    "actions": ";".join(actions),
                    "status": "not_repaired",
                }
            )
        else:
            patched_rows.append(dict(row))

    write_csv(
        manifest_dir / "flow_input_rel_terminal_backbone_repaired.csv",
        ["complex_name", "receptor_pdb", "peptide_pdb"],
        patched_rows,
    )
    write_csv(
        manifest_dir / "terminal_backbone_repair_report.csv",
        ["complex_name", "source_peptide_pdb", "repaired_peptide_pdb", "actions", "status"],
        report_rows,
    )

    two_case_rows = [row for row in patched_rows if any(report["complex_name"] == row["complex_name"] and report["status"] == "repaired" for report in report_rows)]
    write_csv(
        manifest_dir / "flow_input_rel_terminal_backbone_repaired_2cases.csv",
        ["complex_name", "receptor_pdb", "peptide_pdb"],
        two_case_rows,
    )
    print(f"repaired={len(two_case_rows)}")
    print(manifest_dir / "flow_input_rel_terminal_backbone_repaired.csv")
    print(manifest_dir / "flow_input_rel_terminal_backbone_repaired_2cases.csv")
    print(manifest_dir / "terminal_backbone_repair_report.csv")


if __name__ == "__main__":
    main()
