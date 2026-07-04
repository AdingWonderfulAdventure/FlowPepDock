#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from Bio.PDB import PDBParser


STANDARD_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

AF_CHAIN_IDS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MANIFEST_FIELDS = [
    "complex_name",
    "source_receptor_crop_pdb",
    "source_receptor_full_pdb",
    "source_peptide_pdb",
    "copied_receptor_pdb",
    "copied_peptide_pdb",
    "alphafold_multimer_fasta",
    "alphafold3_json",
    "receptor_chain_count",
    "peptide_chain_count",
    "receptor_chain_lengths",
    "peptide_chain_lengths",
    "alphafold_chain_map",
    "sequence_source",
]


@dataclass(frozen=True)
class ChainSequence:
    source_kind: str
    source_chain_id: str
    assigned_chain_id: str
    sequence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将 FlowPepDock 的 complex CSV 批量复制并导出成 "
            "AlphaFold3 JSON 与 AlphaFold-Multimer FASTA 输入。"
        )
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("data/runtime_tables/flow_infer_test536_rel.csv"),
        help="输入 CSV，需包含 complex_name/receptor_pdb/peptide_pdb 三列。",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/alphafold_inputs/test536"),
        help="输出目录；会在其中按 complex_name 建立子目录。",
    )
    parser.add_argument(
        "--af3_version",
        type=int,
        default=4,
        help="AlphaFold3 JSON 版本号，默认 4。",
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=1,
        help="AlphaFold3 JSON 中写入的默认 model seed。",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别。",
    )
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def has_backbone_atoms(residue) -> bool:
    return all(atom_name in residue for atom_name in ("N", "CA", "C", "O"))


def extract_chain_sequences(pdb_path: Path, source_kind: str) -> List[ChainSequence]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = structure[0]
    chains: List[ChainSequence] = []

    for chain in model:
        sequence_chars: List[str] = []
        seen_residues = set()
        source_chain_id = (chain.id or "_").strip() or "_"

        for residue in chain:
            hetflag, resseq, icode = residue.id
            if str(hetflag).strip():
                continue
            if not has_backbone_atoms(residue):
                continue

            residue_key = (resseq, icode)
            if residue_key in seen_residues:
                continue
            seen_residues.add(residue_key)

            residue_name = residue.get_resname().strip()
            one_letter = STANDARD_THREE_TO_ONE.get(residue_name)
            if one_letter is None:
                raise ValueError(
                    f"unsupported_residue:{residue_name} in {pdb_path}"
                )
            sequence_chars.append(one_letter)

        if sequence_chars:
            chains.append(
                ChainSequence(
                    source_kind=source_kind,
                    source_chain_id=source_chain_id,
                    assigned_chain_id="",
                    sequence="".join(sequence_chars),
                )
            )

    if not chains:
        raise ValueError(f"no_valid_protein_chain_found:{pdb_path}")
    return chains


def assign_alphafold_chain_ids(chains: Iterable[ChainSequence]) -> List[ChainSequence]:
    chains = list(chains)
    if len(chains) > len(AF_CHAIN_IDS):
        raise ValueError(
            f"too_many_chains_for_alphafold:{len(chains)}>{len(AF_CHAIN_IDS)}"
        )
    return [
        ChainSequence(
            source_kind=chain.source_kind,
            source_chain_id=chain.source_chain_id,
            assigned_chain_id=AF_CHAIN_IDS[index],
            sequence=chain.sequence,
        )
        for index, chain in enumerate(chains)
    ]


def infer_full_receptor_path(receptor_crop_path: Path) -> Path:
    receptor_full_path = receptor_crop_path.parent / "receptor.pdb"
    if not receptor_full_path.exists():
        raise FileNotFoundError(
            f"missing_full_receptor:{receptor_full_path}"
        )
    return receptor_full_path


def write_multimer_fasta(
    fasta_path: Path,
    complex_name: str,
    chains: List[ChainSequence],
) -> None:
    lines = []
    for chain in chains:
        header = (
            f">{complex_name}|af_id={chain.assigned_chain_id}"
            f"|source={chain.source_kind}|orig_chain={chain.source_chain_id}"
        )
        lines.append(header)
        lines.append(chain.sequence)
    fasta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_af3_payload(
    complex_name: str,
    chains: List[ChainSequence],
    model_seed: int,
    af3_version: int,
) -> dict:
    sequences = []
    for chain in chains:
        description = (
            f"{chain.source_kind} chain from FlowPepDock; "
            f"original_chain={chain.source_chain_id}"
        )
        protein_entry = {
            "id": chain.assigned_chain_id,
            "sequence": chain.sequence,
        }
        if af3_version >= 4:
            protein_entry["description"] = description
        sequences.append({"protein": protein_entry})

    return {
        "name": complex_name,
        "modelSeeds": [model_seed],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": af3_version,
    }


def write_af3_json(
    json_path: Path,
    complex_name: str,
    chains: List[ChainSequence],
    model_seed: int,
    af3_version: int,
) -> None:
    payload = build_af3_payload(
        complex_name=complex_name,
        chains=chains,
        model_seed=model_seed,
        af3_version=af3_version,
    )
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def chain_lengths_text(chains: Iterable[ChainSequence], source_kind: str) -> str:
    values = [
        f"{chain.source_chain_id}:{len(chain.sequence)}"
        for chain in chains
        if chain.source_kind == source_kind
    ]
    return ";".join(values)


def chain_map_text(chains: Iterable[ChainSequence]) -> str:
    values = [
        (
            f"{chain.assigned_chain_id}:{chain.source_kind}:"
            f"{chain.source_chain_id}:{len(chain.sequence)}"
        )
        for chain in chains
    ]
    return ";".join(values)


def export_one_complex(
    row: dict,
    output_root: Path,
    model_seed: int,
    af3_version: int,
) -> dict:
    complex_name = row["complex_name"].strip()
    receptor_crop_path = Path(row["receptor_pdb"])
    peptide_path = Path(row["peptide_pdb"])
    receptor_full_path = infer_full_receptor_path(receptor_crop_path)

    receptor_chains = extract_chain_sequences(
        receptor_full_path, source_kind="receptor"
    )
    peptide_chains = extract_chain_sequences(
        peptide_path, source_kind="peptide"
    )
    assigned_chains = assign_alphafold_chain_ids(
        [*receptor_chains, *peptide_chains]
    )

    complex_output_dir = output_root / complex_name
    complex_output_dir.mkdir(parents=True, exist_ok=True)

    copied_receptor_path = complex_output_dir / "receptor.pdb"
    copied_peptide_path = complex_output_dir / "peptide.pdb"
    multimer_fasta_path = complex_output_dir / "alphafold_multimer.fasta"
    af3_json_path = complex_output_dir / "alphafold3_input.json"

    shutil.copy2(receptor_full_path, copied_receptor_path)
    shutil.copy2(peptide_path, copied_peptide_path)
    write_multimer_fasta(multimer_fasta_path, complex_name, assigned_chains)
    write_af3_json(
        af3_json_path,
        complex_name=complex_name,
        chains=assigned_chains,
        model_seed=model_seed,
        af3_version=af3_version,
    )

    logging.debug(
        "exported %s with %d receptor chains and %d peptide chains",
        complex_name,
        len(receptor_chains),
        len(peptide_chains),
    )

    return {
        "complex_name": complex_name,
        "source_receptor_crop_pdb": str(receptor_crop_path),
        "source_receptor_full_pdb": str(receptor_full_path),
        "source_peptide_pdb": str(peptide_path),
        "copied_receptor_pdb": str(copied_receptor_path),
        "copied_peptide_pdb": str(copied_peptide_path),
        "alphafold_multimer_fasta": str(multimer_fasta_path),
        "alphafold3_json": str(af3_json_path),
        "receptor_chain_count": str(len(receptor_chains)),
        "peptide_chain_count": str(len(peptide_chains)),
        "receptor_chain_lengths": chain_lengths_text(
            assigned_chains, source_kind="receptor"
        ),
        "peptide_chain_lengths": chain_lengths_text(
            assigned_chains, source_kind="peptide"
        ),
        "alphafold_chain_map": chain_map_text(assigned_chains),
        "sequence_source": "observed_atom_records_from_pdb",
    }


def load_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"complex_name", "receptor_pdb", "peptide_pdb"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"missing_required_columns:{sorted(missing_columns)}"
            )
        return list(reader)


def write_manifest(manifest_path: Path, rows: List[dict]) -> None:
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_csv)
    manifest_rows = []
    for row in rows:
        manifest_rows.append(
            export_one_complex(
                row=row,
                output_root=output_dir,
                model_seed=args.model_seed,
                af3_version=args.af3_version,
            )
        )

    manifest_path = output_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    logging.info(
        "done: exported %d complexes to %s",
        len(manifest_rows),
        output_dir,
    )
    logging.info("manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
