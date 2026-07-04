
from rdkit import Chem
import MDAnalysis
import numpy as np
import re
from Bio.PDB import PDBParser


def standard_residue_sort(item):
    """按照数字+插码的组合顺序排序残基编号，保证链条顺序稳定"""
    # convert to str
    if isinstance(item, int):
        return item, 0
    else:
        s = str(item)
        # extract the digital part
        num = "".join([i for i in s if i.isdigit()])

        # extract the non digital part
        non_num = "".join([i for i in s if not i.isdigit()])
        code = ord(non_num)
        if num == "1":
            return (int(num) if num else 0, -code)
        else:
            return (int(num) if num else 0, code)


three_to_one = {
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
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

three_to_one_esm = {
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
    "MSE": "M",  # this is almost the same AA as MET. The sulfur is just replaced by Selen
    "PHE": "F",
    "PRO": "P",
    "PYL": "O",  #
    "SER": "S",
    "SEC": "U",  #
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",  #
    "GLX": "Z",  #
    "XAA": "X",  #
    "XLE": "J",
}  #


def read_pdb_with_connect_labels(
    pdbfile: str, sanitize: bool = True, addHs: bool = False
):
    """读取PDB并严格依据CONECT重建键，适合配体/非肽结构"""
    mol = Chem.MolFromPDBFile(pdbfile, sanitize=False)

    rw_mol = Chem.RWMol(mol)
    while rw_mol.GetNumBonds() > 0:
        bond = rw_mol.GetBondWithIdx(0)
        rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    edges = set()
    for line in open(pdbfile, "r").readlines():
        if line.startswith("CONECT"):
            content = line.strip().split()[1:]
            start = content[0]
            ends = content[1:]
            for end in set(ends):
                edges.add((tuple(sorted((int(start), int(end)))) + (ends.count(end),)))

    for edge in edges:
        atom1_idx, atom2_idx, bond_type = edge
        if bond_type == 1:
            rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, Chem.BondType.SINGLE)
        elif bond_type == 2:
            rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, Chem.BondType.DOUBLE)
        elif bond_type == 3:
            rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, Chem.BondType.TRIPLE)
        else:
            raise RuntimeError

    if sanitize:
        Chem.SanitizeMol(rw_mol)

    if addHs:
        mol = Chem.AddHs(rw_mol, addCoords=True)
    else:
        mol = rw_mol

    return mol


def read_pdb_with_seq(pdbfile: str, sanitize: bool = True, addHs: bool = False):
    """读取肽 PDB，并直接使用当前 PDB 可恢复的连通性。"""
    mol = Chem.MolFromPDBFile(pdbfile, sanitize=False)
    if mol is None:
        raise ValueError(f"failed_to_parse_pdb:{pdbfile}")

    u = MDAnalysis.Universe(pdbfile)
    has_missing_template_atoms = False
    for residue in u.residues:
        res_name = residue.resname.strip()
        res_atoms = residue.atoms.select_atoms("not type H")
        expected_atoms = {
            "GLY": 4,
            "ALA": 5,
            "VAL": 7,
            "LEU": 8,
            "ILE": 8,
            "PRO": 7,
            "PHE": 11,
            "TYR": 12,
            "TRP": 14,
            "SER": 6,
            "THR": 7,
            "CYS": 6,
            "MET": 8,
            "ASN": 8,
            "GLN": 9,
            "ASP": 8,
            "GLU": 9,
            "LYS": 9,
            "ARG": 11,
            "HIS": 10,
        }.get(res_name)
        if expected_atoms is not None and len(res_atoms) < expected_atoms:
            has_missing_template_atoms = True
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            if not has_missing_template_atoms:
                raise

    if addHs:
        return Chem.AddHs(mol, addCoords=True)
    return mol


def get_edges_from_pdb_with_seq(pdbfile: str):
    """Return 0-based directed bond edges consistent with the actual PDB atoms."""
    mol = read_pdb_with_seq(pdbfile, sanitize=False, addHs=False)
    edges = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        edges.append((u, v))
        edges.append((v, u))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64)


def get_sequences_from_pdbfile(file_path, suppress_missed_aa=False):
    """从多链PDB里抽取每条链的序列，非标准残基用-占位"""
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure("random_id", file_path)
    structure = structure[0]
    sequence = None
    for i, chain in enumerate(structure):
        seq_pro = ""
        seq_dic = {}
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                continue
            c_alpha, n, c, o = None, None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
                if atom.name == "O":
                    o = list(atom.get_vector())
            if (
                c_alpha != None and n != None and c != None and o != None
            ):  # only append residue if it is an amino acid
                try:
                    seq_dic[
                        (
                            int(residue.id[1])
                            if residue.id[2] == " "
                            else f"{residue.id[1]}{residue.id[2].strip()}"
                        )
                    ] = three_to_one_esm[residue.get_resname()]
                except Exception as e:
                    seq_dic[
                        (
                            int(residue.id[1])
                            if residue.id[2] == " "
                            else f"{residue.id[1]}{residue.id[2].strip()}"
                        )
                    ] = "-"
                    print(
                        "encountered unknown AA: ",
                        residue.get_resname(),
                        " in the complex. Replacing it with a dash - .",
                    )

        try:
            digit_list = [i for i in seq_dic.keys() if isinstance(i, int)]
            for idx in sorted(
                (
                    set(seq_dic.keys())
                    | set(range(min(digit_list), max(digit_list) + 1))
                    if len(digit_list) > 0
                    else set(seq_dic.keys())
                ),
                key=standard_residue_sort,
            ):
                try:
                    seq_pro += seq_dic[idx]
                except:
                    seq_pro += "-"
                    if not suppress_missed_aa:
                        print(
                            "missed AA: ",
                            idx,
                            " in the complex ",
                            file_path,
                            ". Add it with a dash - .",
                        )
        except:
            print("=========================================" + file_path)

        if sequence is None:
            sequence = seq_pro
        else:
            sequence += ":" + seq_pro

    return sequence


def get_sequences(descriptions, suppress_missed_aa=False):
    """输入支持PDB路径或直接序列文本，统一返回标准序列串"""
    new_sequences = []
    for description in descriptions:
        if "pdb" in description:
            new_sequences.append(
                get_sequences_from_pdbfile(
                    description, suppress_missed_aa=suppress_missed_aa
                )
            )
        else:
            new_sequences.append(re.sub(r"\[.*?\]", "-", description))
    return new_sequences
