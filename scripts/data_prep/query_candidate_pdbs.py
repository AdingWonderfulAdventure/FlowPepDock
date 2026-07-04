#!/usr/bin/env python
"""
Query RCSB for protein–peptide docking candidates.
按 RCSB 自动搜可做蛋白-肽对接的复合物，并导出带中文列名的 CSV。
搜索条件：
- entry 层：链条数>=2，蛋白链数>=min_protein_chains，残基总数>=min_total_residues，分辨率<=3Å或NMR。
- entity 层：至少一条蛋白长链(>=min_receptor_len)，至少一条肽链(长度在[min_peptide_len,max_peptide_len])，
  且最长受体长度 >= ratio * 最长肽长度。
输出 CSV 同时包含：PDB编号、受体/肽链ID、实体ID、长度、序列、描述、物种、实验方法、分辨率、分子组成、是否含非蛋白、筛选备注。
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"


def entry_filter_note(min_protein_chains: int, min_total_residues: int) -> str:
    return (
        f"chains>=2 & protein_chains>={min_protein_chains} & "
        f"residues>={min_total_residues} & (resolution<=3Å or method=NMR)"
    )


def entity_filter_note(min_pep: int, max_pep: int, min_receptor: int, ratio: float) -> str:
    return (
        f"peptide_len[{min_pep},{max_pep}] & receptor>={min_receptor} & "
        f"max_receptor>= {ratio} * max_peptide"
    )


def terminal(attribute: str, operator: str, value) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": attribute,
            "operator": operator,
            "value": value,
        },
    }


def build_entry_query(min_protein_chains: int, min_total_residues: int) -> dict:
    chain_count = terminal(
        "rcsb_entry_info.deposited_polymer_entity_instance_count",
        "greater_or_equal",
        2,
    )
    protein_chains = terminal(
        "rcsb_entry_info.polymer_entity_count_protein",
        "greater_or_equal",
        min_protein_chains,
    )
    residue_count = terminal(
        "rcsb_entry_info.deposited_polymer_monomer_count",
        "greater_or_equal",
        min_total_residues,
    )
    res_le3 = terminal("rcsb_entry_info.resolution_combined", "less_or_equal", 3.0)
    method_nmr1 = terminal("exptl.method", "exact_match", "SOLUTION NMR")
    method_nmr2 = terminal("exptl.method", "exact_match", "SOLID-STATE NMR")
    res_or_nmr = {
        "type": "group",
        "logical_operator": "or",
        "nodes": [res_le3, method_nmr1, method_nmr2],
    }
    return {
        "type": "group",
        "logical_operator": "and",
        "nodes": [chain_count, protein_chains, residue_count, res_or_nmr],
    }


def init_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def load_or_fetch_entry_ids(
    session: requests.Session,
    min_protein_chains: int,
    min_total_residues: int,
    cache_path: Path,
) -> List[str]:
    if cache_path.exists():
        ids: List[str] = []
        if cache_path.suffix.lower() == ".csv":
            with cache_path.open() as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row:
                        ids.append(row[0])
        else:
            ids = [line.strip() for line in cache_path.read_text().splitlines() if line.strip()]
        if ids:
            return ids
    payload = {
        "query": build_entry_query(min_protein_chains, min_total_residues),
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    resp = session.post(SEARCH_URL, json=payload, timeout=60)
    resp.raise_for_status()
    ids = [item["identifier"] for item in resp.json().get("result_set", [])]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.suffix.lower() == ".csv":
        with cache_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pdb_id", "entry_filter"])
            note = entry_filter_note(min_protein_chains, min_total_residues)
            for pid in ids:
                writer.writerow([pid, note])
    else:
        cache_path.write_text("\n".join(ids))
    return ids


def chunked(seq: Iterable[str], size: int):
    seq = list(seq)
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def pick_receptor_peptide(
    entities: List[dict], min_pep: int, max_pep: int, min_receptor: int, ratio: float
) -> Optional[Tuple[dict, int, List[Tuple[int, dict]]]]:
    """选择肽实体（最短的一条）和所有符合条件的受体实体列表（长度>=50且>=3x肽）。"""
    proteins = []
    for ent in entities:
        poly = ent.get("entity_poly", {}) or {}
        poly_type = poly.get("rcsb_entity_polymer_type") or poly.get("type")
        if not poly_type:
            continue
        poly_type_l = str(poly_type).lower()
        if not ("polypeptide" in poly_type_l or "protein" in poly_type_l):
            continue
        length = (
            poly.get("rcsb_sample_sequence_length")
            or poly.get("polymer_length")
            or poly.get("rcsb_entity_polymer_length")
        )
        if not length:
            continue
        proteins.append((int(length), ent))
    if not proteins:
        return None
    peptides = sorted([p for p in proteins if min_pep <= p[0] <= max_pep], key=lambda x: x[0])
    if not peptides:
        return None
    pep_len, pep_ent = peptides[0]
    receptors = [p for p in proteins if p[0] >= min_receptor and p[0] >= ratio * pep_len and p[1] is not pep_ent]
    if not receptors:
        return None
    return pep_ent, pep_len, receptors


def process_batches(
    session: requests.Session,
    entry_ids: List[str],
    processed: Set[str],
    processed_file: Path,
    min_pep: int,
    max_pep: int,
    min_receptor: int,
    ratio: float,
    remaining_limit: int,
    entry_note: str,
    entity_note: str,
) -> List[Dict[str, str]]:
    hits: List[Dict[str, str]] = []
    query = """
    query ($ids:[String!]!) {
      entries(entry_ids:$ids) {
        rcsb_id
        struct { title }
        polymer_entities {
          rcsb_id
          rcsb_polymer_entity { pdbx_description }
          entity_poly {
            rcsb_entity_polymer_type
            rcsb_sample_sequence_length
            pdbx_seq_one_letter_code_can
          }
          rcsb_polymer_entity_container_identifiers {
            auth_asym_ids
            asym_ids
            uniprot_ids
          }
          rcsb_entity_source_organism {
            ncbi_scientific_name
          }
        }
        rcsb_entry_info {
          polymer_composition
          resolution_combined
          experimental_method
        }
        nonpolymer_entities { rcsb_nonpolymer_entity_container_identifiers { auth_asym_ids } }
      }
    }
    """

    pending = [eid for eid in entry_ids if eid not in processed]
    total = len(entry_ids)
    processed_count = len(processed)
    processed_handle = processed_file.open("a")

    for batch in chunked(pending, 100):
        payload = {"query": query, "variables": {"ids": batch}}
        resp = session.post(GRAPHQL_URL, json=payload, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        if "errors" in body and body["errors"]:
            raise RuntimeError(f"GraphQL errors: {body['errors']}")
        entries = body.get("data", {}).get("entries", [])
        for entry in entries:
            entry_id = entry.get("rcsb_id")
            info = entry.get("rcsb_entry_info") or {}
            composition = info.get("polymer_composition", "unknown")
            resolution = (info.get("resolution_combined") or [None])[0]
            method = info.get("experimental_method")
            nonpoly = entry.get("nonpolymer_entities") or []
            has_non_protein = composition != "protein" or len(nonpoly) > 0

            entities = entry.get("polymer_entities") or []
            picked = pick_receptor_peptide(entities, min_pep, max_pep, min_receptor, ratio)
            if not picked:
                processed.add(entry_id)
                processed_handle.write(f"{entry_id}\n")
                processed_handle.flush()
                processed_count += 1
                print(f"Processed {processed_count}/{total}", end="\r")
                continue
            pep_ent, pep_len, rec_list = picked
            pep_poly = pep_ent.get("entity_poly") or {}
            pep_ids = pep_ent.get("rcsb_polymer_entity_container_identifiers") or {}

            def first_chain(d: dict) -> str:
                chains = d.get("auth_asym_ids") or d.get("asym_ids") or []
                return chains[0] if chains else ""

            pep_desc = (pep_ent.get("rcsb_polymer_entity") or {}).get("pdbx_description", "")
            rec_chains = []
            rec_entity_ids = []
            rec_lens = []
            rec_seqs = []
            rec_names = []
            rec_species = []
            for rec_len, rec_ent in rec_list:
                rec_poly = rec_ent.get("entity_poly") or {}
                rec_ids = rec_ent.get("rcsb_polymer_entity_container_identifiers") or {}
                rec_chains.append(first_chain(rec_ids))
                rec_entity_ids.append(rec_ent.get("rcsb_id", ""))
                rec_lens.append(str(rec_len))
                rec_seqs.append((rec_poly.get("pdbx_seq_one_letter_code_can") or "").replace("\n", ""))
                rec_names.append((rec_ent.get("rcsb_polymer_entity") or {}).get("pdbx_description", ""))
                rec_species.append((rec_ent.get("rcsb_entity_source_organism") or [{}])[0].get("ncbi_scientific_name", ""))

            record = {
                "pdb_id": entry_id,
                "受体链ID": ";".join(filter(None, rec_chains)),
                "受体实体ID": ";".join(filter(None, rec_entity_ids)),
                "受体长度": ";".join(filter(None, rec_lens)),
                "受体序列": ";".join(filter(None, rec_seqs)),
                "受体名称": ";".join(filter(None, rec_names)),
                "受体物种": ";".join(filter(None, rec_species)),
                "肽链ID": first_chain(pep_ids),
                "肽实体ID": pep_ent.get("rcsb_id", ""),
                "肽长度": pep_len,
                "肽序列": (pep_poly.get("pdbx_seq_one_letter_code_can") or "").replace("\n", ""),
                "肽名称": pep_desc,
                "实验方法": method,
                "分辨率(A)": resolution,
                "分子组成": composition,
                "是否含非蛋白": has_non_protein,
                "筛选备注_entry": entry_note,
                "筛选备注_entity": entity_note,
            }
            hits.append(record)
            if remaining_limit and len(hits) >= remaining_limit:
                processed.add(entry_id)
                processed_handle.write(f"{entry_id}\n")
                processed_handle.flush()
                print(f"Processed {processed_count+1}/{total}", end="\r")
                processed_handle.close()
                return hits
            processed.add(entry_id)
            processed_handle.write(f"{entry_id}\n")
            processed_handle.flush()
            processed_count += 1
            print(f"Processed {processed_count}/{total}", end="\r")
        time.sleep(0.2)
    processed_handle.close()
    print()
    return hits


def main():
    parser = argparse.ArgumentParser(description="获取符合protein-peptide条件的PDB ID")
    parser.add_argument("--limit", type=int, default=0, help="最多输出多少条（0=全部）")
    parser.add_argument("--min_peptide_len", type=int, default=3)
    parser.add_argument("--max_peptide_len", type=int, default=20)
    parser.add_argument("--min_receptor_len", type=int, default=50)
    parser.add_argument("--ratio", type=float, default=3.0)
    parser.add_argument("--min_protein_chains", type=int, default=2)
    parser.add_argument("--min_total_residues", type=int, default=50)
    parser.add_argument("--output_csv", type=str, default="candidate_pdb_ids.csv")
    parser.add_argument("--entry_cache", type=str, default="candidate_entry_ids.csv")
    parser.add_argument("--checkpoint", type=str, default="data/csv_backup/archive/processed_entries.txt")
    args = parser.parse_args()

    session = init_session()
    entry_cache_path = Path(args.entry_cache)
    entry_ids = load_or_fetch_entry_ids(session, args.min_protein_chains, args.min_total_residues, entry_cache_path)
    print(f"Search API 候选 entry 数量: {len(entry_ids)}")

    checkpoint_path = Path(args.checkpoint)
    processed_set: Set[str] = set()
    if checkpoint_path.exists():
        processed_set = {line.strip() for line in checkpoint_path.read_text().splitlines() if line.strip()}
    else:
        checkpoint_path.touch()

    existing_hits: Set[str] = set()
    output_path = Path(args.output_csv)
    if output_path.exists():
        with output_path.open() as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    existing_hits.add(row[0])

    if args.limit and len(existing_hits) >= args.limit:
        print("已存在的结果数量大于等于limit，不再追加")
        return
    remaining_limit = 0
    if args.limit:
        remaining_limit = args.limit - len(existing_hits)

    entry_note = entry_filter_note(args.min_protein_chains, args.min_total_residues)
    entity_note = entity_filter_note(
        args.min_peptide_len, args.max_peptide_len, args.min_receptor_len, args.ratio
    )

    hits = process_batches(
        session,
        entry_ids,
        processed_set,
        checkpoint_path,
        args.min_peptide_len,
        args.max_peptide_len,
        args.min_receptor_len,
        args.ratio,
        remaining_limit,
        entry_note,
        entity_note,
    )

    write_header = not output_path.exists()
    with open(args.output_csv, "a", newline="") as fout:
        writer = csv.writer(fout)
        if write_header:
            writer.writerow(
                [
                    "PDB编号(pdb_id)",
                    "受体链ID(receptor_chain)",
                    "受体实体ID(receptor_entity_id)",
                    "受体长度(receptor_len)",
                    "受体序列(receptor_seq)",
                    "受体名称(receptor_name)",
                    "受体物种(receptor_species)",
                    "肽链ID(peptide_chain)",
                    "肽实体ID(peptide_entity_id)",
                    "肽长度(peptide_len)",
                    "肽序列(peptide_seq)",
                    "肽名称(peptide_name)",
                    "实验方法(experimental_method)",
                    "分辨率(A)(resolution)",
                    "分子组成(polymer_composition)",
                    "是否含非蛋白(has_non_protein)",
                    "筛选备注_entry",
                    "筛选备注_entity",
                ]
            )
        for record in hits:
            if record["pdb_id"] in existing_hits:
                continue
            writer.writerow(
                [
                    record["pdb_id"],
                    record["受体链ID"],
                    record["受体实体ID"],
                    record["受体长度"],
                    record["受体序列"],
                    record["受体名称"],
                    record["受体物种"],
                    record["肽链ID"],
                    record["肽实体ID"],
                    record["肽长度"],
                    record["肽序列"],
                    record["肽名称"],
                    record["实验方法"],
                    record["分辨率(A)"],
                    record["分子组成"],
                    record["是否含非蛋白"],
                    record["筛选备注_entry"],
                    record["筛选备注_entity"],
                ]
            )
    print(f"本轮新增 {len(hits)} 条，结果追加到 {args.output_csv}")


if __name__ == "__main__":
    main()
