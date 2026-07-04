#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from requests.adapters import HTTPAdapter, Retry
from scipy.spatial import cKDTree


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
BAD_TITLE_WORDS = {
    "ribosome",
    "spliceosome",
    "nucleosome",
    "photosystem",
    "respirasome",
    "capsid",
    "virion",
    "virus-like",
    "microtubule",
}


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def _terminal(attribute: str, operator: str, value) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def _search_payload(args: argparse.Namespace) -> dict:
    nodes = [
        _terminal("rcsb_accession_info.deposit_date", "greater_or_equal", args.deposit_from),
        _terminal("rcsb_entry_info.polymer_entity_count_protein", "greater_or_equal", 2),
        _terminal("rcsb_entry_info.deposited_polymer_entity_instance_count", "greater_or_equal", 2),
        _terminal("rcsb_entry_info.deposited_polymer_monomer_count", "greater_or_equal", args.min_total_residues),
    ]
    if args.max_resolution > 0:
        nodes.append(_terminal("rcsb_entry_info.resolution_combined", "less_or_equal", args.max_resolution))
    if args.xray_only:
        nodes.append(_terminal("exptl.method", "exact_match", "X-RAY DIFFRACTION"))
    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }


def _chunked(seq: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _as_list(v) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _clean_seq(seq: str) -> str:
    return re.sub(r"[^A-Za-z]", "", seq or "").upper()


def _difficulty(length: int) -> str:
    if length <= 12:
        return "简单"
    if length <= 25:
        return "中等"
    return "偏难"


def _download_cif(pdb_id: str, out: Path) -> bool:
    if out.exists() and out.stat().st_size > 0:
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        f"https://files.rcsb.org/download/{pdb_id}.cif",
        f"https://models.rcsb.org/{pdb_id}.bcif?encoding=cif",
    ]
    for url in urls:
        cmd = [
            "curl",
            "-L",
            "--retry",
            "2",
            "--retry-delay",
            "1",
            "--connect-timeout",
            "20",
            "-fS",
            url,
            "-o",
            str(out),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return True
        if out.exists() and out.stat().st_size == 0:
            out.unlink()
    return False


def _atom_table(cif_path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    d = MMCIF2Dict(str(cif_path))
    chains = _as_list(d.get("_atom_site.auth_asym_id") or d.get("_atom_site.label_asym_id"))
    entities = _as_list(d.get("_atom_site.label_entity_id"))
    groups = _as_list(d.get("_atom_site.group_PDB"))
    elems = _as_list(d.get("_atom_site.type_symbol"))
    xs = _as_list(d.get("_atom_site.Cartn_x"))
    ys = _as_list(d.get("_atom_site.Cartn_y"))
    zs = _as_list(d.get("_atom_site.Cartn_z"))
    n = min(len(chains), len(entities), len(groups), len(elems), len(xs), len(ys), len(zs))
    coords_by_chain: dict[str, list[list[float]]] = {}
    entity_by_chain: dict[str, str] = {}
    for i in range(n):
        if str(groups[i]).upper() != "ATOM":
            continue
        elem = str(elems[i]).strip().upper()
        if elem == "H":
            continue
        chain = str(chains[i]).strip()
        entity = str(entities[i]).strip()
        if not chain or not entity:
            continue
        try:
            coord = [float(xs[i]), float(ys[i]), float(zs[i])]
        except ValueError:
            continue
        coords_by_chain.setdefault(chain, []).append(coord)
        entity_by_chain.setdefault(chain, entity)
    return {k: np.asarray(v, dtype=np.float64) for k, v in coords_by_chain.items()}, entity_by_chain


def _min_dist(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return math.inf
    tree = cKDTree(a)
    return float(tree.query(b, k=1)[0].min())


def _fetch_entries(session: requests.Session, ids: list[str]) -> list[dict]:
    query = """
    query ($ids:[String!]!) {
      entries(entry_ids:$ids) {
        rcsb_id
        struct { title }
        rcsb_accession_info { deposit_date initial_release_date }
        rcsb_entry_info {
          resolution_combined
          experimental_method
          deposited_polymer_entity_instance_count
          deposited_polymer_monomer_count
        }
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
          }
        }
      }
    }
    """
    out = []
    for batch in _chunked(ids, 50):
        resp = session.post(GRAPHQL_URL, json={"query": query, "variables": {"ids": batch}}, timeout=90)
        resp.raise_for_status()
        body = resp.json()
        if body.get("errors"):
            raise RuntimeError(body["errors"])
        out.extend(body.get("data", {}).get("entries") or [])
        time.sleep(0.15)
    return out


def _entry_candidates(entry: dict, cif_dir: Path, args: argparse.Namespace) -> list[dict]:
    pdb_id = str(entry["rcsb_id"]).upper()
    title = ((entry.get("struct") or {}).get("title") or "").strip()
    title_l = title.lower()
    if any(word in title_l for word in BAD_TITLE_WORDS):
        return []
    info = entry.get("rcsb_entry_info") or {}
    if int(info.get("deposited_polymer_entity_instance_count") or 0) > args.max_instances:
        return []
    if int(info.get("deposited_polymer_monomer_count") or 0) > args.max_total_residues:
        return []

    entities = []
    for ent in entry.get("polymer_entities") or []:
        poly = ent.get("entity_poly") or {}
        ptype = str(poly.get("rcsb_entity_polymer_type") or "").lower()
        if "polypeptide" not in ptype and "protein" not in ptype:
            continue
        length = int(poly.get("rcsb_sample_sequence_length") or 0)
        ids = ent.get("rcsb_polymer_entity_container_identifiers") or {}
        chains = [str(x).strip() for x in (ids.get("auth_asym_ids") or ids.get("asym_ids") or []) if str(x).strip()]
        if not chains:
            continue
        entities.append(
            {
                "entity_id": str(ent.get("rcsb_id") or "").split("_")[-1],
                "length": length,
                "chains": chains,
                "seq": _clean_seq(poly.get("pdbx_seq_one_letter_code_can") or ""),
                "desc": ((ent.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "").strip(),
            }
        )
    peptides = [e for e in entities if args.min_peptide_len <= e["length"] <= args.max_peptide_len]
    receptors = [e for e in entities if e["length"] >= args.min_receptor_len and e["length"] <= args.max_receptor_len]
    if not peptides or not receptors:
        return []

    cif_path = cif_dir / f"{pdb_id.lower()}.cif"
    if not _download_cif(pdb_id, cif_path):
        return []
    try:
        coords_by_chain, _ = _atom_table(cif_path)
    except Exception:
        return []

    rows = []
    method = ";".join(info.get("experimental_method") or [])
    res_list = info.get("resolution_combined") or []
    resolution = res_list[0] if res_list else ""
    acc = entry.get("rcsb_accession_info") or {}
    for pep in peptides:
        for pep_chain in pep["chains"]:
            pep_coords = coords_by_chain.get(pep_chain)
            if pep_coords is None or len(pep_coords) == 0:
                continue
            contacts = []
            min_contact = math.inf
            rec_desc = []
            for rec in receptors:
                if rec["entity_id"] == pep["entity_id"]:
                    continue
                if rec["length"] < args.ratio * max(1, pep["length"]):
                    continue
                for rec_chain in rec["chains"]:
                    rec_coords = coords_by_chain.get(rec_chain)
                    if rec_coords is None or len(rec_coords) == 0:
                        continue
                    d = _min_dist(rec_coords, pep_coords)
                    min_contact = min(min_contact, d)
                    if d <= args.contact_threshold:
                        contacts.append((d, rec_chain, rec))
                        rec_desc.append(rec["desc"])
            if not contacts:
                continue
            contacts.sort(key=lambda x: x[0])
            kept = []
            seen = set()
            for d, rec_chain, rec in contacts:
                if rec_chain in seen:
                    continue
                kept.append(rec_chain)
                seen.add(rec_chain)
                if len(kept) >= args.max_receptor_chains:
                    break
            rows.append(
                {
                    "category": "RCSB 2026+ contact protein-peptide",
                    "seq_can": pep["seq"],
                    "length": str(pep["length"]),
                    "representative_entry": pdb_id,
                    "all_entries": pdb_id,
                    "description": pep["desc"],
                    "title_example": title,
                    "difficulty": _difficulty(pep["length"]),
                    "contact_pattern": f"{pdb_id}[{pep_chain}:{','.join(kept)}]",
                    "entry_url": f"https://www.rcsb.org/structure/{pdb_id}",
                    "deposit_date": acc.get("deposit_date") or "",
                    "initial_release_date": acc.get("initial_release_date") or "",
                    "experimental_method": method,
                    "resolution": resolution,
                    "peptide_desc": pep["desc"],
                    "receptor_desc": ";".join(sorted(set(x for x in rec_desc if x)))[:500],
                    "min_contact": f"{min_contact:.3f}",
                    "instance_count": str(info.get("deposited_polymer_entity_instance_count") or ""),
                    "total_residues": str(info.get("deposited_polymer_monomer_count") or ""),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Query 2026+ RCSB entries and keep contact-confirmed protein-peptide cases.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--deposit_from", default="2026-01-01")
    ap.add_argument("--limit_entries", type=int, default=0)
    ap.add_argument("--limit_rows", type=int, default=80)
    ap.add_argument("--min_peptide_len", type=int, default=5)
    ap.add_argument("--max_peptide_len", type=int, default=35)
    ap.add_argument("--min_receptor_len", type=int, default=60)
    ap.add_argument("--max_receptor_len", type=int, default=1200)
    ap.add_argument("--min_total_residues", type=int, default=70)
    ap.add_argument("--max_total_residues", type=int, default=3000)
    ap.add_argument("--max_instances", type=int, default=20)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--max_resolution", type=float, default=3.0)
    ap.add_argument("--contact_threshold", type=float, default=8.0)
    ap.add_argument("--max_receptor_chains", type=int, default=3)
    ap.add_argument("--xray_only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_cif_dir = out_dir / "raw_cif"
    session = _session()
    resp = session.post(SEARCH_URL, json=_search_payload(args), timeout=90)
    resp.raise_for_status()
    ids = [item["identifier"].upper() for item in resp.json().get("result_set", [])]
    ids = sorted(set(ids))
    if args.limit_entries:
        ids = ids[: args.limit_entries]
    (out_dir / "entry_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"[search] entries={len(ids)}")

    rows = []
    entries = _fetch_entries(session, ids)
    for i, entry in enumerate(entries, start=1):
        if not entry:
            continue
        found = _entry_candidates(entry, raw_cif_dir, args)
        rows.extend(found)
        print(f"[{i}/{len(entries)}] {entry.get('rcsb_id')} found={len(found)} total={len(rows)}")
        if args.limit_rows and len(rows) >= args.limit_rows:
            rows = rows[: args.limit_rows]
            break

    rows.sort(
        key=lambda r: (
            int(r["length"]) > 25,
            abs(int(r["length"]) - 12),
            float(r["resolution"]) if str(r["resolution"]) else 99.0,
            r["representative_entry"],
        )
    )
    tsv_fields = [
        "category",
        "seq_can",
        "length",
        "representative_entry",
        "all_entries",
        "description",
        "title_example",
        "difficulty",
        "contact_pattern",
        "entry_url",
    ]
    csv_fields = tsv_fields + [
        "deposit_date",
        "initial_release_date",
        "experimental_method",
        "resolution",
        "peptide_desc",
        "receptor_desc",
        "min_contact",
        "instance_count",
        "total_residues",
    ]
    with (out_dir / "rcsb_2026plus_contact_candidates.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tsv_fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in tsv_fields})
    with (out_dir / "rcsb_2026plus_contact_candidates.with_dates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] rows={len(rows)} out={out_dir}")


if __name__ == "__main__":
    main()
