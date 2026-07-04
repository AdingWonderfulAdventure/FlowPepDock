from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from cross_scoring_common import load_cross_pose_rows


_SCORER_STATE: Optional[Dict[str, object]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch score cross-docking poses with GraphPep.")
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--flow_input_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--graphpep_root", type=Path, default=Path(os.environ.get("GRAPHPEP_ROOT", "external/GraphPep_v1.1")))
    parser.add_argument("--ckpt_path", type=Path, default=Path(os.environ.get("GRAPHPEP_CKPT", "external/GraphPep_v1.1/ckpt/model_param.ckpt")))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu_ids", type=str, default="5,6")
    parser.add_argument("--chunk_size", type=int, default=8, help="Number of complexes per chunk.")
    parser.add_argument("--start_chunk", type=int, default=0)
    parser.add_argument("--max_chunks", type=int, default=0, help="0 means run all remaining chunks")
    parser.add_argument("--enable_stage_timing", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def _complex_sort_key(name: str) -> Tuple[str, int]:
    if name.startswith("pose"):
        try:
            return ("pose", int(name.replace("pose", "")))
        except ValueError:
            return ("pose", 0)
    return (name, 0)


def _run_clean_pdb(graphpep_root: Path, input_pdb: Path, output_pdb: Path) -> None:
    renumm = graphpep_root / "bin/renumm.awk"
    cleanpdb = graphpep_root / "bin/cleanpdb.awk"
    cmd = f'"{renumm}" "{input_pdb}" | "{cleanpdb}" > "{output_pdb}"'
    subprocess.run(["bash", "-lc", cmd], check=True)


def _safe_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in name)


def _as_list(value: object) -> List[object]:
    if isinstance(value, list):
        return value
    return [value]


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def _group_complex_tasks(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    for row in rows:
        complex_id = str(row["complex_id"])
        payload = grouped.setdefault(
            complex_id,
            {
                "complex_id": complex_id,
                "group_id": str(row["group_id"]),
                "source_group_id": str(row["source_group_id"]),
                "peptide_id": str(row["peptide_id"]),
                "receptor_id": str(row["receptor_id"]),
                "receptor_pdb": str(row["receptor_pdb"]),
                "pose_rows": [],
            },
        )
        payload["pose_rows"].append(dict(row))
    tasks = list(grouped.values())
    for task in tasks:
        task["pose_rows"] = sorted(task["pose_rows"], key=lambda item: _complex_sort_key(str(item["pose_name"])))
    tasks.sort(key=lambda item: str(item["complex_id"]))
    return tasks


def _build_reference_source_maps(tasks: List[Dict[str, object]], out_dir: Path) -> Dict[str, object]:
    receptor_meta: Dict[str, str] = {}
    peptide_meta: Dict[str, str] = {}
    for task in tasks:
        receptor_id = str(task["receptor_id"])
        receptor_meta.setdefault(receptor_id, str(task["receptor_pdb"]))
        peptide_id = str(task["peptide_id"])
        peptide_meta.setdefault(peptide_id, str(task["pose_rows"][0]["pose_path"]))
    metadata = {
        "receptor_source_map": receptor_meta,
        "peptide_source_map": peptide_meta,
    }
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "reference_source_map.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def _load_completed_chunk_ids(chunk_results_dir: Path) -> set[int]:
    completed = set()
    for path in sorted(chunk_results_dir.glob("chunk_*.csv")):
        try:
            completed.add(int(path.stem.split("_", 1)[1]))
        except Exception:
            continue
    return completed


def _merge_partial_results(out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    chunk_results_dir = out_dir / "chunk_results"
    chunk_errors_dir = out_dir / "chunk_errors"
    ok_frames: List[pd.DataFrame] = []
    err_frames: List[pd.DataFrame] = []
    for path in sorted(chunk_results_dir.glob("chunk_*.csv")):
        if path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if not df.empty:
            ok_frames.append(df)
    for path in sorted(chunk_errors_dir.glob("chunk_*.csv")):
        if path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if not df.empty:
            err_frames.append(df)
    ok_df = pd.concat(ok_frames, ignore_index=True) if ok_frames else pd.DataFrame()
    err_df = pd.concat(err_frames, ignore_index=True) if err_frames else pd.DataFrame()
    if not ok_df.empty:
        ok_df = ok_df.drop_duplicates(subset=["pose_id"], keep="last").reset_index(drop=True)
        ok_df.to_csv(out_dir / "per_pose_scores.partial.csv", index=False)
    if not err_df.empty:
        err_df = err_df.drop_duplicates(subset=["pose_id"], keep="last").reset_index(drop=True)
        err_df.to_csv(out_dir / "errors.partial.csv", index=False)
    return ok_df, err_df


def _write_progress(progress_path: Path, payload: Dict[str, object]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _finalize_outputs(out_dir: Path) -> Dict[str, object]:
    ok_df, err_df = _merge_partial_results(out_dir)
    if ok_df.empty:
        raise RuntimeError(f"no successful GraphPep rows found in {out_dir}")

    ok_df["rank_in_group"] = ok_df.groupby("group_id")["score"].rank(method="first", ascending=False).astype(int)
    ok_df["rank_in_peptide"] = ok_df.groupby("peptide_id")["score"].rank(method="first", ascending=False).astype(int)
    ok_df = ok_df.sort_values(["peptide_id", "score"], ascending=[True, False]).reset_index(drop=True)
    ok_df.to_csv(out_dir / "per_pose_scores.csv", index=False)

    group_best = (
        ok_df.sort_values(["group_id", "score"], ascending=[True, False])
        .groupby("group_id", as_index=False)
        .first()
        .sort_values(["peptide_id", "score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    group_best["group_rank_in_peptide"] = group_best.groupby("peptide_id")["score"].rank(method="first", ascending=False).astype(int)
    group_best.to_csv(out_dir / "group_best_scores.csv", index=False)

    peptide_summary = (
        group_best.groupby("peptide_id")
        .agg(
            num_receptors=("group_id", "nunique"),
            best_score=("score", "max"),
            best_graphpep_pred_score=("graphpep_pred_score", "min"),
            best_group=("group_id", "first"),
        )
        .reset_index()
        .sort_values("best_score", ascending=False)
        .reset_index(drop=True)
    )
    peptide_summary.to_csv(out_dir / "peptide_summary.csv", index=False)

    timing_columns = [column for column in ok_df.columns if column.startswith("graphpep_timing_")]
    if timing_columns:
        timing_df = (
            ok_df[["complex_id", "group_id", "peptide_id", "receptor_id", "graphpep_assigned_gpu", *timing_columns]]
            .drop_duplicates(subset=["complex_id"], keep="last")
            .sort_values(["peptide_id", "receptor_id", "complex_id"])
            .reset_index(drop=True)
        )
        timing_df.to_csv(out_dir / "timing_per_complex.csv", index=False)
        timing_summary = {
            "num_complexes": int(len(timing_df)),
            "total_seconds_sum": float(timing_df["graphpep_timing_total_seconds"].sum()),
            "forward_seconds_sum": float(timing_df["graphpep_timing_forward_seconds"].sum()),
            "graph_build_seconds_sum": float(timing_df["graphpep_timing_graph_build_seconds"].sum()),
            "batch_to_device_seconds_sum": float(timing_df["graphpep_timing_batch_to_device_seconds"].sum()),
        }
        (out_dir / "timing_summary.json").write_text(
            json.dumps(timing_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if not err_df.empty:
        err_df.to_csv(out_dir / "errors.csv", index=False)
    return {
        "ok_rows": int(len(ok_df)),
        "error_rows": int(len(err_df)),
        "num_groups": int(ok_df["group_id"].nunique()),
        "num_peptides": int(ok_df["peptide_id"].nunique()),
    }


def _build_multimodel_decoy_pdb(pose_paths: List[str], output_pdb: Path) -> None:
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb.open("w", encoding="utf-8") as out_handle:
        for model_index, pose_path in enumerate(pose_paths, start=1):
            out_handle.write(f"MODEL     {model_index}\n")
            with Path(pose_path).open("r", encoding="utf-8", errors="ignore") as in_handle:
                for line in in_handle:
                    if line.startswith("END"):
                        continue
                    if line.startswith(("ATOM", "HETATM", "TER")):
                        out_handle.write(line if line.endswith("\n") else line + "\n")
            out_handle.write("ENDMDL\n")
        out_handle.write("END\n")


def _init_worker(
    graphpep_root: str,
    ckpt_path: str,
    gpu_ids: List[int],
    receptor_source_map: Dict[str, str],
    peptide_source_map: Dict[str, str],
    cache_dir: str,
    enable_stage_timing: bool,
) -> None:
    global _SCORER_STATE
    if _SCORER_STATE is not None:
        return

    process_identity = mp.current_process()._identity
    worker_slot = process_identity[0] - 1 if process_identity else 0
    assigned_gpu = gpu_ids[worker_slot % len(gpu_ids)] if gpu_ids else -1

    os.environ["GraphPep_root"] = graphpep_root
    if assigned_gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

    import sys

    bin_dir = Path(graphpep_root) / "bin"
    if str(bin_dir) not in sys.path:
        sys.path.insert(0, str(bin_dir))

    import esm  # noqa: WPS433
    import MDAnalysis as mda  # noqa: WPS433
    import torch  # noqa: WPS433
    from rdkit import Chem  # noqa: WPS433
    from torch_geometric.data import Batch  # noqa: WPS433

    from config import model_config  # noqa: WPS433
    from model import GraphPpScore  # noqa: WPS433
    from pre import AA3_TO_1, inter_graph  # noqa: WPS433
    from utils import get_pocket  # noqa: WPS433

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = copy.deepcopy(model_config)
    model = GraphPpScore(config)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned_state[key[len("model."):]] = value
        else:
            cleaned_state[key] = value
    model.load_state_dict(cleaned_state, strict=True)
    model.to(device)
    model.eval()

    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model.eval()
    esm_model.to(device)
    batch_converter = alphabet.get_batch_converter()

    cache_root = Path(cache_dir)
    (cache_root / "esm_receptor").mkdir(parents=True, exist_ok=True)
    (cache_root / "esm_peptide").mkdir(parents=True, exist_ok=True)

    _SCORER_STATE = {
        "torch": torch,
        "Chem": Chem,
        "Batch": Batch,
        "mda": mda,
        "AA3_TO_1": AA3_TO_1,
        "config": config,
        "device": device,
        "model": model,
        "esm_model": esm_model,
        "batch_converter": batch_converter,
        "inter_graph": inter_graph,
        "get_pocket": get_pocket,
        "graphpep_root": graphpep_root,
        "receptor_source_map": receptor_source_map,
        "peptide_source_map": peptide_source_map,
        "cache_root": cache_root,
        "assigned_gpu": assigned_gpu,
        "enable_stage_timing": bool(enable_stage_timing),
    }


def _maybe_sync_graphpep_device() -> None:
    global _SCORER_STATE
    if _SCORER_STATE is None:
        return
    if not bool(_SCORER_STATE.get("enable_stage_timing", False)):
        return
    torch = _SCORER_STATE["torch"]
    device = _SCORER_STATE["device"]
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _ensure_clean_reference_pdb(kind: str, item_id: str) -> Path:
    global _SCORER_STATE
    if _SCORER_STATE is None:
        raise RuntimeError("GraphPep worker state is not initialized")
    graphpep_root = Path(str(_SCORER_STATE["graphpep_root"]))
    cache_root = Path(str(_SCORER_STATE["cache_root"]))
    source_map_key = "receptor_source_map" if kind == "receptor" else "peptide_source_map"
    source_map = _SCORER_STATE[source_map_key]
    source_path = Path(str(source_map[item_id]))
    clean_dir = cache_root / ("receptors_clean" if kind == "receptor" else "peptides_clean")
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_path = clean_dir / f"{_safe_stem(item_id)}.pdb"
    if clean_path.exists():
        return clean_path
    tmp_path = clean_path.with_suffix(f".tmp.{os.getpid()}.pdb")
    _run_clean_pdb(graphpep_root, source_path, tmp_path)
    try:
        tmp_path.replace(clean_path)
    except FileExistsError:
        if tmp_path.exists():
            tmp_path.unlink()
    return clean_path


def _sequence_from_pdb(pdb_path: Path) -> str:
    global _SCORER_STATE
    if _SCORER_STATE is None:
        raise RuntimeError("GraphPep worker state is not initialized")
    mda = _SCORER_STATE["mda"]
    aa3_to_1 = _SCORER_STATE["AA3_TO_1"]

    universe = mda.Universe(str(pdb_path))
    letters: List[str] = []
    for residue in universe.residues:
        resname = residue.resname.strip().upper()
        if resname == "HOH":
            continue
        letters.append(aa3_to_1.get(resname, "X"))
    return "".join(letters)


def _load_or_compute_esm_embedding(cache_path: Path, sequence: str, label: str):
    global _SCORER_STATE
    if _SCORER_STATE is None:
        raise RuntimeError("GraphPep worker state is not initialized")
    torch = _SCORER_STATE["torch"]
    esm_model = _SCORER_STATE["esm_model"]
    batch_converter = _SCORER_STATE["batch_converter"]
    device = _SCORER_STATE["device"]

    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    batch = [(label, sequence)]
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)
    lengths = (tokens != 1).sum(dim=1)
    with torch.no_grad():
        outputs = esm_model(tokens, repr_layers=[33], return_contacts=False)
    emb = outputs["representations"][33][0, 1 : lengths[0] - 1].detach().cpu()
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    torch.save(emb, tmp_path)
    tmp_path.replace(cache_path)
    return emb


def _score_complex(task: Dict[str, object]) -> List[Dict[str, object]]:
    global _SCORER_STATE
    if _SCORER_STATE is None:
        raise RuntimeError("GraphPep worker state is not initialized")

    torch = _SCORER_STATE["torch"]
    Chem = _SCORER_STATE["Chem"]
    Batch = _SCORER_STATE["Batch"]
    config = _SCORER_STATE["config"]
    device = _SCORER_STATE["device"]
    model = _SCORER_STATE["model"]
    inter_graph = _SCORER_STATE["inter_graph"]
    get_pocket = _SCORER_STATE["get_pocket"]
    graphpep_root = Path(str(_SCORER_STATE["graphpep_root"]))
    cache_root = Path(str(_SCORER_STATE["cache_root"]))
    enable_stage_timing = bool(_SCORER_STATE.get("enable_stage_timing", False))

    receptor_id = str(task["receptor_id"])
    peptide_id = str(task["peptide_id"])
    pose_rows = list(task["pose_rows"])
    timing_payload: Dict[str, object] = {}

    try:
        total_start = time.perf_counter()
        receptor_clean_cache_hit = (cache_root / "receptors_clean" / f"{_safe_stem(receptor_id)}.pdb").exists()
        receptor_clean_start = time.perf_counter()
        clean_receptor = _ensure_clean_reference_pdb("receptor", receptor_id)
        receptor_clean_seconds = time.perf_counter() - receptor_clean_start

        peptide_clean_cache_hit = (cache_root / "peptides_clean" / f"{_safe_stem(peptide_id)}.pdb").exists()
        peptide_clean_start = time.perf_counter()
        clean_peptide = _ensure_clean_reference_pdb("peptide", peptide_id)
        peptide_clean_seconds = time.perf_counter() - peptide_clean_start

        receptor_sequence_start = time.perf_counter()
        receptor_sequence = _sequence_from_pdb(clean_receptor)
        receptor_sequence_seconds = time.perf_counter() - receptor_sequence_start
        peptide_sequence_start = time.perf_counter()
        peptide_sequence = _sequence_from_pdb(clean_peptide)
        peptide_sequence_seconds = time.perf_counter() - peptide_sequence_start
        if not receptor_sequence or not peptide_sequence:
            raise ValueError("empty receptor/peptide sequence after GraphPep cleaning")

        receptor_esm_cache_path = cache_root / "esm_receptor" / f"{_safe_stem(receptor_id)}.pt"
        receptor_esm_cache_hit = receptor_esm_cache_path.exists()
        receptor_esm_start = time.perf_counter()
        receptor_emb = _load_or_compute_esm_embedding(
            receptor_esm_cache_path,
            receptor_sequence,
            f"receptor::{receptor_id}",
        )
        receptor_esm_seconds = time.perf_counter() - receptor_esm_start
        peptide_esm_cache_path = cache_root / "esm_peptide" / f"{_safe_stem(peptide_id)}.pt"
        peptide_esm_cache_hit = peptide_esm_cache_path.exists()
        peptide_esm_start = time.perf_counter()
        peptide_emb = _load_or_compute_esm_embedding(
            peptide_esm_cache_path,
            peptide_sequence,
            f"peptide::{peptide_id}",
        )
        peptide_esm_seconds = time.perf_counter() - peptide_esm_start

        with tempfile.TemporaryDirectory(prefix=f"graphpep_{task['complex_id']}_") as tmpdir:
            tmp_root = Path(tmpdir)
            raw_decoys = tmp_root / "decoys_raw.pdb"
            clean_decoys = tmp_root / "decoys_clean.pdb"
            decoy_build_start = time.perf_counter()
            _build_multimodel_decoy_pdb([str(row["pose_path"]) for row in pose_rows], raw_decoys)
            decoy_build_seconds = time.perf_counter() - decoy_build_start
            decoy_clean_start = time.perf_counter()
            _run_clean_pdb(graphpep_root, raw_decoys, clean_decoys)
            decoy_clean_seconds = time.perf_counter() - decoy_clean_start

            pocket_start = time.perf_counter()
            pocket_pdb_block = get_pocket(str(clean_decoys), str(clean_receptor), config["dis_threshold"])
            pocket_seconds = time.perf_counter() - pocket_start
            rdkit_parse_start = time.perf_counter()
            protein = Chem.MolFromPDBBlock(pocket_pdb_block, removeHs=True, sanitize=False)
            decoys = Chem.MolFromPDBFile(str(clean_decoys), removeHs=True, sanitize=False)
            rdkit_parse_seconds = time.perf_counter() - rdkit_parse_start
            if protein is None or decoys is None:
                raise ValueError("RDKit failed to parse cleaned GraphPep inputs")
            if decoys.GetNumConformers() != len(pose_rows):
                raise ValueError(
                    f"GraphPep decoy conformer count mismatch: got {decoys.GetNumConformers()}, expect {len(pose_rows)}"
                )

            data_list = []
            valid_pose_rows: List[Dict[str, object]] = []
            empty_rows: List[Dict[str, object]] = []
            graph_build_start = time.perf_counter()
            for conf_idx in range(len(pose_rows)):
                decoy = Chem.Mol(decoys, True)
                decoy.AddConformer(decoys.GetConformer(conf_idx), assignId=True)
                try:
                    atom_data, res_data = inter_graph(decoy, protein)
                except RuntimeError as exc:
                    if "non-empty TensorList" not in str(exc):
                        raise
                    payload = dict(pose_rows[conf_idx])
                    payload["status"] = "ok"
                    payload["error_type"] = ""
                    payload["error_message"] = ""
                    payload["graphpep_fnat"] = 0.0
                    payload["graphpep_fnat_atom"] = 0.0
                    payload["graphpep_pred_score"] = 0.0
                    payload["score"] = 0.0
                    payload["graphpep_assigned_gpu"] = int(_SCORER_STATE["assigned_gpu"])
                    empty_rows.append(payload)
                    continue
                residi = res_data.residi.to(dtype=torch.long) - 1
                residj = res_data.residj.to(dtype=torch.long) - 1
                if residi.numel() > 0:
                    if int(residi.max().item()) >= int(peptide_emb.shape[0]) or int(residi.min().item()) < 0:
                        raise IndexError("GraphPep peptide embedding index out of range")
                    if int(residj.max().item()) >= int(receptor_emb.shape[0]) or int(residj.min().item()) < 0:
                        raise IndexError("GraphPep receptor embedding index out of range")
                    res_data.esm2i = peptide_emb.index_select(0, residi)
                    res_data.esm2j = receptor_emb.index_select(0, residj)
                else:
                    res_data.esm2i = peptide_emb.new_zeros((0, peptide_emb.shape[-1]))
                    res_data.esm2j = receptor_emb.new_zeros((0, receptor_emb.shape[-1]))
                atom_data.res_data = res_data
                atom_data.id = conf_idx + 1
                data_list.append(atom_data)
                valid_pose_rows.append(dict(pose_rows[conf_idx]))
            graph_build_seconds = time.perf_counter() - graph_build_start

            scored_rows: List[Dict[str, object]] = list(empty_rows)
            batch_to_device_seconds = 0.0
            forward_seconds = 0.0
            tensor_collect_seconds = 0.0
            if data_list:
                fnat: List[object] = []
                fnat_atom: List[object] = []
                graphpep_pred_score: List[object] = []

                def score_slices(slice_size: int) -> Tuple[List[object], List[object], List[object], float, float, float]:
                    slice_fnat: List[object] = []
                    slice_fnat_atom: List[object] = []
                    slice_pred_score: List[object] = []
                    slice_batch_to_device_seconds = 0.0
                    slice_forward_seconds = 0.0
                    slice_tensor_collect_seconds = 0.0
                    for start_idx in range(0, len(data_list), slice_size):
                        sub_data_list = data_list[start_idx : start_idx + slice_size]
                        batch_to_device_start = time.perf_counter()
                        batch = Batch.from_data_list(sub_data_list).to(device)
                        _maybe_sync_graphpep_device()
                        slice_batch_to_device_seconds += time.perf_counter() - batch_to_device_start
                        forward_start = time.perf_counter()
                        with torch.no_grad():
                            sub_fnat, _edge_logits, sub_fnat_atom, _edge_logits_atom = model(batch)
                            sub_pred_score = -torch.log1p(sub_fnat)
                        _maybe_sync_graphpep_device()
                        slice_forward_seconds += time.perf_counter() - forward_start

                        tensor_collect_start = time.perf_counter()
                        slice_fnat.extend(_as_list(sub_fnat.detach().cpu().tolist()))
                        slice_fnat_atom.extend(_as_list(sub_fnat_atom.detach().cpu().tolist()))
                        slice_pred_score.extend(_as_list(sub_pred_score.detach().cpu().tolist()))
                        slice_tensor_collect_seconds += time.perf_counter() - tensor_collect_start
                        del batch, sub_fnat, sub_fnat_atom, sub_pred_score
                    return (
                        slice_fnat,
                        slice_fnat_atom,
                        slice_pred_score,
                        slice_batch_to_device_seconds,
                        slice_forward_seconds,
                        slice_tensor_collect_seconds,
                    )

                slice_size = len(data_list)
                while True:
                    try:
                        (
                            fnat,
                            fnat_atom,
                            graphpep_pred_score,
                            batch_to_device_seconds,
                            forward_seconds,
                            tensor_collect_seconds,
                        ) = score_slices(slice_size)
                        break
                    except Exception as exc:
                        if not _is_cuda_oom(exc) or slice_size <= 1:
                            raise
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        slice_size = max(1, slice_size // 2)

                for row, fnat_value, fnat_atom_value, raw_score in zip(valid_pose_rows, fnat, fnat_atom, graphpep_pred_score):
                    payload = dict(row)
                    payload["status"] = "ok"
                    payload["error_type"] = ""
                    payload["error_message"] = ""
                    payload["graphpep_fnat"] = float(fnat_value)
                    payload["graphpep_fnat_atom"] = float(fnat_atom_value)
                    payload["graphpep_pred_score"] = float(raw_score)
                    payload["score"] = float(-raw_score)
                    payload["graphpep_assigned_gpu"] = int(_SCORER_STATE["assigned_gpu"])
                    scored_rows.append(payload)

        if enable_stage_timing:
            timing_payload = {
                "graphpep_timing_total_seconds": float(time.perf_counter() - total_start),
                "graphpep_timing_receptor_clean_seconds": float(receptor_clean_seconds),
                "graphpep_timing_peptide_clean_seconds": float(peptide_clean_seconds),
                "graphpep_timing_receptor_sequence_seconds": float(receptor_sequence_seconds),
                "graphpep_timing_peptide_sequence_seconds": float(peptide_sequence_seconds),
                "graphpep_timing_receptor_esm_seconds": float(receptor_esm_seconds),
                "graphpep_timing_peptide_esm_seconds": float(peptide_esm_seconds),
                "graphpep_timing_decoy_build_seconds": float(decoy_build_seconds),
                "graphpep_timing_decoy_clean_seconds": float(decoy_clean_seconds),
                "graphpep_timing_pocket_seconds": float(pocket_seconds),
                "graphpep_timing_rdkit_parse_seconds": float(rdkit_parse_seconds),
                "graphpep_timing_graph_build_seconds": float(graph_build_seconds),
                "graphpep_timing_batch_to_device_seconds": float(batch_to_device_seconds),
                "graphpep_timing_forward_seconds": float(forward_seconds),
                "graphpep_timing_tensor_collect_seconds": float(tensor_collect_seconds),
                "graphpep_timing_receptor_clean_cache_hit": int(receptor_clean_cache_hit),
                "graphpep_timing_peptide_clean_cache_hit": int(peptide_clean_cache_hit),
                "graphpep_timing_receptor_esm_cache_hit": int(receptor_esm_cache_hit),
                "graphpep_timing_peptide_esm_cache_hit": int(peptide_esm_cache_hit),
                "graphpep_timing_num_valid_poses": int(len(valid_pose_rows)),
                "graphpep_timing_num_empty_poses": int(len(empty_rows)),
            }
            for row in scored_rows:
                row.update(timing_payload)

        by_pose = {str(row["pose_id"]): row for row in scored_rows}
        return [by_pose[str(row["pose_id"])] for row in pose_rows]
    except Exception as exc:  # noqa: WPS429
        trace = traceback.format_exc(limit=5)
        error_rows: List[Dict[str, object]] = []
        for row in pose_rows:
            payload = dict(row)
            payload["status"] = "error"
            payload["error_type"] = type(exc).__name__
            payload["error_message"] = f"{exc}\n{trace}"
            payload["graphpep_fnat"] = None
            payload["graphpep_fnat_atom"] = None
            payload["graphpep_pred_score"] = None
            payload["score"] = None
            payload["graphpep_assigned_gpu"] = int(_SCORER_STATE["assigned_gpu"])
            if timing_payload:
                payload.update(timing_payload)
            error_rows.append(payload)
        return error_rows


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    input_dir = out_dir / "input"
    chunk_results_dir = out_dir / "chunk_results"
    chunk_errors_dir = out_dir / "chunk_errors"
    progress_dir = out_dir / "progress"
    for path in [out_dir, input_dir, chunk_results_dir, chunk_errors_dir, progress_dir]:
        path.mkdir(parents=True, exist_ok=True)

    rows = load_cross_pose_rows(args.results_root.resolve(), args.flow_input_csv.resolve())
    all_rows_df = pd.DataFrame(rows)
    input_table = input_dir / "all_pose_rows.csv"
    if not input_table.exists():
        all_rows_df.to_csv(input_table, index=False)

    tasks = _group_complex_tasks(rows)
    pd.DataFrame(
        [
            {
                "complex_id": task["complex_id"],
                "group_id": task["group_id"],
                "source_group_id": task["source_group_id"],
                "peptide_id": task["peptide_id"],
                "receptor_id": task["receptor_id"],
                "receptor_pdb": task["receptor_pdb"],
                "num_poses": len(task["pose_rows"]),
            }
            for task in tasks
        ]
    ).to_csv(input_dir / "all_complex_tasks.csv", index=False)

    reference_maps = _build_reference_source_maps(tasks, out_dir)

    chunk_size = max(1, int(args.chunk_size))
    chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    total_chunks = len(chunks)

    gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty")
    workers = max(1, int(args.workers))

    chunk_indices = list(range(args.start_chunk, total_chunks))
    if args.max_chunks and args.max_chunks > 0:
        chunk_indices = chunk_indices[: args.max_chunks]

    completed = _load_completed_chunk_ids(chunk_results_dir) if args.resume else set()
    pending_chunk_indices = [idx for idx in chunk_indices if idx not in completed]

    start_time = time.perf_counter()
    if pending_chunk_indices:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(
                str(args.graphpep_root.resolve()),
                str(args.ckpt_path.resolve()),
                gpu_ids,
                reference_maps["receptor_source_map"],
                reference_maps["peptide_source_map"],
                str((out_dir / "cache").resolve()),
                bool(args.enable_stage_timing),
            ),
        ) as pool:
            processed_count = 0
            for chunk_idx in pending_chunk_indices:
                chunk = chunks[chunk_idx]
                pd.DataFrame(
                    [
                        {
                            "complex_id": task["complex_id"],
                            "peptide_id": task["peptide_id"],
                            "receptor_id": task["receptor_id"],
                            "num_poses": len(task["pose_rows"]),
                        }
                        for task in chunk
                    ]
                ).to_csv(input_dir / f"chunk_{chunk_idx:04d}.csv", index=False)
                processed_count += 1
                print(
                    json.dumps(
                        {
                            "event": "chunk_start",
                            "chunk_idx": chunk_idx,
                            "chunk_complexes": len(chunk),
                            "processed_chunk_count": processed_count,
                            "scheduled_chunks": len(pending_chunk_indices),
                            "workers": workers,
                            "gpu_ids": gpu_ids,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

                batch_results = list(pool.imap(_score_complex, chunk, chunksize=1))
                flat_rows = [row for result in batch_results for row in result]
                ok_rows = [row for row in flat_rows if row["status"] == "ok"]
                err_rows = [row for row in flat_rows if row["status"] != "ok"]

                ok_path = chunk_results_dir / f"chunk_{chunk_idx:04d}.csv"
                err_path = chunk_errors_dir / f"chunk_{chunk_idx:04d}.csv"
                if ok_rows:
                    pd.DataFrame(ok_rows).to_csv(ok_path, index=False)
                else:
                    ok_path.write_text("", encoding="utf-8")
                if err_rows:
                    pd.DataFrame(err_rows).to_csv(err_path, index=False)
                else:
                    err_path.write_text("", encoding="utf-8")

                partial_ok, partial_err = _merge_partial_results(out_dir)
                progress_payload = {
                    "results_root": str(args.results_root.resolve()),
                    "flow_input_csv": str(args.flow_input_csv.resolve()),
                    "graphpep_root": str(args.graphpep_root.resolve()),
                    "ckpt_path": str(args.ckpt_path.resolve()),
                    "workers": workers,
                    "gpu_ids": gpu_ids,
                    "chunk_size": chunk_size,
                    "completed_chunks": sorted(_load_completed_chunk_ids(chunk_results_dir)),
                    "latest_chunk": chunk_idx,
                    "partial_ok_rows": int(len(partial_ok)),
                    "partial_error_rows": int(len(partial_err)),
                    "elapsed_sec": round(time.perf_counter() - start_time, 2),
                }
                _write_progress(progress_dir / "progress.json", progress_payload)
                print(
                    json.dumps(
                        {
                            "event": "chunk_done",
                            "chunk_idx": chunk_idx,
                            "ok_rows": len(ok_rows),
                            "error_rows": len(err_rows),
                            "partial_ok_rows": int(len(partial_ok)),
                            "partial_error_rows": int(len(partial_err)),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    final_stats = _finalize_outputs(out_dir)
    report = {
        "results_root": str(args.results_root.resolve()),
        "flow_input_csv": str(args.flow_input_csv.resolve()),
        "graphpep_root": str(args.graphpep_root.resolve()),
        "ckpt_path": str(args.ckpt_path.resolve()),
        "workers": workers,
        "gpu_ids": gpu_ids,
        "chunk_size": chunk_size,
        "scheduled_chunks": len(chunk_indices),
        "total_chunks": total_chunks,
        "elapsed_sec": round(time.perf_counter() - start_time, 2),
        "ranking_score_definition": "score = -graphpep_pred_score",
        "enable_stage_timing": bool(args.enable_stage_timing),
        **final_stats,
    }
    (out_dir / "scoring_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
