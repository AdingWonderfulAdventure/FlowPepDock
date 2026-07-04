#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader


RANK_RE = re.compile(r"^rank(\d+)\.pdb$")


@dataclass
class RelaxTask:
    task_id: str
    task_type: str
    rank: int
    frame: Optional[int]
    protein_pdb: str
    input_peptide_pdb: str
    output_peptide_pdb: str


def _relax_one(task: RelaxTask) -> dict:
    # Lazy import in worker process, so each worker initializes PyRosetta once.
    import sys

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from utils.pyrosetta_utils import relax_score

    t0 = time.perf_counter()
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"relax_{task.task_id}_"))
    score = None
    err = ""
    status = "ok"
    try:
        tmp_in = tmp_dir / "input_peptide.pdb"
        tmp_out = tmp_dir / "output_peptide.pdb"
        shutil.copy2(task.input_peptide_pdb, tmp_in)
        score = relax_score(
            (
                task.protein_pdb,
                str(tmp_in),
                str(tmp_out),
                True,
            )
        )
        Path(task.output_peptide_pdb).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_out), task.output_peptide_pdb)
    except Exception as exc:  # noqa: BLE001
        status = "error"
        err = str(exc)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.perf_counter() - t0
    out = asdict(task)
    out.update(
        {
            "elapsed_sec": elapsed,
            "status": status,
            "error": err,
            "score": score,
        }
    )
    return out


def _split_trajectory_frames(traj_pdb: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    u = mda.Universe(str(traj_pdb))
    frame_files = []
    for i, _ in enumerate(u.trajectory):
        fp = out_dir / f"frame_{i:04d}.pdb"
        u.atoms.write(str(fp))
        frame_files.append(fp)
    return frame_files


def _assemble_trajectory(template_traj: Path, relaxed_frame_files: List[Path], out_traj: Path) -> float:
    t0 = time.perf_counter()
    template = mda.Universe(str(template_traj))
    template_keys = [(int(a.resid), str(a.resname), str(a.name)) for a in template.atoms]
    coords = []
    for f in relaxed_frame_files:
        u = mda.Universe(str(f))
        pos_by_key = {}
        pos_by_resid_name = {}
        for a in u.atoms:
            key = (int(a.resid), str(a.resname), str(a.name))
            if key not in pos_by_key:
                pos_by_key[key] = a.position.copy()
            key2 = (int(a.resid), str(a.name))
            if key2 not in pos_by_resid_name:
                pos_by_resid_name[key2] = a.position.copy()

        frame_pos = []
        missing = []
        for resid, resname, atom_name in template_keys:
            p = pos_by_key.get((resid, resname, atom_name))
            if p is None:
                p = pos_by_resid_name.get((resid, atom_name))
            if p is None:
                missing.append((resid, resname, atom_name))
            else:
                frame_pos.append(p)
        if missing:
            raise ValueError(
                f"atom_mapping_failed frame={f.name} missing={missing[:5]} total_missing={len(missing)}"
            )
        coords.append(np.asarray(frame_pos, dtype=np.float32))
    coord_arr = np.asarray(coords, dtype=np.float32)
    template.load_new(coord_arr, format=MemoryReader)
    with mda.Writer(
        str(out_traj),
        multiframe=True,
        bonds=None,
        n_atoms=template.atoms.n_atoms,
    ) as w:
        for _ in template.trajectory:
            w.write(template)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parallel FastRelax for final poses + reverseprocess trajectories, preserving source files."
    )
    ap.add_argument("--input_dir", required=True, help="Directory containing rank*.pdb and rank*_reverseprocess.pdb")
    ap.add_argument("--output_dir", required=True, help="Output directory for relaxed files and timing logs")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Parallel workers")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_split_root = output_dir / "_tmp_split_frames"
    tmp_relaxed_frames_root = output_dir / "_tmp_relaxed_frames"
    tmp_split_root.mkdir(parents=True, exist_ok=True)
    tmp_relaxed_frames_root.mkdir(parents=True, exist_ok=True)

    protein_pdb = input_dir / "3wop_protein_raw.pdb"
    if not protein_pdb.is_file():
        raise FileNotFoundError(f"Missing protein file: {protein_pdb}")

    rank_files = []
    for p in sorted(input_dir.glob("rank*.pdb")):
        m = RANK_RE.match(p.name)
        if not m:
            continue
        rank_files.append((int(m.group(1)), p))

    if not rank_files:
        raise RuntimeError(f"No rank*.pdb found in {input_dir}")

    overall_start = time.perf_counter()

    # 1) Split trajectories into per-frame PDBs.
    split_start = time.perf_counter()
    rank_to_split_frames = {}
    for rank, _ in rank_files:
        traj = input_dir / f"rank{rank}_reverseprocess.pdb"
        if not traj.is_file():
            continue
        frame_dir = tmp_split_root / f"rank{rank}"
        frame_files = _split_trajectory_frames(traj, frame_dir)
        rank_to_split_frames[rank] = frame_files
    split_elapsed = time.perf_counter() - split_start

    # 2) Build tasks.
    tasks: List[RelaxTask] = []
    # Final rank poses.
    for rank, rank_file in rank_files:
        out_final = output_dir / f"rank{rank}_relaxed.pdb"
        tasks.append(
            RelaxTask(
                task_id=f"final_r{rank}",
                task_type="final",
                rank=rank,
                frame=None,
                protein_pdb=str(protein_pdb),
                input_peptide_pdb=str(rank_file),
                output_peptide_pdb=str(out_final),
            )
        )
    # Trajectory frames.
    for rank, frame_files in rank_to_split_frames.items():
        for i, frame_pdb in enumerate(frame_files):
            out_frame = tmp_relaxed_frames_root / f"rank{rank}" / f"frame_{i:04d}_relaxed.pdb"
            tasks.append(
                RelaxTask(
                    task_id=f"traj_r{rank}_f{i:04d}",
                    task_type="trajectory_frame",
                    rank=rank,
                    frame=i,
                    protein_pdb=str(protein_pdb),
                    input_peptide_pdb=str(frame_pdb),
                    output_peptide_pdb=str(out_frame),
                )
            )

    # 3) Parallel relax.
    relax_start = time.perf_counter()
    rows = []
    max_workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_to_task = {ex.submit(_relax_one, t): t for t in tasks}
        for fut in as_completed(fut_to_task):
            rows.append(fut.result())
    relax_elapsed = time.perf_counter() - relax_start

    # 4) Reassemble relaxed continuous trajectories.
    assemble_start = time.perf_counter()
    reassemble_rows = []
    for rank, _ in rank_files:
        traj = input_dir / f"rank{rank}_reverseprocess.pdb"
        if not traj.is_file():
            continue
        relaxed_frame_dir = tmp_relaxed_frames_root / f"rank{rank}"
        relaxed_frames = sorted(relaxed_frame_dir.glob("frame_*_relaxed.pdb"))
        if not relaxed_frames:
            reassemble_rows.append(
                {
                    "rank": rank,
                    "status": "error",
                    "error": "no_relaxed_frames",
                    "frames": 0,
                    "elapsed_sec": 0.0,
                }
            )
            continue
        out_traj = output_dir / f"rank{rank}_reverseprocess_relaxed.pdb"
        try:
            t = _assemble_trajectory(traj, relaxed_frames, out_traj)
            reassemble_rows.append(
                {
                    "rank": rank,
                    "status": "ok",
                    "error": "",
                    "frames": len(relaxed_frames),
                    "elapsed_sec": t,
                }
            )
        except Exception as exc:  # noqa: BLE001
            reassemble_rows.append(
                {
                    "rank": rank,
                    "status": "error",
                    "error": str(exc),
                    "frames": len(relaxed_frames),
                    "elapsed_sec": 0.0,
                }
            )
    assemble_elapsed = time.perf_counter() - assemble_start

    overall_elapsed = time.perf_counter() - overall_start

    # 5) Persist logs.
    task_csv = output_dir / "relax_task_timing.csv"
    with task_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "task_type",
                "rank",
                "frame",
                "protein_pdb",
                "input_peptide_pdb",
                "output_peptide_pdb",
                "elapsed_sec",
                "status",
                "error",
                "score",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    reassemble_csv = output_dir / "trajectory_reassemble_timing.csv"
    with reassemble_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["rank", "status", "error", "frames", "elapsed_sec"],
        )
        w.writeheader()
        for r in reassemble_rows:
            w.writerow(r)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    err_rows = [r for r in rows if r["status"] != "ok"]
    final_ok = [r for r in ok_rows if r["task_type"] == "final"]
    traj_ok = [r for r in ok_rows if r["task_type"] == "trajectory_frame"]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "workers": max_workers,
        "counts": {
            "total_tasks": len(tasks),
            "ok_tasks": len(ok_rows),
            "error_tasks": len(err_rows),
            "final_ok": len(final_ok),
            "trajectory_frame_ok": len(traj_ok),
            "rank_count": len(rank_files),
            "reassemble_ok": sum(1 for r in reassemble_rows if r["status"] == "ok"),
            "reassemble_error": sum(1 for r in reassemble_rows if r["status"] != "ok"),
        },
        "elapsed_sec": {
            "split_trajectory": split_elapsed,
            "parallel_relax": relax_elapsed,
            "reassemble_trajectory": assemble_elapsed,
            "overall": overall_elapsed,
        },
        "throughput": {
            "tasks_per_sec_relax_phase": (len(ok_rows) / relax_elapsed) if relax_elapsed > 0 else None
        },
    }
    summary_json = output_dir / "relax_timing_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"task_timing_csv={task_csv}")
    print(f"reassemble_timing_csv={reassemble_csv}")
    print(f"summary_json={summary_json}")


if __name__ == "__main__":
    main()
