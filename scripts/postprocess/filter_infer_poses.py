"""Filter inference poses by clash/com distance and export best rank1.

Usage:
  python "scripts/postprocess/filter_infer_poses.py" \
    --pred_root "results/infer_run" \
    --out_root "results/infer_run_filtered" \
    --clash_min_dist 1.5 \
    --max_com_dist 20.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import distance_array
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"MDAnalysis import failed: {exc}")


def _load_positions(pdb_path: Path) -> np.ndarray:
    u = mda.Universe(str(pdb_path))
    return u.atoms.positions.astype(np.float64)


def _min_dist(rec_pos: np.ndarray, pep_pos: np.ndarray) -> float:
    d = distance_array(rec_pos, pep_pos)
    return float(d.min())


def _com_dist(rec_pos: np.ndarray, pep_pos: np.ndarray) -> float:
    rec_com = rec_pos.mean(axis=0)
    pep_com = pep_pos.mean(axis=0)
    return float(np.linalg.norm(rec_com - pep_com))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--clash_min_dist", type=float, default=1.5)
    parser.add_argument("--max_com_dist", type=float, default=20.0)
    parser.add_argument("--keep_all", action="store_true", help="Copy all passing poses")
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    kept = 0
    total = 0
    for complex_dir in sorted(p for p in pred_root.iterdir() if p.is_dir()):
        name = complex_dir.name
        rec_pdb = complex_dir / f"{name}_protein_raw.pdb"
        if not rec_pdb.exists():
            # skip non-complex dirs
            continue
        cand = sorted(complex_dir.glob("rank*.pdb"))
        if not cand:
            continue
        try:
            rec_pos = _load_positions(rec_pdb)
        except Exception:
            continue
        best = None
        best_score = None
        pass_list = []
        for p in cand:
            total += 1
            try:
                pep_pos = _load_positions(p)
                min_d = _min_dist(rec_pos, pep_pos)
                com_d = _com_dist(rec_pos, pep_pos)
            except Exception:
                continue
            passed = (min_d >= args.clash_min_dist) and (com_d <= args.max_com_dist)
            rows.append((name, p.name, min_d, com_d, int(passed)))
            if passed:
                pass_list.append(p)
                score = (com_d, -min_d)
                if best_score is None or score < best_score:
                    best_score = score
                    best = p
        if not pass_list:
            # fallback: choose minimal com distance even if not passing
            fallback = None
            fallback_score = None
            for p in cand:
                try:
                    pep_pos = _load_positions(p)
                    min_d = _min_dist(rec_pos, pep_pos)
                    com_d = _com_dist(rec_pos, pep_pos)
                except Exception:
                    continue
                score = (com_d, -min_d)
                if fallback_score is None or score < fallback_score:
                    fallback_score = score
                    fallback = p
            best = fallback
        if best is None:
            continue
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.keep_all:
            for p in pass_list:
                shutil.copy2(p, out_dir / p.name)
        else:
            shutil.copy2(best, out_dir / "rank1.pdb")
        kept += 1

    metrics_csv = out_root / "filter_metrics.csv"
    with metrics_csv.open("w") as f:
        f.write("complex_name,pose,min_dist,com_dist,pass\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]}\n")

    print(f"done complexes={kept} poses={total} metrics={metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
