#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用推理阶段生成的 rank*_ref2015.pdb / rank*_relaxed.pdb 计算复合物 RMSD（受体+肽）。
默认只算 RMSD，DockQ 可选（需要提供 --dockq_cmd）。
输入:
  --pred_root: 推理输出目录，内部按 complex_name/ 存放 rank*.pdb / pose*.pdb
  --csv: 包含 complex_name、peptide_pdb、receptor_pdb 列的 CSV
输出:
  --output: metrics.csv，列为 complex_name,source,pose,complex_rmsd,peptide_ca_rmsd[,dockq,fnat,irmsd,lrmsd]
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms


def _safe_str(val) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    return str(val)


def _load_ca(path: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """返回 CA 坐标 (N,3)。"""
    try:
        u = mda.Universe(str(path))
        sel = u.select_atoms("protein and name CA")
        if sel.n_atoms == 0:
            return None, "no_CA"
        return sel.positions.astype(np.float64), None
    except Exception as e:  # noqa: BLE001
        return None, f"load_fail:{e}"


def _best_rmsd(ref_pos: np.ndarray, pred_pos: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    """允许长度不一致，取前 min_len 计算 RMSD。"""
    min_len = min(len(ref_pos), len(pred_pos))
    if min_len == 0:
        return None, "empty_pair"
    ref = ref_pos[:min_len]
    mob = pred_pos[:min_len]
    try:
        val = rms.rmsd(ref, mob, center=True, superposition=True)
        return float(val), None
    except Exception as e:  # noqa: BLE001
        return None, f"rmsd_fail:{e}"


def _write_complex(rec_pdb: Path, pep_pdb: Path, out_pdb: Path, chain_rec="A", chain_pep="B") -> bool:
    """将受体+肽合并，强制 chainID。用于 DockQ 和复合物 RMSD。"""
    try:
        u_rec = mda.Universe(str(rec_pdb))
        u_pep = mda.Universe(str(pep_pdb))
        # 统一链 ID
        if len(u_rec.segments) == 1:
            u_rec.atoms.chainIDs = chain_rec
            u_rec.segments.segids = [chain_rec]
        else:
            u_rec.atoms.chainIDs = chain_rec
        if len(u_pep.segments) == 1:
            u_pep.atoms.chainIDs = chain_pep
            u_pep.segments.segids = [chain_pep]
        else:
            u_pep.atoms.chainIDs = chain_pep
        # 合并
        new = mda.Merge(u_rec.atoms, u_pep.atoms)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new.atoms.write(str(out_pdb))
        return True
    except Exception:
        return False


def _extract_dockq_metrics(data: dict) -> dict:
    if not isinstance(data, dict):
        return {"dockq": None, "fnat": None, "irmsd": None, "lrmsd": None}

    payload = None
    best_result = data.get("best_result")
    if isinstance(best_result, dict) and best_result:
        if "AB" in best_result and isinstance(best_result.get("AB"), dict):
            payload = best_result.get("AB")
        else:
            first_val = next(iter(best_result.values()), None)
            if isinstance(first_val, dict):
                payload = first_val

    if payload is None and isinstance(data.get("best"), dict):
        payload = data.get("best")
    if payload is None:
        payload = data

    def pick(keys, fallback=None):
        for key in keys:
            if key in payload:
                return payload[key]
            lower = key.lower()
            if lower in payload:
                return payload[lower]
        if fallback:
            for key in fallback:
                if key in data:
                    return data[key]
                lower = key.lower()
                if lower in data:
                    return data[lower]
        return None

    return {
        "dockq": pick(["DockQ", "dockq"], fallback=["best_dockq"]),
        "fnat": pick(["Fnat", "fnat"]),
        "irmsd": pick(["iRMSD", "irmsd", "i_rmsd"]),
        "lrmsd": pick(["LRMSD", "lRMSD", "lrmsd", "l_rmsd"]),
    }


def _run_dockq(cmd: str, model: Path, native: Path) -> Tuple[Optional[dict], Optional[str]]:
    import subprocess
    import json as _json

    try:
        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / "out.json"
            run_cmd = [
                cmd,
                str(model),
                str(native),
                "--json",
                str(out_json),
                "--mapping",
                "AB:AB",
                "--allowed_mismatches",
                "5",
            ]
            res = subprocess.run(run_cmd, capture_output=True, text=True)
            if res.returncode != 0:
                return None, f"dockq_fail:{res.stderr.strip() or res.stdout.strip()}"
            if out_json.is_file():
                data = _json.loads(out_json.read_text())
                return _extract_dockq_metrics(data), None
            return None, "dockq_no_json"
    except Exception as e:  # noqa: BLE001
        return None, f"dockq_err:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", required=True, help="推理输出根目录")
    ap.add_argument("--csv", required=True, help="包含 complex_name / peptide_pdb / receptor_pdb 的 CSV")
    ap.add_argument("--output", required=True, help="输出 metrics.csv")
    ap.add_argument("--dockq_cmd", default=None, help="DockQ 可执行文件（可选）")
    ap.add_argument(
        "--all_poses",
        action="store_true",
        help="输出所有候选 pose 的 RMSD（默认仅保留最优 pose）",
    )
    args = ap.parse_args()

    pred_root = Path(args.pred_root)
    df = pd.read_csv(args.csv)
    rows = []
    skipped = []

    for _, row in df.iterrows():
        raw_name = (row.get("complex_name") or row.get("pdb_id") or "").strip()
        name = raw_name
        source = _safe_str(row.get("source", ""))
        if not name:
            continue
        pred_dir = pred_root / name
        if not pred_dir.is_dir():
            lowered_dir = pred_root / name.lower()
            if lowered_dir.is_dir():
                pred_dir = lowered_dir
            else:
                uppered_dir = pred_root / name.upper()
                if uppered_dir.is_dir():
                    pred_dir = uppered_dir
        if not pred_dir.is_dir():
            skipped.append((name, "no_pred_dir"))
            continue

        # 参考受体
        rec_ref = Path(str(row.get("receptor_pdb") or "")).expanduser()
        if not rec_ref.is_file():
            skipped.append((name, "no_receptor_ref"))
            continue
        # 参考肽
        pep_ref = Path(str(row.get("peptide_pdb") or "")).expanduser()
        if not pep_ref.is_file():
            skipped.append((name, "no_peptide_ref"))
            continue
        # 参考肽 CA
        ref_ca, err = _load_ca(pep_ref)
        if ref_ca is None:
            skipped.append((name, err or "ref_pep_load_fail"))
            continue
        # 参考复合物坐标（全部 protein 原子）
        try:
            u_rec = mda.Universe(str(rec_ref))
            u_pep = mda.Universe(str(pep_ref))
            if len(u_rec.segments) == 1:
                u_rec.atoms.chainIDs = "A"
                u_rec.segments.segids = ["A"]
            else:
                u_rec.atoms.chainIDs = "A"
            if len(u_pep.segments) == 1:
                u_pep.atoms.chainIDs = "B"
                u_pep.segments.segids = ["B"]
            else:
                u_pep.atoms.chainIDs = "B"
            u_ref = mda.Merge(u_rec.atoms, u_pep.atoms)
            ref_all = u_ref.select_atoms("protein").positions.astype(np.float64)
        except Exception as e:  # noqa: BLE001
            skipped.append((name, f"ref_complex_fail:{e}"))
            continue

        # 候选pose
        cand = (
            sorted(pred_dir.glob("rank*_relaxed.pdb"))
            + sorted(pred_dir.glob("rank*_ref2015.pdb"))
            + sorted(pred_dir.glob("rank*.pdb"))
            + sorted(pred_dir.glob("pose*.pdb"))
        )
        if not cand:
            skipped.append((name, "no_rank_or_pose_pdb"))
            continue

        if args.all_poses:
            any_ok = False
            for p in cand:
                mob_ca, err = _load_ca(p)
                if mob_ca is None:
                    continue
                pep_rmsd, err = _best_rmsd(ref_ca, mob_ca)
                if pep_rmsd is None:
                    continue
                try:
                    u_pred_pep = mda.Universe(str(p))
                    if len(u_pred_pep.segments) == 1:
                        u_pred_pep.atoms.chainIDs = "B"
                        u_pred_pep.segments.segids = ["B"]
                    else:
                        u_pred_pep.atoms.chainIDs = "B"
                    u_pred = mda.Merge(u_rec.atoms, u_pred_pep.atoms)
                    mob_all = u_pred.select_atoms("protein").positions.astype(np.float64)
                    comp_rmsd, err2 = _best_rmsd(ref_all, mob_all)
                    if comp_rmsd is None:
                        continue
                except Exception:
                    continue

                dockq_val = ""
                fnat_val = ""
                irmsd_val = ""
                lrmsd_val = ""
                if args.dockq_cmd and rec_ref.is_file():
                    with tempfile.TemporaryDirectory() as td:
                        tmp_native = Path(td) / "native.pdb"
                        tmp_model = Path(td) / "model.pdb"
                        if _write_complex(rec_ref, pep_ref, tmp_native):
                            if _write_complex(rec_ref, p, tmp_model):
                                metrics, derr = _run_dockq(
                                    args.dockq_cmd, tmp_model, tmp_native
                                )
                                if metrics:
                                    dockq_val = metrics.get("dockq") or ""
                                    fnat_val = metrics.get("fnat") or ""
                                    irmsd_val = metrics.get("irmsd") or ""
                                    lrmsd_val = metrics.get("lrmsd") or ""

                rows.append(
                    {
                        "complex_name": name,
                        "source": source,
                        "pose": p.name,
                        "complex_rmsd": comp_rmsd,
                        "peptide_ca_rmsd": pep_rmsd,
                        "dockq": dockq_val,
                        "fnat": fnat_val,
                        "irmsd": irmsd_val,
                        "lrmsd": lrmsd_val,
                    }
                )
                any_ok = True
            if not any_ok:
                skipped.append((name, "rmsd_fail"))
            continue

        best = None
        best_file = None
        best_pep_rmsd = None
        for p in cand:
            mob_ca, err = _load_ca(p)
            if mob_ca is None:
                continue
            pep_rmsd, err = _best_rmsd(ref_ca, mob_ca)
            if pep_rmsd is None:
                continue
            # 复合物 RMSD
            try:
                u_pred_pep = mda.Universe(str(p))
                if len(u_pred_pep.segments) == 1:
                    u_pred_pep.atoms.chainIDs = "B"
                    u_pred_pep.segments.segids = ["B"]
                else:
                    u_pred_pep.atoms.chainIDs = "B"
                u_pred = mda.Merge(u_rec.atoms, u_pred_pep.atoms)
                mob_all = u_pred.select_atoms("protein").positions.astype(np.float64)
                comp_rmsd, err2 = _best_rmsd(ref_all, mob_all)
                if comp_rmsd is None:
                    continue
            except Exception:
                continue
            if best is None or comp_rmsd < best:
                best = comp_rmsd
                best_file = p.name
                best_pep_rmsd = pep_rmsd

        if best is None:
            skipped.append((name, "rmsd_fail"))
            continue

        dockq_val = ""
        fnat_val = ""
        irmsd_val = ""
        lrmsd_val = ""
        if args.dockq_cmd and rec_ref.is_file():
            with tempfile.TemporaryDirectory() as td:
                tmp_native = Path(td) / "native.pdb"
                tmp_model = Path(td) / "model.pdb"
                # 参考复合物
                if _write_complex(rec_ref, pep_ref, tmp_native):
                    # 预测复合物
                    pred_pose_path = pred_dir / best_file
                    if _write_complex(rec_ref, pred_pose_path, tmp_model):
                        metrics, derr = _run_dockq(args.dockq_cmd, tmp_model, tmp_native)
                        if metrics:
                            dockq_val = metrics.get("dockq") or ""
                            fnat_val = metrics.get("fnat") or ""
                            irmsd_val = metrics.get("irmsd") or ""
                            lrmsd_val = metrics.get("lrmsd") or ""

        rows.append(
            {
                "complex_name": name,
                "source": source,
                "pose": best_file,
                "complex_rmsd": best,
                "peptide_ca_rmsd": best_pep_rmsd if best_pep_rmsd is not None else "",
                "dockq": dockq_val,
                "fnat": fnat_val,
                "irmsd": irmsd_val,
                "lrmsd": lrmsd_val,
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {len(rows)} rows -> {out_path}")
    if skipped:
        for n, r in skipped[:10]:
            print(f"[WARN] skipped {n}: {r}")
        print(f"[WARN] skipped total {len(skipped)} complexes")


if __name__ == "__main__":
    main()
