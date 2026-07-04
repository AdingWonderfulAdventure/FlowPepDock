from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..io import write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PoseCred-IPG record snapshots and optionally delete bad files.")
    parser.add_argument("--snapshot_dir", type=Path, required=True)
    parser.add_argument("--delete_bad", action="store_true")
    parser.add_argument("--rewrite_manifests", action="store_true")
    return parser.parse_args()


def _validate_npz(path: Path) -> str | None:
    try:
        data = np.load(path, allow_pickle=False)
        required = ["pose_id", "group_id", "node_feat", "edge_index", "edge_feat", "global_feat"]
        for key in required:
            if key not in data:
                return f"missing_key:{key}"
        edge_index = data["edge_index"]
        if edge_index.ndim == 1 and edge_index.size == 0:
            return None
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            return f"bad_edge_shape:{tuple(edge_index.shape)}"
        return None
    except Exception as exc:
        return repr(exc)


def main() -> None:
    args = parse_args()
    snapshot_dir = args.snapshot_dir
    bad_rows: List[Dict[str, str]] = []

    for split_root in sorted(snapshot_dir.glob("*/")):
        # allow passing either snapshot root or variant root
        if split_root.name in {"train", "val"}:
            variant_roots = [snapshot_dir]
            break
    else:
        variant_roots = sorted([path for path in snapshot_dir.iterdir() if path.is_dir()])

    for variant_root in variant_roots:
        for split in ["train", "val"]:
            records_dir = variant_root / split / "records"
            if not records_dir.exists():
                continue
            for npz_path in sorted(records_dir.glob("*.npz")):
                problem = _validate_npz(npz_path)
                if problem is not None:
                    bad_rows.append(
                        {
                            "variant": variant_root.name,
                            "split": split,
                            "path": str(npz_path),
                            "problem": problem,
                        }
                    )
                    if args.delete_bad:
                        npz_path.unlink(missing_ok=True)
            if args.rewrite_manifests:
                record_paths = sorted(records_dir.glob("*.npz"))
                write_manifest(record_paths, variant_root / split / "manifest.txt")

    report = {
        "snapshot_dir": str(snapshot_dir),
        "bad_count": len(bad_rows),
        "delete_bad": bool(args.delete_bad),
        "rewrite_manifests": bool(args.rewrite_manifests),
        "bad_rows": bad_rows[:1000],
    }
    out_path = snapshot_dir / "validation_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "bad_rows"}, indent=2, ensure_ascii=False))
    if bad_rows:
        for row in bad_rows[:100]:
            print(f'{row["variant"]}\t{row["split"]}\t{row["problem"]}\t{row["path"]}')


if __name__ == "__main__":
    main()
