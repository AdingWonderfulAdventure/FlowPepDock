from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List

import pandas as pd

from ..io import load_pose_record, save_pose_record_shard
from ..train import load_record_refs_from_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack posecred npz records into pt shards.")
    parser.add_argument("--records_index", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--shard_size", type=int, default=512)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = load_record_refs_from_index(args.records_index)
    shard_rows: List[dict] = []
    build_rows: List[dict] = []
    start = time.perf_counter()
    num_shards = math.ceil(len(refs) / max(1, args.shard_size))
    for shard_id in range(num_shards):
        shard_refs = refs[shard_id * args.shard_size : (shard_id + 1) * args.shard_size]
        shard_path = out_dir / f"shard_{shard_id:05d}.pt"
        shard_start = time.perf_counter()
        if not (args.skip_existing and shard_path.exists()):
            records = [load_pose_record(ref.record_path) for ref in shard_refs]
            save_pose_record_shard(records, shard_path)
        shard_seconds = time.perf_counter() - shard_start
        for local_index, ref in enumerate(shard_refs):
            shard_rows.append(
                {
                    "shard_path": str(shard_path),
                    "local_index": local_index,
                    "pose_id": ref.pose_id,
                    "complex_id": ref.complex_id,
                    "group_id": ref.group_id,
                    "receptor_id": ref.receptor_id,
                    "peptide_id": ref.peptide_id,
                    "dockq": ref.dockq,
                    "rmsd": ref.rmsd,
                }
            )
        build_rows.append(
            {
                "shard_id": shard_id,
                "shard_path": str(shard_path),
                "record_count": len(shard_refs),
                "build_seconds": shard_seconds,
            }
        )
        print(
            {
                "shard_id": shard_id,
                "num_shards": num_shards,
                "record_count": len(shard_refs),
                "build_seconds": shard_seconds,
            },
            flush=True,
        )
    total_seconds = time.perf_counter() - start
    pd.DataFrame(shard_rows).to_csv(out_dir / "shard_index.csv", index=False)
    pd.DataFrame(build_rows).to_csv(out_dir / "shard_build_summary.csv", index=False)
    payload = {
        "source_records_index": str(args.records_index),
        "shard_size": args.shard_size,
        "num_records": len(refs),
        "num_shards": num_shards,
        "total_seconds": total_seconds,
    }
    (out_dir / "shard_meta.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
