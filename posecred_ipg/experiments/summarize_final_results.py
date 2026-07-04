from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from ..paths import TMP_ROOT


DEFAULT_ROWS = [
    {
        "tag": "graph_main_npz",
        "report_path": TMP_ROOT / "posecred_full_train_graph_v7_nobad_groupfix" / "eval_report_valonly.json",
        "notes": "current best checkpoint",
    },
    {
        "tag": "graph_shard_seed20260320",
        "report_path": TMP_ROOT / "posecred_full_train_graph_v9_shard_nobad_groupfix" / "eval_report_valonly.json",
        "notes": "shard main seed 20260320",
    },
    {
        "tag": "graph_shard_seed20260321",
        "report_path": TMP_ROOT / "posecred_full_train_graph_v10_shard_nobad_seed20260321" / "train_report.json",
        "notes": "shard main seed 20260321 train report best metric only",
    },
    {
        "tag": "graph_shard_seed20260322",
        "report_path": TMP_ROOT / "posecred_full_train_graph_v11_shard_nobad_seed20260322" / "train_report.json",
        "notes": "shard main seed 20260322 train report best metric only",
    },
    {
        "tag": "stats_baseline_npz",
        "report_path": TMP_ROOT / "posecred_full_train_stats_v5_groupfix" / "eval_report.json",
        "notes": "npz stats baseline",
    },
    {
        "tag": "stats_baseline_shard",
        "report_path": TMP_ROOT / "posecred_full_train_stats_v6_shard_groupfix" / "eval_report.json",
        "notes": "shard stats baseline",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize current posecred final result artifacts.")
    parser.add_argument("--out_csv", type=Path, required=True)
    return parser.parse_args()


def load_payload(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def payload_to_row(tag: str, payload: Dict[str, object], report_path: Path, notes: str) -> Dict[str, object]:
    row = {
        "tag": tag,
        "report_path": str(report_path),
        "notes": notes,
        "val_top1_success": payload.get("val_top1_success", payload.get("best_metric_value", "")),
        "val_global_top1_success": payload.get("val_global_top1_success", ""),
        "val_top5_success": payload.get("val_top5_success", ""),
        "val_mrr": payload.get("val_mrr", ""),
        "val_ndcg": payload.get("val_ndcg", ""),
        "total_seconds": payload.get("total_seconds", ""),
    }
    return row


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []
    for item in DEFAULT_ROWS:
        report_path = Path(item["report_path"])
        if not report_path.exists():
            continue
        payload = load_payload(report_path)
        rows.append(payload_to_row(item["tag"], payload, report_path, item["notes"]))
    if not rows:
        raise FileNotFoundError("未找到任何可汇总的 PoseCred-IPG report_path。")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
