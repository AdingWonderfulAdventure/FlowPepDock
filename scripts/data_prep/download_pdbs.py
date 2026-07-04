#!/usr/bin/env python
"""
从 CSV 读取 PDB 编号，批量下载 PDB/mmCIF 到指定目录。
会自动跳过已存在的文件，便于多次重试把漏下的补齐。
默认并发 128 线程。
"""
import argparse
import csv
import os
import time
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def read_pdb_ids(csv_path: Path) -> List[str]:
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        field_map = {name.lower(): name for name in reader.fieldnames or []}
        key = None
        for cand in ["pdb_id", "pdb编号", "pdb编号(pdb_id)"]:
            if cand.lower() in field_map:
                key = field_map[cand.lower()]
                break
        if key is None:
            raise ValueError(f"CSV里找不到 pdb_id 列，实际列名: {reader.fieldnames}")
        ids = []
        for row in reader:
            pid = (row.get(key) or "").strip()
            if pid:
                ids.append(pid.lower())
    return ids


def download_one(pdb_id: str, out_dir: Path, fmt: str) -> bool:
    url = f"https://files.rcsb.org/download/{pdb_id}.{fmt}"
    out_path = out_dir / f"{pdb_id}.{fmt}"
    if out_path.exists():
        return True
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return True
    except Exception as exc:
        print(f"[WARN] 下载 {pdb_id}.{fmt} 失败: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="从CSV批量下载PDB/mmCIF（可多次重试补齐缺失文件）")
    parser.add_argument("--csv", required=True, help="包含pdb_id/PDB编号列的CSV")
    parser.add_argument("--out_dir", required=True, help="下载保存目录")
    parser.add_argument("--format", choices=["pdb", "cif"], default="pdb", help="下载格式，pdb或cif")
    parser.add_argument("--sleep", type=float, default=0.0, help="每次请求后休眠秒数，防止频率太高")
    parser.add_argument("--workers", type=int, default=128, help="并发线程数（默认128）")
    args = parser.parse_args()

    ids = read_pdb_ids(Path(args.csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def task(pid: str):
        ok = download_one(pid, out_dir, args.format)
        if args.sleep > 0:
            time.sleep(args.sleep)
        return pid, ok

    success = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(task, pid): pid for pid in ids}
        total = len(futures)
        done = 0
        for fut in as_completed(futures):
            pid, ok = fut.result()
            done += 1
            if ok:
                success += 1
            print(f"[{done}/{total}] {pid} -> {'ok' if ok else 'fail'}")
    print(f"完成：成功 {success}/{len(ids)}，保存到 {out_dir}")


if __name__ == "__main__":
    main()

