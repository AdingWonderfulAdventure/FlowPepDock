#!/usr/bin/env python
"""从 split_paths_le10_dedup.csv 生成缺失 features_onehot.pt 的子集 CSV。"""
import csv
from pathlib import Path

def main():
    csv_path = Path('data/csv_backup/split_paths_le10_dedup.csv')
    base = Path('data/processed')
    rows = list(csv.DictReader(csv_path.open()))
    missing = []
    for row in rows:
        pid = (row.get('complex_name') or row.get('PDB编号(pdb_id)') or row.get('pdb_id') or '').strip().lower()
        pt = base / pid / 'features_onehot.pt'
        if not pt.exists() or pt.stat().st_size == 0:
            missing.append(row)
    miss_path = Path('data/csv_backup/split_paths_missing_pt.csv')
    miss_path.parent.mkdir(parents=True, exist_ok=True)
    with miss_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(missing)
    print(f'wrote {len(missing)} missing rows to {miss_path}')

if __name__ == '__main__':
    main()
