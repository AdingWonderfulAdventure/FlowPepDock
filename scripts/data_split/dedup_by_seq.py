#!/usr/bin/env python
"""
按受体链序列 + 肽链序列去重，保留分辨率更高（数值更小，优先 X-ray/EM，NMR 视为无限大）的那一条。

输入：
  - CSV，需包含：pdb_id 或 PDB编号(pdb_id)、受体序列(receptor_seq/受体序列)、肽序列(peptide_seq/肽序列)、分辨率(A)(resolution)、实验方法(experimental_method)。
输出：
  - 去重后的 CSV（同样列），同一受体+肽序列的重复只保留分辨率最优的一条。

用法示例：
  python scripts/data_split/dedup_by_seq.py \
    --input candidate_pdb_ids.csv \
    --output candidate_pdb_ids_dedup.csv
"""

import argparse
import math
import pandas as pd

def parse_resolution(row):
    method = str(row.get('实验方法(experimental_method)', '')).lower()
    res = row.get('分辨率(A)(resolution)', None)
    try:
        res_val = float(res)
    except Exception:
        res_val = math.inf
    # NMR/unknown 给一个很大值，这样 X-ray/EM 会优先
    if 'nmr' in method:
        res_val = math.inf
    return res_val

def main():
    ap = argparse.ArgumentParser(description="按受体/肽序列去重，保留分辨率更高的条目")
    ap.add_argument('--input', required=True, help='输入 CSV')
    ap.add_argument('--output', required=True, help='输出去重后的 CSV')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    # 规范列名
    pdb_col = 'pdb_id'
    if 'PDB编号(pdb_id)' in df.columns:
        pdb_col = 'PDB编号(pdb_id)'
    rec_seq_col = '受体序列(receptor_seq)' if '受体序列(receptor_seq)' in df.columns else 'receptor_seq'
    pep_seq_col = '肽序列(peptide_seq)' if '肽序列(peptide_seq)' in df.columns else 'peptide_seq'

    if rec_seq_col not in df or pep_seq_col not in df:
        raise SystemExit('缺少受体序列/肽序列列')

    df['__res__'] = df.apply(parse_resolution, axis=1)
    df['__key__'] = df[rec_seq_col].astype(str) + '||' + df[pep_seq_col].astype(str)

    # 按 key 分组，取分辨率最小（最好）的一条；如相等则取第一条
    df_sorted = df.sort_values(['__key__', '__res__'])
    dedup = df_sorted.drop_duplicates(subset='__key__', keep='first').drop(columns=['__res__', '__key__'])
    dedup.to_csv(args.output, index=False)
    print(f'输入 {len(df)} 行，输出 {len(dedup)} 行 -> {args.output}')

if __name__ == '__main__':
    main()
