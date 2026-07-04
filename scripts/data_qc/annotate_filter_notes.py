#!/usr/bin/env python
import argparse
import csv
from pathlib import Path


# 描述：给已有筛选结果 CSV 添加筛选备注，用于追踪条件。
# 用法：
#   python scripts/data_qc/annotate_filter_notes.py \
#     --input_csv candidate_pdb_ids.csv \
#     --output_csv candidate_pdb_ids_with_notes.csv \
#     --entry_note "chains>=2 & protein_chains>=2 & residues>=50 & (resolution<=3Å or method=NMR)" \
#     --entity_note "peptide_len[3,30] & receptor>=50 & max_receptor>=3 * max_peptide"
# 输入：input_csv 为已有候选 CSV；entry_note/entity_note 为当时的筛选条件。
# 输出：output_csv 在原列基础上新增 entry_filter/entity_filter 备注。

def main():
    parser = argparse.ArgumentParser(description="为已有的候选 CSV 补充筛选备注")
    parser.add_argument("--csv", required=True, help="目标 CSV 文件")
    parser.add_argument("--entry_note", required=True, help="初筛条件描述")
    parser.add_argument("--entity_note", required=True, help="二次筛条件描述")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open() as f:
        reader = list(csv.reader(f))
    if not reader:
        return

    header = reader[0]
    rows = reader[1:]

    if "entry_filter" not in header:
        header.extend(["entry_filter", "entity_filter"])
        rows = [row + [args.entry_note, args.entity_note] for row in rows]
    else:
        entry_idx = header.index("entry_filter")
        entity_idx = header.index("entity_filter")
        for row in rows:
            while len(row) <= entry_idx:
                row.append("")
            while len(row) <= entity_idx:
                row.append("")
            if not row[entry_idx]:
                row[entry_idx] = args.entry_note
            if not row[entity_idx]:
                row[entity_idx] = args.entity_note

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
