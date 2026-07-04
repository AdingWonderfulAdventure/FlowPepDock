# 536数据集与 RefPepDB-RecentSet / pepset 重合说明

## 数据来源

- 536 主表：`data/runtime_tables/flow_infer_test536_rel.csv`
- RefPepDB-RecentSet：`data/RefPepDB-RecentSet`
- pepset：`data/pepset`

说明：目录中的一级子目录名按 `pdbid` 处理，与 536 主表中的 `complex_name` 做集合比较。

## 基本数量

- 536 主表唯一 `pdbid` 数：`536`
- RefPepDB-RecentSet 唯一 `pdbid` 数：`523`
- pepset 唯一 `pdbid` 数：`185`

## 两两与三者重合

- `536 ∩ RefPepDB-RecentSet`：`398`
- `536 ∩ pepset`：`138`
- `RefPepDB-RecentSet ∩ pepset`：`0`
- `536 ∩ RefPepDB-RecentSet ∩ pepset`：`0`

## 只属于某一侧/某两侧

- 只在 `536`：`0`
- 只在 `RefPepDB-RecentSet`：`125`
- 只在 `pepset`：`47`
- 在 `536` 和 `RefPepDB-RecentSet`，但不在 `pepset`：`398`
- 在 `536` 和 `pepset`，但不在 `RefPepDB-RecentSet`：`138`
- 在 `RefPepDB-RecentSet` 和 `pepset`，但不在 `536`：`0`

## 对 536 的覆盖情况

- `536` 中至少出现在这两个目录之一（`RefPepDB-RecentSet ∪ pepset`）的数量：`536`
- `536` 中完全不在这两个目录里的数量：`0`
- 覆盖率：`100.0000%`

## 名单文件

- `data/runtime_tables/overlap_reports/flow536_and_refpep_recent.csv`
- `data/runtime_tables/overlap_reports/flow536_and_pepset.csv`
- `data/runtime_tables/overlap_reports/all_three.csv`
- `data/runtime_tables/overlap_reports/flow536_only.csv`
- `data/runtime_tables/overlap_reports/flow536_and_refpep_recent_only.csv`
- `data/runtime_tables/overlap_reports/flow536_and_pepset_only.csv`
- `data/runtime_tables/overlap_reports/flow536_not_in_union_recent_pepset.csv`

## 示例

- `536 ∩ RefPepDB-RecentSet` 前10个：7al2, 7arx, 7atr, 7bcy, 7bdu, 7bee, 7bmi, 7bmj, 7bn1, 7bn3
- `536 ∩ pepset` 前10个：1d8d, 1ddv, 1eg4, 1f47, 1j2x, 1jd5, 1l2z, 1lb6, 1mv0, 1p7v
- 三者共同前10个：
- 只在536前10个：

## 带来源标签的536主表

- 已生成：`data/runtime_tables/flow_infer_test536_rel_with_source_set.csv`
- 新增列：`source_set`、`source_membership`
- `source_set=RefPepDB-RecentSet`：`398`
- `source_set=pepset`：`138`
- `source_set=both`：`0`
- `source_set=unmatched`：`0`

## 按来源拆分后的CSV

- `RefPepDB-RecentSet` 子集：`data/runtime_tables/flow_infer_test536_refpep_recent_only.csv`（`398` 条）
- `pepset` 子集：`data/runtime_tables/flow_infer_test536_pepset_only.csv`（`138` 条）
- 拆分摘要：`data/runtime_tables/flow_infer_test536_split_by_source_summary.json`
