# `scripts/data_split/`

本目录存放 **去重、拆分、泄漏审计、测试集清洗统计** 相关脚本。

## 包含内容

- `dedup_by_seq.py`
  - 按序列或组合键去重
- `make_random_split.py`
  - 生成 train / val 随机划分
- `build_stratified_benchmark_subset.py`
  - 仅基于输入属性构建可复现 benchmark 小集，避免按模型结果挑样
- `audit_docking_dataset_leakage.py`
  - 审计训练 / 验证 / 测试之间的泄漏风险
- `build_clean_test_csvs_from_audit.py`
  - 根据审计结果生成更干净的测试 CSV
- `build_rec70_cross_docking_dataset.py`
  - 按当前 `rec70 test 20x50` 口径扩容 cross-docking 数据集，并同时导出增量项目 / 全量项目两套 CSV
- `summarize_exact_peptide_rec70_groups.py`
  - 汇总特定聚类 / 分组规则下的数据统计

## 使用场景

- 构造符合聚类约束或去泄漏要求的数据切分
- 重新组织 benchmark / train / val / test 清单
- 在**不破坏既有 benchmark 项目**的前提下，把小规模 cross-docking 项目平滑扩到更大规模

## 备注

- 这类脚本主要服务于 **数据集组织策略**
- 不是 FlowPepDock 日常训练与推理直接调用的入口
- `build_rec70_cross_docking_dataset.py` 的默认输出目录为
  - `data/cross_docking_rec70_test100x100_20260408/`
  - 其中 `flow_input_*.csv` 供 Flow / RAPiDock 一类生成模型直接读取
  - `selection_manifest_*.csv` 供 PoseCred-IPG / GraphPep / ref2015 等打分模型做标签对齐与 benchmark 统计
