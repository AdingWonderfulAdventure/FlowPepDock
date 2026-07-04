# `scripts/data_qc/`

本目录存放 **数据质量检查、缺失排查、异常样本隔离** 相关脚本。

## 包含内容

- `inspect_processed.py`
  - 巡检 `processed/` 下的 PDB 是否空文件、链异常、肽长度异常等
- `inspect_pt.py`
  - 巡检 `features_*.pt` 是否缺字段、维度错、零字节、NaN/Inf
- `filter_csv_by_existing_processed.py`
  - 只保留已存在 `processed` 目录的 CSV 条目
- `filter_pt_available_by_min_ca_dist.py`
  - 结合 PT 存在性和距离阈值做过滤
- `gen_missing_pt_csv.py`
  - 生成缺少 PT 的样本清单
- `verify_pt_from_report.py`
  - 根据 PT 巡检报告做回查
- `quarantine_bad_processed.py`
  - 将异常 processed 样本隔离
- `annotate_filter_notes.py`
  - 给筛选后的 CSV 增加备注
- `diagnose_rot_observability.py`
  - rot 可观测性等诊断脚本

## 使用场景

- 生成训练 / 推理数据前后做完整性核查
- 快速定位为什么某些样本不能构图或不能训练

## 备注

- 这类脚本用于 **质量控制和排错**
- 不属于 FlowPepDock 正常训练 / 推理的必需入口
