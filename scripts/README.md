# `scripts/` 目录说明

## 根目录：仅保留运行 FlowPepDock 的必须脚本

- `prepare_training_data.py`
  - 从 `complex_name,receptor_pdb,peptide_pdb` CSV 生成 FlowPepDock 训练/推理使用的 `features_*.pt`
- `extract_esm_embedding.py`
  - 为 ESM 特征流程生成 embedding
- `build_features_esm_from_onehot.py`
  - 在 onehot PT 基础上拼接/构建 ESM 版本 PT
- `eval_rmsd_from_preds.py`
  - 训练验证与推理后评估时使用的 RMSD/DockQ 评测脚本

这些脚本直接服务于 FlowPepDock 的数据准备、训练验证或推理评估，因此保留在 `scripts/` 根目录。

## 二级目录：按用途归类的辅助脚本

- `benchmark/`
  - cross-docking 打分、统一 benchmark 评测、README 生成、AF3 本地评测、外部模型本地实验编排等
- `data_prep/`
  - 原始结构下载、挑链、拆链、裁剪口袋、批量 onehot 预处理等
- `data_qc/`
  - processed/PT 巡检、缺失检查、异常样本隔离、备注补充等
- `data_split/`
  - 去重、随机划分、泄漏审计、测试集清洗与统计等
- `feature_tools/`
  - ESM / onehot 特征对齐、替换、校验等
- `ops/`
  - notes 守卫、长任务通知等工程辅助脚本
- `postprocess/`
  - 推理结果过滤、后处理、指标汇总等
- `figures/`
  - 论文/答辩用结果表出图脚本；例如 `plot_pose_rank.py` 可把 pose 级打分表直接导出为 Illustrator 可编辑的 `PDF/SVG`

## 整理原则

- 根目录脚本必须与 FlowPepDock 的主训练/推理链路直接相关
- 非主链路脚本一律下沉到二级目录，并以“用途”命名
- 后续新增脚本时，优先放入对应用途目录；只有确实属于主链路入口时，才允许放在根目录
- 临时产物、缓存、误放文件不要留在根目录；先移到 `trash/staging/YYYYMMDD/` 留档，再决定是否彻底删除
