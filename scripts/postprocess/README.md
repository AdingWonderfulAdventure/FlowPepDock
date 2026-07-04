# `scripts/postprocess/`

本目录存放 **推理结果后处理与统计汇总** 相关脚本。

## 包含内容

- `filter_infer_poses.py`
  - 对推理 pose 按距离等规则做过滤
- `relax_with_trajectory_parallel.py`
  - 对生成构象做后处理 / 并行 relax
- `summarize_metrics_rmsd_dockq.py`
  - 汇总 RMSD / DockQ 等后处理指标

## 使用场景

- 对 FlowPepDock 输出结果做筛选或后处理
- 汇总不同实验的推理指标

## 备注

- 真正基础的推理评估入口仍然是根目录下的 `scripts/eval_rmsd_from_preds.py`
- 本目录脚本更偏向 **推理之后的进一步处理**
