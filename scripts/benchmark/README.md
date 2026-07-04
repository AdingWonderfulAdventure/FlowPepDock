# `scripts/benchmark/`

本目录存放 **FlowPepDock 结果评测 / cross-docking 基准打分** 相关脚本。

## 包含内容

- `cross_scoring_common.py`
  - cross benchmark 打分脚本共用的复合物构建、pose 行读取等工具
- `score_cross_ref2015.py`
  - 使用 `ref2015` 对 cross-docking 结果逐 pose 打分
- `score_cross_rosetta_interface.py`
  - 使用 Rosetta interface 指标对 cross-docking 结果打分
- `score_cross_haddock3.py`
  - 使用 `haddock3-score` 对 cross-docking 结果打分
- `score_cross_interpepscore.py`
  - 使用 `InterPepScore` 对 cross-docking 结果打分
- `score_cross_flexpepdock.py`
  - 使用 `FlexPepDock` 对 cross-docking 结果打分
- `evaluate_cross_benchmark.py`
  - 统一计算受体层 / pose 层 benchmark 指标
- `write_cross_benchmark_readme.py`
  - 根据评测结果自动生成对应结果目录 README，并可通过 `--selection_manifest` 适配不同规模的 cross-docking 项目
- `benchmark_rapidock_ori.py`
  - 用 FlowPepDock 的 same-io 口径包装 `RAPiDock_ori` 纯推理测速，并输出机器可读汇总
- `eval_af3_test536_metrics.py`
  - 对 AF3 `test536` 最小回传结果批量计算 RMSD / DockQ
- `prepare_haddock3_local_pocket_experiment.py` / `run_haddock3_local_pocket_queue.py`
  - 生成并执行本地 `HADDOCK3 local-pocket` 对比实验
- `prepare_flexpepdock_independent_local_experiment.py` / `run_flexpepdock_independent_local.py`
  - 生成并执行本地 `FlexPepDock independent-local` 对比实验
- `build_flexpepdock_binary_full_manifest.py`
  - 生成 `FlexPepDock binary-full receptor` 实验清单
- `build_haddock3_retry_manifest.py`
  - 为 `HADDOCK3` retry / 补跑生成 manifest
- `prepare_rapidock_ori_benchmark_inputs.py` / `run_rapidock_ori_gpu_mode.py`
  - 准备并执行 `RAPiDock_ori` 基线 benchmark 输入与 GPU 模式任务
- `eval_selected_complex_rmsd.py` / `eval_selected_complex_rmsd_dockq.py`
  - 对选定复合物子集补算 RMSD / DockQ，便于局部抽检
- `watch_and_launch_when_idle.py`
  - 监控机器负载，在 CPU / 内存空闲时自动启动后台 benchmark 任务

## 使用场景

- 对 FlowPepDock 推理产物做横向模型对比
- 统一输出 `topk`、`top%`、`MRR`、`MAP`、`NDCG` 等指标
- 为 `results/infer/.../other_model_scoring/` 自动补说明文档
- 在 `20x50`、`100x100` 等不同 cross-docking 项目之间复用统一 benchmark README 模板
- 对 AF3 / HADDOCK3 / FlexPepDock / RAPiDock_ori 的本地对比实验做准备、调度与局部复核

## 备注

- 这些脚本服务于 **benchmark 和横向评测**，不是 FlowPepDock 训练 / 推理主链入口，所以不放在 `scripts/` 根目录
