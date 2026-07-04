# Quickstart

> 文档类型：当前代码说明
>
> 本文只保留当前 `PoseCred-IPG v1` 的实际使用方式。
> 开发过程、已废弃方案和历史实验脚本统一见 `posecred_ipg/STATUS_AND_PLAN.md` 与 `posecred_ipg/legacy/README.md`。

当前目录已经按 `core / data / engine / evaluation / pipelines / experiments / models` 分层，`python -m posecred_ipg.*` 旧命令仍然可用，但根目录模块现在主要只是兼容壳子。

## 当前该用什么

当前建议分成两部分：

- **最佳 checkpoint**
  - [graph_main_best.pt](/root/FlowPepDock/posecred_ipg/final_exports/graph_main_best.pt)
  - 独立评估：
    [graph_main_eval_report.json](/root/FlowPepDock/posecred_ipg/final_exports/graph_main_eval_report.json)

- **默认训练/评估数据通路**
  - `shard`
  - 通过 `--use_default_shard_snapshot` 自动加载
  - 实际展开为：
    - `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/train_shards_v1/shard_index.csv`
    - `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/val_shards_v1/shard_index.csv`
  - `posecred_ipg/final_exports/default_train_shard_index.csv` / `default_val_shard_index.csv` 只是归档副本

固定导出目录：

- [final_exports/README.md](/root/FlowPepDock/posecred_ipg/final_exports/README.md)

当前 snapshot 留档（不是训练命令里必须显式传的超参数）：

- `prune_strategy = hybrid`
- `N_pair = 32`
- `bad head` 已在当前主线代码中废弃并移除
- `clash_penalty_scale = 0.0`
- 与 `bad head` 配套的 `bad BCE`、`bad_loss_weight`、4 项 `loss_weights` 的第 4 项也都已同步废弃；旧 checkpoint 里若仍带 `bad_head.*` 权重，只会在兼容加载时自动忽略

## 训练

图模型：

```bash
python -m posecred_ipg.train \
  --model_name posecred_ipg \
  --use_default_shard_snapshot \
  --out_dir /root/FlowPepDock/tmp/posecred_run_new \
  --epochs 5 \
  --device cuda \
  --gpu_ids 1 \
  --clash_penalty_scale 0.0 \
  --eval_poses_per_group 0
```

`Stats+MLP` 基线：

```bash
python -m posecred_ipg.train \
  --model_name stats_mlp \
  --use_default_shard_snapshot \
  --out_dir /root/FlowPepDock/tmp/posecred_stats_run_new \
  --epochs 5 \
  --device cuda \
  --gpu_ids 1 \
  --clash_penalty_scale 0.0 \
  --eval_poses_per_group 0
```

## 单 checkpoint 评估

```bash
python -m posecred_ipg.evaluate_checkpoint \
  --checkpoint /root/FlowPepDock/tmp/posecred_run_new/checkpoints/best.pt \
  --out_path /root/FlowPepDock/tmp/posecred_run_new/eval_report.json \
  --model_name posecred_ipg \
  --use_default_shard_snapshot \
  --groups_per_batch 32 \
  --poses_per_group 0 \
  --device cuda \
  --gpu_ids 1 \
  --clash_penalty_scale 0.0
```

## pooled/global 评估

```bash
python -m posecred_ipg.evaluate_pooled_global \
  --checkpoint /root/FlowPepDock/tmp/posecred_run_new/checkpoints/best.pt \
  --out_dir /root/FlowPepDock/tmp/posecred_run_new/pooled_eval \
  --model_name posecred_ipg \
  --use_default_shard_snapshot \
  --groups_per_batch 32 \
  --poses_per_group 0 \
  --device cuda \
  --gpu_ids 1 \
  --clash_penalty_scale 0.0
```

## 导出逐 pose 打分表

```bash
python -m posecred_ipg.export_scores \
  --checkpoint /root/FlowPepDock/posecred_ipg/final_exports/graph_main_best.pt \
  --out_csv /root/FlowPepDock/tmp/posecred_scores.csv \
  --model_name posecred_ipg \
  --use_default_shard_snapshot \
  --split val \
  --groups_per_batch 32 \
  --poses_per_group 0 \
  --device cuda \
  --gpu_ids 1 \
  --clash_penalty_scale 0.0
```

## 直接对 cross pose table 打分

默认推荐的新入口：

```bash
python -m posecred_ipg.score_cross_pose_table \
  --cross_pose_table results/infer/cross_20pep_50rec_N10_step10/ipg_scoring/cross_pose_table.csv \
  --out_dir /root/FlowPepDock/tmp/ipg_cross_score_run \
  --checkpoint /root/FlowPepDock/posecred_ipg/final_exports/graph_main_best.pt \
  --config_snapshot /root/FlowPepDock/posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/config_snapshot.json \
  --build_workers 4 \
  --device cuda \
  --gpu_ids 1
```

如果已经提前构建好标准 `npz` records，可直接复用：

```bash
python -m posecred_ipg.score_cross_pose_table \
  --cross_pose_table results/infer/cross_20pep_50rec_N10_step10/ipg_scoring/cross_pose_table.csv \
  --records_manifest results/infer/cross_20pep_50rec_N10_step10/ipg_scoring/manifest.txt \
  --out_dir /root/FlowPepDock/tmp/ipg_cross_score_run_from_npz \
  --checkpoint /root/FlowPepDock/posecred_ipg/final_exports/graph_main_best.pt \
  --config_snapshot /root/FlowPepDock/results/infer/cross_20pep_50rec_N10_step10/ipg_scoring/config_snapshot.json \
  --device cuda \
  --gpu_ids 1
```

如果你想在在线构图时顺手把 records 落盘，方便后续复用，可加：

- `--save_records_dir /path/to/records_dir`
- `--save_uncompressed_npz`（更快，但更占磁盘）
- `--full_model_forward`（关闭 `score-only fastpath`，强制走完整模型前向）
- `--build_workers 4`（按 `receptor_pdb` 分组并行构建）

当前默认行为：

- `--poses_per_group 0`：不截断，保留每个受体下的全部 poses
- `--groups_per_batch 0`：自动解析为当前输入里全部 `group`，也就是默认把该多肽的全 poses 一次性喂给打分入口

### 当前推荐用法

- **只跑一次、追求端到端速度**
  - 直接用 `score_cross_pose_table`
  - 不传 `--records_manifest` / `--records_index`
  - 让入口在线构图后直接打分

- **后续会反复重跑同一批 poses**
  - 首次运行加 `--save_records_dir`
  - 推荐同时加 `--save_uncompressed_npz`
  - 后续改用 `--records_manifest` 或 `--records_index` 直接复用标准 `npz`

- **想做严格对照或排查数值路径**
  - 加 `--full_model_forward`
  - 强制关闭快路径，回退到完整模型前向

### 当前已验证的速度结论

- 纯 `npz -> score` 小样本导出：
  - 相比旧导出链路，当前快路径约 `3x` 级加速

- `PDB/cross_pose_table -> score` 端到端：
  - 在线构图默认走全 pose
  - 结合残基缓存、`cdist` 和按 `receptor_pdb` 分组并行构建后，端到端明显快于最初版本

- 在线构图阶段：
  - `--build_workers` 现在按 `receptor_pdb` 分组并行
  - 这比逐 pose 并行更能吃到受体缓存收益

## 当前最好结果

当前最佳模型是 `graph_main_npz`：

- `val_top1_success = 0.5470798569725864`
- `val_global_top1_success = 0.5136518771331058`
- `val_top5_success = 0.7318235995232419`
- `val_mrr = 0.6256527044667688`
- `val_ndcg = 0.9345137940941872`

完整总表：

- [final_results_snapshot.csv](/root/FlowPepDock/posecred_ipg/final_exports/final_results_snapshot.csv)

## 当前判断

- 如果你要直接拿现成最优模型，用 `graph_main_npz best`
- 如果你要继续训练和评估，默认走 `shard`
- 当前不建议：
  - 开启 `clash` 后处理
  - 改回 `N_pair = 64`
- 任何偏离当前默认配置的研发性尝试，都应转到 `posecred_ipg/STATUS_AND_PLAN.md` 记录，不应混入当前使用说明
