# Final Artifacts

> 文档类型：当前代码说明
>
> 本文只描述当前正式交付物、当前固定结果口径和当前推荐用法。
> 历史运行目录、阶段性实验产物和已废弃方案不在本文范围内；相关内容统一见 `posecred_ipg/STATUS_AND_PLAN.md`。

## 当前固定出口

当前统一固定出口目录：

- `posecred_ipg/final_exports/`

其中正式文件为：

- `posecred_ipg/final_exports/graph_main_best.pt`
- `posecred_ipg/final_exports/graph_main_eval_report.json`
- `posecred_ipg/final_exports/default_train_shard_index.csv`
- `posecred_ipg/final_exports/default_val_shard_index.csv`
- `posecred_ipg/final_exports/final_results_snapshot.csv`

其中 `default_train_shard_index.csv` / `default_val_shard_index.csv` 只是归档副本；`posecred_ipg.train --use_default_shard_snapshot` 和 `posecred_ipg.evaluate_checkpoint --use_default_shard_snapshot` 的实际展开路径以 `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/train_shards_v1/shard_index.csv` 与 `.../val_shards_v1/shard_index.csv` 为准。

## 当前 snapshot 留档

以下内容是当前 snapshot/config 留档，不是训练命令里必须显式传的超参数：

- `prune_strategy = hybrid`
- `N_pair = 32`
- `loss_weights = (1.0, 0.5, 0.2)`
- `clash_penalty_scale = 0.0`

当前现役训练/推理结构固定为：

- `score_head + dockq_head`
- `listwise + pairwise + dockq`

## 当前正式结果

当前最佳 checkpoint：

- `posecred_ipg/final_exports/graph_main_best.pt`

该 checkpoint 是外部发布资产，默认不进 Git；若源码包里没有该文件，按
`docs/RELEASE_ASSETS.md` 下载并校验后放回上述路径。

对应正式评估：

- `posecred_ipg/final_exports/graph_main_eval_report.json`

核心结果：

- `val_top1_success = 0.5470798569725864`
- `val_global_top1_success = 0.5136518771331058`
- `val_top5_success = 0.7318235995232419`
- `val_mrr = 0.6256527044667688`
- `val_ndcg = 0.9345137940941872`

统一结果总表：

- `posecred_ipg/final_exports/final_results_snapshot.csv`

## 当前默认数据通路

当前默认训练/评估数据通路为 shard：

- `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/train_shards_v1/shard_index.csv`
- `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/val_shards_v1/shard_index.csv`

归档副本同步保留为：

- `posecred_ipg/final_exports/default_train_shard_index.csv`
- `posecred_ipg/final_exports/default_val_shard_index.csv`

当前最佳 checkpoint 继续使用固定导出目录中的正式模型文件，不再以历史 `tmp/` 路径作为正式引用口径。

## 当前推荐命令

图模型训练：

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

`Stats+MLP` 基线训练：

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

单 checkpoint 评估：

```bash
python -m posecred_ipg.evaluate_checkpoint \
  --checkpoint /root/FlowPepDock/posecred_ipg/final_exports/graph_main_best.pt \
  --out_path /root/FlowPepDock/tmp/posecred_run_new/eval_report.json \
  --model_name posecred_ipg \
  --use_default_shard_snapshot \
  --groups_per_batch 32 \
  --poses_per_group 0 \
  --device cuda \
  --gpu_ids 1 \
  --clash_penalty_scale 0.0
```

pooled/global 评估：

```bash
python -m posecred_ipg.evaluate_pooled_global \
  --checkpoint /root/FlowPepDock/posecred_ipg/final_exports/graph_main_best.pt \
  --out_dir /root/FlowPepDock/tmp/posecred_run_new/pooled_eval \
  --model_name posecred_ipg \
  --use_default_shard_snapshot \
  --groups_per_batch 32 \
  --poses_per_group 0 \
  --device cuda \
  --gpu_ids 1 \
  --clash_penalty_scale 0.0
```

逐 pose 分数导出：

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

## 当前文档跳转

- 当前快速入口：`posecred_ipg/QUICKSTART.md`
- 当前文档总索引：`posecred_ipg/docs/00_INDEX.md`
- 固定导出目录说明：`posecred_ipg/final_exports/README.md`
- 开发归档：`posecred_ipg/STATUS_AND_PLAN.md`
