# 6. 实验、正式结果与当前结论

> 文档类型：当前代码说明
>
> 本文只保留当前正式结果、当前现役对照和当前正式结论。
> 开发过程中的消融、旧 baseline、lite 路线和已废弃方案统一见 `posecred_ipg/STATUS_AND_PLAN.md`。

## 6.1 当前纳入正式口径的模型

当前文档中纳入正式口径的模型只有两类：

- `posecred_ipg` 图模型
- `stats_mlp` 非图基线

以下路线不再属于当前正式结果说明：

- `GBDT ranker`
- `lite schema`
- `bad head / clash` 历史消融路线
- `repeat_*` 历史重复实验

## 6.2 当前最佳正式结果

当前最佳 checkpoint：

- `posecred_ipg/final_exports/graph_main_best.pt`

对应正式评估：

- `posecred_ipg/final_exports/graph_main_eval_report.json`

核心结果：

- `val_top1_success = 0.5470798569725864`
- `val_global_top1_success = 0.5136518771331058`
- `val_top5_success = 0.7318235995232419`
- `val_mrr = 0.6256527044667688`
- `val_ndcg = 0.9345137940941872`

## 6.3 当前与 `Stats+MLP` 基线对比

当前正式保留的非图基线大致为：

- `val_top1_success = 0.3766`
- `val_global_top1_success = 0.3481`

因此当前图模型相对当前基线的提升是明确且实质性的。

## 6.4 当前正式结论

当前正式结论固定为：

- 最佳 checkpoint 使用 `posecred_ipg/final_exports/graph_main_best.pt`
- 默认训练/评估数据通路使用 shard
- 当前 snapshot 留档为 `hybrid + N_pair=32 + clash_penalty_scale=0.0`
- 当前现役结构为 `score_head + dockq_head`
- 当前现役损失为 `listwise + pairwise + dockq`

## 6.5 当前统一结果表

当前统一结果表：

- `posecred_ipg/final_exports/final_results_snapshot.csv`

正式引用结果时，优先使用该总表和 `graph_main_eval_report.json`，不要再从历史 `tmp/` 运行目录中手工摘数。

## 6.6 开发归档入口

如果需要追溯下面这些问题，请转到开发归档文档：

- `N_pair` 和 pruning 的选择过程
- `bad head` / `clash` 的历史消融过程
- `GBDT` 与其他旧 baseline 的阶段性结果
- `lite schema` 的裁维尝试
- 多轮 repeat / repeat_* 历史实验

统一入口：

- `posecred_ipg/STATUS_AND_PLAN.md`
- `posecred_ipg/legacy/README.md`
