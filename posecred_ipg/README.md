# PoseCred-IPG v1

> 文档类型：当前代码说明
>
> 本文只描述当前仓库中仍可执行、可复现、可交付的内容。
> 开发过程、问题排查、阶段性尝试和已废弃方案统一见 `posecred_ipg/STATUS_AND_PLAN.md`；
> 已退出正式主线的脚本和模块统一见 `posecred_ipg/legacy/README.md`。

## 当前这套代码是什么

`PoseCred-IPG v1` 当前是一套面向 **protein-peptide docking pose reranking** 的独立模型实现。

它的任务定义是：

- 输入同一个 peptide 对应的多个 receptor 候选及其 poses；
- 为每个 `receptor x pose` 输出一个 `score_pose`；
- 按分数统一排序，优先把更接近真实高质量构象的 pose 排到前面。

当前任务口径是：

- `positive-only`
- `open-world`
- `pose ranking`

它不是：

- binder / non-binder 分类器
- 真实亲和力回归器
- 真实结合概率预测器
- 大规模序列级互作筛选器

## 当前代码能做什么

当前仓库内仍然可直接使用的能力主要有：

- 训练图模型 `posecred_ipg`
- 训练非图基线 `stats_mlp`
- 对 checkpoint 做独立评估与 pooled/global 评估
- 导出逐 pose 分数表
- 对 cross-docking pose table 执行在线构图 + 打分
- 构建 `PoseRecord`、record snapshot 和 shard snapshot
- 使用固定导出目录中的正式 checkpoint、结果表和 shard 索引归档副本

## 当前正式主线口径

当前 snapshot / 结构留档：

- `prune_strategy = hybrid`
- `N_pair = 32`
- `loss_weights = (1.0, 0.5, 0.2)`
- `clash_penalty_scale = 0.0`

当前现役模型结构：

- `score_head + dockq_head`
- `listwise + pairwise + dockq` 三项损失

当前固定出口：

- 最佳 checkpoint：`posecred_ipg/final_exports/graph_main_best.pt`
- 独立评估结果：`posecred_ipg/final_exports/graph_main_eval_report.json`
- 训练 shard 索引归档副本：`posecred_ipg/final_exports/default_train_shard_index.csv`
- 验证 shard 索引归档副本：`posecred_ipg/final_exports/default_val_shard_index.csv`
- 当前统一结果表：`posecred_ipg/final_exports/final_results_snapshot.csv`
- 当前训练/评估实际 shard 入口：`posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/train_shards_v1/shard_index.csv` + `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/val_shards_v1/shard_index.csv`
- `posecred_ipg/final_exports/default_train_shard_index.csv` / `default_val_shard_index.csv` 只是归档副本

## 当前明确不属于主线的内容

以下项目不再属于当前正式主线说明：

- `bad_head`
- `bad BCE`
- `bad_loss_weight`
- `loss_weights` 第 4 项
- `physical_bad_label -> bad_head` 训练监督链路
- `lite / ablation / repeat / old baseline / gbdt` 历史实验脚本

当前代码对旧资产仅保留兼容行为：

- 旧 checkpoint 中若仍带 `bad_head.*` 权重，加载时自动忽略
- 旧配置若仍传 4 项 `loss_weights`，第 4 项只能为 `0.0`
- `physical_bad_label` 仅保留为数据构建阶段的物理异常标签，不参与当前训练损失

## 当前推荐阅读入口

如果你要直接使用当前代码，按这个顺序看：

1. `posecred_ipg/QUICKSTART.md`
2. `posecred_ipg/FINAL_ARTIFACTS.md`
3. `posecred_ipg/docs/00_INDEX.md`
4. `posecred_ipg/final_exports/README.md`

如果你要追溯研发过程或历史路线，去看：

- `posecred_ipg/STATUS_AND_PLAN.md`
- `posecred_ipg/legacy/README.md`

如果你要写论文中的方法定位补充，去看：

- `posecred_ipg/GRAPHPEP_COMPARISON.md`
- `posecred_ipg/docs/09_THESIS_WRITEUP_GUIDE.md`

如果你要解释模型导出的 `score` 数值含义，去看：

- `posecred_ipg/docs/11_SCORE_SEMANTICS.md`

## 当前目录结构

当前目录的职责分层如下：

- `posecred_ipg/core/`：配置、常量、路径、特征布局、`PoseRecord` 等公共定义
- `posecred_ipg/data/`：PDB 解析、特征构建、record 序列化、dataset / sampler / collator
- `posecred_ipg/models/`：`stats_mlp` 基线与 `PoseCred-IPG` 图模型
- `posecred_ipg/engine/`：训练主循环、损失函数、指标与设备包装
- `posecred_ipg/evaluation/`：checkpoint 评估、pooled/global 评估、逐 pose 打分导出、cross pose 打分
- `posecred_ipg/pipelines/`：record / snapshot / shard 构建与校验入口
- `posecred_ipg/experiments/`：当前仍保留的 smoke、benchmark、结果汇总脚本
- `posecred_ipg/final_exports/`：当前正式交付文件
- `posecred_ipg/legacy/`：已退出正式主线的历史脚本与归档模块
- 根目录同名 `*.py`：兼容入口壳，便于沿用旧命令

## 当前结论

一句话说完：

- 当前 `PoseCred-IPG v1` 已经收敛到一条清晰主线：
  `hybrid + N_pair=32 + score_head/dockq_head + listwise/pairwise/dockq + no_bad_head + no_clash_post`。
