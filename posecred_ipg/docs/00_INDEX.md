# PoseCred-IPG 文档索引

> 文档类型：当前代码说明
>
> 本索引只组织当前仓库仍然有效的说明文档，并明确区分当前代码说明、开发归档和历史归档。

## 当前代码说明

这些文档用于描述当前代码仍能做什么、当前固定口径是什么、当前怎么用：

1. `posecred_ipg/docs/01_TASK_AND_SCOPE.md`
2. `posecred_ipg/docs/02_DATA_AND_SNAPSHOTS.md`
3. `posecred_ipg/docs/03_FEATURES_AND_REPRESENTATION.md`
4. `posecred_ipg/docs/04_MODEL_AND_LOSSES.md`
5. `posecred_ipg/docs/05_TRAINING_AND_EVALUATION.md`
6. `posecred_ipg/docs/06_EXPERIMENTS_AND_RESULTS.md`
7. `posecred_ipg/docs/07_DEPLOYMENT_AND_ARTIFACTS.md`
8. `posecred_ipg/docs/08_LIMITATIONS_AND_NEXT_STEPS.md`
9. `posecred_ipg/docs/09_THESIS_WRITEUP_GUIDE.md`
10. `posecred_ipg/docs/10_DATA_PIPELINE_TABLE.md`
11. `posecred_ipg/docs/11_SCORE_SEMANTICS.md`

## 当前使用入口

如果你是直接用代码，优先看：

- `posecred_ipg/README.md`
- `posecred_ipg/QUICKSTART.md`
- `posecred_ipg/FINAL_ARTIFACTS.md`
- `posecred_ipg/final_exports/README.md`

## 开发归档与历史归档

以下文档不属于“当前代码说明”，只用于追溯开发过程与旧路线：

- 开发归档：`posecred_ipg/STATUS_AND_PLAN.md`
- 历史归档：`posecred_ipg/legacy/README.md`

## 建议阅读顺序

建议按下面顺序阅读：

1. `posecred_ipg/README.md`
2. `posecred_ipg/QUICKSTART.md`
3. `posecred_ipg/docs/01_TASK_AND_SCOPE.md`
4. `posecred_ipg/docs/02_DATA_AND_SNAPSHOTS.md`
5. `posecred_ipg/docs/03_FEATURES_AND_REPRESENTATION.md`
6. `posecred_ipg/docs/04_MODEL_AND_LOSSES.md`
7. `posecred_ipg/docs/05_TRAINING_AND_EVALUATION.md`
8. `posecred_ipg/docs/06_EXPERIMENTS_AND_RESULTS.md`
9. `posecred_ipg/docs/07_DEPLOYMENT_AND_ARTIFACTS.md`
10. `posecred_ipg/docs/08_LIMITATIONS_AND_NEXT_STEPS.md`
11. `posecred_ipg/docs/09_THESIS_WRITEUP_GUIDE.md`
12. `posecred_ipg/docs/10_DATA_PIPELINE_TABLE.md`
13. `posecred_ipg/docs/11_SCORE_SEMANTICS.md`

## 当前一句话结论

- 当前 `PoseCred-IPG v1` 的正式主线已经固定为：
  `hybrid + N_pair=32 + score_head/dockq_head + listwise/pairwise/dockq + default shard + no_bad_head + no_clash_post`。
