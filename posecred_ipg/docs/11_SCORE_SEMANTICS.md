# PoseCred-IPG Score 语义说明

> 文档类型：当前代码说明
>
> 本文说明当前 `PoseCred-IPG v1` 输出列 `score` / 主输出 `score_pose` 的来源、训练语义、数值范围、推荐用法和常见误解。

## 1. 一句话结论

`PoseCred-IPG` 的 `score` 是一个用于 pose reranking 的相对排序分数。

- 分数越大，模型越倾向于认为该 pose 更接近高质量真实结合构象。
- 它不是成功概率。
- 它不是 DockQ 本身。
- 它不是 binding affinity。
- 它没有固定的理论上下界。

当前正式推理口径下，`clash_penalty_scale = 0.0`，因此导出的 `score` 就是模型 `score_head` 的裸输出。

## 2. 代码里的分数从哪里来

当前正式主模型实现位于：

- `posecred_ipg/models/graph.py`

模型先把一个 pose 编码为 `pose_repr`：

```text
node_feat / edge_index / edge_feat / global_feat
  -> GINE/MPNN encoder
  -> attention pool + mean pool + max pool
  -> readout
  -> pose_repr
```

然后通过两个输出头：

```text
score_head(pose_repr) -> score_pose
dockq_head(pose_repr) -> DockQ auxiliary prediction
```

当前主输出是：

```python
self.score_head(pose_repr).squeeze(-1)
```

这里没有 `sigmoid`、没有 `softmax`、没有 `clamp`。也就是说，`score_pose` 是线性层直接输出的实数。

## 3. 导出表里的 `score` 是什么

逐 pose 导出、pooled/global 评估和 cross pose 打分最终都会走如下形式：

```text
final_score = model_score - clash_penalty_scale * clash_penalty
```

其中：

```text
clash_penalty = sum(clash_penalty_feat * clash_penalty_weights)
```

当前正式口径是：

```text
clash_penalty_scale = 0.0
```

所以当前正式导出的：

```text
score = model_score
```

如果后续手动把 `clash_penalty_scale` 设为大于 0，则 `score` 会变成带物理 clash 后处理的扣分结果，数值会整体降低，且不能再和 `clash_penalty_scale = 0.0` 的结果直接混比。

## 4. 这个分数是怎么被训练出来的

当前训练损失由三部分组成：

1. `listwise ranking`
2. `pairwise ranking`
3. `DockQ auxiliary regression`

对应代码：

- `posecred_ipg/engine/losses.py`

### 4.1 Listwise ranking

同一个 `group_id` 内的一组候选 pose 会一起参与排序监督。

目标分布来自 DockQ：

```text
target = softmax(DockQ / dockq_temperature)
```

模型预测分布来自 score：

```text
pred = log_softmax(score / score_temperature)
```

训练目标是让高 DockQ 的 pose 在同组内拿到更高的 score。

### 4.2 Pairwise ranking

如果同组内两个 pose 的 DockQ 差异足够大：

```text
DockQ_i - DockQ_j > pair_margin_dockq
```

则训练会要求：

```text
score_i - score_j >= pair_margin_score
```

当前默认配置：

```text
pair_margin_dockq = 0.1
pair_margin_score = 0.2
```

这进一步强化了“更好 DockQ 应该更高 score”的相对顺序。

### 4.3 DockQ auxiliary regression

`dockq_head` 会额外预测 DockQ，并用 Huber loss 监督。

这只是辅助监督，不等于 `score_head` 被训练成 DockQ 回归器。正式排序、导出和 cross pose reranking 使用的是 `score_head` 输出。

## 5. 数值范围

### 5.1 理论范围

`score_head` 是线性层裸输出，因此理论范围是：

```text
(-inf, +inf)
```

负数、0、大于 1、大于 2 都是合法值。

不要把 `score` 当成 `[0, 1]` 区间内的置信度。

### 5.2 当前 checkpoint 的实测范围

使用当前正式 checkpoint：

- `posecred_ipg/final_exports/graph_main_best.pt`

使用当前正式 val shard：

- `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/val_shards_v1/shard_index.csv`

在 `clash_penalty_scale = 0.0` 下导出 `8390` 条 val pose，得到当前实测分布：

```text
count    8390
mean     0.388
std      0.466
min     -1.513
p5      -0.276
p10     -0.134
p25      0.099
p50      0.347
p75      0.635
p90      0.971
p95      1.238
p99      1.781
max      2.562
```

这不是模型的硬范围，只是当前 checkpoint、当前数据口径、当前特征构建方式下的经验分布。

## 6. 怎么解释一个具体 score

推荐按下面方式解释：

| score 区间 | 经验解释 |
| --- | --- |
| `< 0` | 相对偏低，模型通常不太看好 |
| `0 ~ 0.5` | 常见中低区间，需要看同组排序 |
| `0.5 ~ 1.0` | 常见中高区间，有一定排序优势 |
| `1.0 ~ 1.5` | 当前 val 分布里偏高 |
| `> 1.5` | 当前 val 分布里很高，但仍不是概率 |

这个表只能用于当前正式 checkpoint 的经验解释，不能当成跨模型、跨数据集、跨后处理参数的通用标尺。

## 7. 推荐使用方式

### 7.1 同一批候选 pose 内排序

这是最推荐、也最符合训练目标的用法：

```text
同一个 peptide / receptor / group 的多个 pose
  -> 按 score 降序
  -> 取 top1 / top5
```

### 7.2 Cross pose / cross receptor reranking

当前 `score_cross_pose_table` 会输出：

- `per_pose_scores.csv`
- `group_best_scores.csv`
- `peptide_summary.csv`

推荐先看 `per_pose_scores.csv` 的逐 pose 排名，再看 `group_best_scores.csv` 的每个 receptor/group 最优 pose。

### 7.3 设阈值筛选

如果必须用固定阈值，建议先在当前任务的验证集或历史结果上重新校准。

当前 val 分布下可以粗略参考：

- `score > 1.0`：较强候选
- `score > 1.5`：高分候选

但这只是经验阈值，不是训练时定义的分类边界。

## 8. 不推荐的用法

不要这样解释：

```text
score = 0.8，所以成功概率是 80%
```

不要这样解释：

```text
score = 0.8，所以 DockQ 约等于 0.8
```

不要这样直接比较：

```text
模型 A 的 score = 1.2
模型 B 的 score = 1.0
所以模型 A 更好
```

除非两个分数来自同一个 checkpoint、同一套特征构建、同一个 `clash_penalty_scale`、同一种输入分布，否则绝对数值不可直接比较。

## 9. score 与 DockQ 的关系

`score` 的训练监督来自 DockQ，因此它和 DockQ 有正相关关系。

但它不是 DockQ 回归值。当前 val 导出结果上：

```text
Pearson(score, DockQ) 约 0.364
Spearman(score, DockQ) 约 0.336
```

这说明 score 有排序信号，但不能当成 DockQ 的线性替代。

按 score 分箱看，当前 val 上的经验趋势如下：

| score 区间 | pose 数 | 平均 DockQ | DockQ >= 0.49 比例 |
| --- | ---: | ---: | ---: |
| `<= -0.5` | 151 | 0.277 | 0.113 |
| `-0.5 ~ 0.0` | 1308 | 0.326 | 0.151 |
| `0.0 ~ 0.5` | 3968 | 0.345 | 0.168 |
| `0.5 ~ 1.0` | 2186 | 0.410 | 0.288 |
| `1.0 ~ 1.5` | 586 | 0.502 | 0.549 |
| `> 1.5` | 191 | 0.593 | 0.812 |

趋势是分数越高，高 DockQ pose 的比例越高；但每个区间内部仍然会有误差。

## 10. 为什么 score 没有绝对物理单位

当前 `score` 是神经网络排序分数，不对应能量单位、距离单位或概率单位。

原因包括：

- `listwise` 主要约束同组内 softmax 排序。
- `pairwise` 主要约束有明显 DockQ 差异的 pose 之间的分数间隔。
- 线性输出头没有概率校准。
- 训练没有要求 score 落在固定区间。
- 不同 checkpoint 的 score 尺度可能不同。

因此，score 的正确身份是：

```text
learned ranking score
```

不是：

```text
calibrated confidence
```

## 11. 复现实测分布的命令

当前文档中的 val 分布可用以下命令复核：

```bash
python -m posecred_ipg.export_scores \
  --checkpoint posecred_ipg/final_exports/graph_main_best.pt \
  --out_csv tmp/ipg_val_scores_for_analysis.csv \
  --model_name posecred_ipg \
  --use_default_shard_snapshot \
  --split val \
  --groups_per_batch 32 \
  --poses_per_group 0 \
  --device cpu \
  --num_workers 0 \
  --clash_penalty_scale 0.0
```

统计命令：

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("tmp/ipg_val_scores_for_analysis.csv")
print(df["score"].describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
print(df[["score", "dockq"]].corr(method="pearson"))
print(df[["score", "dockq"]].corr(method="spearman"))
PY
```

## 12. 论文或报告中的推荐表述

可以这样写：

```text
PoseCred-IPG outputs a learned pose ranking score for each candidate protein-peptide pose. The score is produced by a linear score head on top of the interface-pair graph representation and is trained with listwise and pairwise DockQ-based ranking losses, together with an auxiliary DockQ regression head. The score is used for relative reranking and is not a calibrated probability or a direct DockQ estimate.
```

中文表述：

```text
PoseCred-IPG 为每个候选 protein-peptide pose 输出一个学习得到的排序分数。该分数由界面残基对图表示上的线性 score head 产生，并通过基于 DockQ 的 listwise/pairwise 排序损失训练，同时保留 DockQ 辅助回归头。该分数用于候选构象的相对重排序，不是经过校准的成功概率，也不是 DockQ 的直接估计值。
```

