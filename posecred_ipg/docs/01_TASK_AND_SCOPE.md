# 1. 任务定义与适用范围

> 文档类型：当前代码说明
>
> 本文只定义当前任务口径和当前适用范围；历史路线与开发过程统一见 `posecred_ipg/STATUS_AND_PLAN.md`。

## 1.1 这个模型在做什么

`PoseCred-IPG` 当前做的是蛋白-多肽 docking 后的 **pose 重打分与排序**。

输入场景是：

- 一个 peptide
- 多个候选 receptor
- 每个 receptor 上若干 docking poses

模型输出每个 `receptor x pose` 的单一分数 `score_pose`，然后按分数统一排序，优先筛出最像高质量真实结合构象的候选。

## 1.2 这不是在做什么

这个模型不是：

- binder / non-binder 分类器
- 真实亲和力预测器
- 真实结合概率预测器
- 生物学是否发生互作的判断器

它的分数语义应解释为：

- 该 pose 与训练集中高质量真实构象的相似度
- 该 pose 的结构可信度 / 构象合理性
- 该 pose 看起来像不像真实共晶体系里的好 pose

## 1.3 为什么要这样定义

训练数据来自真实共晶体系的重对接：

- 已知蛋白-多肽共晶复合物
- 拆分后重新 docking
- 对每个 pose 计算相对原始共晶的 `DockQ` 和 `RMSD`

因此监督本质上是：

- “这个 pose 与真实高质量构象接近吗”

而不是：

- “这个 receptor-peptide 配对在自然界里到底是真是假”

所以这个任务天然更适合定义成：

- `positive-only`
- `open-world`
- `pose ranking`

## 1.4 当前模型的适用范围

当前版本适合：

- 大量 docking poses 的后重打分
- 同一个 peptide 在多个 receptor 上的候选姿态统一排序
- 高质量 pose 的 top-1 / top-k 检出

当前版本不直接覆盖：

- 真实结合自由能预测
- 实验亲和力回归
- binder discovery 的真假分类
- 大规模序列级互作筛选

## 1.5 当前版本的 snapshot / 结构留档

以下内容是当前版本的 snapshot / 结构留档，不是训练或推理命令里必须显式传的运行超参数：

- `prune_strategy = hybrid`
- `N_pair = 32`
- `bad head` 已在当前主线代码中废弃并移除
- `clash_penalty_scale = 0.0`

其中：

- `bad head` 当前已不属于主线代码
- 显式 `clash` 后处理当前不作为默认推理步骤使用

## 1.6 当前版本的两个现役模型入口

当前仓库内保留的两个现役模型入口是：

### 正式主模型

- `graph_main`
- 用于追求当前最佳精度

### 当前基线模型

- `stats_mlp`
- 用于提供当前仍可直接复现的非图基线口径
