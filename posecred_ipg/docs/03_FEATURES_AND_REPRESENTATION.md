# 3. 特征与表示

> 文档类型：当前代码说明
>
> 本文只描述当前现役特征表示和当前固定特征口径；历史裁维路线与实验记录不随公开代码仓库发布。

## 3.1 整体表示

当前主模型使用的是 **residue-pair interface graph**。

节点不是单个原子，也不是单个残基，而是一个界面残基对：

- `(receptor residue, peptide residue)`

这个选择的核心原因：

- docking pose 的关键不在单残基本身，而在界面接触对
- residue-pair 比 atom-level 轻得多
- 对 pose reranking 足够直接

## 3.2 candidate generation 与 pruning

### Step A: candidate generation

候选 pair 的初筛条件：

- `min heavy atom distance < 8A`

### Step B: pruning

当前 snapshot / 表示留档使用：

- `prune_strategy = hybrid`
- `N_pair = 32`

`hybrid` 的设计目标是同时兼顾：

- 最近接触
- peptide coverage
- clash 区域保留
- 局部界面连续性

当前表示层留档固定为：

- `hybrid pruning`
- `N_pair = 32`

## 3.3 当前完整主特征

当前正式主模型 `graph_main` 使用的 node 特征总维度是：

- `node_feat = 70`

global 特征总维度是：

- `global_feat = 12`

定义在：

- `posecred_ipg/feature_layout.py`

### 节点特征切片

- `residue_identity: 0:52`
- `geometry: 52:59`
- `physchem: 59:64`
- `clash: 64:67`
- `direction: 67:70`

### 全局特征切片

- `basic: 0:4`
- `clash: 4:12`

## 3.4 每类特征的实际含义

### residue_identity

包括：

- receptor aa one-hot
- peptide aa one-hot
- residue flags
- peptide 相对位置

作用：

- 区分不同残基身份与基本理化类别
- 保留 peptide 序列位置信息

### geometry

包括：

- `CA/CB/min-heavy` 距离
- 均值距离
- overlap 相关几何量
- 一些基础相对几何量

作用：

- 提供接触几何合理性
- 区分近接触、松散接触与错位接触

### physchem

包括：

- 电荷互补
- 疏水互补
- 芳香相关互补
- 局部 density 类统计

作用：

- 提供接触是否“化学上更像真实界面”的补充信息

### clash

包括：

- clash flag
- severe clash flag
- overlap

作用：

- 显式描述物理异常
- 当前实验表明这类输入特征应保留

### direction

包括：

- 简化方向向量

作用：

- 补充局部几何朝向信息
- 当前影响相对较小

## 3.5 edge 与全局表示

边关系主要包括：

- 共享 receptor residue
- 共享 peptide residue
- 空间近邻 edge

全局特征主要覆盖：

- 基础接口规模摘要
- 全局 clash 摘要

## 3.6 当前关于特征的正式结论

- `clash` 输入应保留
- `physchem` 和 `direction` 不是当前最强刚需
- 真正的速度瓶颈不在这几维特征
- 若继续优化速度，应优先从 IO / batch / graph forward 入手
