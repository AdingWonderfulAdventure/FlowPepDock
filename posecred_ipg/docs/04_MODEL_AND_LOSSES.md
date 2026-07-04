# 4. 模型结构与训练目标

> 文档类型：当前代码说明
>
> 本文只描述当前现役模型结构、当前训练目标和当前兼容口径；相关开发历史不随公开代码仓库发布。

## 4.1 当前正式主模型

当前正式主模型是一个轻量 residue-pair 图排序器：

- 2 层 GINE/MPNN 风格编码器
- hidden dim `96`
- attention + mean + max pooling
- `score head` 为主输出，`DockQ` 为当前启用的辅助监督

核心实现：

- `posecred_ipg/model.py / posecred_ipg/graph implementation files`

## 4.2 模型输入

每个 pose 输入为：

- `node_feat`
- `edge_index`
- `edge_feat`
- `global_feat`

训练时按 group 组织 batch，不是简单逐 pose 独立回归。

## 4.3 输出

正式主输出：

- `score_pose`

语义：

- pose 的结构可信度 / 结构合理性 / 与高质量真实构象的相似度

当前仍保留的辅助监督：

- `DockQ aux head`

当前现役模型只包含 `score_head` 和 `dockq_head`，不包含 `bad head`。

## 4.4 当前损失设计

当前损失只由三部分组成：

1. `listwise ranking`
2. `pairwise ranking`
3. `DockQ aux regression`

对应实现：

- `posecred_ipg/losses.py`

## 4.5 当前正式主版本损失权重

当前默认权重：

- `listwise = 1.0`
- `pairwise = 0.5`
- `dockq = 0.2`

也就是：

- `loss_weights = (1.0, 0.5, 0.2)`

兼容说明：

- 旧快照里如果仍出现 4 项 `loss_weights`，当前代码只接受第 4 项为 `0.0`
- 旧 checkpoint 若仍带 `bad_head.*` 权重，加载时会自动忽略

## 4.5.1 `bad head` 废弃后的同步处理项

`bad head` 从当前主线代码中移除后，以下项目也同步废弃或在运行时被显式跳过：

- `bad_head` 输出头：已从现役模型结构中移除
- `bad BCE`：已从当前主线训练损失中移除
- `bad_loss_weight`：已废弃；当前训练入口若再传入会直接报错
- `loss_weights` 第 4 项：已废弃；仅保留“旧配置兼容时必须为 `0.0`”这一例外
- `physical_bad_label -> bad_head` 监督链路：已断开；`physical_bad_label` 只保留为数据构建阶段的物理异常标签
- 旧 checkpoint 中的 `bad_head.*` 参数：加载时自动忽略，不再参与现役模型恢复

因此，当前主线里凡是涉及 `bad head / bad BCE / bad_loss_weight / 4th loss weight` 的内容，都应按历史兼容项理解，而不是现役训练配置。

## 4.6 为什么主监督只用 DockQ

当前主排序监督只用 `DockQ`，不混入 `RMSD`。

原因是：

- `DockQ` 更直接描述界面质量
- 更适合作为 pose ranking 主标签
- `RMSD` 可用于辅助分析，但不适合作为当前主排序定义

## 4.7 当前不纳入主线的项

当前正式主版本：

- 保留 clash 输入
- 不保留 `bad head`
- 不保留 `bad BCE`
- 不保留 `bad_loss_weight`
- 不保留 `loss_weights` 第 4 项
- `physical_bad_label` 仅作为 record 构建阶段的物理异常标签保留，不再对应训练期输出头

## 4.8 当前不做 clash 后处理

`clash` 后处理指的是：

- `final_score = model_score - beta * clash_penalty`

当前默认关闭该后处理，当前正式推理口径直接使用 `clash_penalty_scale = 0.0`。

## 4.9 当前模型线的正式结论

- 当前模型路线已经成立
- 主模型不需要继续大改结构
- 当前最优解不是更复杂模型，而是保持现结构并做工程优化
