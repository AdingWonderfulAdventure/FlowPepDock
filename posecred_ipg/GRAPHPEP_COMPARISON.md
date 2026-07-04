# PoseCred-IPG 与 GraphPep 的差异对比

> 文档类型：当前代码说明（方法定位补充）
>
> 本文用于说明当前 `PoseCred-IPG` 与 `GraphPep` 的方法定位差异，不记录开发过程或历史实验路线。

## 1. 结论先行

可以把两者的关系概括成一句话：

> `PoseCred-IPG` 明确继承了 `GraphPep` 的 **interaction-derived / residue-pair graph** 思想，但并不是对 `GraphPep` 的直接复现，而是一个针对 **protein-peptide docking pose reranking** 场景做出的轻量化、排序化和工程化改造版本。

如果后面论文里需要一句较稳妥的表述，建议写成：

> 本工作受到 `GraphPep` 提出的相互作用派生图表示启发，不再以单原子或单残基作为唯一建图单元，而是以蛋白-多肽界面残基对作为核心表示对象；但不同于 `GraphPep` 面向通用复合物打分的双尺度图框架，本文进一步围绕 docking pose 重打分任务，在图构建、监督目标、训练批组织和物理约束建模上进行了针对性改造。

## 2. 共同点：PoseCred-IPG 与 GraphPep 的方法继承关系

两者最核心的共性只有一条，但这一条很关键：

- 都没有把整个复合物直接建成传统“全原子图”或“单残基图”。
- 都把 **界面相互作用单元** 当成更重要的建模对象。
- 都认为蛋白-多肽构象质量的关键，不在全局坐标本身，而在界面接触是否合理。

换句话说，`PoseCred-IPG` 的方法论源头确实可以追溯到 `GraphPep`，这一点完全可以在论文中明确写明。

## 3. 核心差异总表

| 维度 | GraphPep | PoseCred-IPG | 该差异的意义 |
| --- | --- | --- | --- |
| 核心任务定位 | 面向 protein-peptide complex 的通用打分与 decoy 评分 | 面向 docking 输出的 **pose reranking / 重打分** | 从“通用复合物评分”进一步收缩到“组内排序” |
| 图表示层级 | **双尺度图**：atom-level interaction graph + residue-level interaction graph | **单尺度图**：只保留 residue-pair interface graph | 结构更轻，推理和工程维护更直接 |
| 节点定义 | 原子相互作用对、残基相互作用对都参与建图 | 节点直接定义为 `(receptor residue, peptide residue)` | 保留 GraphPep 的核心灵感，但砍掉 atom 级复杂度 |
| 候选界面控制 | 公开程序默认 `dis_threshold = 6`，按阈值生成 interaction-derived 图 | 先用 `min heavy atom distance < 8A` 生成候选，再做 `hybrid + N_pair=32` pruning | 明确把界面图压缩到少量关键 pair，更适合 reranking |
| 特征来源 | 原子/残基 one-hot、距离与角度、统计势，并融合 `ESM-2` | 手工构造 `70D node + 12D global`，包含 identity/geometry/physchem/clash/direction | 更强调 docking 场景下的几何与物理可解释性 |
| 语言模型使用 | 明确引入 `ESM-2` 增强 residue 表示 | 当前正式主线 **不引入 ESM** | 降低依赖和成本，保持主线稳定 |
| 输出语义 | 先预测 interaction/contact 置信度，再把高阈值节点计数聚合为类似 `fnat` 的分数 | 直接输出 `score_pose`，并保留 `DockQ aux head` | 从“接触计数式评分”转为“直接 pose 排序分数” |
| 监督目标 | 论文摘要强调：关注 residue-residue contact，而不是单一 peptide RMSD | 主监督是 **group 内 DockQ 排序**，损失由 `listwise + pairwise + DockQ aux` 组成 | 明确针对“同组 pose 谁该排前面”来优化 |
| batch 组织方式 | 公开程序主要展示逐复合物 decoy 评分流程 | 训练时按 `group` 组织 batch，同组 pose 一起进网络 | 天然适配 docking reranking 场景 |
| 物理异常建模 | 公开程序支持可选 plausibility check / PoseBusters 输出 | 直接把 `clash` 做成节点特征、全局特征和物理坏样本标签 | 物理约束被前移进模型，而不只是外部后处理 |
| 最终主指标 | 文献主打不同 decoy benchmark 上的排序能力 | 当前主指标固定为 `val_top1_success`，即组内 Top-1 是否满足 `DockQ >= 0.49` | 评价口径更贴合对接工作流的“第一名是否可用” |

## 4. 差异展开：论文中值得重点阐述的部分

### 4.1 从双尺度 GraphPep 变成单尺度 IPG

`GraphPep` 的公开程序显示，它并不是只做 residue-pair 图，而是先构建 atom-level interaction graph，再把 atom 图信息汇聚到 residue-level interaction graph 上，最后输出分数。  
而 `PoseCred-IPG` 则明确放弃了 atom-level 分支，只保留 residue-pair interface graph。

这并不是简单的结构裁剪，而是围绕任务目标做出的重构：

- `GraphPep` 更强调通用复合物打分能力；
- `PoseCred-IPG` 更强调 docking decoy 的组内重排序；
- 在这个任务里，界面残基对往往已经足够承载主要判别信息；
- 因此去掉 atom-level 图后，模型复杂度、IO 压力和工程成本都会明显下降。

这部分在论文里可以解释成：

> 本文保留了 `GraphPep` 的相互作用派生表示思想，但未沿用其双尺度原子-残基图结构，而是将建模重点收敛到 residue-pair interface graph，以降低模型复杂度并强化其在 pose reranking 任务上的针对性。

### 4.2 从 contact-centric scoring 变成 group-wise ranking

`GraphPep` 的摘要明确指出，它关注的是 **residue-residue contacts**，而不是使用单一 peptide RMSD 作为损失核心。  
公开 `v1.1` 程序进一步表明：它先预测 interaction/contact 的置信度，再通过多个阈值上的节点计数聚合出最终 decoy 分数。

`PoseCred-IPG` 则走了另一条路：

- 主输出直接定义为 `score_pose`；
- 主监督不是“接触计数”；
- 而是针对同一个 `group` 内多个 decoy pose 的 **相对排序**；
- 以 `DockQ` 作为主标签；
- 并组合 `listwise ranking + pairwise ranking + DockQ auxiliary regression`。

这意味着：

- `GraphPep` 更像是“基于局部 contact 可信度推导整体构象分数”；
- `PoseCred-IPG` 更像是“直接学习同组 pose 的优先级关系”。

对你的论文来说，这一条是最重要的创新差异之一。

### 4.3 从 ESM 增强转向轻量、可控、可复现路线

`GraphPep` 公开程序默认支持 `ESM-2`，而且把受体残基和肽残基的 `ESM-2` 向量一起并入 residue graph。

`PoseCred-IPG` 当前主线则明确不引入 `ESM`，理由不是“忘了做”，而是主动取舍：

- 当前主线已经成立；
- 当前短板并不主要来自序列语义；
- 更重要的是交付稳定性、训练吞吐和工程落地；
- 过早引入 `ESM` 会增加数据准备与部署复杂度。

因此，和 `GraphPep` 相比，你的方法更偏向：

- 轻量化；
- 易部署；
- 依赖更少；
- 更适合和现有 docking 流水线拼接。

### 4.4 从“外部 plausibility 检查”转成“模型内部显式 clash 建模”

`GraphPep` 的公开程序支持可选 plausibility check，并能输出额外的通过项统计。  
`PoseCred-IPG` 则把这类物理合理性信息直接前置到了模型输入中：

- 节点特征里显式保留 `clash`；
- 全局特征里显式统计 `severe_clash_count`、`max_overlap`、`min_heavy_dist` 等；
- 还构造了 `physical_bad_label`；
- 但它在当前代码里只作为物理异常标签保留，不再对应训练期 `bad head`；
- 最终正式主版本也没有采用额外的 clash 后处理惩罚。

这个区别很适合在论文里写成：

> 与 `GraphPep` 倾向于在模型外侧进行可选合理性校验不同，本文将 docking pose 的物理异常信息内生化到图特征中，使模型能够在排序过程中直接感知 steric clash 等不合理界面模式。

### 4.5 从“全量 interaction graph”转向“hybrid pruning + N_pair=32”

你这个版本还有一个很明显的工程创新点：**candidate pruning**。

`PoseCred-IPG` 不是把所有可能界面 pair 一股脑塞进图里，而是：

1. 先按距离生成候选 pair；
2. 再用 `hybrid` 策略综合考虑：
   - 最近接触；
   - peptide 覆盖；
   - clash 区域保留；
   - 局部界面连续性；
3. 最终把图压到 `N_pair = 32`。

这一步是 `GraphPep` 公开程序中没有重点强调的，也是 `PoseCred-IPG` 特别适合在论文中展开说明的部分。  
它说明本文并非只是“借鉴表示”，而是针对 docking reranking 的资源约束和噪声特点，进一步提出了 **面向任务的界面压缩设计**。

## 5. 一句句话术：论文里怎么写最稳

### 5.1 一句话交代方法来源

> 本文方法受到 `GraphPep` 的启发，采用界面相互作用派生图而非传统单残基图来表征蛋白-多肽复合物。

### 5.2 一句话交代关键不同

> 不同于 `GraphPep` 的双尺度 atom/residue 图打分框架，本文面向 docking pose 重打分任务，采用单尺度 residue-pair interface graph，并引入基于 `DockQ` 的 group-wise 排序训练目标。

### 5.3 一句话交代你的贡献

> 在保留 interaction-derived 表示优点的同时，本文进一步引入 `hybrid` 候选裁剪、显式 clash 特征与组内排序监督，使模型更适合蛋白-多肽 docking decoy 的高效重打分。

## 6. 建议你在论文里避免的写法

下面这些写法不太稳，建议别用：

- “本文复现了 `GraphPep` 并做了少量修改。”
- “本文与 `GraphPep` 基本一致，只是损失函数不同。”
- “本文只是去掉了 `ESM` 和 atom graph。”

更稳妥的写法应该是：

- “本文受到 `GraphPep` 启发，但围绕 pose reranking 做了任务重定义。”
- “本文保留 interaction-derived residue-pair 思路，同时对图层级、特征体系和监督目标进行了重构。”
- “本文不是 `GraphPep` 的工程裁剪版，而是一个面向 docking 重打分场景的定制化变体。”

## 7. 本文档的依据来源

### 7.1 你仓库里的 PoseCred-IPG 依据

- `posecred_ipg/docs/03_FEATURES_AND_REPRESENTATION.md`
- `posecred_ipg/docs/04_MODEL_AND_LOSSES.md`
- `posecred_ipg/docs/05_TRAINING_AND_EVALUATION.md`
- `posecred_ipg/docs/08_LIMITATIONS_AND_NEXT_STEPS.md`
- `posecred_ipg/data/features.py`
- `posecred_ipg/models/graph.py`
- `posecred_ipg/engine/losses.py`
- `posecred_ipg/data/dataset.py`

### 7.2 GraphPep 的公开依据

- 论文摘要：`An interaction-derived graph learning framework for scoring protein-peptide complexes`  
  `https://www.nature.com/articles/s42256-025-01136-1`
- GraphPep 程序包：`GraphPep program (Zenodo, v1.1)`  
  `https://zenodo.org/records/17099863`

### 7.3 需要注意的边界

- `GraphPep v1.1` 的公开程序主要展示了推理代码和模型定义；
- 它没有在程序包里完整附带训练脚本；
- 因此本文档关于 `GraphPep` 训练目标的表述，凡涉及“contact 而非 RMSD”的部分，主要依据论文摘要；
- 凡涉及双尺度图、ESM-2、阈值聚合打分的部分，主要依据公开程序源码与 README。

## 8. 论文写作建议

如果后续论文需要强调“创新性”，更稳妥的说法不是简单宣称“优于 `GraphPep`”，而是：

1. 先承认方法灵感来自 `GraphPep` 的 interaction-derived graph；
2. 再明确指出你把它改造成了 **面向 pose reranking 的单尺度 residue-pair 图排序器**；
3. 然后突出你自己的三件事：
   - `hybrid + N_pair=32` 的界面裁剪；
   - 基于 `DockQ` 的 group-wise ranking 训练；
   - 显式 clash / 物理异常特征建模。

这样写逻辑更完整，也更容易把“受启发”和“有独立改造”这两层关系交代清楚。
