# Flow / IPG 运行口径硬性约束

虽然文件名还叫 `FLOW_DATASET_CONTRACT.md`，但从现在开始，这份文档同时约束：

- `Flow` 的训练 / 推理口径
- `PoseCred-IPG` 的训练 / 评估 / 重打分口径

这份文档是当前仓库 `Flow + PoseCred-IPG` 主线在 **训练 / 推理 / 评估** 三个阶段的**唯一正式数据与超参数约束**。

只要任务涉及以下任一事项，都必须先读这份文档：

- `Flow` 训练
- `Flow` 验证
- `Flow` 推理
- `Flow` benchmark
- `PoseCred-IPG` 训练
- `PoseCred-IPG` 评估
- `PoseCred-IPG` cross pose 重打分
- 修改脚本、配置、README、AGENTS、笔记或自动化流程
- 任何 AI / 代理 / 接手者尝试判断“Flow / IPG 现在到底该用哪套数据和超参数”

## 1. 唯一正式 CSV

### 1.1 Train

- **正式训练 CSV**：`data/runtime_tables/flow_train_rel.csv`
- 行数：`6037`
- 字段：`complex_name,pdb_dir`

### 1.2 Val

- **正式验证 CSV**：`data/runtime_tables/flow_val_rel.csv`
- 行数：`600`
- 字段：`complex_name,pdb_dir`

### 1.3 Test / Inference

- **正式测试 / 推理 CSV**：`data/runtime_tables/flow_infer_test536_rel.csv`
- 行数：`536`
- 字段：`complex_name,receptor_pdb,peptide_pdb`

## 2. 源表对应关系

正式运行时 CSV 使用的是当前仓库可直接消费的**相对路径版**主表：

- `data/runtime_tables/flow_train_rel.csv`
  - 对应历史主拆分：`data/rebuild_isolated/rebuild_20251221_163301/11_split_train_val_9to1/train.csv`
- `data/runtime_tables/flow_val_rel.csv`
  - 对应历史主拆分：`data/rebuild_isolated/rebuild_20251221_163301/11_split_train_val_9to1/val.csv`
- `data/runtime_tables/flow_infer_test536_rel.csv`
  - 对应 clean `536` 源表：`data/processed_test30/pt_available.fully_clean.csv`
  - `inference.py` 对命中正式 strict-536 `complex_name` 的输入行，会强制收口到这张表里的 canonical `receptor_pdb / peptide_pdb`

## 3. Flow 强制规则

以下规则不是建议，是硬约束：

- **任何 Flow 训练命令**，默认只能写 `data/runtime_tables/flow_train_rel.csv`
- **任何 Flow 验证命令**，默认只能写 `data/runtime_tables/flow_val_rel.csv`
- **任何 Flow 正式测试 / 推理 / strict536 benchmark 命令**，默认只能写 `data/runtime_tables/flow_infer_test536_rel.csv`
- 即便外部子集 / 历史诊断 CSV 误把 strict-536 的某些行改回 `RefPepDB-RecentSet` / `pepset` raw 路径，只要 `complex_name` 命中正式 strict-536 主表，推理入口也必须回收为 `data/processed_test30/` 的 canonical 资产
- 如果文档、脚本、注释、笔记、对话答案里出现了别的 Flow 主 CSV，除非明确写出“历史 / 诊断 / 子集 / 废弃”，否则一律视为错误
- 如果后续需要改动这三张正式 CSV，必须同步更新：
  - `FLOW_DATASET_CONTRACT.md`
  - `README.md`
  - `AGENTS.md`
  - `notes/AI_CONTEXT.md`
  - `docs/FILE_STRUCTURE.md`
- **任何 Flow 正式训练命令**，都必须同时写明：
  - 数据集目录
  - train / val CSV
  - 训练配置文件
  - 当前实际生效的训练关键超参数
- **任何 Flow 正式推理命令**，都必须同时写明：
  - 数据集目录
  - test CSV
  - `model_dir`
  - `ckpt`
  - 当前实际生效的推理关键超参数
- 不允许只写“用默认配置跑一下”这种憨批表述而不把关键超参数落盘

## 4. 明确不要混用的 CSV

下面这些文件**不是**当前 Flow 主线的 train / val / test 正式入口，不能拿来替代上面的三张主表：

- `data/processed_test30/pt_available.csv`
- `data/processed_test30/pt_available.strict_exact_clean.csv`
- `data/processed_test30/pt_available.unseen_peptide_clean.csv`
- `data/runtime_tables/flow_infer_test536_rel_with_source_set.csv`
- `data/runtime_tables/flow_infer_test536_rel_2chains.csv`
- `data/runtime_tables/flow_infer_test536_pepset_only.csv`
- `data/runtime_tables/flow_infer_test536_refpep_recent_only.csv`
- `data/benchmark_subsets/test536_known_pocket_stratified_*/flow_infer_subset_rel.csv`
- `data/rebuild_isolated/rebuild_20251221_163301/11_split_train_val_9to1/train_single_random_not_highloss.csv`
- 任何 `tmp/`、`results/diagnostics/`、`legacy/` 下的 CSV
- 任何 `ipg / posecred_ipg` 相关 CSV

## 5. 当前 checkpoint 与这三张表的关系

- 当前仓库默认 Flow checkpoint：`train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt`
- 当前 `tclip` checkpoint：`logs/full_abs_tight_tclipped028_ddp1234567_bs10_20260414_153322/flow_esm_best.pt`

这两条 Flow 线在当前仓库确认使用的是**同一套 train / val 内容**：

- Train：`data/runtime_tables/flow_train_rel.csv`
- Val：`data/runtime_tables/flow_val_rel.csv`

它们的主要差异在训练策略，而**不是**当前仓库记录下来的 train / val 数据划分。

### 5.1 `tclip` checkpoint 的 step 口径

- 当 `Flow` 使用 `tclip` checkpoint  
  `logs/full_abs_tight_tclipped028_ddp1234567_bs10_20260414_153322/flow_esm_best.pt`
  做正式推理时，当前**优先 step = 3**
- 这里的“优先”含义是：
  - 它是当前仓库给 `tclip` 这条线保留的**运行折中默认值**
  - 不是“所有评测榜单上单项精度绝对第一的 step”
- 原因必须回指这两批结果目录，不允许口头瞎编：
  - `results/tclip028_step_sweep_stratified128_g123456_20260424_174749/full536_val300_step01to10_n5_bs32_20260424_final/reports/strict_step1to10_combined_report.md`
  - `results/tclip028_step_sweep_stratified128_g123456_20260424_174749/full536_val300_step01to10_n5_bs32_20260424_final/reports/test536_vs_val300_step_consistency.md`
  - `results/tclip028_n_sweep_step03_stratified96_n5to30_bs160_g123_20260424_213246/README.md`
- 这几份结果给出的结论是：
  - `2026-04-24` 这轮 `full536 + val300` 的 step sweep 里，单看 `mean_dockq`：
    - `test536` 最优是 `step=10`
    - `val300` 最优是 `step=7`
  - 但 `step=3` 是当前保留的**精度 / 速度折中位**：
    - 在 `test536` 上，`step=3` 的 `complex<=2A` 命中率是这轮里最高档之一，而且显著快于 `step=7/10`
    - 在 `val300` 上，`step=3` 的 `mean_dockq` 排名第 `2`，但耗时明显低于 `step=7/10`
  - 后续 `N` sweep 也正是固定在 `step=3` 上继续做的，因此当前仓库把 `step=3` 作为 `tclip` 线的优先运行口径保留：
    - `results/tclip028_n_sweep_step03_stratified96_n5to30_bs160_g123_20260424_213246/reports/n_sweep_summary.md`

## 6. Flow 当前正式运行口径

### 6.1 Flow 正式训练口径

> 下面这一节只保留 **Flow 训练主环真实读取** 的数据入口和超参数。  
> 不属于训练主环、只在推理或训练期可选小评估里生效的参数，不算“正式训练口径”。

#### 数据集目录

- 结构资产根目录：`data/rebuild_isolated/rebuild_20251221_163301/`
- 训练样本物理目录：`data/rebuild_isolated/rebuild_20251221_163301/processed/`
- 训练 CSV：`data/runtime_tables/flow_train_rel.csv`
- 验证 CSV：`data/runtime_tables/flow_val_rel.csv`

#### 正式配置文件

- 配置：`train_models/CGTensorProductEquivariantModel/model_parameters.yml`

#### 当前正式训练关键超参数

- `embedding_mode = esm`
- `batch_size = 10`
- `lr = 5e-5`
- `w_decay = 1e-4`
- `n_epochs = 300`
- `scheduler = cosine`
- `use_ema = true`
- `num_dataloader_workers = 0`

#### 当前正式时间采样 / 扰动口径

- `flow.time_sampling = mixed`
- `flow.t_min = 0.2`
- `flow.t_max = 1.0`
- `flow.mix_fixed_t = 0.6`
- `flow.mix_fixed_prob = 0.6`
- `flow.mix_beta_alpha = 8.0`
- `flow.mix_beta_beta = 4.0`
- `flow.mix_beta_min = 0.5`
- `flow.mix_beta_max = 0.9`
- `flow.mix_beta_prob = 0.3`
- `flow.mix_small_min = 0.2`
- `flow.mix_small_max = 0.4`
- `flow.mix_small_prob = 0.1`
- `flow.tr_sampling = gaussian_shell`
- `flow.tr_r_min = 6.0`
- `flow.tr_r_max = 16.0`
- `flow.tr_r_mu = 12.0`
- `flow.tr_r_sigma = 2.0`
- `flow.tr_center_mode = pep_com`
- `flow.tr_min_dist = 1.5`
- `flow.tr_reject_max_tries = 30`
- `flow.sigma_tr_max = 10.0`
- `flow.sigma_rot_max = 3.14`
- `flow.sigma_tor_bb_max = 0.8`
- `flow.sigma_tor_sc_max = 0.5`

#### 当前正式 loss 口径

- `flow.loss_weights.tr = 12.0`
- `flow.loss_weights.rot = 16.0`
- `flow.loss_weights.tor_bb = 0.5`
- `flow.loss_weights.tor_sc = 0.2`
- `flow.loss_weights.clash = 5.0`

#### 这些参数在训练里到底怎么生效

- `embedding_mode / batch_size / lr / w_decay / n_epochs / scheduler / use_ema / num_dataloader_workers`
  - 由 `train_flow.py` 主训练流程和 `utils/utils.py` 真实读取
- `flow.time_sampling / flow.t_min / flow.t_max / flow.mix_* / flow.tr_* / flow.sigma_*`
  - 由 `utils/transform.py` 真实读取，决定训练时噪声时间采样、位移扰动和噪声尺度
- `flow.loss_weights.*`
  - 由 `utils/flow_matching.py` 真实读取，决定各项训练 loss 的加权方式

#### 不在本合同逐条展开、但仍受训练配置文件约束的项

- `model_parameters.yml` 里还有一批**确实会被训练主环读取**的实现细节字段，例如部分旋转目标相关 `flow.*` 项
- 这些字段当前不单独列成“正式口径”，是为了避免把实现细节、废弃字段和必须留档口径混成一锅
- 但只要有人改动这些**实际被训练主环读取**的字段，也必须同步更新本文件留档

#### 明确不算训练主环口径的参数

以下参数**不决定 Flow 训练主环本身**，不要再写进“正式训练超参数”里冒充训练口径：

- `sampling.num_steps_flow`
- `sampling.solver_flow`
- `default_inference_args.yaml` 里的 `N`
- `default_inference_args.yaml` 里的 `batch_size`
- `default_inference_args.yaml` 里的 `flow_num_steps`
- `default_inference_args.yaml` 里的 `flow_solver`
- `default_inference_args.yaml` 里的 `amp`
- `default_inference_args.yaml` 里的 `scoring_function`
- `default_inference_args.yaml` 里的 `prealign_to_native_center`

这些东西的作用是：

- `sampling.num_steps_flow` / `sampling.solver_flow`
  - 只作为 **模型配置里的推理 fallback**
  - 当 `inference.py` 没从 CLI / 推理 YAML 收到显式覆盖时，才会回退读这里
- 它们**不参与训练 loss 主环**
- 训练中只有开启 `train_flow.py --eval_every ... --eval_csv ...` 的**可选小推理评估**时，才可能间接走到推理链

#### 训练最小硬规则

- 训练时必须明确写出：
  - `config = train_models/CGTensorProductEquivariantModel/model_parameters.yml`
  - `train_csv = data/runtime_tables/flow_train_rel.csv`
  - `val_csv = data/runtime_tables/flow_val_rel.csv`
  - 本节列出的当前正式训练关键超参数
- 如果有人把训练 CSV 改成别的、或者把本节列出的训练关键超参数偷偷改了但没留档，默认视为**跑歪**

### 6.2 Flow 正式推理口径

> 下面这一节写的才是 `Flow` 推理时真实控制采样行为的参数。

#### 数据集目录

- 推理资产根目录：`data/processed_test30/`
- 正式推理 CSV：`data/runtime_tables/flow_infer_test536_rel.csv`

#### 正式模型入口

- `model_dir = train_models/CGTensorProductEquivariantModel`
- `ckpt = flowpepdock_best.pt`
- 推理配置：`default_inference_args.yaml`

#### 当前正式推理超参数

- `N = 10`
- `batch_size = 16`
- `flow_num_steps = 10`
- `flow_solver = euler`
- `scoring_function = none`
- `amp = false`
- `prealign_to_native_center = true`

#### `tclip` checkpoint 的推理特例

- 如果 `ckpt = logs/full_abs_tight_tclipped028_ddp1234567_bs10_20260414_153322/flow_esm_best.pt`
  - 当前优先写：`flow_num_steps = 3`
  - 理由不准只写“经验上更好”，必须指向：
    - `results/tclip028_step_sweep_stratified128_g123456_20260424_174749/full536_val300_step01to10_n5_bs32_20260424_final/reports/strict_step1to10_combined_report.md`
    - `results/tclip028_step_sweep_stratified128_g123456_20260424_174749/full536_val300_step01to10_n5_bs32_20260424_final/reports/test536_vs_val300_step_consistency.md`
    - `results/tclip028_n_sweep_step03_stratified96_n5to30_bs160_g123_20260424_213246/reports/n_sweep_summary.md`
- 如果不是 `tclip` checkpoint，就不要把这条 `step=3` 口径乱套到别的 Flow checkpoint 上

#### 这些参数在推理里到底决定什么

- `N`
  - 每个复合物最终采多少个候选 pose
- `batch_size`
  - 每轮送进采样/模型前向的图数量，主要影响推理吞吐和显存
- `flow_num_steps`
  - 真正生效位置在 `inference.py` 和 `utils/sampling.py`
  - 决定 Flow 推理一共做多少次离散时间步更新
  - 每一步时间步长近似为 `dt = (t_max - t_min) / flow_num_steps`
- `flow_solver`
  - 决定采样积分器；当前正式口径是 `euler`
- `scoring_function`
  - 决定是否走额外重排 / 打分分支；当前正式口径是 `none`
- `amp`
  - 决定推理前向是否开启混合精度
- `prealign_to_native_center`
  - 决定是否在采样前把初始肽中心对齐到 native 中心附近

#### `sampling.num_steps_flow = 50` 为什么不能写成训练口径

- 因为它不在 `train_flow.py` 的训练主环里直接控制 loss 或参数更新
- 它只是在推理阶段，当你**没有显式传** `flow_num_steps` 时，才会被当作默认 fallback
- 而当前正式推理口径已经在 `default_inference_args.yaml` 里显式写死：
  - `flow_num_steps = 10`
- 所以当前仓库正式默认推理时，真正生效的是 `10`，不是模型配置里的 `50`

#### 推理最小硬规则

- 推理时必须明确写出：
  - `protein_peptide_csv = data/runtime_tables/flow_infer_test536_rel.csv`
  - `model_dir = train_models/CGTensorProductEquivariantModel`
  - `ckpt = flowpepdock_best.pt`
  - `N = 10`
  - `batch_size = 16`
  - `flow_num_steps = 10`
  - `flow_solver = euler`
  - `scoring_function = none`
  - `amp = false`
- 如果 `ckpt` 换成 `tclip` checkpoint，则必须额外明确：
  - `ckpt = logs/full_abs_tight_tclipped028_ddp1234567_bs10_20260414_153322/flow_esm_best.pt`
  - `flow_num_steps = 3`
  - 以及上面那三个 step-sweep / N-sweep 结果目录
- 如果以后改成别的 `N / batch_size / flow_num_steps / flow_solver / ckpt`，必须留档；不留档默认视为**非正式口径**

## 7. PoseCred-IPG 强制规则

- **任何 IPG 正式训练 / 评估 / 重打分命令**，都必须同时写明：
  - 数据集目录
  - shard index / records index / cross pose table
  - 当前实际生效的数据入口
  - `checkpoint`（仅评估 / 重打分需要）
  - `config_snapshot`（仅 cross pose 重打分需要显式写）
  - 关键超参数
- 不允许只写“沿用默认 IPG”而不把 `groups_per_batch / poses_per_group / clash_penalty_scale / shard snapshot` 说清楚
- 如果后续更改 IPG 的默认 shard snapshot、默认 checkpoint 或正式评估超参数，必须同步更新：
  - `FLOW_DATASET_CONTRACT.md`
  - `README.md`
  - `AGENTS.md`
  - `notes/AI_CONTEXT.md`
  - `docs/POSECRED_IPG_RECOVERY_DATASET_REGISTRY_20260416.md`

## 8. PoseCred-IPG 当前正式运行口径

### 8.1 IPG 正式训练口径

> 下面这一节只写 **IPG 训练主环真实读取** 的数据入口和超参数。

#### 数据集目录

- snapshot 根目录：`posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32`
- `posecred_ipg.train --use_default_shard_snapshot` 实际展开到：
  - `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/train_shards_v1/shard_index.csv`
  - `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/val_shards_v1/shard_index.csv`
- 为了归档备查，仓库还保留了内容等价的固定导出副本：
  - `posecred_ipg/final_exports/default_train_shard_index.csv`
  - `posecred_ipg/final_exports/default_val_shard_index.csv`

#### 正式模型入口

- 正式 checkpoint：`posecred_ipg/final_exports/graph_main_best.pt`

#### 当前正式训练超参数

- `model_name = posecred_ipg`
- `use_default_shard_snapshot = true`
- `epochs = 5`
- `lr = 1e-3`
- `weight_decay = 1e-4`
- `groups_per_batch = 16`
- `poses_per_group = 8`
- `eval_poses_per_group = 0`
- `clash_penalty_scale = 0.0`
- `main_metric = val_top1_success`
- `seed = 20260320`

#### 这些值为什么算“真口径”

- `use_default_shard_snapshot`
  - 会真实展开成：
    - `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/train_shards_v1/shard_index.csv`
    - `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/val_shards_v1/shard_index.csv`
- `epochs / lr / weight_decay / groups_per_batch / poses_per_group / eval_poses_per_group`
  - 都是 `posecred_ipg.train` 训练主环真实读取的参数
- `clash_penalty_scale / main_metric / seed / model_name`
  - 也是 `posecred_ipg.train` 主流程真实读取的参数

#### 明确不要误写进训练口径的项

- `config_snapshot`
  - 不是 `posecred_ipg.train --use_default_shard_snapshot` 的显式训练参数
  - 训练时它是通过 shard index 所在 snapshot 目录自动反查加载的，不该写成“训练命令必须显式传的参数”
- `final_exports/default_train_shard_index.csv` / `default_val_shard_index.csv`
  - 是归档副本，不是 `--use_default_shard_snapshot` 这条训练命令实际展开到的路径
- `prune_strategy / N_pair / loss_weights`
  - 这些属于 snapshot/config 内容留档，不是当前正式训练命令里需要单独显式填写的训练超参数

#### 训练最小硬规则

- IPG 正式训练时必须明确写出：
  - `use_default_shard_snapshot = true`
  - `train_shard = posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/train_shards_v1/shard_index.csv`
  - `val_shard = posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/val_shards_v1/shard_index.csv`
  - `epochs = 5`
  - `lr = 1e-3`
  - `weight_decay = 1e-4`
  - `groups_per_batch = 16`
  - `poses_per_group = 8`
  - `eval_poses_per_group = 0`
  - `clash_penalty_scale = 0.0`
  - `main_metric = val_top1_success`
  - `seed = 20260320`

### 8.2 IPG 正式评估 / 推理 / 重打分口径

#### 默认 checkpoint 评估

- 数据集目录：
  - `posecred_ipg.evaluate_checkpoint --use_default_shard_snapshot` 实际展开到：
    - `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/train_shards_v1/shard_index.csv`
    - `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/val_shards_v1/shard_index.csv`
- checkpoint：
  - `posecred_ipg/final_exports/graph_main_best.pt`
- 超参数：
  - `model_name = posecred_ipg`
  - `use_default_shard_snapshot = true`
  - `groups_per_batch = 32`
  - `poses_per_group = 0`
  - `clash_penalty_scale = 0.0`
  - `seed = 20260320`

#### 这些参数到底决定什么

- `groups_per_batch`
  - 每个 batch 同时装多少个 group
  - 直接影响显存占用和吞吐
- `poses_per_group`
  - 每个 group 抽多少个 pose 参与当前 loader / forward
  - `0` 表示吃该 group 全部 pose
- `clash_penalty_scale`
  - 评估 / 重打分时是否以及多大程度对 clash penalty 做后处理修正
- 这些都是真正会影响 IPG 评估 / 重打分输出的参数

#### cross pose 重打分

- 必须明确写出：
  - `checkpoint`
  - `config_snapshot = posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/config_snapshot.json`
  - `cross_pose_table`
  - `records_index`（若已有）
  - `groups_per_batch`
  - `poses_per_group`
  - `clash_penalty_scale`
- 当前正式推荐超参数：
  - `groups_per_batch = 0`
  - `poses_per_group = 0`
  - `clash_penalty_scale = 0.0`

#### `groups_per_batch = 0` 在 cross pose 重打分里到底是什么意思

- 它不是“不用这个参数”
- `posecred_ipg/evaluation/score_cross_pose_table.py` 会把 `groups_per_batch <= 0` 解析成：
  - 当前这张 `cross_pose_table` 里一共有多少个 group，就一次性按这个数量展开
- 所以当前正式重打分口径里：
  - `groups_per_batch = 0`
  - 实际含义是“自动吃完整张当前表的 group 数”
- 这和评估脚本里的 `groups_per_batch = 32` 不是一回事，不能混写

#### IPG 评估最小硬规则

- 如果只是默认正式评估，必须写：
  - `checkpoint = posecred_ipg/final_exports/graph_main_best.pt`
  - `train_shard = posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/train_shards_v1/shard_index.csv`
  - `val_shard = posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/val_shards_v1/shard_index.csv`
  - `model_name = posecred_ipg`
  - `groups_per_batch = 32`
  - `poses_per_group = 0`
  - `clash_penalty_scale = 0.0`
- 如果是 rec70 / strict536 / hardneg / recovery 这类特殊数据集，必须额外留档到：
  - `docs/POSECRED_IPG_RECOVERY_DATASET_REGISTRY_20260416.md`

## 9. 给人和 AI 的一句话规则

如果你只记一件事，就记这句：

> `Flow train = data/runtime_tables/flow_train_rel.csv`
> `Flow val = data/runtime_tables/flow_val_rel.csv`
> `Flow test = data/runtime_tables/flow_infer_test536_rel.csv`
>
> `IPG train/eval actual shards = posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/train_shards_v1/shard_index.csv + val_shards_v1/shard_index.csv`
> `IPG archive shard copies = posecred_ipg/final_exports/default_train_shard_index.csv + default_val_shard_index.csv`
> `IPG default ckpt = posecred_ipg/final_exports/graph_main_best.pt`
> `IPG config snapshot = posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/config_snapshot.json`
