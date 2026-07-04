# FlowPepDock 当前目录结构

只记录 **当前主线仍建议直接使用** 的目录与文件。  
历史实验、废弃链路和旧台账见 `legacy/`。

## 根目录
- `train_flow.py`：Flow 训练入口。
- `inference.py`：Flow 推理入口。
- `scoreing.py`：PyRosetta 重打分入口；不属于主训练链，但仍可独立使用。
- `FLOW_DATASET_CONTRACT.md`：Flow / IPG 主线训练、推理、评估数据与关键超参数的硬性约束文档。
- `default_inference_args.yaml`：Flow 推理默认参数；当前正式默认口径为 `ckpt: flowpepdock_best.pt / scoring_function: none / batch_size: 16 / N: 10 / flow_num_steps: 10 / amp: false`。
- `flowpepdock_env.yaml`：环境定义。
- `requirement.txt`：额外 pip 依赖。
- `README.md`：当前主流程说明。
- `trash/`：本地可追溯隔离区；误放文件、缓存和待人工确认是否彻底删除的资产先放这里，不作为运行时输入；少量为保留 Git 级移动历史而迁入的历史软链接也放这里。

## 示例输入
- `examples/csv/inference_smoke_2_cases.csv`：推理 smoke CSV；只引用仓库内置
  `examples/pdb/` 最小 PDB 示例，不依赖本机私有数据目录。
- `examples/pdb/`：GitHub quickstart smoke-test 使用的小型受体/肽 PDB 输入。
- `examples/csv/5_items.csv`：历史示例；当前不再作为推理 smoke 基线。
- `examples/csv/prepare_training_data_example.csv`：`prepare_training_data.py` 输入字段示例。

## 当前主线代码
- `models/`
  - `models/model.py`：模型装配入口。
  - `models/flow_model.py`：Flow 主模型实现。
- `dataset/`
  - 当前 Flow 训练/推理使用的数据特征与图构造逻辑。
- `utils/`
  - `utils/flow_utils.py`：Flow 相关工具。
  - `utils/flow_matching.py`：Flow Matching 训练逻辑。
  - `utils/so3.py`：SO(3) 数值与采样工具。
  - 其他被 `train_flow.py` / `inference.py` 直接引用的通用模块。

## 当前保留脚本

### `scripts/` 根目录：仅保留主链入口
- `scripts/prepare_training_data.py`：从结构文件生成 PT。
- `scripts/extract_esm_embedding.py`：生成 ESM embedding。
- `scripts/build_features_esm_from_onehot.py`：在 onehot PT 基础上构建 ESM 版本 PT。
- `scripts/eval_rmsd_from_preds.py`：从预测结果评估 RMSD / DockQ。

### `scripts/data_prep/`：原始数据准备
- `scripts/data_prep/split_cif_chains.py`：拆 CIF、挑链并裁剪受体。
- `scripts/data_prep/run_prepare_onehot.sh`：批量生成 onehot PT。
- `scripts/data_prep/query_candidate_pdbs.py`：查询候选 PDB。
- `scripts/data_prep/download_pdbs.py`：下载 CIF。
- `scripts/data_prep/pick_chains_from_cif.py`：从 CIF 中挑链。
- `scripts/data_prep/crop_receptor_around_peptide.py`：按肽裁剪受体。

### `scripts/data_qc/`：质量检查
- `scripts/data_qc/inspect_processed.py`：检查处理后的 PDB 目录。
- `scripts/data_qc/inspect_pt.py`：检查 PT 文件完整性与字段。
- `scripts/data_qc/filter_csv_by_existing_processed.py`：按已有 processed 过滤 CSV。
- `scripts/data_qc/filter_pt_available_by_min_ca_dist.py`：按 CA 距离过滤 PT 可用表。
- `scripts/data_qc/gen_missing_pt_csv.py`：生成缺失 PT 清单。
- `scripts/data_qc/quarantine_bad_processed.py`：隔离坏样本。
- `scripts/data_qc/verify_pt_from_report.py`：基于报告回查 PT。

### `scripts/data_split/`：去重与切分
- `scripts/data_split/audit_docking_dataset_leakage.py`：clean test 审计。
- `scripts/data_split/build_clean_test_csvs_from_audit.py`：基于审计结果产出 clean test CSV。
- `scripts/data_split/build_rec70_cross_docking_dataset.py`：按既有 `20x50` 口径扩容 `rec70 test` cross-docking 项目，并导出增量 / 全量 CSV。
- `scripts/data_split/summarize_exact_peptide_rec70_groups.py`：按 peptide / receptor cluster 汇总。
- `scripts/data_split/make_random_split.py`：构造随机切分。

### `scripts/feature_tools/`：特征辅助工具
- `scripts/feature_tools/compare_pt_embeddings.py`：检查 onehot / ESM PT 前缀一致性。
- `scripts/feature_tools/replace_onehot_with_esm.py`：替换 PT 中 embedding 尾部。
- `scripts/feature_tools/validate_esm_pt_pipeline.py`：检查 ESM PT 流程。

### `scripts/postprocess/` / `scripts/benchmark/` / `scripts/ops/`
- `scripts/postprocess/filter_infer_poses.py`：筛选推理 pose。
- `scripts/benchmark/`：cross-docking 打分、统一 benchmark 评测与结果 README 生成、AF3 本地评测、外部模型本地实验编排与局部复核。
- `scripts/benchmark/eval_af3_test536_metrics.py`：AF3 `test536` 最小回传结果 RMSD / DockQ 评测脚本。
- `scripts/ops/note_guard.py`：台账同步守卫。

## 文档
- `docs/QUICKSTART_RUNTIME.md`：当前仓库开箱即跑入口（只保留现役命令）。
- `docs/SHARED_OBJECTS.md`：仓库内 `.so` 二进制扩展的作用、现役使用点与独立性影响。
- `docs/MIGRATION_AND_ARCHIVE.md`：新仓库相对旧仓库的命名、归档与兼容说明。
- `docs/REPO_STATUS.md`：当前仓库总状态、归档边界与外部数据补齐说明。
- `docs/DATA_SYMLINKS.md`：当前从旧仓库接入的数据软链接清单。
- `docs/ACTIVE_FLOW_POSECRED_DATASETS.md`：当前 Flow / PoseCred-IPG 现役资产总表。
- `docs/POSECRED_IPG_RECOVERY_DATASET_REGISTRY_20260416.md`：`2026-04-16` 这轮 PoseCred-IPG hardneg / recovery 流水线的数据集总留档。
- `docs/FILE_STRUCTURE.md`：本文件，只描述当前主线。
- `docs/RELEASE_ASSETS.md`：外部 checkpoint、可选 SO(3) 缓存、目标路径与
  SHA256 校验清单。
- `docs/FLOWPEP_SAME_IO_BENCHMARK_CHANGELOG_20260330.md`：same-io 正式测速口径、结果与提交链摘要说明。
- `docs/thesis/`：论文写作支撑文档（审稿清单、结构笔记、delta 审计、作图参考、实验待办等），不与 `legacy/Thesis/big/` 章节主稿混放。
- `docs/thesis/README.md`：`docs/thesis/` 范围说明。
- `docs/project_overview.md`：项目总览。
- `docs/training_onehot_to_esm.md`：onehot → ESM 训练说明。
- `docs/csv_pipeline.md`：CSV 管线说明。
- `docs/MODEL_BLUEPRINT.md`：模型蓝图。
- `docs/pt_generation_notes.md`：PT 生成补充说明。

## 正式结果目录
- `results/infer/FlowPep_Strict_536`
  - strict `536` 的当前正式默认结果目录
  - 对应当前正式默认 `flow_num_steps=10`
- `results/infer/RAPi_Strict_536`
  - strict `536` 的 `RAPiDock_ori` 基线目录
  - 当前已补齐到 `536/536`
- `results/infer/FLOWPEP_VS_RAPI_STRICT_536.md`
  - `FlowPep` vs `RAPi` 的 strict `536` 主线整理稿
- `results/infer/FLOWPEP_VS_RAPI_SPEED_SAME_IO_FINAL_20260329.md`
  - `FlowPepDock` vs `RAPiDock_ori` 的 same-io 速度冻结主表
  - 统一口径：`Flow step=10`、`RAPiDock_ori step=16`、`N=10`、单卡总 CPU=`2`
  - 当前主结论：`FlowPepDock` 在 `K=1/2` 等价设置下分别快 `1.31x / 1.35x`
- `results/infer/FlowPep_Strict_536_step_9`
  - strict `536` 的 `step=9` 正式复跑目录
- `results/infer/FlowPep_Strict_536_step_11`
  - strict `536` 的 `step=11` 正式复跑目录
- `results/infer/speedtest/`
  - 正式结果的测速、评测、启动日志与三方对比产物目录
  - `FlowPep` 当前正式对比文件：
    - `results/infer/speedtest/FlowPep_Strict_536_step_11/step9_step10_step11_comparison.md`
    - `results/infer/speedtest/FlowPep_Strict_536_step10_same_io_20260329/FINAL_SPEED_FREEZE_20260329.md`
    - `results/infer/speedtest/RAPiDock_ori_step16_n10/SPEED_SUMMARY_20260329.md`
- `results/infer/selection_rules/`
  - strict `536` 下不同 pose 选择规则的评测留档
  - 当前重点文件：
    - `results/infer/selection_rules/FLOWPEP_RAPI_TOP1_VS_ORACLE_STRICT_536.md`

## 当前正式结论
- strict `536` 上，`flow_num_steps` 当前正式排序为：**`step10 > step11 > step9`**
- 因此默认正式入口应使用 `results/infer/FlowPep_Strict_536`
- `FlowPep` 与 `RAPi` 的当前主线对照不要写成单指标全面领先：
  - `FlowPep` 强在 RMSD / hit rate
  - `RAPi` 的 `DockQ` 均值 / 中位数略高
- 若讨论生成阶段效率，应优先引用 same-io 速度冻结结果：
  - `results/infer/FLOWPEP_VS_RAPI_SPEED_SAME_IO_FINAL_20260329.md`
  - 不要把带 `ref2015`、多卡混跑或不同 CPU 预算的旧测速口径混进正式结论
- 当前常引用主表更接近 `oracle by complex RMSD` 上界；
  - 更接近部署态的 `top1` 口径留档见：
    - `results/infer/selection_rules/FLOWPEP_RAPI_TOP1_VS_ORACLE_STRICT_536.md`

## 台账
- `notes/AI_CONTEXT.md`：新对话入口。
- `notes/INDEX.md`：台账索引。
- `notes/STATE.md`：当前状态。
- `notes/RECENT.md`：最近摘要。
- `notes/MAIN.md`：当前主线已完成项目。
- `notes/ARCHIVE.md`：当前主线总档案。
- `notes/CODEMAP.md`：代码索引。
- `notes/ISSUES.md` / `notes/ISSUES_RESOLVED.md`：问题台账。
- `notes/RUNS.md`：完整运行记录。
- `notes/LOSS_METRICS.md`：当前仍可复用的训练指标说明。

## PoseCred-IPG
- `posecred_ipg/`：当前保留的 PoseCred-IPG 主目录。
- `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/`：默认 snapshot。
- `posecred_ipg/final_exports/`：固定导出入口。
- `posecred_ipg/legacy/`：已经确认退出当前正式主线的 IPG 实验脚本。

## 模型与数据资产
- `train_models/CGTensorProductEquivariantModel/model_parameters.yml`：Flow 配置。
- `train_models/CGTensorProductEquivariantModel/README.md`：默认 checkpoint
  放置和校验说明。
- `train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt`：当前默认
  Flow checkpoint；不进 Git，作为外部 release asset 发布。
- `data/`：当前主线数据入口；现役大数据已通过软链接接入，见 `docs/DATA_SYMLINKS.md`。
- `data/diagnostics/`：当前 PoseCred-IPG 仍在使用的 docking pose / test-groups 评测资产入口。
- `data/runtime_tables/`：当前仓库生成的相对路径版主 CSV（Flow / PoseCred-IPG）。

## 归档区
- `legacy/`：历史脚本、旧文档、废弃链路与旧项目台账。
- `legacy/utils/`：已归档的历史 `.so` 二进制扩展。
- `legacy/scripts/ranker_scorehead/`：已废弃的 `ranker/scorehead` 相关脚本。
- `legacy/scripts/extra_models/`：已废弃的非 Flow / 非 PoseCred-IPG 辅助模型脚本（旧 confidence、受体排序构表等）。
- `legacy/notes/`：已归档的历史台账。
- `legacy/notes/bak/`：旧版台账备份快照。
- `legacy/notes/debug/`：已归档的临时调试笔记与阶段性排查总结。
- `legacy/docs/`：已归档的历史结构索引与旧任务文档。
- `legacy/Thesis/`：已归档的论文主稿目录与旧链路写作材料；默认不作为当前主线工作目录。
