# GitHub Upload Decision Notes

这份清单只回答一个问题：现在这些已修改、未跟踪、论文相关、结果相关文件，到底该不该随代码仓库上传 GitHub。

## 建议纳入代码仓库

这些是公开科研代码仓库的门面和复现说明，建议纳入：

- `.gitignore`
- `.dockerignore`
- `Dockerfile`
- `README.md`
- `docs/INSTALL.md`
- `docs/RELEASE_ASSETS.md`
- `docs/GITHUB_RELEASE_CHECKLIST.md`
- `docs/PAPER_REPOSITORY_ALIGNMENT.md`
- `docs/GITHUB_UPLOAD_DECISION.md`
- `CONTRIBUTING.md`
- `CITATION.cff`
- `NOTICE.md`
- `FLOW_DATASET_CONTRACT.md`
- `FLOW_RESULT_CONTRACT.md`
- `default_inference_args.yaml`
- `flowpepdock_env.yaml`
- `requirement.txt`

## 当前既有代码改动取舍

本轮 `git diff` 里既有主线代码改动，也有文档/发布配置改动。建议按下面口径处理：

### 建议保留的代码改动

- `scripts/benchmark/score_cross_graphpep.py`
  - 唯一有实际行为变化的既有脚本。
  - 新增 GraphPep 批内分片打分和 CUDA OOM 自动降片逻辑，能避免大批 pose 一次性送入 GPU 时直接失败。
  - 属于论文 cross-docking/GraphPep 对照复现链路，建议保留。
- `scripts/README.md`
  - 新增 `scripts/figures/` 目录说明，和新增论文出图脚本配套。
  - 建议保留。
- `posecred_ipg/README.md`
- `posecred_ipg/docs/00_INDEX.md`
  - 新增 `11_SCORE_SEMANTICS.md` 索引，属于 PoseCred-IPG 文档补齐。
  - 建议保留。
- `.gitignore`
- `.dockerignore`
- `Dockerfile`
- `README.md`
- `RESULT.md`
- `result.md`
- `docs/ACTIVE_FLOW_POSECRED_DATASETS.md`
  - 属于 GitHub 发布整理、小论文口径对齐、Docker 环境修正。
  - 建议保留。

### 可以保留，但需要确认署名/版权口径的改动

- `LICENSE`
  - 当前从 `Copyright (c) 2024 huifengzhao` 改成了 `Copyright (c) 2026 FlowPepDock contributors`。
  - 如果原作者/团队同意以项目贡献者名义发布，可以保留；如果要严谨保守，建议改成 `Copyright (c) 2024-2026 FlowPepDock contributors`
    或保留原作者并补充 contributors。

### 建议复核后再决定的纯清理改动

以下文件当前改动只是删除旧文件头注释，没有发现代码行为变化：

- `dataset/peptide_feature.py`
- `dataset/protein_feature.py`
- `inference.py`
- `models/flow_model.py`
- `models/model.py`
- `utils/dataset_utils.py`
- `utils/flow_utils.py`
- `utils/inference_parsing.py`
- `utils/inference_utils.py`
- `utils/peptide_updater.py`
- `utils/pyrosetta_utils.py`
- `utils/sampling.py`
- `utils/transform.py`
- `utils/utils.py`

这些可以随公开仓库一起保留，用来清理私人邮箱/旧时间戳；如果需要保留历史作者信息，则不要通过这种方式删除，而是在 `NOTICE.md` 或 README attribution 中统一说明。

主线代码目录应纳入：

- `inference.py`
- `train_flow.py`
- `scoreing.py`
- `models/`
- `dataset/`
- `utils/`
- `posecred_ipg/`
- `scripts/` 中通用数据准备、评估和后处理脚本
- `examples/csv/`
- `examples/pdb/`
- `train_models/CGTensorProductEquivariantModel/model_parameters.yml`
- `train_models/CGTensorProductEquivariantModel/README.md`

PoseCred-IPG 当前分层源码目录也必须纳入，否则旧兼容壳子会出现
`ModuleNotFoundError: No module named 'posecred_ipg.core'` 这类发布包缺源码
错误：

- `posecred_ipg/core/`
- `posecred_ipg/data/`
- `posecred_ipg/engine/`
- `posecred_ipg/evaluation/`
- `posecred_ipg/experiments/`
- `posecred_ipg/models/`
- `posecred_ipg/pipelines/`

## 建议作为可选论文复现脚本纳入

这些未跟踪脚本属于“小论文/论文图表复现”而不是模型主链。可以纳入，但要在 README 或脚本注释里写清楚：运行它们需要 `results/`、`thesis_result/` 或外部补充材料。

### 建议优先纳入的通用/轻量脚本

- `scripts/figures/plot_pose_rank.py`
  - 通用 pose rank-vs-score 曲线脚本，参数化程度较高，可复用到新 CSV/TSV。
  - 建议纳入，并在 `scripts/README.md` 里作为推荐出图入口。
- `scripts/figures/render_small_paper_graphical_abstract_vector.py`
  - 直接生成 Illustrator 可编辑 SVG，不依赖私有 docx/AI 源文件。
  - 建议纳入。

### 可纳入，但属于论文 artifact 脚本

- `scripts/figures/export_figure2_illustrator_clean_svg.py`
- `scripts/figures/plot_figure2_panel_c_efficiency_variants.py`
- `scripts/figures/plot_figure2_runtime_profile_draft.py`
- `scripts/figures/plot_figure2_runtime_profile_merged_cd.py`
- `scripts/figures/plot_figure4c_posecred_ipg_rvt.py`
- `scripts/figures/render_ai_cropped_vector_abstract.py`
- `scripts/figures/render_flow_ipg_graphical_abstract.py`
- `scripts/figures/render_small_paper_docx_graphical_abstract.py`
- `scripts/benchmark/redraw_figure3_heatmaps.py`
- `scripts/benchmark/redraw_thesis_crossdock_heatmaps.py`

这些脚本语法检查通过，但硬编码了 `results/`、`thesis_result/`、`lunwen/`、`paper_extra_figures/` 或 `legacy/` 下的论文中间产物。若上传 GitHub，建议把它们标注为 `paper artifact scripts`，不要放进主 README 的 quickstart。

如果 GitHub 仓库要保持“纯代码 + 最小示例”，这些脚本可以先不上传，放到论文 supplement 或 artifact archive。

## 建议谨慎纳入

这些脚本有本地路径、外部服务或补救性质，上传前必须确认依赖和说明：

- `scripts/benchmark/rescue_graphpep_errors.py`
  - 用于 GraphPep 失败/缺失分数补救，依赖 `/root/GraphPep/GraphPep_v1.1`、GPU 空闲检测和本地布局。
  - 不适合作为普通用户入口。若上传，建议放在 `scripts/benchmark/experimental/` 并写清楚外部依赖。
- `scripts/data_prep/query_rcsb_2026plus_contact_peptides.py`
  - 会访问 RCSB 网络接口并下载 mmCIF。
  - 可作为数据准备工具上传，但 README 要说明网络访问、缓存目录和结果只作为候选筛选，不是正式 runtime CSV 替代品。

## 不建议纳入普通 GitHub 仓库

这些内容建议继续由 `.gitignore` 拦住，或作为 DOI/Zenodo/Release asset 单独发布：

- `logs/`
- `results/`
- `thesis_result/`
- `lunwen/`
- `paper_extra_figures/`
- `dabian/`
- `manual_review_*`
- `strict536_case_bundle_*`
- `*.pt`
- `*.pth`
- `*.ckpt`
- `*.npy`
- `*.npz`
- `*.pdb`
- `*.cif`
- `*.docx`
- `*.pptx`
- `*.pdf`
- `*.tif`
- `.tmp_*`

例外：`examples/pdb/` 下的最小 smoke-test PDB 应随源码发布；它们是
README quickstart 的输入，不是正式数据集或生成结果。
另外，为兼容旧版 smoke CSV，以下三份小型 PDB 也应随源码发布：

- `data/rebuild_isolated/rebuild_20251221_163301/processed/2kid/receptor.pdb`
- `data/rebuild_isolated/rebuild_20251221_163301/processed/2kid/peptide.pdb`
- `data/rebuild_isolated/rebuild_20251221_163301/processed/2rui/receptor.pdb`

不要因此放开整个 `data/rebuild_isolated/`，否则又会把本机大数据集塞进 Git。

## 外部发布资产

以下文件不进 Git，但必须作为 release asset 或外部 artifact 明确提供，并按
`docs/RELEASE_ASSETS.md` 放回目标路径后校验 SHA256：

- `train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt`
- `posecred_ipg/final_exports/graph_main_best.pt`
- 可选 SO(3) 首次运行缓存：
  - `.so3_omegas_array2.npy`
  - `.so3_cdf_vals2.npy`
  - `.so3_score_norms2.npy`
  - `.so3_exp_score_norms2.npy`

## 当前已跟踪大文件风险

当前仓库历史里已经有不少大文件被 Git 跟踪，例如：

- `data/runtime_tables/posecred_ipg_train_docked_rel.csv`，约 41.9 MB
- `posecred_ipg/*/records_index.csv`，约 2-11 MB
- `posecred_ipg/*/shard_index.csv`，约 1.7-10 MB
- `docs/presentation/*.pptx`
- `legacy/Thesis/**/*.docx`
- `thesis_result/**/*.png`
- `legacy/utils/*.so`

`.gitignore` 只能阻止新文件继续进来，不能让已跟踪历史文件自动消失。若要做真正干净的公开仓库，建议新建一个 release-clean 分支或干净目录，只拷贝代码、配置、轻量 CSV、文档和必要示例；不要直接把整个历史工作目录原样推上 GitHub。

## 小论文口径注意

当前 `lunwen/小论文.docx` 的主结果已经和 `docs/PAPER_REPOSITORY_ALIGNMENT.md` 对齐：

- Stage I 主效果使用 tclip-step3 full strict-536：
  - Flow `pep<=2A=0.7481`
  - Flow `complex<=2A=0.8209`
  - Flow `DockQ mean=0.5432`
- Stage I 速度使用：
  - fair32：Flow `1.24 s/complex` vs RAPi `3.54 s/complex`
  - full strict-536 segmented：Flow `0.95 s/complex` vs RAPi `1.97 s/complex`
- Stage II held-out 使用：
  - Top-1 `0.4763`
  - Top-5 `0.6852`
  - MRR `0.5655`
  - NDCG `0.9232`
- Stage II cross-docking 使用 sample-04 Table 4 口径：
  - IPG receptor Top-10 `0.55`
  - IPG receptor Top-20% `0.55`
  - IPG pose Top-50 `0.70`
  - IPG 1,000 poses `8.29 s`
  - IPG 10,000 poses `39.95 s`

不要再把历史 `flowpepdock_best.pt` strict-536 的 `0.7257 / 0.8078 / 0.5342` 当成小论文 headline。
