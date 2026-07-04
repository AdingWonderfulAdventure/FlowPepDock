# `scripts/data_prep/`

本目录存放 **原始结构整理、拆链、裁剪、批量预处理** 相关脚本。

## 包含内容

- `query_candidate_pdbs.py`
  - 在 RCSB 检索候选复合物
- `download_pdbs.py`
  - 下载原始 CIF/PDB 文件
- `pick_chains_from_cif.py`
  - 从 CIF 中选择受体链 / 肽链
- `auto_chain_picker.py`
  - 自动挑链逻辑
- `split_cif_chains.py`
  - 将 CIF 拆成 `receptor.pdb` / `peptide.pdb`
- `crop_receptor_around_peptide.py`
  - 按肽附近空间范围裁剪受体
- `export_alphafold_inputs.py`
  - 把 FlowPepDock 的 `complex_name,receptor_pdb,peptide_pdb` CSV 复制并导出为 `AlphaFold3 JSON` 与 `AlphaFold-Multimer FASTA`
- `run_prepare_onehot.sh`
  - 批量生成 onehot PT 的辅助 shell 脚本

## 使用场景

- 从原始结构数据构建 FlowPepDock 可训练 / 可推理的 PDB 输入
- 重建 `processed/` 或抽取新的测试集 / 数据集
- 把现有测试集批量转成 AlphaFold 系列可直接消费的输入

## 备注

- 这些脚本属于 **数据准备链路前段**
- 真正把 `receptor_pdb + peptide_pdb` 变成 `features_*.pt` 的主入口仍然是根目录下的 `scripts/prepare_training_data.py`
- `export_alphafold_inputs.py` 默认会从 `receptor_crop.pdb` 同目录自动找到全长 `receptor.pdb`，避免把裁剪口袋误喂给 AlphaFold
