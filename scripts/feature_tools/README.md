# `scripts/feature_tools/`

本目录存放 **onehot / ESM 特征对齐、替换、校验** 相关脚本。

## 包含内容

- `compare_pt_embeddings.py`
  - 比较 onehot PT 与 ESM PT 的前缀是否一致
- `replace_onehot_with_esm.py`
  - 用 ESM embedding 替换已有 PT 的尾部特征
- `validate_esm_pt_pipeline.py`
  - 校验 onehot → ESM 特征流程是否一致

## 使用场景

- 需要验证 ESM 特征流程没有把图结构搞坏
- 需要在已有 onehot PT 基础上构造或替换成 ESM PT

## 备注

- 根目录下的 `scripts/extract_esm_embedding.py` 和 `scripts/build_features_esm_from_onehot.py` 属于主特征构建入口
- 本目录更偏向 **辅助比对 / 校验 / 替换工具**
