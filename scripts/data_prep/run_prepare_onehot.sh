#!/usr/bin/env bash
set -euo pipefail

# 使用 onehot embedding 批量生成 PT 特征。
# 默认查找仓库内 data/csv_backup/train_all_pt.csv。
# 若当前仓库未附带该 CSV，请显式传入 FLOWPEPDOCK_TRAIN_CSV 或手动运行 prepare_training_data.py。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CSV="${ROOT_DIR}/data/csv_backup/train_all_pt.csv"
TRAIN_CSV="${FLOWPEPDOCK_TRAIN_CSV:-${DEFAULT_CSV}}"

if [ ! -f "${TRAIN_CSV}" ]; then
  echo "训练 CSV 不存在: ${TRAIN_CSV}" >&2
  echo "当前仓库不内置 train_all_pt.csv；请先准备数据后重试。" >&2
  exit 1
fi

PYTHONPATH="${ROOT_DIR}" python "${ROOT_DIR}/scripts/prepare_training_data.py" \
  --csv "${TRAIN_CSV}" \
  --output_dir data/processed \
  --cache_dir preprocess_cache/tmp_cache \
  --embedding onehot \
  --num_workers 32
