# Final Exports

> 文档类型：当前代码说明
>
> 本文只描述当前 `PoseCred-IPG v1` 的固定出口文件，不引用历史运行目录。

这个目录提供当前 `PoseCred-IPG v1` 的固定出口，不需要再去 `tmp/` 或长路径里找文件。

## 文件说明

- [graph_main_best.pt](graph_main_best.pt)
  - 当前最佳 checkpoint
  - 这是大型二进制发布资产，Git 默认不跟踪；若源码包中没有该文件，请按
    `../../docs/RELEASE_ASSETS.md` 下载并校验后放回本目录

- [graph_main_eval_report.json](graph_main_eval_report.json)
  - 当前最佳 checkpoint 的独立评估结果

- [default_train_shard_index.csv](default_train_shard_index.csv)
  - 默认训练 shard index 的归档副本

- [default_val_shard_index.csv](default_val_shard_index.csv)
  - 默认验证 shard index 的归档副本

- [final_results_snapshot.csv](final_results_snapshot.csv)
  - 当前正式结果总表

## 当前口径

- 最佳 checkpoint 继续使用 `graph_main_best.pt`
- `default_train_shard_index.csv` / `default_val_shard_index.csv` 是固定导出的归档副本
- `--use_default_shard_snapshot` 的实际训练/评估入口以 `posecred_ipg/record_snapshots/cocrystal_positive_only_hybrid_npair32_v1/hybrid_npair32/train_shards_v1/shard_index.csv` 与 `.../val_shards_v1/shard_index.csv` 为准
- 当前 snapshot 留档：
  - `prune_strategy = hybrid`
  - `N_pair = 32`
  - `bad head` 已在当前主线代码中废弃并移除
  - `clash_penalty_scale = 0.0`

## 相关文档

- [../QUICKSTART.md](../QUICKSTART.md)
- [../FINAL_ARTIFACTS.md](../FINAL_ARTIFACTS.md)
- [../README.md](../README.md)
