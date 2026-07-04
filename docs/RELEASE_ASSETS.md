# Release Assets

> 文档类型：GitHub 发布资产清单

本仓库只跟踪代码、配置、文档和最小 smoke-test 示例。大型 checkpoint、
正式数据集、运行缓存和生成结构文件不进 Git，必须通过 release assets 或外部
artifact store 单独下载。

## 必需资产

推荐把必需 checkpoint 打成一个外部资产包：

```text
FlowPepDock_external_assets.tar.gz
```

该包应从仓库根目录解压，内部路径必须保持为下表的目标路径。当前下载入口在正式
GitHub Release、Zenodo、Hugging Face 或其他 artifact store 发布后补齐；不要把
这些 `.pt` 文件作为普通 Git 文件提交。

| 用途 | 目标路径 | 本机参考大小 | SHA256 |
| --- | --- | ---: | --- |
| Flow 默认推理 checkpoint | `train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt` | 91,757,338 bytes | `d7dfdb0e5189d498c1fa2845924fee62a1703faf0a47953506909e713db0ebfd` |
| PoseCred-IPG 默认 checkpoint | `posecred_ipg/final_exports/graph_main_best.pt` | 2,039,230 bytes | `e70211f6f31990cf89712fe33ac2b509f974ad66174874767520bed69177028c` |

## 可选预计算缓存

首次运行 FlowPepDock 时可能会生成 SO(3) 缓存。低配 CPU 或慢盘环境下这一步
可能超过 300 秒。若发布包提供缓存，可放在仓库根目录：

| 文件 | 本机参考大小 | SHA256 |
| --- | ---: | --- |
| `.so3_omegas_array2.npy` | 16,128 bytes | `d3381e706474f0f0ce8c6d90bc6cb6931f14853792b1a718a01141a01d58fbe7` |
| `.so3_cdf_vals2.npy` | 16,000,128 bytes | `bcfc6e4edffcb2158e633099482e215dc20c716bc4eda3d79e5d782d315da783` |
| `.so3_score_norms2.npy` | 16,000,128 bytes | `0bb8f11d4ee0e9b2d5bbe546675877784dd17ac7fb1b9ecec7529218abd66bb9` |
| `.so3_exp_score_norms2.npy` | 8,128 bytes | `8e0b26983bd68deddede749beae97ce33e394d3a12947fbabe0086bce0c81621` |

这些缓存不是模型权重；缺失时程序会尝试自动生成。

## 下载后校验

如果使用推荐资产包，从仓库根目录运行：

```bash
tar -xzf FlowPepDock_external_assets.tar.gz -C .
sha256sum -c SHA256SUMS.txt
```

也可以直接校验两个目标文件：

```bash
sha256sum train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt
sha256sum posecred_ipg/final_exports/graph_main_best.pt
```

输出应与上表一致。若哈希不一致，不要继续跑正式 benchmark，先重新下载。

## GitHub smoke-test 输入

`examples/csv/inference_smoke_2_cases.csv` 只引用仓库内置最小 PDB 示例：

```text
examples/pdb/7bbg/receptor.pdb
examples/pdb/7bbg/peptide.pdb
examples/pdb/7al2/receptor.pdb
examples/pdb/7al2/peptide.pdb
```

这些文件必须随 GitHub 源码一起发布。不要把 smoke CSV 指到
`data/rebuild_isolated/`、`data/processed_test30/` 或其他本机私有数据目录。

为兼容旧版 smoke CSV 或外部文档里仍保留的 `2kid / 2rui` 路径，源码包也应带上
以下三份小型 PDB：

```text
data/rebuild_isolated/rebuild_20251221_163301/processed/2kid/receptor.pdb
data/rebuild_isolated/rebuild_20251221_163301/processed/2kid/peptide.pdb
data/rebuild_isolated/rebuild_20251221_163301/processed/2rui/receptor.pdb
```

这三份文件只是 smoke-test 兼容输入，不代表正式 `data/rebuild_isolated/`
数据集要整体进 Git。
