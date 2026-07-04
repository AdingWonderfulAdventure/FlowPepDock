# FLOW Result Contract

这份文档定义 `Flow` 结果目录、扫 step 汇总、可视化和结论归口的硬规则。处理任何 `Flow` 的结果分析、benchmark、扫参报告或结论输出前，**必须先读这里**。

## 必读范围

- 任何 `Flow` / `tclip` / `step sweep` 的结果汇总
- 任何 `test536` / `val300` / `536` 相关的推理评测
- 任何需要判断“哪个 step 最优”的结论
- 任何会写入 README、报告或留档的结果文档

## 当前正式总目录

- 正式总目录：`results/tclip028_step_sweep_stratified128_g123456_20260424_174749/full536_val300_step01to10_n5_bs32_20260424_final`
- 该目录是 `full536 + val300` 这套 `step=1..10` 扫描的唯一正式入口
- 该目录下 `reports/` 是当前统一汇总与可视化的唯一结论来源

## 结论口径

- 速度与精度一起综合看，默认推荐 `step=3`
- 如果精度优先、能接受更慢，可以把 `step=7` 作为备选
- `step>=8` 不建议作为默认值

## N sweep 口径

- 当前 N sweep 正式目录：`results/tclip028_n_sweep_step03_stratified96_n5to30_bs160_g123_20260424_213246`
- 该轮固定 `step=3`，在 `stratified96` 小集上扫 `N=5/10/15/20/25/30`
- 默认推荐 `N=25`
- `N=20` 可作为保守折中档
- `N=5/10` 只适合烟测或早期 pilot，不建议作为默认正式口径
- `N=30` 相比 `N=25` 没有形成足够的新增收益，不建议默认上

## 目录边界

- `full536_val300_step568_n5_bs32_20260424_194120`：历史中间目录，只保留 `step=1/2/3/4/5/6/8`
- `full536_val300_step7910_n5_bs32_20260424_182525`：历史分段目录，只保留 `step=7/9/10`
- `full536_val300_step568_n5_bs32_20260424_193838`：废弃中间目录，只保留烟测和输入留档
- 上述历史目录都不得再作为最终结论入口

## 结果写作规则

- 写 README / 报告时，必须明确说明当前正式总目录
- 写 step 结论时，必须同时考虑 `test536` 和 `val300`
- 写“最优 step”时，默认以 `mean_dockq`、`complex_rmsd<=2A`、`median_complex_rmsd`、`peptide_ca_rmsd<=2A`、`pdb->pdb` 时间综合判断
- 不能只引用单一 cohort 或单一指标下结论
