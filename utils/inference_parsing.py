from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
    FileType,
    RawTextHelpFormatter,
)


class _InferenceHelpFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    """给 inference --help 用的格式器：保留换行并显示默认值。"""


def _add_input_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("输入与输出")
    group.add_argument(
        "--config",
        type=FileType(mode="r"),
        default=None,
        help="YAML 运行配置。只会回填 CLI 未显式设置的字段；CLI 显式传值优先。",
    )
    group.add_argument(
        "--protein_peptide_csv",
        type=str,
        default=None,
        help=(
            "推荐入口：批量输入 CSV。\n"
            "常用列：complex_name, protein_description/receptor_pdb, peptide_description/peptide_pdb, receptor_pt。"
        ),
    )
    group.add_argument(
        "--complex_name",
        type=str,
        default=None,
        help="单样本模式下的复合物名称；使用 --protein_peptide_csv 时通常忽略。",
    )
    group.add_argument(
        "--protein_description",
        type=str,
        default=None,
        help=(
            "单样本模式下的受体输入。\n"
            "支持：受体 .pdb、预制受体 .pt，或受体氨基酸序列（会调用 ESMFold 补结构）。"
        ),
    )
    group.add_argument(
        "--peptide_description",
        type=str,
        default=None,
        help=(
            "单样本模式下的肽输入。\n"
            "当前只支持现成 peptide.pdb 路径；旧版“直接给肽序列”入口已废弃。"
        ),
    )
    group.add_argument(
        "--output_dir",
        type=str,
        default="outputs/default_result",
        help="结果输出目录。",
    )
    group.add_argument(
        "--save_visualisation",
        action="store_true",
        default=False,
        help="保存 reverse process 轨迹 PDB（每个 pose 额外输出 *_reverseprocess.pdb）。",
    )


def _add_model_and_scoring_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("模型与打分")
    group.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="主模型目录；其中应包含 model_parameters.yml 与 checkpoint。",
    )
    group.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="主模型 checkpoint 文件名；相对于 --model_dir 解析。",
    )
    group.add_argument(
        "--scoring_function",
        type=str,
        choices=["none", "confidence", "ref2015"],
        default="none",
        help="采样完成后的重排/打分方式。",
    )
    group.add_argument(
        "--fastrelax",
        action="store_true",
        default=False,
        help="在 ref2015 路线中额外执行 FastRelax。",
    )
    group.add_argument(
        "--confidence_model_dir",
        type=str,
        default=None,
        help="confidence 打分模型目录；仅在 --scoring_function confidence 时使用。",
    )
    group.add_argument(
        "--confidence_ckpt",
        type=str,
        default=None,
        help="confidence 打分模型 checkpoint；仅在 --scoring_function confidence 时使用。",
    )


def _add_sampling_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("采样主参数")
    group.add_argument(
        "--N",
        type=int,
        default=5,
        help="每个复合物生成多少个 pose。",
    )
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="每批复合物数量；一次进入 GPU 的 pose 数大约为 batch_size × N。",
    )
    group.add_argument(
        "--flow_num_steps",
        type=int,
        default=None,
        help="覆盖采样步数；未传时沿用 model_parameters.yml / ckpt config。",
    )
    group.add_argument(
        "--flow_solver",
        type=str,
        choices=["euler", "heun"],
        default=None,
        help="覆盖 flow solver；未传时沿用模型配置。",
    )


def _add_flow_override_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("Flow 推理覆盖参数")
    group.add_argument(
        "--flow_self_condition_infer",
        action=BooleanOptionalAction,
        default=None,
        help="保留实验项：仅在 light-enhancement 组合验证过，未见稳定净收益；默认关闭。",
    )
    group.add_argument(
        "--flow_sparse_interface",
        action=BooleanOptionalAction,
        default=None,
        help="保留实验项：仅在 light-enhancement 组合验证过，未见稳定净收益；默认关闭。",
    )
    group.add_argument(
        "--flow_sparse_interface_topk",
        type=int,
        default=None,
        help="保留实验项：关联 sparse_interface；当前仅用于复现实验，不建议直接切主线。",
    )
    group.add_argument(
        "--flow_final_refine",
        action=BooleanOptionalAction,
        default=None,
        help="保留实验项：历史 light-enhancement 组合未证明稳定净收益，默认关闭。",
    )
    group.add_argument(
        "--flow_final_refine_scale",
        type=float,
        default=None,
        help="保留实验项：关联 final_refine；当前仅用于复现实验，不建议直接切主线。",
    )
    group.add_argument(
        "--flow_final_refine_tr_scale",
        type=float,
        default=None,
        help="保留实验项：关联 final_refine；当前仅用于复现实验，不建议直接切主线。",
    )
    group.add_argument(
        "--flow_final_refine_rot_scale",
        type=float,
        default=None,
        help="保留实验项：关联 final_refine；当前仅用于复现实验，不建议直接切主线。",
    )
    group.add_argument(
        "--flow_final_refine_tor_scale",
        type=float,
        default=None,
        help="保留实验项：关联 final_refine；当前仅用于复现实验，不建议直接切主线。",
    )
    group.add_argument(
        "--flow_steric_guidance",
        action=BooleanOptionalAction,
        default=None,
        help="保留实验项：已有实现且当前配置可能默认开启，但历史 A/B 收益不稳定，正文需单独报告。",
    )
    group.add_argument(
        "--flow_steric_guidance_scale",
        type=float,
        default=None,
        help="steric guidance 全局强度缩放；未传时沿用模型配置。",
    )
    group.add_argument(
        "--flow_steric_guidance_cutoff",
        type=float,
        default=None,
        help="steric guidance 排斥感知距离阈值（Å）；未传时沿用模型配置。",
    )
    group.add_argument(
        "--flow_steric_guidance_temperature",
        type=float,
        default=None,
        help="steric guidance 软阈值温度；未传时沿用模型配置。",
    )
    group.add_argument(
        "--flow_steric_guidance_torque_scale",
        type=float,
        default=None,
        help="steric guidance 中旋转力矩缩放；未传时沿用模型配置。",
    )
    group.add_argument(
        "--flow_steric_guidance_max_tr",
        type=float,
        default=None,
        help="单步 steric translation guidance 最大范数；未传时沿用模型配置。",
    )
    group.add_argument(
        "--flow_steric_guidance_max_rot",
        type=float,
        default=None,
        help="单步 steric rotation guidance 最大范数；未传时沿用模型配置。",
    )
    group.add_argument(
        "--flow_hard_overlap_guard",
        action=BooleanOptionalAction,
        default=None,
        help="保留实验项：尚缺系统性 A/B 验证，且当前仅 Euler 路径生效；默认关闭。",
    )
    group.add_argument(
        "--flow_hard_overlap_guard_min_dist",
        type=float,
        default=None,
        help="保留实验项：关联 hard-overlap guard；当前仅用于未定论实验复现。",
    )
    group.add_argument(
        "--flow_hard_overlap_guard_backoff",
        type=float,
        default=None,
        help="保留实验项：关联 hard-overlap guard；当前仅用于未定论实验复现。",
    )
    group.add_argument(
        "--flow_hard_overlap_guard_max_backtracks",
        type=int,
        default=None,
        help="保留实验项：关联 hard-overlap guard；当前仅用于未定论实验复现。",
    )
    group.add_argument(
        "--flow_hard_overlap_guard_last_steps",
        type=int,
        default=None,
        help="保留实验项：关联 hard-overlap guard；当前仅用于未定论实验复现。",
    )
    group.add_argument(
        "--flow_tr_direction_mode",
        type=str,
        choices=["isotropic", "local_hemisphere"],
        default=None,
        help=(
            "初始化 shell 的方向分布；"
            "isotropic=全空间随机，local_hemisphere=只保留朝向受体局部一侧的半球。"
        ),
    )
    group.add_argument(
        "--flow_tr_direction_local_k",
        type=int,
        default=None,
        help="local_hemisphere 模式下，用于估计局部受体方向的最近 receptor 原子数。",
    )


def _add_runtime_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("运行设备与性能")
    group.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="推理设备选择：auto=有 CUDA 则 GPU，否则 CPU。",
    )
    group.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "自动多卡入口：逗号分隔的 GPU 编号列表，例如 1,2,3,4。\n"
            "提供后会自动稳定切分 CSV 并启动对应数量的子进程。"
        ),
    )
    group.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="高级/内部参数：手工指定稳定分片总数；通常由 --gpus 自动设置。",
    )
    group.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="高级/内部参数：当前运行第几片（从 1 开始）；通常由 --gpus 自动设置。",
    )
    group.add_argument(
        "--cpu",
        type=int,
        default=5,
        help="CPU 预算，用于预检查、多进程和 DataLoader worker 自动推断。",
    )
    group.add_argument(
        "--loader_workers",
        type=int,
        default=None,
        help="前处理 DataLoader worker 数；为空时按 cpu / batch_size 自动推断。",
    )
    group.add_argument(
        "--loader_prefetch_factor",
        type=int,
        default=2,
        help="每个 DataLoader worker 预取批次数；仅在 loader_workers > 0 时生效。",
    )
    group.add_argument(
        "--loader_persistent_workers",
        action=BooleanOptionalAction,
        default=True,
        help="启用持久化 DataLoader workers，减少重复拉起进程的开销。",
    )
    group.add_argument(
        "--eager_graph_loading",
        action=BooleanOptionalAction,
        default=False,
        help="实验开关：先在主进程物化全部 graph；当前默认关闭，只有显式传入才启用。",
    )
    group.add_argument(
        "--eager_graph_max_items",
        type=int,
        default=64,
        help="保留给 eager graph loading 实验的阈值参数；当前默认链不会自动启用 eager。",
    )
    group.add_argument(
        "--amp",
        action=BooleanOptionalAction,
        default=False,
        help="推理阶段启用 AMP（默认关闭）。",
    )
    group.add_argument(
        "--timing",
        action=BooleanOptionalAction,
        default=False,
        help="打印分阶段耗时统计。",
    )
    group.add_argument(
        "--timing_output",
        type=str,
        default=None,
        help="把机器可读的耗时汇总写到 JSON 文件；适合后续横向对比。",
    )
    group.add_argument(
        "--timing_force_cuda_sync",
        action=BooleanOptionalAction,
        default=False,
        help="计时时强制在 CUDA 前向前后同步；默认关闭以避免额外阻塞，分步耗时会更近似。",
    )
    group.add_argument(
        "--gpu_update_fastpath",
        action=BooleanOptionalAction,
        default=None,
        help="尽量把采样更新留在 GPU 上；未传时只要是 CUDA 就自动开启。",
    )
    group.add_argument(
        "--torsion_device",
        type=str,
        choices=["cpu", "gpu"],
        default=None,
        help="torsion 更新所在设备；未传时按 fastpath 自动决策（CUDA 默认走 gpu）。",
    )
    group.add_argument(
        "--cache_peptide_graph",
        action=BooleanOptionalAction,
        default=True,
        help="相同 peptide_description 复用肽图与 torsion mask，减少重复预处理。",
    )
    group.add_argument(
        "--peptide_esm_path",
        type=str,
        default=None,
        help="给所有样本复用同一个 peptide ESM embedding 文件（torch.load 可读）。",
    )
    group.add_argument(
        "--receptor_cache_dir",
        type=str,
        default=None,
        help="受体图缓存查找目录；传入后优先尝试从该目录复用受体图缓存。",
    )
    group.add_argument(
        "--graph_cache_dir",
        type=str,
        default=None,
        help="完整复合物图缓存查找目录；传入后可直接加载预制好的样本图。",
    )
    group.add_argument(
        "--save_receptor_cache_dir",
        type=str,
        default=None,
        help="显式指定受体图缓存落盘目录；未传则绝不保存新缓存。",
    )
    group.add_argument(
        "--save_graph_cache_dir",
        type=str,
        default=None,
        help="显式指定完整复合物图缓存落盘目录；未传则绝不保存新缓存。",
    )
    group.add_argument(
        "--receptor_build_workers",
        type=int,
        default=None,
        help="受体缓存 miss 的预构建 worker 数；为空时按 cpu 预算保守自动推断。",
    )
    group.add_argument(
        "--cpu_math_threads",
        type=int,
        default=1,
        help="限制单进程内 Torch/BLAS 线程数，避免 GPU 推理时 CPU 过度抢核。",
    )


def _add_eval_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("自动评测")
    group.add_argument(
        "--auto_eval_metrics",
        action=BooleanOptionalAction,
        default=False,
        help="推理结束后自动计算 RMSD / DockQ。",
    )
    group.add_argument(
        "--auto_eval_csv",
        type=str,
        default=None,
        help="自动评测使用的 CSV；为空时回退到 --protein_peptide_csv。",
    )
    group.add_argument(
        "--auto_eval_output",
        type=str,
        default="metrics_rmsd_dockq.csv",
        help="自动评测输出文件名或绝对路径。",
    )
    group.add_argument(
        "--auto_eval_dockq_cmd",
        type=str,
        default=None,
        help="DockQ 可执行文件路径；为空时自动探测。",
    )


def _add_debug_and_compat_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("调试、兼容与已废弃语义")
    group.add_argument(
        "--prealign_to_native_center",
        action=BooleanOptionalAction,
        default=True,
        help="推理前默认把肽中心平移到 receptor.pdb 同目录 peptide.pdb 的中心。",
    )
    group.add_argument(
        "--rot_oracle",
        action="store_true",
        default=False,
        help="调试用：以 x_t->x0 的刚体旋转替换模型 rot 更新，只用于定位瓶颈。",
    )
    group.add_argument(
        "--torsion_debug",
        action="store_true",
        default=False,
        help="调试 torsion CPU/GPU 更新的一致性。",
    )
    group.add_argument(
        "--peptide_source",
        type=str,
        choices=["pdb"],
        default="pdb",
        help=(
            "[兼容保留/已废弃] 旧版肽来源参数。\n"
            "当前仓库只接受现成 peptide.pdb；该参数仅保留为兼容入口，固定只能是 pdb。"
        ),
    )


def get_parser():
    """构建推理阶段命令行解析器，把所有入口参数集中在一起。"""
    description = (
        "FlowPepDock inference 入口。\n"
        "--help 展示的是推理脚本可直接控制的 CLI 参数；\n"
        "模型结构/训练超参数来自 --model_dir/model_parameters.yml 与 checkpoint 内 config。"
    )
    epilog = (
        "常用示例：\n"
        "  单卡：python inference.py --config default_inference_args.yaml "
        "--protein_peptide_csv data/runtime_tables/flow_infer_test536_rel.csv "
        "--output_dir results/infer/demo --N 10 --batch_size 16\n"
        "  多卡：python inference.py --config default_inference_args.yaml "
        "--protein_peptide_csv data/runtime_tables/flow_infer_test536_rel.csv "
        "--output_dir results/infer/demo --gpus 1,2,3,4,5,6,7 --N 10 --batch_size 16\n"
        "  受体缓存：python inference.py --config default_inference_args.yaml "
        "--protein_peptide_csv data/runtime_tables/flow_infer_test536_rel.csv "
        "--output_dir results/infer/cache_build --receptor_cache_dir /tmp/flow_rec_cache "
        "--save_receptor_cache_dir /tmp/flow_rec_cache --receptor_build_workers 4\n\n"
        "兼容/废弃说明：\n"
        "  1. --peptide_source 仍会显示，但当前只接受 pdb；传其他值会直接报错。\n"
        "  2. --peptide_description 的旧版“直接给肽序列”语义已废弃；现在必须提供 peptide.pdb。\n"
        "  3. --num_shards / --shard_index 仍在用，但主要是多卡 launcher 的高级/内部参数。"
    )
    parser = ArgumentParser(
        prog="inference.py",
        description=description,
        epilog=epilog,
        formatter_class=_InferenceHelpFormatter,
    )

    _add_input_args(parser)
    _add_model_and_scoring_args(parser)
    _add_sampling_args(parser)
    _add_flow_override_args(parser)
    _add_runtime_args(parser)
    _add_eval_args(parser)
    _add_debug_and_compat_args(parser)
    return parser


def infer_explicit_cli_dests(parser: ArgumentParser, argv=None) -> set[str]:
    """找出用户在 CLI 里显式传过的参数 dest，避免被 config 回填覆盖。"""
    argv = list(argv or [])
    option_to_dest = {}
    for action in parser._actions:
        for option in getattr(action, "option_strings", []):
            option_to_dest[option] = action.dest

    explicit = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit.add(dest)
    return explicit
