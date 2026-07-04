#!/usr/bin/env python
##########################################################################
# File Name: train_flow.py
# Author: FlowPepDock contributors
# Description: Flow Matching training入口，支持多卡DDP和AMP
#########################################################################

import argparse
import glob
import os
import sys
import shlex
import time
import csv
import subprocess
import shutil
import warnings
import copy
warnings.filterwarnings("once", category=FutureWarning)
# 屏蔽常见的噪音告警
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch.cuda.amp\.GradScaler\(.*\)` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch.cuda.amp\.autocast\(.*\)` is deprecated.*",
    category=FutureWarning,
)
import pandas as pd
import numpy as np
from types import SimpleNamespace
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import torch
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from e3nn.nn import BatchNorm as E3BatchNorm

from models.model import ScoreModel
from utils.transform import FlowMatchingTransform
from utils.flow_matching import flow_matching_loss
from utils.utils import ExponentialMovingAverage, get_optimizer_and_scheduler
from utils.geometry import rot6d_to_matrix, axis_angle_to_matrix

# 屏蔽无关紧要的第三方版本提示
warnings.filterwarnings(
    "ignore",
    message="Boto3 will no longer support Python 3.9",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FlowPepDock with Flow Matching")
    parser.add_argument(
        "--config",
        type=str,
        default="train_models/CGTensorProductEquivariantModel/model_parameters.yml",
        help="模型/优化相关配置文件",
    )
    parser.add_argument("--train_dir", type=str, help="训练集Graph目录(.pt)，若提供 --train_csv 可不填")
    parser.add_argument("--val_dir", type=str, help="验证集Graph目录(.pt)，若提供 --val_csv 可不填")
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="训练集CSV，需含 complex_name，可选 pdb_dir；当前仓库不内置训练数据，需显式传入",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=None,
        help="验证集CSV，需含 complex_name，可选 pdb_dir；当前仓库不内置验证数据，需显式传入",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        choices=["onehot", "esm"],
        default=None,
        help="指定 embedding 模式（onehot/esm）；若不传则使用 config.embedding_mode",
    )
    parser.add_argument("--log_dir", type=str, default="training_logs", help="日志/ckpt输出目录")
    parser.add_argument("--epochs", type=int, default=None, help="训练epoch覆盖配置中的n_epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="单卡batch size")
    parser.add_argument("--lr", type=float, default=None, help="学习率覆盖配置中的lr")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader的worker数量（默认跟随cfg.num_dataloader_workers；不填更稳，避免open files/shm翻车）",
    )
    parser.add_argument("--save_every", type=int, default=10, help="多少个epoch保存一次ckpt")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint路径")
    parser.add_argument(
        "--resume_model_only",
        action="store_true",
        help="只加载 --resume 的模型权重，不加载 optimizer/scheduler/scaler（用于跨版本参数变更后继续训练）",
    )
    parser.add_argument("--clip_grad", type=float, default=1.0, help="梯度裁剪max_norm，<=0则不裁剪")
    parser.add_argument(
        "--no_batch_norm",
        action="store_true",
        help="覆盖配置：禁用BatchNorm（batch_size很小/overfit测试更稳）",
    )
    parser.add_argument(
        "--val_same_as_train",
        action="store_true",
        help="直接复用 train loss 作为 val（仅用于同集对齐调试，避免额外计算）",
    )
    parser.add_argument(
        "--val_fixed_t",
        type=float,
        default=None,
        help="val 固定时间 t（0-1），仅用于评估稳定性诊断；None 表示按配置采样",
    )
    parser.add_argument(
        "--val_fixed_seed",
        type=int,
        default=None,
        help="val 固定噪声随机种子（按样本确定性复现），仅用于评估稳定性诊断",
    )
    parser.add_argument(
        "--val_log_loss_threshold",
        type=float,
        default=None,
        help="val loss 超过该阈值时额外记录到 bad_batches.txt（None 表示不做阈值记录）",
    )
    # Flow 超参覆盖（避免每次手改 yml）
    parser.add_argument("--flow_sigma_tr_max", type=float, default=None, help="覆盖 cfg.flow.sigma_tr_max（None不覆盖）")
    parser.add_argument("--flow_sigma_rot_max", type=float, default=None, help="覆盖 cfg.flow.sigma_rot_max（None不覆盖）")
    parser.add_argument("--flow_sigma_tor_bb_max", type=float, default=None, help="覆盖 cfg.flow.sigma_tor_bb_max（None不覆盖）")
    parser.add_argument("--flow_sigma_tor_sc_max", type=float, default=None, help="覆盖 cfg.flow.sigma_tor_sc_max（None不覆盖）")
    parser.add_argument("--flow_w_tr", type=float, default=None, help="覆盖 cfg.flow.loss_weights.tr（None不覆盖）")
    parser.add_argument("--flow_w_rot", type=float, default=None, help="覆盖 cfg.flow.loss_weights.rot（None不覆盖）")
    parser.add_argument(
        "--flow_w_rot_kabsch",
        type=float,
        default=None,
        help="覆盖 cfg.flow.loss_weights.rot_kabsch（None不覆盖）",
    )
    parser.add_argument(
        "--flow_rot_ref_input",
        type=int,
        default=None,
        help="已废弃：rot_ref_input 历史 rot-only 诊断未带来验证收益；保留参数名仅为兼容旧脚本，训练入口会拒绝启用。",
    )
    parser.add_argument(
        "--flow_rot_ref_mode",
        type=str,
        default=None,
        help="已废弃：关联旧 rot_ref_input 实验；保留字段仅为兼容旧配置，训练入口会拒绝继续使用该分支。",
    )
    parser.add_argument(
        "--flow_rot_ref_dist",
        type=float,
        default=None,
        help="已废弃：关联旧 rot_ref_input 实验；保留字段仅为兼容旧配置，训练入口会拒绝继续使用该分支。",
    )
    parser.add_argument("--flow_rot_target_scale", type=float, default=None, help="覆盖 cfg.flow.loss_weights.rot_target_scale（None不覆盖）")
    parser.add_argument(
        "--flow_rot_loss_mode",
        type=str,
        default=None,
        help="覆盖 cfg.flow.loss_weights.rot_loss_mode（mse/geodesic/quat/axis_angle_sep/dual_axis/swing_twist/vmf_axis/frame_vector）",
    )
    parser.add_argument(
        "--flow_rot_frame_axis",
        type=str,
        default=None,
        help="覆盖 cfg.flow.loss_weights.rot_frame_axis（如 xz/z/xyz）",
    )
    parser.add_argument(
        "--flow_rot_target_mode",
        type=str,
        default=None,
        help="覆盖 cfg.flow.rot_target_mode（noise/score/rel/x1/kabsch/kabsch_rigid/frame/frame_vector/frame_xt/frame_local/frame_local_xt/frame_rel_iface/anchor/anchor_ref/anchor_seq_cb2/anchor_ref_seq_cb2/anchor_ref_iface_sc/anchor_ref_iface_sc_twist/anchor_ref_seq_weighted_twist/contact_*）",
    )
    parser.add_argument("--flow_rot_noise_mode", type=str, default=None, help="覆盖 cfg.flow.rot_noise_mode（isotropic/anchor）")
    parser.add_argument("--flow_rot_repr", type=str, default=None, help="覆盖 cfg.flow.rot_repr（axis_angle/rot6d/rot6d_x1）")
    parser.add_argument("--flow_rot_head_mode", type=str, default=None, help="覆盖 cfg.flow.rot_head_mode（equivariant/absolute）")
    parser.add_argument("--flow_rot_axis_ref_mode", type=str, default=None, help="覆盖 cfg.flow.rot_axis_ref_mode（rec_to_pep_com/none）")
    parser.add_argument("--flow_rot_loss_t_min", type=float, default=None, help="覆盖 cfg.flow.loss_weights.rot_loss_t_min（None不覆盖）")
    parser.add_argument("--flow_rot_frame_contact_only", type=int, default=None, help="覆盖 cfg.flow.rot_frame_contact_only（1=仅接口残基帧）")
    parser.add_argument("--flow_rot_frame_contact_cutoff", type=float, default=None, help="覆盖 cfg.flow.rot_frame_contact_cutoff（Å）")
    parser.add_argument("--flow_rot_frame_contact_min_points", type=int, default=None, help="覆盖 cfg.flow.rot_frame_contact_min_points")
    parser.add_argument("--flow_rot_frame_contact_fallback", type=int, default=None, help="覆盖 cfg.flow.rot_frame_contact_fallback（1=接口不足回退全受体）")
    parser.add_argument("--flow_rot_frame_input", type=int, default=None, help="已废弃：显式 frame 输入在 rot-only 诊断与 subset512/e8 复核中均未提升验证指标；训练入口会拒绝启用。")
    parser.add_argument("--flow_rot_frame_input_mode", type=str, default=None, help="已废弃：关联旧 rot_frame_input 实验；保留字段仅为兼容旧脚本。")
    parser.add_argument("--flow_rot_frame_input_scale", type=float, default=None, help="已废弃：关联旧 rot_frame_input 实验；保留字段仅为兼容旧脚本。")
    parser.add_argument("--flow_rot_frame_multi_anchor", type=int, default=None, help="已废弃：multi-anchor 在 8epoch 复核中未复现早期收益，baseline 更稳；训练入口会拒绝启用。")
    parser.add_argument("--flow_rot_frame_multi_anchor_k", type=int, default=None, help="已废弃：关联旧 rot_frame_multi_anchor 实验；保留字段仅为兼容旧脚本。")
    parser.add_argument("--flow_rot_frame_multi_anchor_neighbors", type=int, default=None, help="已废弃：关联旧 rot_frame_multi_anchor 实验；保留字段仅为兼容旧脚本。")
    parser.add_argument("--flow_rot_anchor_input", type=int, default=None, help="已废弃：锚点输入历史诊断未带来明显提升；训练入口会拒绝启用。")
    parser.add_argument("--flow_rot_anchor_input_scale", type=float, default=None, help="已废弃：关联旧 rot_anchor_input 实验；保留字段仅为兼容旧脚本。")
    parser.add_argument(
        "--flow_rot_anchor_input_mode",
        type=str,
        default=None,
        help="已废弃：关联旧 rot_anchor_input 实验；保留字段仅为兼容旧脚本。",
    )
    parser.add_argument("--flow_coord_align_weight", type=float, default=None, help="覆盖 cfg.flow.loss_weights.coord_align（None不覆盖）")
    parser.add_argument("--flow_ca_coord_weight", type=float, default=None, help="覆盖 cfg.flow.loss_weights.ca_coord（None不覆盖）")
    parser.add_argument("--flow_pose_align_weight", type=float, default=None, help="覆盖 cfg.flow.loss_weights.pose_align（None不覆盖）")
    parser.add_argument("--flow_triplet_weight", type=float, default=None, help="覆盖 cfg.flow.loss_weights.triplet_align（None不覆盖）")
    parser.add_argument("--flow_rot_cls_bins", type=int, default=None, help="覆盖 cfg.flow.rot_cls_bins（0=关闭）")
    parser.add_argument("--flow_rot_cls_target_mode", type=str, default=None, help="覆盖 cfg.flow.rot_cls_target_mode（kabsch/frame/anchor/anchor_ref）")
    parser.add_argument("--flow_rot_cls_weight", type=float, default=None, help="覆盖 cfg.flow.loss_weights.rot_cls_weight（None不覆盖）")
    parser.add_argument("--flow_interface_contact_head", type=int, default=None, help="保留实验项：接口辅助监督可学到，但未稳定转化为 infer_mix 主指标提升，默认关闭。")
    parser.add_argument("--flow_interface_pairdist_head", type=int, default=None, help="保留实验项：接口辅助监督可学到，但未稳定转化为 infer_mix 主指标提升，默认关闭。")
    parser.add_argument("--flow_interface_head_dim", type=int, default=None, help="覆盖 cfg.flow.interface_head_dim（默认ns）")
    parser.add_argument("--flow_interface_contact_weight", type=float, default=None, help="覆盖 cfg.flow.loss_weights.interface_contact（None不覆盖）")
    parser.add_argument("--flow_interface_pairdist_weight", type=float, default=None, help="覆盖 cfg.flow.loss_weights.interface_pairdist（None不覆盖）")
    parser.add_argument("--flow_interface_contact_cutoff", type=float, default=None, help="覆盖 cfg.flow.loss_weights.interface_contact_cutoff（Å）")
    parser.add_argument("--flow_interface_pairdist_max", type=float, default=None, help="覆盖 cfg.flow.loss_weights.interface_pairdist_max（距离裁剪上限）")
    parser.add_argument("--flow_rot_tor_refiner", type=int, default=None, help="保留实验项：仅在组合实验中验证为可运行，但同预算明显不如 baseline，未形成独立正收益。")
    parser.add_argument("--flow_rot_tor_refiner_rot_scale", type=float, default=None, help="保留实验项：关联 rot_tor_refiner；当前未形成独立正收益。")
    parser.add_argument("--flow_rot_tor_refiner_tor_scale", type=float, default=None, help="保留实验项：关联 rot_tor_refiner；当前未形成独立正收益。")
    parser.add_argument("--flow_tor_pairdist_inject", type=int, default=None, help="保留实验项：与 rot_tor_refiner 组合可运行，但同预算不如 baseline，默认关闭。")
    parser.add_argument("--flow_inter_edge_update", type=int, default=None, help="保留实验项：seed1/seed2 方向相反，收益不稳定，默认关闭。")
    parser.add_argument("--flow_inter_edge_update_scale", type=float, default=None, help="保留实验项：关联 inter_edge_update；当前仍在稳定化筛选。")
    parser.add_argument("--flow_self_condition", type=int, default=None, help="保留实验项：仅在 light-enhancement 组合验证过，未见稳定净收益，默认关闭。")
    parser.add_argument("--flow_self_condition_prob", type=float, default=None, help="覆盖 cfg.flow.self_condition_prob（训练时注入概率）")
    parser.add_argument("--flow_self_condition_scale", type=float, default=None, help="保留实验项：关联 self_condition；当前仅用于复现实验，不建议直接切主线。")
    parser.add_argument("--flow_sparse_interface", type=int, default=None, help="保留实验项：仅在 light-enhancement 组合验证过，未见稳定净收益，默认关闭。")
    parser.add_argument("--flow_sparse_interface_topk", type=int, default=None, help="保留实验项：关联 sparse_interface；当前仅用于复现实验，不建议直接切主线。")
    parser.add_argument("--flow_contact_weight", type=float, default=None, help="覆盖 cfg.flow.loss_weights.contact（None不覆盖）")
    parser.add_argument("--flow_contact_cutoff", type=float, default=None, help="覆盖 cfg.flow.loss_weights.contact_cutoff（None不覆盖）")
    parser.add_argument("--flow_contact_top_k", type=int, default=None, help="覆盖 cfg.flow.loss_weights.contact_top_k（None不覆盖）")
    parser.add_argument("--flow_contact_true_max", type=float, default=None, help="覆盖 cfg.flow.loss_weights.contact_true_max（None不覆盖）")
    parser.add_argument("--flow_contact_t_min", type=float, default=None, help="覆盖 cfg.flow.loss_weights.contact_t_min（None不覆盖）")
    parser.add_argument("--flow_contact_noncontact_weight", type=float, default=None, help="覆盖 cfg.flow.loss_weights.contact_noncontact（None不覆盖）")
    parser.add_argument("--flow_contact_noncontact_min", type=float, default=None, help="覆盖 cfg.flow.loss_weights.contact_noncontact_min（None不覆盖）")
    parser.add_argument("--flow_clash_weight", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash（None不覆盖）")
    parser.add_argument("--flow_clash_min_dist", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_min_dist（A）")
    parser.add_argument("--flow_clash_soft_dist", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_soft_dist（A）")
    parser.add_argument("--flow_clash_soft_weight", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_soft_weight（None不覆盖）")
    parser.add_argument("--flow_clash_density_cutoff", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_density_cutoff（A）")
    parser.add_argument("--flow_clash_density_weight", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_density_weight（None不覆盖）")
    parser.add_argument("--flow_clash_density_allowance", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_density_allowance（None不覆盖）")
    parser.add_argument("--flow_clash_density_temperature", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_density_temperature（None不覆盖）")
    parser.add_argument("--flow_clash_unroll_steps", type=int, default=None, help="覆盖 cfg.flow.loss_weights.clash_unroll_steps（None不覆盖）")
    parser.add_argument("--flow_clash_local_rec_radius", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_local_rec_radius（A，0=关闭局部受体mask）")
    parser.add_argument("--flow_clash_local_pep_radius", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_local_pep_radius（A，0=关闭局部肽mask）")
    parser.add_argument("--flow_clash_local_min_rec_atoms", type=int, default=None, help="覆盖 cfg.flow.loss_weights.clash_local_min_rec_atoms")
    parser.add_argument("--flow_clash_local_min_pep_atoms", type=int, default=None, help="覆盖 cfg.flow.loss_weights.clash_local_min_pep_atoms")
    parser.add_argument("--flow_clash_local_fallback_global", type=int, default=None, help="覆盖 cfg.flow.loss_weights.clash_local_fallback_global（1=局部不足时回退全局）")
    parser.add_argument("--flow_clash_adaptive_metric", type=str, default=None, help="覆盖 cfg.flow.loss_weights.clash_adaptive_metric（collide_ratio/shell_ratio/min_dist/density）")
    parser.add_argument("--flow_clash_adaptive_center", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_adaptive_center")
    parser.add_argument("--flow_clash_adaptive_temperature", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_adaptive_temperature")
    parser.add_argument("--flow_clash_adaptive_min_factor", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_adaptive_min_factor（安全样本的最小steric系数）")
    parser.add_argument("--flow_clash_hard_overlap_center", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_hard_overlap_center（A）")
    parser.add_argument("--flow_clash_hard_overlap_temperature", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_hard_overlap_temperature")
    parser.add_argument("--flow_clash_hard_overlap_max_factor", type=float, default=None, help="覆盖 cfg.flow.loss_weights.clash_hard_overlap_max_factor")
    parser.add_argument("--flow_w_tor_bb", type=float, default=None, help="覆盖 cfg.flow.loss_weights.tor_bb（None不覆盖）")
    parser.add_argument("--flow_w_tor_sc", type=float, default=None, help="覆盖 cfg.flow.loss_weights.tor_sc（None不覆盖）")
    parser.add_argument("--flow_tr_scale", type=float, default=None, help="覆盖 cfg.flow.loss_weights.tr_scale（None不覆盖）")
    parser.add_argument("--flow_stage1_epochs", type=int, default=0, help="前期阶段 epoch 数（0=关闭分阶段权重）")
    parser.add_argument("--flow_stage1_w_tr", type=float, default=None, help="阶段1覆盖 loss_weights.tr（None不覆盖）")
    parser.add_argument("--flow_stage1_w_rot", type=float, default=None, help="阶段1覆盖 loss_weights.rot（None不覆盖）")
    parser.add_argument("--flow_stage1_w_tor_bb", type=float, default=None, help="阶段1覆盖 loss_weights.tor_bb（None不覆盖）")
    parser.add_argument("--flow_stage1_w_tor_sc", type=float, default=None, help="阶段1覆盖 loss_weights.tor_sc（None不覆盖）")
    parser.add_argument("--flow_time_sampling", type=str, default=None, help="覆盖 cfg.flow.time_sampling（uniform/sqrt/beta/clipped）")
    parser.add_argument("--flow_t_min", type=float, default=None, help="覆盖 cfg.flow.t_min（截断避免过多小t）")
    parser.add_argument("--flow_t_max", type=float, default=None, help="覆盖 cfg.flow.t_max（默认1.0）")
    parser.add_argument(
        "--flow_deterministic_seed",
        type=int,
        default=None,
        help="覆盖 cfg.flow.deterministic_seed（固定训练噪声，仅用于诊断）",
    )
    parser.add_argument(
        "--flow_fixed_t",
        type=float,
        default=None,
        help="覆盖 cfg.flow.fixed_t（固定训练t，仅用于诊断）",
    )
    parser.add_argument("--flow_beta_alpha", type=float, default=None, help="覆盖 cfg.flow.beta_alpha（time_sampling=beta）")
    parser.add_argument("--flow_beta_beta", type=float, default=None, help="覆盖 cfg.flow.beta_beta（time_sampling=beta）")
    parser.add_argument("--flow_tr_sampling", type=str, default=None, help="覆盖 cfg.flow.tr_sampling（gaussian/shell）")
    parser.add_argument("--flow_tr_r_min", type=float, default=None, help="覆盖 cfg.flow.tr_r_min（shell模式，最小半径）")
    parser.add_argument("--flow_tr_r_max", type=float, default=None, help="覆盖 cfg.flow.tr_r_max（shell模式，最大半径）")
    parser.add_argument("--flow_tr_r_mu", type=float, default=None, help="覆盖 cfg.flow.tr_r_mu（gaussian_shell 半径均值）")
    parser.add_argument("--flow_tr_r_sigma", type=float, default=None, help="覆盖 cfg.flow.tr_r_sigma（gaussian_shell 半径方差）")
    parser.add_argument("--flow_tr_center_mode", type=str, default=None, help="覆盖 cfg.flow.tr_center_mode（receptor_com/pep_com）")
    parser.add_argument("--flow_tr_min_dist", type=float, default=None, help="覆盖 cfg.flow.tr_min_dist（最小原子距离阈值）")
    parser.add_argument("--flow_tr_reject_max_tries", type=int, default=None, help="覆盖 cfg.flow.tr_reject_max_tries（拒绝采样最大尝试）")
    parser.add_argument(
        "--flow_target_mode",
        type=str,
        default=None,
        choices=["velocity", "noise"],
        help="覆盖 cfg.flow.target_mode：velocity（回归速度v）/noise（回归噪声eps）",
    )
    parser.add_argument(
        "--flow_time_weighting",
        type=str,
        default=None,
        help="设置 cfg.flow.loss_time_weighting：none/t/t2/sqrt_t（仅对 tr/rot 生效）",
    )
    parser.add_argument("--swanlab_project", type=str, default=None, help="SwanLab项目名，留空则不记录")
    parser.add_argument("--swanlab_run_name", type=str, default=None, help="SwanLab run名称")
    parser.add_argument("--swanlab_tags", type=str, nargs="*", default=None, help="SwanLab tags列表")
    parser.add_argument(
        "--swanlab_resume_id",
        type=str,
        default=None,
        help="指定已有 SwanLab run id 续跑；若为空且 log_dir 下有 swanlab_run_id.txt 则自动续跑",
    )
    parser.add_argument("--bark_id", type=str, default=None, help="Bark 设备 key；传入后训练结束发送通知")
    parser.add_argument(
        "--bark_min_minutes",
        type=float,
        default=20.0,
        help="仅当训练耗时≥该阈值时发送 Bark（分钟）",
    )
    # 评估：每隔 N 个 epoch 使用当前 ckpt 在小子集上跑推理+RMSD
    parser.add_argument("--eval_every", type=int, default=0, help="每隔多少个epoch触发一次推理评估（0则不评估）")
    parser.add_argument("--eval_csv", type=str, default=None, help="小评估子集 CSV（20-50 条），用于推理+RMSD")
    parser.add_argument("--eval_infer_config", type=str, default="default_inference_args.yaml", help="推理使用的 config")
    parser.add_argument("--model_dir", type=str, default="train_models/CGTensorProductEquivariantModel", help="推理时读取 model_parameters.yml 的目录")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="评估推理 batch size（默认1，显存允许可加大以提速）")
    parser.add_argument("--eval_cpu", type=int, default=0, help="评估推理 CPU 核数；0 表示用 GPU，>0 表示 CPU 推理")
    parser.add_argument("--eval_dockq", action="store_true", help="评估阶段同时计算 DockQ（需要已安装 DockQ）")
    parser.add_argument("--skip_val", action="store_true", help="跳过每个epoch的val循环（省时省资源）；此时best按train_loss保存")
    return parser.parse_args()


def _raise_on_deprecated_flow_switches(args):
    deprecated_reasons = {
        "flow_rot_frame_input": "显式 frame 输入在 rot-only 诊断与 subset512/e8 复核中均未提升验证端指标。",
        "flow_rot_frame_input_mode": "关联已废弃的 flow_rot_frame_input。",
        "flow_rot_frame_input_scale": "关联已废弃的 flow_rot_frame_input。",
        "flow_rot_frame_multi_anchor": "multi-anchor 在 8epoch 复核中未复现早期收益，baseline 更稳。",
        "flow_rot_frame_multi_anchor_k": "关联已废弃的 flow_rot_frame_multi_anchor。",
        "flow_rot_frame_multi_anchor_neighbors": "关联已废弃的 flow_rot_frame_multi_anchor。",
        "flow_rot_anchor_input": "锚点输入历史诊断未带来明显提升。",
        "flow_rot_anchor_input_scale": "关联已废弃的 flow_rot_anchor_input。",
        "flow_rot_anchor_input_mode": "关联已废弃的 flow_rot_anchor_input。",
        "flow_rot_ref_input": "参考点输入重跑后仍未提升验证指标。",
        "flow_rot_ref_mode": "关联已废弃的 flow_rot_ref_input。",
        "flow_rot_ref_dist": "关联已废弃的 flow_rot_ref_input。",
    }
    triggered = []
    for name, reason in deprecated_reasons.items():
        value = getattr(args, name, None)
        if value is None:
            continue
        if isinstance(value, bool) and value is False:
            continue
        if isinstance(value, int) and value == 0:
            continue
        triggered.append((name, reason))
    if triggered:
        details = "\n".join(f"- {name}: {reason}" for name, reason in triggered)
        raise SystemExit(
            "检测到已废弃的 flow 开关，训练入口已禁止继续启用：\n"
            f"{details}\n"
            "这些代码块仍保留仅为兼容旧 checkpoint / 旧脚本，不再作为可用训练分支。"
        )


def _raise_on_deprecated_flow_config(flow_cfg):
    deprecated_flags = {
        "rot_frame_input": "显式 frame 输入在 rot-only 诊断与 subset512/e8 复核中均未提升验证端指标。",
        "rot_frame_multi_anchor": "multi-anchor 在 8epoch 复核中未复现早期收益，baseline 更稳。",
        "rot_anchor_input": "锚点输入历史诊断未带来明显提升。",
        "rot_ref_input": "参考点输入重跑后仍未提升验证指标。",
    }
    triggered = [
        (name, reason)
        for name, reason in deprecated_flags.items()
        if bool((flow_cfg or {}).get(name, False))
    ]
    if triggered:
        details = "\n".join(f"- flow.{name}: {reason}" for name, reason in triggered)
        raise SystemExit(
            "配置文件中启用了已废弃的 flow 分支：\n"
            f"{details}\n"
            "请关闭这些分支后再训练；保留实现仅为兼容旧实验记录。"
        )


def load_config(path):
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)


def apply_embedding_mode(cfg, mode):
    if mode is None or str(mode).strip() == "":
        mode = getattr(cfg, "embedding_mode", None)
    if mode is None or str(mode).strip() == "":
        mode = "onehot"
    mode = str(mode).lower()
    if mode not in {"onehot", "esm"}:
        raise ValueError(f"不支持的 embedding_mode={mode}（支持 onehot/esm）")
    cfg.embedding_mode = mode
    use_esm = mode == "esm"
    esm_value = "inline" if use_esm else None
    for key in (
        "esm_embeddings_path_train",
        "esm_embeddings_path_val",
        "esm_embeddings_peptide_train",
        "esm_embeddings_peptide_val",
        "esm_embeddings_peptide_test",
    ):
        setattr(cfg, key, esm_value)
    return mode


class GraphFolderDataset(Dataset):
    """从目录或CSV加载torch.save的HeteroData"""

    def __init__(self, root, transform=None, csv_path=None, embedding="onehot"):
        super().__init__(root, transform)
        self.files = []
        self.embedding = embedding
        if csv_path:
            self._load_from_csv(csv_path, root)
        else:
            self.files = sorted(glob.glob(os.path.join(root, f"features_{embedding}.pt")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"找不到 features_{embedding}.pt 图数据")

    def _load_from_csv(self, csv_path, default_root):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "complex_name" not in (reader.fieldnames or []):
                raise ValueError("CSV需要包含列 complex_name，可选 pdb_dir")
            for row in reader:
                name = (row.get("complex_name") or "").strip().lower()
                if not name:
                    continue
                pdb_dir = (row.get("pdb_dir") or "").strip()
                if not pdb_dir:
                    rec_pdb = (row.get("receptor_pdb") or "").strip()
                    if rec_pdb:
                        pdb_dir = os.path.dirname(rec_pdb)
                if pdb_dir:
                    base = pdb_dir
                elif default_root:
                    base = os.path.join(default_root, name)
                else:
                    base = None
                if base:
                    pt_path = os.path.join(base, f"features_{self.embedding}.pt")
                    self.files.append(pt_path)

    def len(self):
        return len(self.files)

    def get(self, idx):
        path = self.files[idx]
        data = torch.load(path, map_location="cpu")
        # 兜底补充 complex_name/idx，便于调试日志打印
        try:
            cname = getattr(data, "complex_name", None)
            if not cname:
                cname = os.path.basename(os.path.dirname(path)).lower()
                data.complex_name = cname
            data.idx = idx
        except Exception:
            pass
        return data


def init_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        backend = os.environ.get("TORCH_BACKEND", "nccl")
        dist.init_process_group(backend=backend)
        return True, local_rank, world_size
    return False, 0, 1


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(value, world_size):
    if world_size == 1:
        return value
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / world_size


def _format_duration(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}小时")
    if minutes:
        parts.append(f"{minutes}分")
    parts.append(f"{secs}秒")
    return "".join(parts)


def _format_cmd(argv):
    cmd_str = " ".join(shlex.quote(arg) for arg in argv)
    if len(cmd_str) > 200:
        return cmd_str[:197] + "..."
    return cmd_str


def _send_bark(bark_id, title, body, timeout=10):
    url = f"https://api.day.app/{bark_id}/{quote(title)}/{quote(body)}"
    with urlopen(url, timeout=timeout) as resp:
        resp.read()


def _set_bn_batch_stats(model):
    module = model.module if hasattr(model, "module") else model
    bn_modules = []
    for m in module.modules():
        if isinstance(m, (E3BatchNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_modules.append((m, m.training, getattr(m, "momentum", None)))
            m.train()
            if getattr(m, "momentum", None) is not None:
                m.momentum = 0.0
    return bn_modules


def _restore_bn_state(bn_modules):
    for m, was_training, momentum in bn_modules:
        if was_training:
            m.train()
        else:
            m.eval()
        if momentum is not None:
            m.momentum = momentum




def prepare_dataloader(
    dataset, batch_size, num_workers, distributed, shuffle=True, pin_memory: bool = False
):
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=False,
    )
    return loader, sampler


def _build_pep_res_orig_pos(pep_a, pep, device):
    """
    构建与 pep 残基节点一一对应的原始坐标（优先 CA，缺失时回退该残基原子均值）。
    返回形状: [num_pep_res, 3]，失败时返回 None。
    """
    if pep_a is None or not hasattr(pep_a, "orig_pos") or not hasattr(pep_a, "atom2res_index"):
        return None
    orig_pos = pep_a.orig_pos.to(device)
    if orig_pos.numel() == 0:
        return None
    atom2res = pep_a.atom2res_index.to(device).long()
    atom_batch = (
        pep_a.batch.to(device).long()
        if hasattr(pep_a, "batch") and pep_a.batch is not None
        else torch.zeros(atom2res.shape[0], device=device, dtype=torch.long)
    )
    pep_res_batch = (
        pep.batch.to(device).long()
        if pep is not None and hasattr(pep, "batch") and pep.batch is not None
        else None
    )
    if pep_res_batch is not None and pep_res_batch.numel() > 0:
        num_res = int(pep_res_batch.shape[0])
        n_graph = int(pep_res_batch.max().item()) + 1
        counts = torch.bincount(pep_res_batch, minlength=n_graph).long()
        offsets = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long), counts.cumsum(dim=0)[:-1]], dim=0
        )
        if atom_batch.numel() > 0 and int(atom_batch.max().item()) < offsets.shape[0]:
            res_global = atom2res + offsets[atom_batch]
        else:
            res_global = atom2res
    else:
        res_global = atom2res
        num_res = int(res_global.max().item()) + 1 if res_global.numel() > 0 else 0
    if num_res <= 0:
        return None
    valid = (res_global >= 0) & (res_global < num_res)
    if valid.sum().item() == 0:
        return None
    res_global = res_global[valid]
    pos = orig_pos[valid]
    atom_ids = (
        pep_a.atom2atomid_index.to(device).long()[valid]
        if hasattr(pep_a, "atom2atomid_index")
        else None
    )

    res_pos = torch.zeros((num_res, 3), device=device, dtype=pos.dtype)
    res_cnt = torch.zeros((num_res,), device=device, dtype=pos.dtype)

    if atom_ids is not None:
        ca_mask = atom_ids == 1
        if ca_mask.any():
            ca_idx = res_global[ca_mask]
            ca_pos = pos[ca_mask]
            res_pos.index_add_(0, ca_idx, ca_pos)
            res_cnt.index_add_(0, ca_idx, torch.ones_like(ca_idx, dtype=pos.dtype))

    missing = res_cnt <= 0
    if missing.any():
        all_sum = torch.zeros_like(res_pos)
        all_cnt = torch.zeros_like(res_cnt)
        all_sum.index_add_(0, res_global, pos)
        all_cnt.index_add_(0, res_global, torch.ones_like(res_global, dtype=pos.dtype))
        fill_mask = missing & (all_cnt > 0)
        if fill_mask.any():
            all_mean = all_sum / all_cnt.clamp_min(1.0).unsqueeze(-1)
            res_pos[fill_mask] = all_mean[fill_mask]
            res_cnt[fill_mask] = 1.0

    res_pos = res_pos / res_cnt.clamp_min(1.0).unsqueeze(-1)
    return res_pos


def extract_targets(batch, device):
    pep_a = batch["pep_a"] if "pep_a" in getattr(batch, "node_types", []) else None
    pep = batch["pep"] if "pep" in getattr(batch, "node_types", []) else None
    rec = batch["receptor"] if "receptor" in getattr(batch, "node_types", []) else None
    pep_res_orig_pos = _build_pep_res_orig_pos(pep_a, pep, device)
    targets = {
        "tr": batch.flow_tr_target.to(device) if hasattr(batch, "flow_tr_target") else None,
        "rot": batch.flow_rot_target.to(device) if hasattr(batch, "flow_rot_target") else None,
        "rot_mat": batch.flow_rot_target_mat.to(device)
        if hasattr(batch, "flow_rot_target_mat")
        else None,
        "rot_valid": batch.flow_rot_valid.to(device)
        if hasattr(batch, "flow_rot_valid")
        else None,
        "rot_axis_ref": batch.flow_rot_axis_ref.to(device)
        if hasattr(batch, "flow_rot_axis_ref")
        else None,
        "rot_cls_target": batch.flow_rot_cls_target.to(device)
        if hasattr(batch, "flow_rot_cls_target")
        else None,
        "pep_pos": pep_a.pos.to(device) if pep_a is not None and hasattr(pep_a, "pos") else None,
        "pep_orig_pos": pep_a.orig_pos.to(device)
        if pep_a is not None and hasattr(pep_a, "orig_pos")
        else None,
        "pep_res_orig_pos": pep_res_orig_pos,
        "pep_batch": pep_a.batch.to(device) if pep_a is not None and hasattr(pep_a, "batch") else None,
        "pep_res_batch": pep.batch.to(device) if pep is not None and hasattr(pep, "batch") else None,
        "rec_pos": rec.pos.to(device) if rec is not None and hasattr(rec, "pos") else None,
        "rec_batch": rec.batch.to(device) if rec is not None and hasattr(rec, "batch") else None,
        "pep_atom2res_index": pep_a.atom2res_index.to(device)
        if pep_a is not None and hasattr(pep_a, "atom2res_index")
        else None,
        "pep_atom2atomid_index": pep_a.atom2atomid_index.to(device)
        if pep_a is not None and hasattr(pep_a, "atom2atomid_index")
        else None,
        "tor_backbone": batch.flow_tor_backbone_target.to(device)
        if hasattr(batch, "flow_tor_backbone_target")
        else None,
        "tor_sidechain": batch.flow_tor_sidechain_target.to(device)
        if hasattr(batch, "flow_tor_sidechain_target")
        else None,
    }
    return targets


def maybe_unwrap(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def maybe_apply_self_condition(batch, model, enabled: bool, prob: float) -> None:
    """按概率注入 self-conditioning（仅 tr/rot，detach，不引入额外反传图）。"""
    if not enabled:
        if hasattr(batch, "self_cond_tr"):
            delattr(batch, "self_cond_tr")
        if hasattr(batch, "self_cond_rot"):
            delattr(batch, "self_cond_rot")
        return
    if prob <= 0.0 or torch.rand(1).item() >= prob:
        if hasattr(batch, "self_cond_tr"):
            delattr(batch, "self_cond_tr")
        if hasattr(batch, "self_cond_rot"):
            delattr(batch, "self_cond_rot")
        return
    with torch.no_grad():
        sc_outputs = model(batch)
    tr_sc = sc_outputs.get("tr_pred") if isinstance(sc_outputs, dict) else None
    rot_sc = sc_outputs.get("rot_pred") if isinstance(sc_outputs, dict) else None
    if tr_sc is not None and rot_sc is not None:
        batch.self_cond_tr = tr_sc.detach()
        batch.self_cond_rot = rot_sc.detach()
    else:
        if hasattr(batch, "self_cond_tr"):
            delattr(batch, "self_cond_tr")
        if hasattr(batch, "self_cond_rot"):
            delattr(batch, "self_cond_rot")


def main():
    t_main_start = time.time()
    args = parse_args()
    _raise_on_deprecated_flow_switches(args)
    distributed, local_rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if not args.train_dir and not args.train_csv:
        raise SystemExit("必须提供 --train_dir 或 --train_csv 之一")
    if (not args.skip_val) and (not args.val_dir and not args.val_csv):
        raise SystemExit("必须提供 --val_dir 或 --val_csv 之一")

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.n_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.no_batch_norm:
        cfg.no_batch_norm = True
    resolved_embedding = apply_embedding_mode(cfg, args.embedding)
    args.embedding = resolved_embedding
    if not hasattr(cfg, "n_epochs"):
        cfg.n_epochs = 100
    if not hasattr(cfg, "batch_size"):
        cfg.batch_size = 4
    cfg.flow = getattr(cfg, "flow", {})
    os.makedirs(args.log_dir, exist_ok=True)
    flow_target_mode = str((cfg.flow or {}).get("target_mode", "velocity") or "velocity").lower()

    # dataloader 策略：默认跟随配置；不乱开多进程/不强制 pin_memory，避免 “Too many open files/Pin memory thread exited”
    effective_num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(getattr(cfg, "num_dataloader_workers", 0) or 0)
    )
    effective_pin_memory = bool(getattr(cfg, "pin_memory", False))

    # 命令行覆盖 flow 超参（只在 flow 模式下生效）
    if isinstance(cfg.flow, dict):
        if args.flow_sigma_tr_max is not None:
            cfg.flow["sigma_tr_max"] = float(args.flow_sigma_tr_max)
        if args.flow_sigma_rot_max is not None:
            cfg.flow["sigma_rot_max"] = float(args.flow_sigma_rot_max)
        if args.flow_sigma_tor_bb_max is not None:
            cfg.flow["sigma_tor_bb_max"] = float(args.flow_sigma_tor_bb_max)
        if args.flow_sigma_tor_sc_max is not None:
            cfg.flow["sigma_tor_sc_max"] = float(args.flow_sigma_tor_sc_max)
        if args.flow_time_sampling is not None:
            cfg.flow["time_sampling"] = str(args.flow_time_sampling)
        if args.flow_t_min is not None:
            cfg.flow["t_min"] = float(args.flow_t_min)
        if args.flow_t_max is not None:
            cfg.flow["t_max"] = float(args.flow_t_max)
        if args.flow_rot_target_mode is not None:
            cfg.flow["rot_target_mode"] = str(args.flow_rot_target_mode)
        if args.flow_rot_noise_mode is not None:
            cfg.flow["rot_noise_mode"] = str(args.flow_rot_noise_mode)
        if args.flow_rot_repr is not None:
            cfg.flow["rot_repr"] = str(args.flow_rot_repr)
        if args.flow_rot_head_mode is not None:
            cfg.flow["rot_head_mode"] = str(args.flow_rot_head_mode)
        if args.flow_rot_axis_ref_mode is not None:
            cfg.flow["rot_axis_ref_mode"] = str(args.flow_rot_axis_ref_mode)
        if args.flow_rot_frame_contact_only is not None:
            cfg.flow["rot_frame_contact_only"] = bool(int(args.flow_rot_frame_contact_only))
        if args.flow_rot_frame_contact_cutoff is not None:
            cfg.flow["rot_frame_contact_cutoff"] = float(args.flow_rot_frame_contact_cutoff)
        if args.flow_rot_frame_contact_min_points is not None:
            cfg.flow["rot_frame_contact_min_points"] = int(args.flow_rot_frame_contact_min_points)
        if args.flow_rot_frame_contact_fallback is not None:
            cfg.flow["rot_frame_contact_fallback"] = bool(int(args.flow_rot_frame_contact_fallback))
        if args.flow_rot_frame_input is not None:
            cfg.flow["rot_frame_input"] = bool(int(args.flow_rot_frame_input))
        if args.flow_rot_frame_input_mode is not None:
            cfg.flow["rot_frame_input_mode"] = str(args.flow_rot_frame_input_mode)
        if args.flow_rot_frame_input_scale is not None:
            cfg.flow["rot_frame_input_scale"] = float(args.flow_rot_frame_input_scale)
        if args.flow_rot_frame_multi_anchor is not None:
            cfg.flow["rot_frame_multi_anchor"] = bool(int(args.flow_rot_frame_multi_anchor))
        if args.flow_rot_frame_multi_anchor_k is not None:
            cfg.flow["rot_frame_multi_anchor_k"] = int(args.flow_rot_frame_multi_anchor_k)
        if args.flow_rot_frame_multi_anchor_neighbors is not None:
            cfg.flow["rot_frame_multi_anchor_neighbors"] = int(args.flow_rot_frame_multi_anchor_neighbors)
        if args.flow_rot_anchor_input is not None:
            cfg.flow["rot_anchor_input"] = bool(int(args.flow_rot_anchor_input))
        if args.flow_rot_anchor_input_scale is not None:
            cfg.flow["rot_anchor_input_scale"] = float(args.flow_rot_anchor_input_scale)
        if args.flow_rot_anchor_input_mode is not None:
            cfg.flow["rot_anchor_input_mode"] = str(args.flow_rot_anchor_input_mode)
        if args.flow_rot_cls_bins is not None:
            cfg.flow["rot_cls_bins"] = int(args.flow_rot_cls_bins)
        if args.flow_rot_cls_target_mode is not None:
            cfg.flow["rot_cls_target_mode"] = str(args.flow_rot_cls_target_mode)
        if args.flow_interface_contact_head is not None:
            cfg.flow["interface_contact_head"] = bool(int(args.flow_interface_contact_head))
        if args.flow_interface_pairdist_head is not None:
            cfg.flow["interface_pairdist_head"] = bool(int(args.flow_interface_pairdist_head))
        if args.flow_interface_head_dim is not None:
            cfg.flow["interface_head_dim"] = int(args.flow_interface_head_dim)
        if args.flow_rot_tor_refiner is not None:
            cfg.flow["rot_tor_refiner"] = bool(int(args.flow_rot_tor_refiner))
        if args.flow_rot_tor_refiner_rot_scale is not None:
            cfg.flow["rot_tor_refiner_rot_scale"] = float(args.flow_rot_tor_refiner_rot_scale)
        if args.flow_rot_tor_refiner_tor_scale is not None:
            cfg.flow["rot_tor_refiner_tor_scale"] = float(args.flow_rot_tor_refiner_tor_scale)
        if args.flow_tor_pairdist_inject is not None:
            cfg.flow["tor_pairdist_inject"] = bool(int(args.flow_tor_pairdist_inject))
        if args.flow_inter_edge_update is not None:
            cfg.flow["inter_edge_update"] = bool(int(args.flow_inter_edge_update))
        if args.flow_inter_edge_update_scale is not None:
            cfg.flow["inter_edge_update_scale"] = float(args.flow_inter_edge_update_scale)
        if args.flow_self_condition is not None:
            cfg.flow["self_condition"] = bool(int(args.flow_self_condition))
        if args.flow_self_condition_prob is not None:
            cfg.flow["self_condition_prob"] = float(args.flow_self_condition_prob)
        if args.flow_self_condition_scale is not None:
            cfg.flow["self_condition_scale"] = float(args.flow_self_condition_scale)
        if args.flow_sparse_interface is not None:
            cfg.flow["sparse_interface"] = bool(int(args.flow_sparse_interface))
        if args.flow_sparse_interface_topk is not None:
            cfg.flow["sparse_interface_topk"] = int(args.flow_sparse_interface_topk)
        if args.flow_deterministic_seed is not None:
            cfg.flow["deterministic_seed"] = int(args.flow_deterministic_seed)
        if args.flow_fixed_t is not None:
            fixed_t = float(args.flow_fixed_t)
            if fixed_t < 0.0 or fixed_t > 1.0:
                raise ValueError(f"[flow] 非法 fixed_t={fixed_t}，需满足 0<=fixed_t<=1")
            cfg.flow["fixed_t"] = fixed_t
        if args.flow_beta_alpha is not None:
            cfg.flow["beta_alpha"] = float(args.flow_beta_alpha)
        if args.flow_beta_beta is not None:
            cfg.flow["beta_beta"] = float(args.flow_beta_beta)
        if args.flow_tr_sampling is not None:
            cfg.flow["tr_sampling"] = str(args.flow_tr_sampling).lower()
        if args.flow_tr_r_min is not None:
            cfg.flow["tr_r_min"] = float(args.flow_tr_r_min)
        if args.flow_tr_r_max is not None:
            cfg.flow["tr_r_max"] = float(args.flow_tr_r_max)
        if args.flow_tr_r_mu is not None:
            cfg.flow["tr_r_mu"] = float(args.flow_tr_r_mu)
        if args.flow_tr_r_sigma is not None:
            cfg.flow["tr_r_sigma"] = float(args.flow_tr_r_sigma)
        if args.flow_tr_center_mode is not None:
            cfg.flow["tr_center_mode"] = str(args.flow_tr_center_mode).lower()
        if args.flow_tr_min_dist is not None:
            cfg.flow["tr_min_dist"] = float(args.flow_tr_min_dist)
        if args.flow_tr_reject_max_tries is not None:
            cfg.flow["tr_reject_max_tries"] = int(args.flow_tr_reject_max_tries)
        if args.flow_target_mode is not None:
            cfg.flow["target_mode"] = str(args.flow_target_mode).lower()
        lw = cfg.flow.get("loss_weights", {})
        if not isinstance(lw, dict):
            lw = {}
        if args.flow_w_tr is not None:
            lw["tr"] = float(args.flow_w_tr)
        if args.flow_w_rot is not None:
            lw["rot"] = float(args.flow_w_rot)
        if args.flow_w_rot_kabsch is not None:
            lw["rot_kabsch"] = float(args.flow_w_rot_kabsch)
        if args.flow_rot_ref_input is not None:
            cfg.flow["rot_ref_input"] = bool(int(args.flow_rot_ref_input))
        if args.flow_rot_ref_mode is not None:
            cfg.flow["rot_ref_mode"] = str(args.flow_rot_ref_mode)
        if args.flow_rot_ref_dist is not None:
            cfg.flow["rot_ref_dist"] = float(args.flow_rot_ref_dist)
        if args.flow_rot_target_scale is not None:
            lw["rot_target_scale"] = float(args.flow_rot_target_scale)
        if args.flow_rot_loss_mode is not None:
            lw["rot_loss_mode"] = str(args.flow_rot_loss_mode)
        if args.flow_rot_frame_axis is not None:
            lw["rot_frame_axis"] = str(args.flow_rot_frame_axis)
        if args.flow_rot_loss_t_min is not None:
            lw["rot_loss_t_min"] = float(args.flow_rot_loss_t_min)
        if args.flow_coord_align_weight is not None:
            lw["coord_align"] = float(args.flow_coord_align_weight)
        if args.flow_ca_coord_weight is not None:
            lw["ca_coord"] = float(args.flow_ca_coord_weight)
        if args.flow_pose_align_weight is not None:
            lw["pose_align"] = float(args.flow_pose_align_weight)
        if args.flow_triplet_weight is not None:
            lw["triplet_align"] = float(args.flow_triplet_weight)
        if args.flow_rot_cls_weight is not None:
            lw["rot_cls_weight"] = float(args.flow_rot_cls_weight)
        if args.flow_interface_contact_weight is not None:
            lw["interface_contact"] = float(args.flow_interface_contact_weight)
        if args.flow_interface_pairdist_weight is not None:
            lw["interface_pairdist"] = float(args.flow_interface_pairdist_weight)
        if args.flow_interface_contact_cutoff is not None:
            lw["interface_contact_cutoff"] = float(args.flow_interface_contact_cutoff)
        if args.flow_interface_pairdist_max is not None:
            lw["interface_pairdist_max"] = float(args.flow_interface_pairdist_max)
        if args.flow_contact_weight is not None:
            lw["contact"] = float(args.flow_contact_weight)
        if args.flow_contact_cutoff is not None:
            lw["contact_cutoff"] = float(args.flow_contact_cutoff)
        if args.flow_contact_top_k is not None:
            lw["contact_top_k"] = int(args.flow_contact_top_k)
        if args.flow_contact_true_max is not None:
            lw["contact_true_max"] = float(args.flow_contact_true_max)
        if args.flow_contact_t_min is not None:
            lw["contact_t_min"] = float(args.flow_contact_t_min)
        if args.flow_contact_noncontact_weight is not None:
            lw["contact_noncontact"] = float(args.flow_contact_noncontact_weight)
        if args.flow_contact_noncontact_min is not None:
            lw["contact_noncontact_min"] = float(args.flow_contact_noncontact_min)
        if args.flow_clash_weight is not None:
            lw["clash"] = float(args.flow_clash_weight)
        if args.flow_clash_min_dist is not None:
            lw["clash_min_dist"] = float(args.flow_clash_min_dist)
        if args.flow_clash_soft_dist is not None:
            lw["clash_soft_dist"] = float(args.flow_clash_soft_dist)
        if args.flow_clash_soft_weight is not None:
            lw["clash_soft_weight"] = float(args.flow_clash_soft_weight)
        if args.flow_clash_density_cutoff is not None:
            lw["clash_density_cutoff"] = float(args.flow_clash_density_cutoff)
        if args.flow_clash_density_weight is not None:
            lw["clash_density_weight"] = float(args.flow_clash_density_weight)
        if args.flow_clash_density_allowance is not None:
            lw["clash_density_allowance"] = float(args.flow_clash_density_allowance)
        if args.flow_clash_density_temperature is not None:
            lw["clash_density_temperature"] = float(args.flow_clash_density_temperature)
        if args.flow_clash_unroll_steps is not None:
            lw["clash_unroll_steps"] = int(args.flow_clash_unroll_steps)
        if args.flow_clash_local_rec_radius is not None:
            lw["clash_local_rec_radius"] = float(args.flow_clash_local_rec_radius)
        if args.flow_clash_local_pep_radius is not None:
            lw["clash_local_pep_radius"] = float(args.flow_clash_local_pep_radius)
        if args.flow_clash_local_min_rec_atoms is not None:
            lw["clash_local_min_rec_atoms"] = int(args.flow_clash_local_min_rec_atoms)
        if args.flow_clash_local_min_pep_atoms is not None:
            lw["clash_local_min_pep_atoms"] = int(args.flow_clash_local_min_pep_atoms)
        if args.flow_clash_local_fallback_global is not None:
            lw["clash_local_fallback_global"] = bool(int(args.flow_clash_local_fallback_global))
        if args.flow_clash_adaptive_metric is not None:
            lw["clash_adaptive_metric"] = str(args.flow_clash_adaptive_metric)
        if args.flow_clash_adaptive_center is not None:
            lw["clash_adaptive_center"] = float(args.flow_clash_adaptive_center)
        if args.flow_clash_adaptive_temperature is not None:
            lw["clash_adaptive_temperature"] = float(args.flow_clash_adaptive_temperature)
        if args.flow_clash_adaptive_min_factor is not None:
            lw["clash_adaptive_min_factor"] = float(args.flow_clash_adaptive_min_factor)
        if args.flow_clash_hard_overlap_center is not None:
            lw["clash_hard_overlap_center"] = float(args.flow_clash_hard_overlap_center)
        if args.flow_clash_hard_overlap_temperature is not None:
            lw["clash_hard_overlap_temperature"] = float(args.flow_clash_hard_overlap_temperature)
        if args.flow_clash_hard_overlap_max_factor is not None:
            lw["clash_hard_overlap_max_factor"] = float(args.flow_clash_hard_overlap_max_factor)
        if args.flow_w_tor_bb is not None:
            lw["tor_bb"] = float(args.flow_w_tor_bb)
        if args.flow_w_tor_sc is not None:
            lw["tor_sc"] = float(args.flow_w_tor_sc)
        if args.flow_tr_scale is not None:
            lw["tr_scale"] = float(args.flow_tr_scale)
        cfg.flow["loss_weights"] = lw
        _raise_on_deprecated_flow_config(cfg.flow)
        if args.flow_pose_align_weight is not None:
            pose_val = float(lw.get("pose_align", 0.0) or 0.0)
            if pose_val <= 0:
                raise ValueError(
                    f"[flow] pose_align 权重覆盖失败，当前={pose_val}，请检查 --flow_pose_align_weight"
                )

        if args.flow_time_weighting is not None:
            mode = str(args.flow_time_weighting).lower()
            if mode in {"none", "off", "0", "false"}:
                cfg.flow["loss_time_weighting"] = {}
            elif mode in {"t"}:
                cfg.flow["loss_time_weighting"] = {"tr": "t", "rot": "t"}
            elif mode in {"t2", "t^2"}:
                cfg.flow["loss_time_weighting"] = {"tr": "t2", "rot": "t2"}
            elif mode in {"sqrt_t", "sqrt"}:
                cfg.flow["loss_time_weighting"] = {"tr": "sqrt_t", "rot": "sqrt_t"}
            else:
                raise ValueError(
                    f"不支持的 --flow_time_weighting={args.flow_time_weighting}（支持 none/t/t2/sqrt_t）"
                )
        flow_target_mode = str(cfg.flow.get("target_mode", flow_target_mode) or flow_target_mode).lower()

    val_cfg = cfg
    if not args.skip_val and (args.val_fixed_t is not None or args.val_fixed_seed is not None):
        val_cfg = copy.deepcopy(cfg)
        if not isinstance(getattr(val_cfg, "flow", {}), dict):
            val_cfg.flow = {}
        else:
            val_cfg.flow = dict(val_cfg.flow)
        if args.val_fixed_t is not None:
            val_cfg.flow["fixed_t"] = float(args.val_fixed_t)
        if args.val_fixed_seed is not None:
            val_cfg.flow["deterministic_seed"] = int(args.val_fixed_seed)

    if local_rank == 0:
        print(f"[stage] main start at {time.strftime('%Y-%m-%d %H:%M:%S')} device={device}")
        print(
            f"[dataloader] num_workers={effective_num_workers} pin_memory={effective_pin_memory}",
            flush=True,
        )
        if isinstance(cfg.flow, dict):
            lw = cfg.flow.get("loss_weights", {}) if isinstance(cfg.flow.get("loss_weights", {}), dict) else {}
            print(
                "[flow] sigma_max "
                f"tr={cfg.flow.get('sigma_tr_max')} rot={cfg.flow.get('sigma_rot_max')} "
                f"tor_bb={cfg.flow.get('sigma_tor_bb_max')} tor_sc={cfg.flow.get('sigma_tor_sc_max')}"
            )
            print(
                "[flow] loss_weights "
                f"tr={lw.get('tr', 1.0)} rot={lw.get('rot', 1.0)} "
                f"tor_bb={lw.get('tor_bb', 1.0)} tor_sc={lw.get('tor_sc', 1.0)} "
                f"tr_scale={lw.get('tr_scale', 10.0)} ca_coord={lw.get('ca_coord', 0.0)} "
                f"pose_align={lw.get('pose_align', 0.0)} triplet_align={lw.get('triplet_align', 0.0)} "
                f"rot_cls={lw.get('rot_cls_weight', 0.0)}"
            )
            print(
                "[flow] time_sampling "
                f"mode={cfg.flow.get('time_sampling','uniform')} t_min={cfg.flow.get('t_min',0.0)} t_max={cfg.flow.get('t_max',1.0)} "
                f"beta_alpha={cfg.flow.get('beta_alpha',2.0)} beta_beta={cfg.flow.get('beta_beta',1.0)} "
                f"loss_time_weighting={cfg.flow.get('loss_time_weighting',{})} target_mode={cfg.flow.get('target_mode','velocity')}"
            )
            print(
                "[flow] light_struct "
                f"self_condition={cfg.flow.get('self_condition', False)} "
                f"self_condition_prob={cfg.flow.get('self_condition_prob', 0.0)} "
                f"sparse_interface={cfg.flow.get('sparse_interface', False)} "
                f"sparse_interface_topk={cfg.flow.get('sparse_interface_topk', 0)}"
            )
        if not args.skip_val and isinstance(getattr(val_cfg, "flow", {}), dict):
            if args.val_fixed_t is not None or args.val_fixed_seed is not None:
                print(
                    "[val] fixed eval "
                    f"t={val_cfg.flow.get('fixed_t', None)} seed={val_cfg.flow.get('deterministic_seed', None)}"
                )

    t_load_start = time.time()
    train_transform = FlowMatchingTransform(cfg)
    val_transform = FlowMatchingTransform(val_cfg)
    train_dataset = GraphFolderDataset(
        args.train_dir or "", transform=train_transform, csv_path=args.train_csv, embedding=args.embedding
    )
    val_dataset = None
    if not args.skip_val:
        val_dataset = GraphFolderDataset(
            args.val_dir or "", transform=val_transform, csv_path=args.val_csv, embedding=args.embedding
        )

    def clean_dict(d):
        if not isinstance(d, dict):
            return {}
        out = {}
        for k, v in d.items():
            if v is None:
                continue
            v = torch.nan_to_num(v, nan=0.0, posinf=1e9, neginf=-1e9)
            out[k] = v
        return out

    bad_batch_log = Path(args.log_dir) / "bad_batches.txt"

    train_loader, train_sampler = prepare_dataloader(
        train_dataset,
        cfg.batch_size,
        effective_num_workers,
        distributed,
        shuffle=True,
        pin_memory=effective_pin_memory,
    )
    val_loader, val_sampler = None, None
    if val_dataset is not None:
        val_loader, val_sampler = prepare_dataloader(
            val_dataset,
            cfg.batch_size,
            effective_num_workers,
            distributed,
            shuffle=False,
            pin_memory=effective_pin_memory,
        )
    if local_rank == 0:
        val_batches = len(val_loader) if val_loader is not None else 0
        print(
            f"[stage] dataloaders ready in {time.time() - t_load_start:.1f}s "
            f"#train_batches={len(train_loader)} #val_batches={val_batches}",
            flush=True,
        )

    model = ScoreModel(cfg)
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    accum_steps = max(1, args.accumulation_steps)
    # 估算总训练步数用于 warmup/cosine 调度，支持比率/epoch 自动计算
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.n_epochs * steps_per_epoch // accum_steps
    cfg.total_steps = total_steps
    warmup_steps = getattr(cfg, "warmup_steps", None)
    warmup_ratio = getattr(cfg, "warmup_ratio", None)
    warmup_epochs = getattr(cfg, "warmup_epochs", None)
    if warmup_ratio is not None:
        warmup_steps = int(total_steps * float(warmup_ratio))
    elif warmup_epochs is not None:
        warmup_steps = int(warmup_epochs * steps_per_epoch // accum_steps)
    if warmup_steps is None:
        # 默认用总步数的 5% 但至少 1，最多总步数-1，避免一直处于 warmup
        warmup_steps = max(1, min(total_steps - 1, int(total_steps * 0.05)))
    warmup_steps = max(0, min(warmup_steps, max(0, total_steps - 1)))
    cfg.warmup_steps = warmup_steps

    optimizer, scheduler = get_optimizer_and_scheduler(cfg, maybe_unwrap(model), scheduler_mode="min")
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    scheduler_is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) if scheduler is not None else False
    use_ema = bool(getattr(cfg, "use_ema", False))
    ema = None
    if use_ema:
        ema_decay = float(getattr(cfg, "ema_rate", 0.999) or 0.999)
        ema = ExponentialMovingAverage(maybe_unwrap(model).parameters(), decay=ema_decay)
        if local_rank == 0:
            print(f"[ema] enabled decay={ema_decay}", flush=True)
    if local_rank == 0:
        print(f"[stage] model/optim ready in {time.time() - t_main_start:.1f}s", flush=True)
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        # 允许少量“历史ckpt多出来的参数”（例如旧版本里 forward 动态创建过的投影层）
        maybe_unwrap(model).load_state_dict(ckpt["model"], strict=False)  # type: ignore
        if ema is not None and "ema_weights" in ckpt and ckpt["ema_weights"] is not None:
            ema.load_state_dict(ckpt["ema_weights"], device=device)
            if local_rank == 0:
                print(f"恢复 EMA 权重，resume_from={args.resume}")
        if args.resume_model_only:
            start_epoch = 0
            if local_rank == 0:
                print(f"仅恢复模型权重（不加载优化器），resume_from={args.resume}")
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            if "scheduler" in ckpt and scheduler is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0)
            if local_rank == 0:
                print(f"恢复训练，起始epoch={start_epoch}")

    loss_weights = cfg.flow.get("loss_weights", {}) if isinstance(cfg.flow, dict) else {}
    stage1_epochs = max(0, int(getattr(args, "flow_stage1_epochs", 0) or 0))
    stage1_weights = dict(loss_weights)
    if args.flow_stage1_w_tr is not None:
        stage1_weights["tr"] = float(args.flow_stage1_w_tr)
    if args.flow_stage1_w_rot is not None:
        stage1_weights["rot"] = float(args.flow_stage1_w_rot)
    if args.flow_stage1_w_tor_bb is not None:
        stage1_weights["tor_bb"] = float(args.flow_stage1_w_tor_bb)
    if args.flow_stage1_w_tor_sc is not None:
        stage1_weights["tor_sc"] = float(args.flow_stage1_w_tor_sc)
    time_weighting = cfg.flow.get("loss_time_weighting", {}) if isinstance(cfg.flow, dict) else {}
    self_cond_enabled = bool((cfg.flow or {}).get("self_condition", False)) if isinstance(cfg.flow, dict) else False
    self_cond_prob = float((cfg.flow or {}).get("self_condition_prob", 0.5) or 0.0) if isinstance(cfg.flow, dict) else 0.0
    self_cond_prob = max(0.0, min(1.0, self_cond_prob))


    sw_run = None
    if local_rank == 0 and args.swanlab_project:
        import swanlab

        run_id_file = Path(args.log_dir) / "swanlab_run_id.txt"
        resume_id = args.swanlab_resume_id
        if args.swanlab_project and resume_id is None and run_id_file.exists():
            resume_id = run_id_file.read_text().strip()
        sw_cfg = {
            "config": vars(cfg),
            "project": args.swanlab_project or "FlowPepDock_notify",
        }
        if not args.swanlab_project:
            sw_cfg["mode"] = "local"
        if args.swanlab_run_name:
            sw_cfg["name"] = args.swanlab_run_name
        if args.swanlab_tags:
            sw_cfg["tags"] = args.swanlab_tags
        if resume_id:
            sw_cfg["id"] = resume_id
            sw_cfg["resume"] = True
        sw_run = swanlab.init(**sw_cfg)
        if sw_run and not run_id_file.exists():
            try:
                run_id_file.parent.mkdir(parents=True, exist_ok=True)
                run_id_file.write_text(getattr(sw_run, "id", ""))
            except Exception:
                pass

    best_metric = float("inf")
    consecutive_eval_fail = 0
    best_ckpt_path = Path(args.log_dir) / f"flow_{args.embedding}_best.pt"

    for epoch in range(start_epoch, cfg.n_epochs):
        loop_start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        if stage1_epochs > 0 and epoch < stage1_epochs:
            current_loss_weights = stage1_weights
            if local_rank == 0 and epoch == start_epoch:
                print(f"[flow] stage1 loss_weights {current_loss_weights}", flush=True)
        else:
            current_loss_weights = loss_weights
            if local_rank == 0 and stage1_epochs > 0 and epoch == stage1_epochs:
                print(f"[flow] stage2 loss_weights {current_loss_weights}", flush=True)

        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        train_comp_acc = {}
        train_frame_cos_sum = 0.0
        train_frame_cos_abs_sum = 0.0
        train_frame_ortho_sum = 0.0
        train_frame_batches = 0
        num_batches = 0
        start_time = time.time()
        global_step = epoch * max(1, len(train_loader))
        train_pbar = tqdm(
            train_loader,
            total=len(train_loader),
            disable=local_rank != 0,
            desc=f"Train {epoch+1}",
        )
        first_step_logged = False
        for step, batch in enumerate(train_pbar):
            if not first_step_logged and local_rank == 0:
                tqdm.write(f"[stage] first train batch epoch {epoch+1} after {time.time()-loop_start:.1f}s")
                first_step_logged = True
            batch = batch.to(device)
            maybe_apply_self_condition(batch, model, self_cond_enabled, self_cond_prob)
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(batch)
                targets = extract_targets(batch, device)
                # 清理 NaN/Inf，遇到空张量时跳过对应分量
                outputs = clean_dict(outputs)
                targets = clean_dict(targets)
                time_samples = getattr(batch, "flow_time", None) if hasattr(batch, "flow_time") else None
                if time_samples is not None:
                    try:
                        time_samples = time_samples.to(device)
                    except Exception:
                        pass
                lw_batch = dict(current_loss_weights)
                if ("tor_pred_sidechain" not in outputs) or outputs["tor_pred_sidechain"].numel() == 0:
                    lw_batch["tor_sc"] = 0.0
                if ("tor_sc" in targets) and targets["tor_sc"] is not None and targets["tor_sc"].numel() == 0:
                    lw_batch["tor_sc"] = 0.0

                loss, loss_dict = flow_matching_loss(
                    outputs,
                    targets,
                    lw_batch,
                    time_samples=time_samples,
                    time_weighting=time_weighting,
                    target_mode=flow_target_mode,
                )
                loss = loss / accum_steps
            if not torch.isfinite(loss):
                names = getattr(batch, "complex_name", "")
                if isinstance(names, (list, tuple)):
                    names = ",".join([str(n) for n in names])
                if not names:
                    idx = getattr(batch, "idx", None)
                    if idx is not None:
                        names = str(idx)
                def _summ(t):
                    t_det = t.detach()
                    if t_det.numel() == 0:
                        return 0.0, 0.0
                    finite = torch.isfinite(t_det)
                    finite_ratio = float(finite.sum().item()) / max(1, finite.numel())
                    amax = float(
                        torch.nan_to_num(t_det, nan=0.0, posinf=1e9, neginf=-1e9)
                        .abs()
                        .max()
                        .item()
                    )
                    return finite_ratio, amax
                dbg_parts = []
                for k in ("tr_pred", "rot_pred", "tor_pred_backbone", "tor_pred_sidechain"):
                    if isinstance(outputs, dict) and k in outputs and outputs[k] is not None:
                        fr, am = _summ(outputs[k])
                        dbg_parts.append(f"{k}[fin={fr:.2f},max={am:.2e}]")
                # 目标分量的统计
                target_parts = []
                for k in ("tr", "rot", "tor_bb", "tor_sc"):
                    if isinstance(targets, dict) and k in targets and targets[k] is not None:
                        fr, am = _summ(targets[k])
                        target_parts.append(f"t_{k}[fin={fr:.2f},max={am:.2e}]")
                tqdm.write(
                        f"[WARN] non-finite train loss at epoch {epoch+1} step {step+1} "
                        f"pdb:{names} {' '.join(dbg_parts)} {' '.join(target_parts)}"
                    )
                try:
                    with bad_batch_log.open("a") as f:
                        f.write(
                            f"train,epoch={epoch+1},step={step+1},pdb={names},"
                            f"{' '.join(dbg_parts)} {' '.join(target_parts)}\n"
                        )
                except Exception:
                    pass
                optimizer.zero_grad(set_to_none=True)
                continue
            # 记录 step 级别 loss（简洁版），便于复查振荡
            if local_rank == 0:
                step_metrics_csv = Path(args.log_dir) / "step_metrics.csv"
                step_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
                write_header = not step_metrics_csv.exists()
                rot_frame_cos = None
                rot_frame_cos_abs = None
                rot_frame_ortho = None
                def _vec_norm_mean(x):
                    try:
                        if x is None:
                            return 0.0
                        x = x.detach()
                        if x.numel() == 0:
                            return 0.0
                        if x.ndim == 1:
                            return float(x.abs().mean().item())
                        if x.ndim >= 2:
                            return float(torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1).mean().item())
                        return 0.0
                    except Exception:
                        return 0.0

                def _cos_mean(pred, target):
                    try:
                        if pred is None or target is None:
                            return 0.0, 0.0
                        pred = pred.detach()
                        target = target.detach()
                        if pred.numel() == 0 or target.numel() == 0:
                            return 0.0, 0.0
                        if pred.shape[0] != target.shape[0]:
                            if pred.shape[0] == 1 and target.shape[0] > 1:
                                pred = pred.expand(target.shape[0], *pred.shape[1:])
                            elif target.shape[0] == 1 and pred.shape[0] > 1:
                                target = target.expand(pred.shape[0], *target.shape[1:])
                            else:
                                return 0.0, 0.0
                        pred = pred.reshape(pred.shape[0], -1)
                        target = target.reshape(target.shape[0], -1)
                        denom = torch.linalg.vector_norm(pred, dim=1) * torch.linalg.vector_norm(target, dim=1)
                        denom = denom.clamp_min(1e-8)
                        cos = (pred * target).sum(dim=1) / denom
                        cos = torch.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)
                        return float(cos.mean().item()), float(cos.abs().mean().item())
                    except Exception:
                        return 0.0, 0.0

                def _frame_metrics(pred, target):
                    try:
                        if pred is None or target is None:
                            return 0.0, 0.0, 0.0
                        pred = pred.detach()
                        target = target.detach()
                        if pred.ndim == 1:
                            pred = pred.unsqueeze(0)
                        if target.ndim == 1:
                            target = target.unsqueeze(0)
                        if pred.ndim == 2 and pred.shape == (3, 3):
                            pred = pred.unsqueeze(0)
                        if target.ndim == 2 and target.shape == (3, 3):
                            target = target.unsqueeze(0)
                        if pred.shape[0] != target.shape[0]:
                            if pred.shape[0] == 1 and target.shape[0] > 1:
                                pred = pred.expand(target.shape[0], -1)
                            elif target.shape[0] == 1 and pred.shape[0] > 1:
                                target = target.expand(pred.shape[0], -1)
                            else:
                                return 0.0, 0.0, 0.0
                        if pred.ndim == 3 and pred.shape[-2:] == (3, 3):
                            pred_mat = pred
                        elif pred.shape[-1] == 3:
                            pred_mat = axis_angle_to_matrix(pred)
                        elif pred.shape[-1] == 6:
                            pred_mat = rot6d_to_matrix(pred)
                        elif pred.shape[-1] == 9:
                            pred_mat = pred.reshape(-1, 3, 3)
                        else:
                            return 0.0, 0.0, 0.0
                        if target.ndim == 3 and target.shape[-2:] == (3, 3):
                            tgt_mat = target
                        elif target.shape[-1] == 3:
                            tgt_mat = axis_angle_to_matrix(target)
                        elif target.shape[-1] == 9:
                            tgt_mat = target.reshape(-1, 3, 3)
                        elif target.shape[-1] == 6:
                            tgt_mat = rot6d_to_matrix(target)
                        else:
                            return 0.0, 0.0, 0.0
                        pred_norm = pred_mat.norm(dim=1).clamp_min(1e-8)
                        tgt_norm = tgt_mat.norm(dim=1).clamp_min(1e-8)
                        pred_unit = pred_mat / pred_norm.unsqueeze(1)
                        tgt_unit = tgt_mat / tgt_norm.unsqueeze(1)
                        cos = (pred_unit * tgt_unit).sum(dim=1).clamp(min=-1.0, max=1.0)
                        cos_mean = cos.mean(dim=1)
                        cos_abs_mean = cos.abs().mean(dim=1)
                        u = pred_unit[:, :, 0]
                        v = pred_unit[:, :, 1]
                        w = pred_unit[:, :, 2]
                        ortho = (
                            (u * v).sum(dim=-1).abs()
                            + (u * w).sum(dim=-1).abs()
                            + (v * w).sum(dim=-1).abs()
                        )
                        return (
                            float(cos_mean.mean().item()),
                            float(cos_abs_mean.mean().item()),
                            float(ortho.mean().item()),
                        )
                    except Exception:
                        return 0.0, 0.0, 0.0
                try:
                    with step_metrics_csv.open("a", newline="") as f:
                        import csv as _csv
                        tr_cos, tr_cos_abs = _cos_mean(
                            outputs.get("tr_pred") if isinstance(outputs, dict) else None,
                            targets.get("tr") if isinstance(targets, dict) else None,
                        )
                        rot_frame_cos, rot_frame_cos_abs, rot_frame_ortho = _frame_metrics(
                            outputs.get("rot_pred") if isinstance(outputs, dict) else None,
                            targets.get("rot_mat") if isinstance(targets, dict) and targets.get("rot_mat") is not None else targets.get("rot") if isinstance(targets, dict) else None,
                        )
                        if rot_frame_cos is not None:
                            train_frame_cos_sum += float(rot_frame_cos)
                            train_frame_cos_abs_sum += float(rot_frame_cos_abs)
                            train_frame_ortho_sum += float(rot_frame_ortho)
                            train_frame_batches += 1
                        w = _csv.DictWriter(
                            f,
                            fieldnames=[
                                "epoch",
                                "step",
                                "global_step",
                                "loss",
                                "tr",
                                "rot",
                                "tor_bb",
                                "tor_sc",
                                "clash",
                                "clash_min_dist",
                                "clash_collide_ratio",
                                "clash_shell_ratio",
                                "clash_density",
                                "t",
                                "tr_pred_norm",
                                "rot_pred_norm",
                                "tr_cos",
                                "tr_cos_abs",
                                # rot_cos/rot_cos_abs 已废弃，保留 frame 指标为准
                                "rot_frame_cos",
                                "rot_frame_cos_abs",
                                "rot_frame_ortho",
                                "tr_tgt_norm",
                                "rot_tgt_norm",
                                "ca_coord",
                                "ca_coord_ratio",
                                "ca_pred_norm",
                                "ca_tgt_norm",
                                "rot_mask_ratio",
                                "pose_align",
                                "triplet_align",
                                "rot_cls",
                                "rot_cls_acc",
                                "iface_contact",
                                "iface_contact_acc",
                                "iface_pairdist",
                                "iface_pairs",
                            ],
                        )
                        if write_header:
                            w.writeheader()
                        w.writerow(
                            {
                                "epoch": epoch + 1,
                                "step": step + 1,
                                "global_step": global_step + step + 1,
                                # loss 当前被除以 accum_steps，恢复原始 batch loss
                                "loss": float(loss.item() * accum_steps),
                                "tr": float(loss_dict.get("tr", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "rot": float(loss_dict.get("rot", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "tor_bb": float(loss_dict.get("tor_bb", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "tor_sc": float(loss_dict.get("tor_sc", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "clash": float(loss_dict.get("clash", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "clash_min_dist": float(loss_dict.get("clash_min_dist", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "clash_collide_ratio": float(loss_dict.get("clash_collide_ratio", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "clash_shell_ratio": float(loss_dict.get("clash_shell_ratio", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "clash_density": float(loss_dict.get("clash_density", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "t": float(time_samples.mean().item()) if time_samples is not None else 0.0,
                                "tr_pred_norm": _vec_norm_mean(outputs.get("tr_pred") if isinstance(outputs, dict) else None),
                                "rot_pred_norm": _vec_norm_mean(outputs.get("rot_pred") if isinstance(outputs, dict) else None),
                                "tr_cos": tr_cos,
                                "tr_cos_abs": tr_cos_abs,
                                "rot_frame_cos": rot_frame_cos,
                                "rot_frame_cos_abs": rot_frame_cos_abs,
                                "rot_frame_ortho": rot_frame_ortho,
                                "tr_tgt_norm": _vec_norm_mean(targets.get("tr") if isinstance(targets, dict) else None),
                                "rot_tgt_norm": _vec_norm_mean(targets.get("rot") if isinstance(targets, dict) else None),
                                "ca_coord": float(loss_dict.get("ca_coord", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "ca_coord_ratio": float(loss_dict.get("ca_coord_ratio", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "ca_pred_norm": _vec_norm_mean(outputs.get("ca_pred") if isinstance(outputs, dict) else None),
                                "ca_tgt_norm": float(loss_dict.get("ca_tgt_norm", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "rot_mask_ratio": (
                                    float(loss_dict["rot_mask_ratio"].item())
                                    if isinstance(loss_dict, dict) and "rot_mask_ratio" in loss_dict
                                    else ""
                                ),
                                "pose_align": float(loss_dict.get("pose_align", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "triplet_align": float(loss_dict.get("triplet_align", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "rot_cls": float(loss_dict.get("rot_cls", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "rot_cls_acc": float(loss_dict.get("rot_cls_acc", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "iface_contact": float(loss_dict.get("iface_contact", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "iface_contact_acc": float(loss_dict.get("iface_contact_acc", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "iface_pairdist": float(loss_dict.get("iface_pairdist", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                "iface_pairs": float(loss_dict.get("iface_pairs", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                            }
                        )
                except Exception as e:
                    tqdm.write(f"[WARN] failed to write step_metrics.csv: {e}")

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                if args.clip_grad and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # scheduler：cosine(LambdaLR) 需要按 optimizer step 更新；plateau 按 epoch 的 val_loss 更新
                if scheduler is not None and not scheduler_is_plateau:
                    try:
                        scheduler.step()
                    except Exception as e:
                        if local_rank == 0:
                            tqdm.write(f"[WARN] scheduler step failed (per-opt): {e}")
                if ema is not None:
                    ema.update(maybe_unwrap(model).parameters())
                if local_rank == 0:
                    raw_loss = loss.item() * accum_steps
                    comp_parts = []
                    if isinstance(loss_dict, dict):
                        for k in ("tr", "rot", "tor_bb", "tor_sc", "ca_coord", "iface_contact", "iface_pairdist"):
                            if k in loss_dict:
                                comp_parts.append(f"{k}:{loss_dict[k].item():.4f}")
                    comp_str = " | " + " ".join(comp_parts) if comp_parts else ""
                    # 打印当前 batch 的 pdb ids 便于定位问题样本，使用 tqdm.write 避免破坏进度条
                    names_str = ""
                    try:
                        names = getattr(batch, "complex_name", None)
                        if names is not None:
                            if isinstance(names, (list, tuple)):
                                names_str = ",".join([str(n) for n in names])
                            else:
                                names_str = str(names)
                    except Exception:
                        names_str = ""
                    msg = f"[Epoch {epoch+1} Step {step+1}] opt_step loss {raw_loss:.4f}{comp_str}"
                    if names_str:
                        msg += f" | pdb:{names_str}"
                    tqdm.write(msg)

            raw_loss = loss.item() * accum_steps
            epoch_loss += raw_loss
            if isinstance(loss_dict, dict):
                for k, v in loss_dict.items():
                    train_comp_acc[k] = train_comp_acc.get(k, 0.0) + float(v.item())
            num_batches += 1
            if sw_run is not None and local_rank == 0:
                batch_log = {"batch/train_loss": raw_loss}
                if isinstance(loss_dict, dict):
                    for k in ("tr", "rot", "tor_bb", "tor_sc", "ca_coord", "clash"):
                        if k in loss_dict:
                            batch_log[f"batch/train_loss_{k}"] = loss_dict[k].item()
                    for k in ("clash_min_dist", "clash_collide_ratio"):
                        if k in loss_dict:
                            batch_log[f"batch/train_{k}"] = loss_dict[k].item()
                if local_rank == 0:
                    rot_frame_cos_val = locals().get("rot_frame_cos", None)
                    rot_frame_cos_abs_val = locals().get("rot_frame_cos_abs", None)
                    rot_frame_ortho_val = locals().get("rot_frame_ortho", None)
                    if rot_frame_cos_val is not None:
                        batch_log["batch/train_rot_frame_cos"] = rot_frame_cos_val
                        batch_log["batch/train_rot_frame_cos_abs"] = rot_frame_cos_abs_val
                        batch_log["batch/train_rot_frame_ortho"] = rot_frame_ortho_val
                sw_run.log(batch_log, step=global_step + step + 1)

        train_loss_avg = epoch_loss / max(1, num_batches)
        train_loss_avg = reduce_tensor(train_loss_avg, world_size)
        train_loss_dict = {k: (v / max(1, num_batches)) for k, v in train_comp_acc.items()}
        train_frame_cos_avg = None
        train_frame_cos_abs_avg = None
        train_frame_ortho_avg = None
        if train_frame_batches > 0:
            train_frame_cos_avg = train_frame_cos_sum / train_frame_batches
            train_frame_cos_abs_avg = train_frame_cos_abs_sum / train_frame_batches
            train_frame_ortho_avg = train_frame_ortho_sum / train_frame_batches

        # 清理训练阶段残留的激活/缓存，避免占用显存影响后续 eval
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        val_loss = float("nan")
        val_loss_dict = {}
        val_loss_stats = {}
        val_frame_cos_avg = None
        val_frame_cos_abs_avg = None
        val_frame_ortho_avg = None
        if args.val_same_as_train:
            val_loss = float(train_loss_avg)
            val_loss_dict = dict(train_loss_dict)
        elif val_loader is not None:
            model.eval()
            bn_state = _set_bn_batch_stats(model)
            if local_rank == 0:
                tqdm.write("[val] BatchNorm 使用 batch 统计（momentum=0）")
            val_loss_acc = 0.0
            val_batches = 0
            val_comp_acc = {}
            val_loss_list = []
            val_nonfinite = 0
            val_over_threshold = 0
            val_frame_cos_sum = 0.0
            val_frame_cos_abs_sum = 0.0
            val_frame_ortho_sum = 0.0
            val_frame_batches = 0
            val_loss_threshold = None
            if args.val_log_loss_threshold is not None:
                val_loss_threshold = float(args.val_log_loss_threshold)

            def _batch_names(batch_obj):
                names = getattr(batch_obj, "complex_name", "")
                if isinstance(names, (list, tuple)):
                    names = ",".join([str(n) for n in names])
                if not names:
                    idx = getattr(batch_obj, "idx", None)
                    if idx is not None:
                        names = str(idx)
                return names

            def _summ(t):
                t_det = t.detach()
                if t_det.numel() == 0:
                    return 0.0, 0.0
                finite = torch.isfinite(t_det)
                finite_ratio = float(finite.sum().item()) / max(1, finite.numel())
                amax = float(
                    torch.nan_to_num(t_det, nan=0.0, posinf=1e9, neginf=-1e9)
                    .abs()
                    .max()
                    .item()
                )
                return finite_ratio, amax

            def _format_debug(outputs_dict, targets_dict):
                dbg_parts = []
                for k in ("tr_pred", "rot_pred", "tor_pred_backbone", "tor_pred_sidechain"):
                    if isinstance(outputs_dict, dict) and k in outputs_dict and outputs_dict[k] is not None:
                        fr, am = _summ(outputs_dict[k])
                        dbg_parts.append(f"{k}[fin={fr:.2f},max={am:.2e}]")
                target_parts = []
                for k in ("tr", "rot", "tor_bb", "tor_sc"):
                    if isinstance(targets_dict, dict) and targets_dict.get(k) is not None:
                        fr, am = _summ(targets_dict[k])
                        target_parts.append(f"t_{k}[fin={fr:.2f},max={am:.2e}]")
                return dbg_parts, target_parts

            with torch.no_grad():
                val_pbar = tqdm(
                    val_loader,
                    total=len(val_loader),
                    disable=local_rank != 0,
                    desc=f"Val {epoch+1}",
                )
                for step, batch in enumerate(val_pbar):
                    batch = batch.to(device)
                    maybe_apply_self_condition(batch, model, self_cond_enabled, self_cond_prob)
                    outputs = model(batch)
                    targets = extract_targets(batch, device)
                    # 清理 NaN/Inf，遇到空张量时跳过对应分量
                    outputs = clean_dict(outputs)
                    targets = clean_dict(targets)
                    time_samples = getattr(batch, "flow_time", None) if hasattr(batch, "flow_time") else None
                    if time_samples is not None:
                        try:
                            time_samples = time_samples.to(device)
                        except Exception:
                            pass
                    lw_batch = dict(current_loss_weights)
                    if ("tor_pred_sidechain" not in outputs) or outputs["tor_pred_sidechain"].numel() == 0:
                        lw_batch["tor_sc"] = 0.0
                    if ("tor_sc" in targets) and targets["tor_sc"] is not None and targets["tor_sc"].numel() == 0:
                        lw_batch["tor_sc"] = 0.0

                    loss, loss_dict = flow_matching_loss(
                        outputs,
                        targets,
                        lw_batch,
                        time_samples=time_samples,
                        time_weighting=time_weighting,
                        target_mode=flow_target_mode,
                    )
                    if not torch.isfinite(loss):
                        val_nonfinite += 1
                        names = _batch_names(batch)
                        dbg_parts, target_parts = _format_debug(outputs, targets)
                        tqdm.write(
                            f"[WARN] non-finite val loss at epoch {epoch+1} pdb:{names} "
                            f"{' '.join(dbg_parts)} {' '.join(target_parts)}"
                        )
                        try:
                            with bad_batch_log.open("a") as f:
                                f.write(
                                    f"val,epoch={epoch+1},pdb={names},"
                                    f"{' '.join(dbg_parts)} {' '.join(target_parts)}\n"
                                )
                        except Exception:
                            pass
                        continue
                    if val_loss_threshold is not None and loss.item() > val_loss_threshold:
                        val_over_threshold += 1
                        names = _batch_names(batch)
                        dbg_parts, target_parts = _format_debug(outputs, targets)
                        tqdm.write(
                            f"[WARN] val loss>{val_loss_threshold:.2f} at epoch {epoch+1} pdb:{names} "
                            f"{' '.join(dbg_parts)} {' '.join(target_parts)}"
                        )
                        try:
                            with bad_batch_log.open("a") as f:
                                f.write(
                                    f"val_loss_gt_threshold,epoch={epoch+1},pdb={names},loss={loss.item():.6f},"
                                    f"{' '.join(dbg_parts)} {' '.join(target_parts)}\n"
                                )
                        except Exception:
                            pass
                    if local_rank == 0:
                        step_metrics_csv = Path(args.log_dir) / "val_step_metrics.csv"
                        step_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
                        write_header = not step_metrics_csv.exists()
                        def _vec_norm_mean(x):
                            try:
                                if x is None:
                                    return 0.0
                                x = x.detach()
                                if x.numel() == 0:
                                    return 0.0
                                if x.ndim == 1:
                                    return float(x.abs().mean().item())
                                if x.ndim >= 2:
                                    return float(torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1).mean().item())
                                return 0.0
                            except Exception:
                                return 0.0

                        def _cos_mean(pred, target):
                            try:
                                if pred is None or target is None:
                                    return 0.0, 0.0
                                pred = pred.detach()
                                target = target.detach()
                                if pred.numel() == 0 or target.numel() == 0:
                                    return 0.0, 0.0
                                if pred.shape[0] != target.shape[0]:
                                    return 0.0, 0.0
                                pred = pred.reshape(pred.shape[0], -1)
                                target = target.reshape(target.shape[0], -1)
                                denom = torch.linalg.vector_norm(pred, dim=1) * torch.linalg.vector_norm(target, dim=1)
                                denom = denom.clamp_min(1e-8)
                                cos = (pred * target).sum(dim=1) / denom
                                cos = torch.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)
                                return float(cos.mean().item()), float(cos.abs().mean().item())
                            except Exception:
                                return 0.0, 0.0

                        def _frame_metrics(pred, target):
                            try:
                                if pred is None or target is None:
                                    return 0.0, 0.0, 0.0
                                pred = pred.detach()
                                target = target.detach()
                                if pred.ndim == 1:
                                    pred = pred.unsqueeze(0)
                                if target.ndim == 1:
                                    target = target.unsqueeze(0)
                                if pred.ndim == 2 and pred.shape == (3, 3):
                                    pred = pred.unsqueeze(0)
                                if target.ndim == 2 and target.shape == (3, 3):
                                    target = target.unsqueeze(0)
                                if pred.shape[0] != target.shape[0]:
                                    if pred.shape[0] == 1 and target.shape[0] > 1:
                                        pred = pred.expand(target.shape[0], -1)
                                    elif target.shape[0] == 1 and pred.shape[0] > 1:
                                        target = target.expand(pred.shape[0], -1)
                                    else:
                                        return 0.0, 0.0, 0.0
                                if pred.ndim == 3 and pred.shape[-2:] == (3, 3):
                                    pred_mat = pred
                                elif pred.shape[-1] == 3:
                                    pred_mat = axis_angle_to_matrix(pred)
                                elif pred.shape[-1] == 6:
                                    pred_mat = rot6d_to_matrix(pred)
                                elif pred.shape[-1] == 9:
                                    pred_mat = pred.reshape(-1, 3, 3)
                                else:
                                    return 0.0, 0.0, 0.0
                                if target.ndim == 3 and target.shape[-2:] == (3, 3):
                                    tgt_mat = target
                                elif target.shape[-1] == 3:
                                    tgt_mat = axis_angle_to_matrix(target)
                                elif target.shape[-1] == 9:
                                    tgt_mat = target.reshape(-1, 3, 3)
                                elif target.shape[-1] == 6:
                                    tgt_mat = rot6d_to_matrix(target)
                                else:
                                    return 0.0, 0.0, 0.0
                                pred_norm = pred_mat.norm(dim=1).clamp_min(1e-8)
                                tgt_norm = tgt_mat.norm(dim=1).clamp_min(1e-8)
                                pred_unit = pred_mat / pred_norm.unsqueeze(1)
                                tgt_unit = tgt_mat / tgt_norm.unsqueeze(1)
                                cos = (pred_unit * tgt_unit).sum(dim=1).clamp(min=-1.0, max=1.0)
                                cos_mean = cos.mean(dim=1)
                                cos_abs_mean = cos.abs().mean(dim=1)
                                u = pred_unit[:, :, 0]
                                v = pred_unit[:, :, 1]
                                w = pred_unit[:, :, 2]
                                ortho = (
                                    (u * v).sum(dim=-1).abs()
                                    + (u * w).sum(dim=-1).abs()
                                    + (v * w).sum(dim=-1).abs()
                                )
                                return (
                                    float(cos_mean.mean().item()),
                                    float(cos_abs_mean.mean().item()),
                                    float(ortho.mean().item()),
                                )
                            except Exception:
                                return 0.0, 0.0, 0.0

                        try:
                            with step_metrics_csv.open("a", newline="") as f:
                                import csv as _csv
                                tr_cos, tr_cos_abs = _cos_mean(
                                    outputs.get("tr_pred") if isinstance(outputs, dict) else None,
                                    targets.get("tr") if isinstance(targets, dict) else None,
                                )
                                rot_frame_cos, rot_frame_cos_abs, rot_frame_ortho = _frame_metrics(
                                    outputs.get("rot_pred") if isinstance(outputs, dict) else None,
                                    targets.get("rot_mat") if isinstance(targets, dict) and targets.get("rot_mat") is not None else targets.get("rot") if isinstance(targets, dict) else None,
                                )
                                if rot_frame_cos is not None:
                                    val_frame_cos_sum += float(rot_frame_cos)
                                    val_frame_cos_abs_sum += float(rot_frame_cos_abs)
                                    val_frame_ortho_sum += float(rot_frame_ortho)
                                    val_frame_batches += 1
                                w = _csv.DictWriter(
                                    f,
                                    fieldnames=[
                                        "epoch",
                                        "step",
                                        "global_step",
                                        "loss",
                                        "tr",
                                    "rot",
                                    "tor_bb",
                                    "tor_sc",
                                    "clash",
                                    "clash_min_dist",
                                    "clash_collide_ratio",
                                    "t",
                                        "tr_pred_norm",
                                        "rot_pred_norm",
                                        "tr_cos",
                                        "tr_cos_abs",
                                        # rot_cos/rot_cos_abs 已废弃，保留 frame 指标为准
                                        "rot_frame_cos",
                                        "rot_frame_cos_abs",
                                        "rot_frame_ortho",
                                        "tr_tgt_norm",
                                    "rot_tgt_norm",
                                    "ca_coord",
                                    "ca_coord_ratio",
                                        "ca_pred_norm",
                                        "ca_tgt_norm",
                                        "rot_mask_ratio",
                                        "pose_align",
                                    "triplet_align",
                                    "rot_cls",
                                    "rot_cls_acc",
                                    "iface_contact",
                                    "iface_contact_acc",
                                    "iface_pairdist",
                                    "iface_pairs",
                                ],
                                )
                                if write_header:
                                    w.writeheader()
                                w.writerow(
                                    {
                                        "epoch": epoch + 1,
                                        "step": step + 1,
                                        "global_step": epoch * max(1, len(val_loader)) + step + 1,
                                        "loss": float(loss.item()),
                                        "tr": float(loss_dict.get("tr", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                    "rot": float(loss_dict.get("rot", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                    "tor_bb": float(loss_dict.get("tor_bb", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                    "tor_sc": float(loss_dict.get("tor_sc", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                    "clash": float(loss_dict.get("clash", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                    "clash_min_dist": float(loss_dict.get("clash_min_dist", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                    "clash_collide_ratio": float(loss_dict.get("clash_collide_ratio", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                    "t": float(time_samples.mean().item()) if time_samples is not None else 0.0,
                                        "tr_pred_norm": _vec_norm_mean(outputs.get("tr_pred") if isinstance(outputs, dict) else None),
                                        "rot_pred_norm": _vec_norm_mean(outputs.get("rot_pred") if isinstance(outputs, dict) else None),
                                        "tr_cos": tr_cos,
                                        "tr_cos_abs": tr_cos_abs,
                                        "rot_frame_cos": rot_frame_cos,
                                        "rot_frame_cos_abs": rot_frame_cos_abs,
                                        "rot_frame_ortho": rot_frame_ortho,
                                        "tr_tgt_norm": _vec_norm_mean(targets.get("tr") if isinstance(targets, dict) else None),
                                        "rot_tgt_norm": _vec_norm_mean(targets.get("rot") if isinstance(targets, dict) else None),
                                        "ca_coord": float(loss_dict.get("ca_coord", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                        "ca_coord_ratio": float(loss_dict.get("ca_coord_ratio", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                        "ca_pred_norm": _vec_norm_mean(outputs.get("ca_pred") if isinstance(outputs, dict) else None),
                                        "ca_tgt_norm": float(loss_dict.get("ca_tgt_norm", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                        "rot_mask_ratio": (
                                            float(loss_dict["rot_mask_ratio"].item())
                                            if isinstance(loss_dict, dict) and "rot_mask_ratio" in loss_dict
                                            else ""
                                        ),
                                        "pose_align": float(loss_dict.get("pose_align", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                        "triplet_align": float(loss_dict.get("triplet_align", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                        "rot_cls": float(loss_dict.get("rot_cls", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                        "rot_cls_acc": float(loss_dict.get("rot_cls_acc", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                        "iface_contact": float(loss_dict.get("iface_contact", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                        "iface_contact_acc": float(loss_dict.get("iface_contact_acc", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                        "iface_pairdist": float(loss_dict.get("iface_pairdist", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                        "iface_pairs": float(loss_dict.get("iface_pairs", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                                    }
                                )
                        except Exception as e:
                            tqdm.write(f"[WARN] failed to write val_step_metrics.csv: {e}")

                        comp_parts = []
                        if isinstance(loss_dict, dict):
                            for k in ("tr", "rot", "tor_bb", "tor_sc", "ca_coord", "iface_contact", "iface_pairdist"):
                                if k in loss_dict:
                                    comp_parts.append(f"{k}:{loss_dict[k].item():.4f}")
                        comp_str = " | " + " ".join(comp_parts) if comp_parts else ""
                        names_str = ""
                        try:
                            names = getattr(batch, "complex_name", None)
                            if names is not None:
                                if isinstance(names, (list, tuple)):
                                    names_str = ",".join([str(n) for n in names])
                                else:
                                    names_str = str(names)
                        except Exception:
                            names_str = ""
                        msg = f"[Val {epoch+1} Step {step+1}] loss {loss.item():.4f}{comp_str}"
                        if names_str:
                            msg += f" | pdb:{names_str}"
                        tqdm.write(msg)
                    val_loss_acc += loss.item()
                    val_loss_list.append(float(loss.item()))
                    if isinstance(loss_dict, dict):
                        for k, v in loss_dict.items():
                            val_comp_acc[k] = val_comp_acc.get(k, 0.0) + float(v.item())
                    val_batches += 1
            _restore_bn_state(bn_state)

            val_loss = val_loss_acc / max(1, val_batches)
            val_loss = reduce_tensor(val_loss, world_size)
            val_loss_dict = {k: (v / max(1, val_batches)) for k, v in val_comp_acc.items()}
            val_frame_cos_avg = None
            val_frame_cos_abs_avg = None
            val_frame_ortho_avg = None
            if val_frame_batches > 0:
                val_frame_cos_avg = val_frame_cos_sum / val_frame_batches
                val_frame_cos_abs_avg = val_frame_cos_abs_sum / val_frame_batches
                val_frame_ortho_avg = val_frame_ortho_sum / val_frame_batches
            val_loss_stats = {}
            if val_loss_list:
                loss_arr = np.asarray(val_loss_list, dtype=np.float64)
                val_loss_stats = {
                    "median": float(np.median(loss_arr)),
                    "p90": float(np.quantile(loss_arr, 0.9)),
                    "p99": float(np.quantile(loss_arr, 0.99)),
                    "max": float(loss_arr.max()),
                    "nonfinite": float(val_nonfinite),
                    "over_threshold": float(val_over_threshold),
                }
        elapsed = time.time() - start_time

        if local_rank == 0:
            if val_loader is None:
                print(
                    f"Epoch {epoch+1}/{cfg.n_epochs} | train loss {train_loss_avg:.4f} | val skipped | time {elapsed:.1f}s"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{cfg.n_epochs} | train loss {train_loss_avg:.4f} | val loss {val_loss:.4f} | time {elapsed:.1f}s"
                )
            # 追加简洁 CSV 记录，便于快速查看 epoch 级别指标
            metrics_csv = Path(args.log_dir) / "epoch_metrics.csv"
            metrics_csv.parent.mkdir(parents=True, exist_ok=True)
            try:
                metrics_fields = [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_clash",
                    "train_clash_min_dist",
                    "train_clash_collide_ratio",
                    "train_clash_local_rec_ratio",
                    "train_clash_local_pep_ratio",
                    "train_clash_adaptive_gate",
                    "train_clash_hard_overlap_gate",
                    "val_clash",
                    "val_clash_min_dist",
                    "val_clash_collide_ratio",
                    "val_clash_local_rec_ratio",
                    "val_clash_local_pep_ratio",
                    "val_clash_adaptive_gate",
                    "val_clash_hard_overlap_gate",
                    "time_sec",
                    "val_loss_median",
                    "val_loss_p90",
                    "val_loss_p99",
                    "val_loss_max",
                    "val_loss_nonfinite",
                    "val_loss_over_threshold",
                ]
                write_header = not metrics_csv.exists()
                if metrics_csv.exists():
                    import csv

                    with metrics_csv.open("r", newline="") as f:
                        reader = csv.reader(f)
                        existing_header = next(reader, None)
                    if existing_header:
                        merged = existing_header + [f for f in metrics_fields if f not in existing_header]
                        if merged != existing_header:
                            with metrics_csv.open("r", newline="") as f:
                                reader = csv.DictReader(f)
                                rows = list(reader)
                            with metrics_csv.open("w", newline="") as f:
                                w = csv.DictWriter(f, fieldnames=merged)
                                w.writeheader()
                                for row in rows:
                                    w.writerow(row)
                        metrics_fields = merged
                        write_header = not metrics_csv.exists()
                    else:
                        write_header = True
                with metrics_csv.open("a", newline="") as f:
                    import csv
                    w = csv.DictWriter(f, fieldnames=metrics_fields)
                    if write_header:
                        w.writeheader()
                    w.writerow(
                        {
                            "epoch": epoch + 1,
                            "train_loss": float(train_loss_avg),
                            "val_loss": float(val_loss) if val_loader is not None else "",
                            "train_clash": float(train_loss_dict.get("clash", 0.0)),
                            "train_clash_min_dist": float(train_loss_dict.get("clash_min_dist", 0.0)),
                            "train_clash_collide_ratio": float(train_loss_dict.get("clash_collide_ratio", 0.0)),
                            "train_clash_local_rec_ratio": float(train_loss_dict.get("clash_local_rec_ratio", 0.0)),
                            "train_clash_local_pep_ratio": float(train_loss_dict.get("clash_local_pep_ratio", 0.0)),
                            "train_clash_adaptive_gate": float(train_loss_dict.get("clash_adaptive_gate", 1.0)),
                            "train_clash_hard_overlap_gate": float(train_loss_dict.get("clash_hard_overlap_gate", 1.0)),
                            "val_clash": float(val_loss_dict.get("clash", 0.0)) if val_loader is not None else "",
                            "val_clash_min_dist": float(val_loss_dict.get("clash_min_dist", 0.0)) if val_loader is not None else "",
                            "val_clash_collide_ratio": float(val_loss_dict.get("clash_collide_ratio", 0.0)) if val_loader is not None else "",
                            "val_clash_local_rec_ratio": float(val_loss_dict.get("clash_local_rec_ratio", 0.0)) if val_loader is not None else "",
                            "val_clash_local_pep_ratio": float(val_loss_dict.get("clash_local_pep_ratio", 0.0)) if val_loader is not None else "",
                            "val_clash_adaptive_gate": float(val_loss_dict.get("clash_adaptive_gate", 1.0)) if val_loader is not None else "",
                            "val_clash_hard_overlap_gate": float(val_loss_dict.get("clash_hard_overlap_gate", 1.0)) if val_loader is not None else "",
                            "time_sec": float(elapsed),
                            "val_loss_median": val_loss_stats.get("median", "") if val_loader is not None else "",
                            "val_loss_p90": val_loss_stats.get("p90", "") if val_loader is not None else "",
                            "val_loss_p99": val_loss_stats.get("p99", "") if val_loader is not None else "",
                            "val_loss_max": val_loss_stats.get("max", "") if val_loader is not None else "",
                            "val_loss_nonfinite": val_loss_stats.get("nonfinite", "") if val_loader is not None else "",
                            "val_loss_over_threshold": val_loss_stats.get("over_threshold", "") if val_loader is not None else "",
                        }
                    )
            except Exception as e:
                tqdm.write(f"[WARN] failed to write epoch_metrics.csv: {e}")

            if sw_run is not None:
                log_dict = {
                    "train/loss": train_loss_avg,
                }
                if val_loader is not None:
                    log_dict["val/loss"] = val_loss
                    if isinstance(val_loss_stats, dict) and val_loss_stats:
                        log_dict["val/loss_median"] = val_loss_stats.get("median", 0.0)
                        log_dict["val/loss_p90"] = val_loss_stats.get("p90", 0.0)
                        log_dict["val/loss_p99"] = val_loss_stats.get("p99", 0.0)
                        log_dict["val/loss_max"] = val_loss_stats.get("max", 0.0)
                        log_dict["val/loss_nonfinite"] = val_loss_stats.get("nonfinite", 0.0)
                        log_dict["val/loss_over_threshold"] = val_loss_stats.get("over_threshold", 0.0)
                if train_frame_cos_avg is not None:
                    log_dict["train/rot_frame_cos"] = train_frame_cos_avg
                    log_dict["train/rot_frame_cos_abs"] = train_frame_cos_abs_avg
                    log_dict["train/rot_frame_ortho"] = train_frame_ortho_avg
                if val_frame_cos_avg is not None:
                    log_dict["val/rot_frame_cos"] = val_frame_cos_avg
                    log_dict["val/rot_frame_cos_abs"] = val_frame_cos_abs_avg
                    log_dict["val/rot_frame_ortho"] = val_frame_ortho_avg
                # 分项 loss（flow_matching_loss 返回四个分量）
                for prefix, ld in (("train", train_loss_dict), ("val", val_loss_dict)):
                    if isinstance(ld, dict):
                        for k in (
                            "tr",
                            "rot",
                            "tor_bb",
                            "tor_sc",
                            "ca_coord",
                            "clash",
                            "clash_min_dist",
                            "clash_collide_ratio",
                            "clash_local_rec_ratio",
                            "clash_local_pep_ratio",
                            "clash_adaptive_gate",
                            "clash_hard_overlap_gate",
                        ):
                            if k in ld:
                                v = ld[k]
                                log_dict[f"{prefix}/loss_{k}"] = (
                                    v.item() if torch.is_tensor(v) else float(v)
                                )
                log_dict["time/epoch_sec"] = elapsed
                if scheduler is not None:
                    last_lr = optimizer.param_groups[0]["lr"]
                    log_dict["lr"] = last_lr
                sw_run.log(log_dict, step=epoch + 1)

        if scheduler is not None:
            try:
                if scheduler_is_plateau:
                    metric = val_loss if val_loader is not None else float(train_loss_avg)
                    scheduler.step(metric)
            except Exception as e:
                if local_rank == 0:
                    tqdm.write(f"[WARN] scheduler step failed: {e}")

        # 可选推理评估（小子集）
        if (
            local_rank == 0
            and args.eval_every > 0
            and args.eval_csv
            and ((epoch + 1) % args.eval_every == 0)
        ):
            if consecutive_eval_fail >= 2:
                tqdm.write(f"[WARN] eval skipped at epoch {epoch+1} due to consecutive failures")
                torch.cuda.empty_cache()
                continue
            eval_dir = Path(args.log_dir) / f"eval_epoch_{epoch+1}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            tmp_ckpt = (eval_dir / f"flow_{args.embedding}_eval_tmp.pt").resolve()
            eval_ckpt = {
                "epoch": epoch + 1,
                "model": maybe_unwrap(model).state_dict(),
                "config": vars(cfg),
            }
            if ema is not None:
                eval_ckpt["ema_weights"] = ema.state_dict()
            torch.save(eval_ckpt, tmp_ckpt)
            tqdm.write(f"[eval] saved tmp ckpt {tmp_ckpt}")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path.cwd())
            env.setdefault("TORUS_CACHE_DIR", str(Path.cwd()))
            # 准备评估输入 CSV，若缺少描述列则补空
            eval_input_csv = eval_dir / "eval_input.csv"
            eval_rows = 0
            try:
                df_eval = pd.read_csv(args.eval_csv)
                df_eval = df_eval.fillna("")
                eval_rows = len(df_eval)
                if "complex_name" not in df_eval.columns and "pdb_id" in df_eval.columns:
                    df_eval["complex_name"] = df_eval["pdb_id"]
                for col in ["protein_description", "peptide_description"]:
                    if col not in df_eval.columns:
                        df_eval[col] = ""
                # 若描述为空且有 pdb 路径，则直接用 pdb 路径避免 inference 误判去跑 ESMFold
                for desc_col, pdb_col in [
                    ("protein_description", "receptor_pdb"),
                    ("peptide_description", "peptide_pdb"),
                ]:
                    if pdb_col in df_eval.columns:
                        df_eval.loc[df_eval[desc_col] == "", desc_col] = df_eval[pdb_col]
                df_eval.to_csv(eval_input_csv, index=False)
            except Exception as e:
                tqdm.write(f"[WARN] failed to prepare eval CSV: {e}")
                eval_input_csv = Path(args.eval_csv)
            # 推理
            infer_cpu = max(args.eval_cpu or 1, 1)
            infer_bs = args.eval_batch_size or 1
            infer_t0 = time.time()
            infer_cmd = [
                "python",
                "inference.py",
                "--config",
                args.eval_infer_config,
                "--protein_peptide_csv",
                str(eval_input_csv),
                "--output_dir",
                str(eval_dir.resolve()),
                "--ckpt",
                str(tmp_ckpt),
                "--model_dir",
                args.model_dir,
                "--batch_size",
                str(infer_bs),
                "--cpu",
                str(infer_cpu),
            ]
            # 超时：每条样本留 45s 余量，至少 600s
            import math
            timeout_eval = max(600, math.ceil((eval_rows or 1) * 45))
            try:
                subprocess.run(infer_cmd, check=True, env=env, timeout=timeout_eval)
                tqdm.write(f"[eval] inference bs={infer_bs} cpu={infer_cpu} took {time.time()-infer_t0:.1f}s -> {eval_dir}")
            except Exception as e:
                tqdm.write(f"[WARN] eval inference failed at epoch {epoch+1}: {e}")
                consecutive_eval_fail += 1
                torch.cuda.empty_cache()
                continue
            # 评估 RMSD
            metrics_csv = eval_dir / "metrics.csv"
            dockq_cmd = shutil.which("dockq") if args.eval_dockq else None
            eval_cmd = [
                "python",
                "scripts/eval_rmsd_from_preds.py",
                "--pred_root",
                str(eval_dir),
                "--csv",
                args.eval_csv,
                "--output",
                str(metrics_csv),
            ]
            if dockq_cmd:
                eval_cmd += ["--dockq_cmd", dockq_cmd]
            try:
                subprocess.run(eval_cmd, check=True, env=env, timeout=timeout_eval)
                rows = []
                if metrics_csv.is_file():
                    with open(metrics_csv, "r", newline="") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                if rows:
                    rmsd_key = "complex_rmsd" if "complex_rmsd" in rows[0] else "ca_rmsd"
                    vals = [float(r[rmsd_key]) for r in rows if r.get(rmsd_key) not in (None, "", "nan")]
                    if vals:
                        rmsd_mean = sum(vals) / len(vals)
                        rmsd_min = min(vals)
                        tqdm.write(f"[eval] epoch {epoch+1} {rmsd_key} mean {rmsd_mean:.3f} min {rmsd_min:.3f}")
                        if sw_run is not None:
                            sw_run.log({f"eval/{rmsd_key}_mean": rmsd_mean, f"eval/{rmsd_key}_min": rmsd_min}, step=epoch + 1)
                    dockq_vals = [float(r["dockq"]) for r in rows if r.get("dockq") not in (None, "", "nan")]
                    if dockq_vals:
                        dockq_mean = sum(dockq_vals) / len(dockq_vals)
                        dockq_max = max(dockq_vals)
                        tqdm.write(f"[eval] epoch {epoch+1} DockQ mean {dockq_mean:.3f} max {dockq_max:.3f}")
                        if sw_run is not None:
                            sw_run.log({"eval/dockq_mean": dockq_mean, "eval/dockq_max": dockq_max}, step=epoch + 1)
                else:
                    tqdm.write(f"[WARN] eval metrics empty at epoch {epoch+1}")
            except Exception as e:
                tqdm.write(f"[WARN] eval rmsd failed at epoch {epoch+1}: {e}")
                consecutive_eval_fail += 1
                torch.cuda.empty_cache()
                continue

            # 成功则清零失败计数
            consecutive_eval_fail = 0

        # 保存最优 ckpt（默认按 val_loss；skip_val 时按 train_loss）
        metric = float(val_loss) if val_loader is not None else float(train_loss_avg)
        if local_rank == 0 and metric < best_metric:
            best_metric = metric
            best_payload = {
                "epoch": epoch + 1,
                "model": maybe_unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict(),
                "config": vars(cfg),
                "best_metric": best_metric,
            }
            if ema is not None:
                best_payload["ema_weights"] = ema.state_dict()
            torch.save(best_payload, best_ckpt_path)
            if val_loader is None:
                tqdm.write(f"[best] epoch {epoch+1} train_loss {best_metric:.4f} -> {best_ckpt_path}")
            else:
                tqdm.write(f"[best] epoch {epoch+1} val_loss {best_metric:.4f} -> {best_ckpt_path}")

        if local_rank == 0 and ((epoch + 1) % args.save_every == 0 or epoch + 1 == cfg.n_epochs):
            ckpt_path = os.path.join(args.log_dir, f"flow_{args.embedding}_epoch{epoch+1}.pt")
            epoch_payload = {
                "epoch": epoch + 1,
                "model": maybe_unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict(),
                "config": vars(cfg),
            }
            if ema is not None:
                epoch_payload["ema_weights"] = ema.state_dict()
            torch.save(epoch_payload, ckpt_path)
    if sw_run is not None and local_rank == 0:
        sw_run.finish()
    if local_rank == 0 and args.bark_id and args.bark_min_minutes > 0:
        elapsed = time.time() - t_main_start
        if elapsed >= args.bark_min_minutes * 60.0:
            cmd_str = _format_cmd(sys.argv)
            duration = _format_duration(elapsed)
            msg = f"{cmd_str}已完成，耗时{duration}。"
            try:
                _send_bark(args.bark_id, "FlowPepDock", msg)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Bark notify failed: {exc}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
