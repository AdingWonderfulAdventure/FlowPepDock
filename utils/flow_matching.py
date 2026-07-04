##########################################################################
# File Name: flow_matching.py
# Author: FlowPepDock contributors
# Created Time: Tue 10 Dec 2024 10:00:00 AM CST
#########################################################################

"""
Rectified Flow Matching utilities.

这些工具函数把用户写在AGENTS.md里的规范落地：在不改模型forward的情况下，
根据干净姿态采样(x0, x1, x_t)，并计算flow loss需要的目标速度。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean

from utils.geometry import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    kabsch_torch,
    matrix_to_axis_angle,
    rot6d_to_matrix,
)
from utils.steric import apply_rigid_body_updates, resolve_time_scale, steric_penalty_from_positions


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """把角度裁剪到[-pi, pi]，避免torsion爆炸。"""
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def _expand_time(
    t: torch.Tensor, target: torch.Tensor, splits: Optional[Sequence[int]] = None
) -> torch.Tensor:
    """
    Broadcast per-graph time t到任意形状的目标张量。
    splits: list of counts per graph，用于torsion这种每个complex数量不同的自由度。
    """
    if splits is not None:
        device = t.device
        repeats = torch.as_tensor(splits, device=device, dtype=torch.long)
        time_vector = torch.repeat_interleave(t, repeats)
    else:
        time_vector = t
    while time_vector.ndim < target.ndim:
        time_vector = time_vector.unsqueeze(-1)
    return time_vector


def _geodesic_rot_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mat: Optional[torch.Tensor],
    time_weights: Optional[torch.Tensor],
    mode: str,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    pred_mat = axis_angle_to_matrix(pred)
    if target_mat is None:
        if target is None:
            return torch.tensor(0.0, device=pred.device)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        target_mat = axis_angle_to_matrix(target)
    if target_mat.ndim == 2:
        target_mat = target_mat.unsqueeze(0)
    rel = torch.matmul(pred_mat, target_mat.transpose(-1, -2))
    trace = torch.diagonal(rel, dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
    angle = torch.acos(cos)
    per = angle.pow(2)
    if mask is not None:
        if mask.numel() == 0 or mask.sum().item() == 0:
            return torch.tensor(0.0, device=pred.device)
        per = per[mask]
        if time_weights is not None:
            time_weights = time_weights[mask]
    if time_weights is None or not mode:
        return per.mean()
    if mode == "t":
        w = time_weights
    elif mode in {"t2", "t^2"}:
        w = time_weights * time_weights
    elif mode in {"sqrt_t", "sqrt"}:
        w = torch.sqrt(time_weights)
    else:
        raise ValueError(f"[flow] 不支持的 time_weighting[rot]={mode}（支持 t/t2/sqrt_t 或留空）")
    return (per * w).mean()


def _reduce_per_sample(
    per: torch.Tensor,
    time_weights: Optional[torch.Tensor],
    mode: str,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if mask is not None:
        if mask.numel() == 0 or mask.sum().item() == 0:
            return torch.tensor(0.0, device=per.device)
        per = per[mask]
        if time_weights is not None:
            time_weights = time_weights[mask]
    if time_weights is None or not mode:
        return per.mean()
    if mode == "t":
        w = time_weights
    elif mode in {"t2", "t^2"}:
        w = time_weights * time_weights
    elif mode in {"sqrt_t", "sqrt"}:
        w = torch.sqrt(time_weights)
    else:
        raise ValueError(f"[flow] 不支持的 time_weighting[rot]={mode}（支持 t/t2/sqrt_t 或留空）")
    return (per * w).mean()


def _quat_rot_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mat: Optional[torch.Tensor],
    time_weights: Optional[torch.Tensor],
    mode: str,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    pred_q = axis_angle_to_quaternion(pred)
    if target_mat is None:
        if target is None:
            return torch.tensor(0.0, device=pred.device)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        target_q = axis_angle_to_quaternion(target)
    else:
        target_q = axis_angle_to_quaternion(matrix_to_axis_angle(target_mat))
    pred_q = pred_q / pred_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    target_q = target_q / target_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dot = (pred_q * target_q).sum(dim=-1).abs().clamp_max(1.0)
    per = 1.0 - dot
    return _reduce_per_sample(per, time_weights, mode, mask)


def _axis_angle_sep_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    time_weights: Optional[torch.Tensor],
    mode: str,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    pred_angle = pred.norm(dim=-1)
    target_angle = target.norm(dim=-1)
    pred_axis = pred / pred_angle.clamp_min(1e-8).unsqueeze(-1)
    target_axis = target / target_angle.clamp_min(1e-8).unsqueeze(-1)
    axis_cos = (pred_axis * target_axis).sum(dim=-1).clamp(min=-1.0, max=1.0)
    axis_loss = 1.0 - axis_cos
    angle_loss = (pred_angle - target_angle).abs()
    per = axis_loss + angle_loss
    small = (pred_angle < 1e-6) & (target_angle < 1e-6)
    per = torch.where(small, torch.zeros_like(per), per)
    return _reduce_per_sample(per, time_weights, mode, mask)


def _dual_axis_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mat: Optional[torch.Tensor],
    time_weights: Optional[torch.Tensor],
    mode: str,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    pred_mat = axis_angle_to_matrix(pred)
    if target_mat is None:
        if target is None:
            return torch.tensor(0.0, device=pred.device)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        target_mat = axis_angle_to_matrix(target)
    if target_mat.ndim == 2:
        target_mat = target_mat.unsqueeze(0)
    pred_x = pred_mat[:, :, 0]
    pred_y = pred_mat[:, :, 1]
    tgt_x = target_mat[:, :, 0]
    tgt_y = target_mat[:, :, 1]
    cos_x = (pred_x * tgt_x).sum(dim=-1).clamp(min=-1.0, max=1.0)
    cos_y = (pred_y * tgt_y).sum(dim=-1).clamp(min=-1.0, max=1.0)
    per = (1.0 - cos_x) + (1.0 - cos_y)
    return _reduce_per_sample(per, time_weights, mode, mask)


def _frame_vector_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    time_weights: Optional[torch.Tensor],
    mode: str,
    mask: Optional[torch.Tensor],
    ortho_weight: float = 0.1,
    axis_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if pred.shape[-1] == 3:
        pred_mat = axis_angle_to_matrix(pred)
    elif pred.shape[-1] == 6:
        pred_mat = rot6d_to_matrix(pred)
    elif pred.shape[-1] == 9:
        pred_mat = pred.reshape(-1, 3, 3)
    else:
        return torch.tensor(0.0, device=pred.device)
    if target.shape[-1] == 3:
        target_mat = axis_angle_to_matrix(target)
    elif target.shape[-1] == 9:
        target_mat = target.reshape(-1, 3, 3)
    elif target.shape[-1] == 6:
        target_mat = rot6d_to_matrix(target)
    else:
        return torch.tensor(0.0, device=pred.device)
    pred_norm = pred_mat.norm(dim=1, keepdim=False).clamp_min(1e-8)
    target_norm = target_mat.norm(dim=1, keepdim=False).clamp_min(1e-8)
    pred_unit = pred_mat / pred_norm.unsqueeze(1)
    target_unit = target_mat / target_norm.unsqueeze(1)
    cos = (pred_unit * target_unit).sum(dim=1).clamp(min=-1.0, max=1.0)
    per = 1.0 - cos
    if axis_weights is not None:
        if axis_weights.numel() != 3:
            axis_weights = None
        else:
            w = axis_weights.to(dtype=per.dtype, device=per.device).reshape(1, 3)
            denom = w.sum().clamp_min(1e-6)
            per = (per * w).sum(dim=1) / denom
    if axis_weights is None:
        per = per.mean(dim=1)
    if ortho_weight > 0:
        u = pred_unit[:, :, 0]
        v = pred_unit[:, :, 1]
        w = pred_unit[:, :, 2]
        ortho = (
            (u * v).sum(dim=-1).abs()
            + (u * w).sum(dim=-1).abs()
            + (v * w).sum(dim=-1).abs()
        )
        per = per + ortho_weight * ortho
    return _reduce_per_sample(per, time_weights, mode, mask)


def _vmf_axis_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    time_weights: Optional[torch.Tensor],
    mode: str,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    pred_angle = pred.norm(dim=-1)
    target_angle = target.norm(dim=-1)
    pred_axis = pred / pred_angle.clamp_min(1e-8).unsqueeze(-1)
    target_axis = target / target_angle.clamp_min(1e-8).unsqueeze(-1)
    axis_cos = (pred_axis * target_axis).sum(dim=-1).clamp(min=-1.0, max=1.0)
    axis_loss = 1.0 - axis_cos
    angle_loss = (pred_angle - target_angle).abs()
    per = axis_loss + angle_loss
    small = (pred_angle < 1e-6) & (target_angle < 1e-6)
    per = torch.where(small, torch.zeros_like(per), per)
    return _reduce_per_sample(per, time_weights, mode, mask)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = torch.unbind(q1, -1)
    w2, x2, y2, z2 = torch.unbind(q2, -1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _swing_twist_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mat: Optional[torch.Tensor],
    axis_ref: Optional[torch.Tensor],
    time_weights: Optional[torch.Tensor],
    mode: str,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if axis_ref is None:
        return torch.tensor(0.0, device=pred.device)
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    if axis_ref.ndim == 1:
        axis_ref = axis_ref.unsqueeze(0)
    axis_ref = axis_ref / axis_ref.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    pred_q = axis_angle_to_quaternion(pred)
    if target_mat is None:
        if target is None:
            return torch.tensor(0.0, device=pred.device)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        target_q = axis_angle_to_quaternion(target)
    else:
        target_q = axis_angle_to_quaternion(matrix_to_axis_angle(target_mat))

    def _decompose(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = q[:, :1]
        v = q[:, 1:]
        proj = axis_ref * (v * axis_ref).sum(dim=-1, keepdim=True)
        q_twist = torch.cat([w, proj], dim=-1)
        q_twist = q_twist / q_twist.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        q_twist_conj = q_twist.clone()
        q_twist_conj[:, 1:] = -q_twist_conj[:, 1:]
        q_swing = _quat_mul(q, q_twist_conj)
        swing_axis = q_swing[:, 1:]
        swing_norm = swing_axis.norm(dim=-1)
        swing_axis = swing_axis / swing_norm.clamp_min(1e-8).unsqueeze(-1)
        twist_angle = 2.0 * torch.atan2(
            q_twist[:, 1:].norm(dim=-1), q_twist[:, 0].abs() + 1e-8
        )
        return swing_axis, twist_angle, swing_norm

    swing_p, twist_p, swing_norm_p = _decompose(pred_q)
    swing_t, twist_t, swing_norm_t = _decompose(target_q)
    cos_axis = (swing_p * swing_t).sum(dim=-1).clamp(min=-1.0, max=1.0)
    axis_loss = 1.0 - cos_axis
    angle_loss = (twist_p - twist_t).abs()
    per = axis_loss + angle_loss
    small = (swing_norm_p < 1e-6) | (swing_norm_t < 1e-6)
    per = torch.where(small, torch.zeros_like(per), per)
    return _reduce_per_sample(per, time_weights, mode, mask)


def _coord_align_loss(
    pep_pos: torch.Tensor,
    pep_orig_pos: torch.Tensor,
    pep_batch: torch.Tensor,
    atom2atomid_index: torch.Tensor,
    tr_pred: torch.Tensor,
    rot_pred: torch.Tensor,
    time_samples: torch.Tensor,
) -> torch.Tensor:
    if pep_pos is None or pep_orig_pos is None or pep_batch is None:
        return torch.tensor(0.0, device=tr_pred.device)
    if pep_pos.numel() == 0:
        return torch.tensor(0.0, device=tr_pred.device)
    if time_samples is None:
        return torch.tensor(0.0, device=tr_pred.device)
    t = time_samples[pep_batch].unsqueeze(-1)
    center = scatter_mean(pep_pos, pep_batch, dim=0)
    center_atoms = center[pep_batch]
    tr_update = tr_pred[pep_batch] * t
    rot_update = rot_pred[pep_batch] * t
    rot_mat = axis_angle_to_matrix(rot_update)
    rel = pep_pos - center_atoms
    pred_pos = torch.einsum("ni,nij->nj", rel, rot_mat) + center_atoms - tr_update
    if atom2atomid_index is None:
        mask = torch.ones(pred_pos.shape[0], dtype=torch.bool, device=pred_pos.device)
    else:
        bb_mask = torch.isin(atom2atomid_index, torch.tensor([0, 1, 2, 3], device=pred_pos.device))
        mask = bb_mask
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=pred_pos.device)
    diff = pred_pos[mask] - pep_orig_pos[mask]
    return diff.pow(2).mean()


def _ca_coord_loss(
    ca_pred: torch.Tensor,
    pep_pos: torch.Tensor,
    pep_orig_pos: torch.Tensor,
    atom2res_index: torch.Tensor,
    atom2atomid_index: torch.Tensor,
    pep_atom_batch: torch.Tensor,
    pep_res_batch: torch.Tensor,
    time_samples: torch.Tensor,
    target_mode: str,
) -> Tuple[torch.Tensor, float, float, float]:
    if ca_pred is None or pep_pos is None or pep_orig_pos is None:
        return torch.tensor(0.0, device=pep_pos.device if pep_pos is not None else ca_pred.device), 0.0, 0.0, 0.0
    if atom2res_index is None or atom2atomid_index is None or pep_atom_batch is None or pep_res_batch is None:
        return torch.tensor(0.0, device=ca_pred.device), 0.0, 0.0, 0.0
    if ca_pred.numel() == 0:
        return torch.tensor(0.0, device=ca_pred.device), 0.0, 0.0, 0.0
    if time_samples is None or time_samples.numel() == 0:
        return torch.tensor(0.0, device=ca_pred.device), 0.0, 0.0, 0.0

    ca_mask = atom2atomid_index == 1
    pred_list = []
    target_list = []
    total_res = 0
    used_res = 0
    num_graphs = int(pep_res_batch.max().item()) + 1 if pep_res_batch.numel() > 0 else 0
    for g in range(num_graphs):
        res_mask = pep_res_batch == g
        if res_mask.sum().item() == 0:
            continue
        atom_mask = (pep_atom_batch == g) & ca_mask
        if atom_mask.sum().item() == 0:
            continue
        res_idx = atom2res_index[atom_mask]
        order = torch.argsort(res_idx)
        cur_ca = pep_pos[atom_mask][order]
        orig_ca = pep_orig_pos[atom_mask][order]
        ca_pred_g = ca_pred[res_mask]
        total_res += int(ca_pred_g.shape[0])
        if ca_pred_g.shape[0] != cur_ca.shape[0]:
            continue
        used_res += int(ca_pred_g.shape[0])
        t_val = float(time_samples[g].item())
        time_scale = t_val if str(target_mode or "velocity").lower() == "velocity" else (1.0 - t_val)
        if time_scale < 1e-6:
            time_scale = 1e-6
        target = (orig_ca - cur_ca) / time_scale
        pred_list.append(ca_pred_g)
        target_list.append(target)

    if not pred_list:
        return torch.tensor(0.0, device=ca_pred.device), 0.0, 0.0, 0.0
    pred_cat = torch.cat(pred_list, dim=0)
    target_cat = torch.cat(target_list, dim=0)
    loss = F.mse_loss(pred_cat, target_cat)
    ratio = float(used_res) / float(max(1, total_res))
    pred_norm = float(torch.linalg.vector_norm(pred_cat, dim=1).mean().item()) if pred_cat.numel() else 0.0
    tgt_norm = float(torch.linalg.vector_norm(target_cat, dim=1).mean().item()) if target_cat.numel() else 0.0
    return loss, ratio, pred_norm, tgt_norm


def _resolve_sigma_max(flow_cfg: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """根据配置推断四种自由度的最大噪声幅度。"""
    sigma_tuple = (
        flow_cfg.get("sigma_tr_max"),
        flow_cfg.get("sigma_rot_max"),
        flow_cfg.get("sigma_tor_bb_max"),
        flow_cfg.get("sigma_tor_sc_max"),
    )

    names = ("tr", "rot", "tor_backbone", "tor_sidechain")
    result = {}
    for name, sigma_value in zip(names, sigma_tuple):
        if sigma_value is None:
            raise ValueError(
                f"[flow] 缺少{name}的sigma_max，请在 flow 配置里补齐。"
            )
        result[name] = torch.as_tensor(sigma_value, device=device, dtype=torch.float32)
    return result


def sample_rectified_flow_batch(
    clean_states: Dict[str, Optional[torch.Tensor]],
    flow_cfg: Dict,
    torsion_splits: Optional[Dict[str, Sequence[int]]] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, Optional[torch.Tensor]], torch.Tensor]:
    """
    采样Rectified Flow状态：
        x0 -> x1 -> x_t，并返回目标速度v* = x1 - x0。

    Args:
        clean_states: {'tr': Tensor[B,3], 'rot': Tensor[B,3], 'tor_backbone': Tensor[sum T_bb], ...}
            任意缺失自由度可填None。
        flow_cfg: args.flow配置字典。
        torsion_splits: {'backbone': [T_bb1, T_bb2, ...], 'sidechain': [...]}，帮助把t广播到torsion上。
        generator: torch.Generator，用于可复现采样。

    Returns:
        x_t_states, target_velocities, time_samples
    """
    device = next(
        (tensor.device for tensor in clean_states.values() if tensor is not None),
        torch.device("cpu"),
    )
    batch_size = (
        clean_states["tr"].shape[0]
        if clean_states.get("tr") is not None
        else clean_states["rot"].shape[0]
    )
    time_sampling = flow_cfg.get("time_sampling", "uniform")
    if time_sampling != "uniform":
        raise NotImplementedError(f"暂时只支持uniform时间采样，收到{time_sampling}")
    t = torch.rand(batch_size, device=device, generator=generator)
    sigma_max = _resolve_sigma_max(flow_cfg, device)

    torsion_splits = torsion_splits or {}
    backbone_splits = torsion_splits.get("backbone")
    sidechain_splits = torsion_splits.get("sidechain")

    # 终点x1
    noisy_states = {}
    if clean_states.get("tr") is not None:
        eps = torch.randn_like(clean_states["tr"], generator=generator)
        noisy_states["tr"] = clean_states["tr"] + sigma_max["tr"] * eps
    else:
        noisy_states["tr"] = None

    if clean_states.get("rot") is not None:
        eps = torch.randn_like(clean_states["rot"], generator=generator)
        noisy_states["rot"] = clean_states["rot"] + sigma_max["rot"] * eps
    else:
        noisy_states["rot"] = None

    if clean_states.get("tor_backbone") is not None:
        rand = torch.rand_like(clean_states["tor_backbone"], generator=generator)
        delta = (rand * 2 - 1) * sigma_max["tor_backbone"]
        noisy_states["tor_backbone"] = wrap_to_pi(clean_states["tor_backbone"] + delta)
    else:
        noisy_states["tor_backbone"] = None

    if clean_states.get("tor_sidechain") is not None:
        rand = torch.rand_like(clean_states["tor_sidechain"], generator=generator)
        delta = (rand * 2 - 1) * sigma_max["tor_sidechain"]
        noisy_states["tor_sidechain"] = wrap_to_pi(clean_states["tor_sidechain"] + delta)
    else:
        noisy_states["tor_sidechain"] = None

    # 构造x_t
    xt_states = {}
    if clean_states.get("tr") is not None:
        weight = _expand_time(t, clean_states["tr"])
        xt_states["tr"] = (1 - weight) * clean_states["tr"] + weight * noisy_states["tr"]
    else:
        xt_states["tr"] = None

    if clean_states.get("rot") is not None:
        weight = _expand_time(t, clean_states["rot"])
        xt_states["rot"] = (1 - weight) * clean_states["rot"] + weight * noisy_states["rot"]
    else:
        xt_states["rot"] = None

    if clean_states.get("tor_backbone") is not None:
        weight = _expand_time(t, clean_states["tor_backbone"], backbone_splits)
        xt_states["tor_backbone"] = wrap_to_pi(
            (1 - weight) * clean_states["tor_backbone"] + weight * noisy_states["tor_backbone"]
        )
    else:
        xt_states["tor_backbone"] = None

    if clean_states.get("tor_sidechain") is not None:
        weight = _expand_time(t, clean_states["tor_sidechain"], sidechain_splits)
        xt_states["tor_sidechain"] = wrap_to_pi(
            (1 - weight) * clean_states["tor_sidechain"] + weight * noisy_states["tor_sidechain"]
        )
    else:
        xt_states["tor_sidechain"] = None

    # 目标速度
    targets = {}
    targets["tr"] = (
        noisy_states["tr"] - clean_states["tr"] if clean_states.get("tr") is not None else None
    )
    targets["rot"] = (
        noisy_states["rot"] - clean_states["rot"] if clean_states.get("rot") is not None else None
    )
    targets["tor_backbone"] = (
        wrap_to_pi(noisy_states["tor_backbone"] - clean_states["tor_backbone"])
        if clean_states.get("tor_backbone") is not None
        else None
    )
    targets["tor_sidechain"] = (
        wrap_to_pi(noisy_states["tor_sidechain"] - clean_states["tor_sidechain"])
        if clean_states.get("tor_sidechain") is not None
        else None
    )

    return xt_states, targets, t


def _pose_align_loss(
    pep_pos: torch.Tensor,
    pep_orig_pos: torch.Tensor,
    pep_batch: torch.Tensor,
    tr_pred: torch.Tensor,
    rot_pred: torch.Tensor,
    time_samples: torch.Tensor,
    target_mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pep_pos is None or pep_orig_pos is None or pep_batch is None:
        return torch.tensor(0.0, device=tr_pred.device)
    if pep_pos.numel() == 0 or pep_orig_pos.numel() == 0:
        return torch.tensor(0.0, device=tr_pred.device)
    if time_samples.ndim == 2 and time_samples.shape[1] == 1:
        time_samples = time_samples.squeeze(1)
    time_scale = time_samples
    if target_mode != "velocity":
        time_scale = 1.0 - time_scale
    time_scale = time_scale.clamp_min(eps)
    tr_update = tr_pred * time_scale[:, None]
    rot_update = rot_pred * time_scale[:, None]
    rot_mat = axis_angle_to_matrix(rot_update)
    num_graphs = tr_pred.shape[0]
    centers = scatter_mean(pep_pos, pep_batch, dim=0, dim_size=num_graphs)
    pos_centered = pep_pos - centers[pep_batch]
    rot_mat_per = rot_mat[pep_batch]
    rotated = torch.einsum("nij,nj->ni", rot_mat_per, pos_centered)
    moved = rotated + centers[pep_batch] + tr_update[pep_batch]
    diff = moved - pep_orig_pos
    per_atom = diff.pow(2).sum(dim=-1)
    per_graph = scatter_mean(per_atom, pep_batch, dim=0, dim_size=num_graphs)
    # 归一化到肽自身尺度，避免对齐损失量级爆炸
    orig_centers = scatter_mean(pep_orig_pos, pep_batch, dim=0, dim_size=num_graphs)
    orig_centered = pep_orig_pos - orig_centers[pep_batch]
    scale_per_atom = orig_centered.pow(2).sum(dim=-1)
    scale_per_graph = scatter_mean(scale_per_atom, pep_batch, dim=0, dim_size=num_graphs).clamp_min(eps)
    per_graph = per_graph / scale_per_graph
    return per_graph.mean()


def _kabsch_rot_loss(
    pep_pos: torch.Tensor,
    pep_orig_pos: torch.Tensor,
    pep_batch: torch.Tensor,
    atom2atomid_index: Optional[torch.Tensor],
    rot_pred: torch.Tensor,
    time_samples: torch.Tensor,
    target_mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pep_pos is None or pep_orig_pos is None or pep_batch is None or rot_pred is None:
        return torch.tensor(0.0, device=rot_pred.device if rot_pred is not None else pep_pos.device)
    if pep_pos.numel() == 0 or pep_orig_pos.numel() == 0:
        return torch.tensor(0.0, device=rot_pred.device)
    if time_samples.ndim == 2 and time_samples.shape[1] == 1:
        time_samples = time_samples.squeeze(1)
    time_scale = time_samples
    if target_mode != "velocity":
        time_scale = 1.0 - time_scale
    time_scale = time_scale.clamp_min(eps)
    if rot_pred.ndim == 1:
        rot_pred = rot_pred.unsqueeze(0)
    if rot_pred.shape[-1] != 3:
        return torch.tensor(0.0, device=rot_pred.device)
    rot_update = rot_pred * time_scale[:, None]

    num_graphs = int(pep_batch.max().item() + 1) if pep_batch.numel() > 0 else rot_pred.shape[0]
    target_mats = []
    valid_mask = torch.zeros((num_graphs,), dtype=torch.bool, device=rot_pred.device)
    for g in range(num_graphs):
        mask = pep_batch == g
        if mask.sum().item() < 3:
            target_mats.append(torch.eye(3, device=rot_pred.device))
            continue
        pos_g = pep_pos[mask]
        orig_g = pep_orig_pos[mask]
        if atom2atomid_index is not None:
            atom_ids = atom2atomid_index[mask]
            ca_mask = atom_ids == 1
        else:
            ca_mask = torch.zeros(pos_g.shape[0], dtype=torch.bool, device=rot_pred.device)
        if ca_mask.sum().item() >= 3:
            cur = pos_g[ca_mask]
            ref = orig_g[ca_mask]
        else:
            cur = pos_g
            ref = orig_g
        if cur.shape[0] < 3 or ref.shape[0] < 3:
            target_mats.append(torch.eye(3, device=rot_pred.device))
            continue
        try:
            R_hat, _t_hat = kabsch_torch(cur.T, ref.T)
            target_mats.append(R_hat.to(dtype=torch.float32))
            valid_mask[g] = True
        except Exception:
            target_mats.append(torch.eye(3, device=rot_pred.device))
            continue
    target_mat = torch.stack(target_mats, dim=0)
    loss = _geodesic_rot_loss(
        rot_update,
        target=None,
        target_mat=target_mat,
        time_weights=None,
        mode="",
        mask=valid_mask,
    )
    return loss


def _select_triplet_indices(
    pep_batch: torch.Tensor,
    atom2res_index: torch.Tensor,
    atom2atomid_index: torch.Tensor,
    n_graphs: int,
) -> torch.Tensor:
    """为每个graph选3个锚点（首/中/末残基的CA），返回全局索引。"""
    idx_list = []
    ca_mask = atom2atomid_index == 1
    for gid in range(n_graphs):
        mask = pep_batch == gid
        if mask.sum().item() == 0:
            continue
        local_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        local_ca = local_idx[ca_mask[mask]]
        if local_ca.numel() >= 3:
            res_idx = atom2res_index[local_ca]
            order = torch.argsort(res_idx)
            local_ca = local_ca[order]
            sel = torch.tensor(
                [0, local_ca.numel() // 2, local_ca.numel() - 1],
                device=local_ca.device,
                dtype=torch.long,
            )
            idx_list.append(local_ca[sel])
        else:
            # CA 不够就退化为任意原子（仍保证 3 个）
            if local_idx.numel() < 3:
                continue
            sel = torch.tensor(
                [0, local_idx.numel() // 2, local_idx.numel() - 1],
                device=local_idx.device,
                dtype=torch.long,
            )
            idx_list.append(local_idx[sel])
    if not idx_list:
        return torch.empty(0, dtype=torch.long, device=pep_batch.device)
    return torch.cat(idx_list, dim=0)


def _triplet_align_loss(
    pep_pos: torch.Tensor,
    pep_orig_pos: torch.Tensor,
    pep_batch: torch.Tensor,
    atom2res_index: torch.Tensor,
    atom2atomid_index: torch.Tensor,
    tr_pred: torch.Tensor,
    rot_pred: torch.Tensor,
    time_samples: torch.Tensor,
    target_mode: str = "velocity",
) -> torch.Tensor:
    """三锚点对齐损失：用首/中/末残基CA监督刚体旋转。"""
    if (
        pep_pos is None
        or pep_orig_pos is None
        or pep_batch is None
        or atom2res_index is None
        or atom2atomid_index is None
    ):
        return torch.tensor(0.0, device=tr_pred.device)
    if pep_pos.numel() == 0 or pep_orig_pos.numel() == 0:
        return torch.tensor(0.0, device=tr_pred.device)
    time_scale = time_samples.float().clamp_min(1e-6)
    if target_mode != "velocity":
        time_scale = 1.0 - time_scale
    time_scale = time_scale.clamp_min(1e-6)
    tr_update = tr_pred * time_scale[:, None]
    rot_update = rot_pred * time_scale[:, None]
    rot_mat = axis_angle_to_matrix(rot_update)
    num_graphs = tr_pred.shape[0]
    idx = _select_triplet_indices(pep_batch, atom2res_index, atom2atomid_index, num_graphs)
    if idx.numel() == 0:
        return torch.tensor(0.0, device=tr_pred.device)
    anchor_batch = pep_batch[idx]
    centers = scatter_mean(pep_pos, pep_batch, dim=0, dim_size=num_graphs)
    pos_centered = pep_pos[idx] - centers[anchor_batch]
    rot_mat_per = rot_mat[anchor_batch]
    rotated = torch.einsum("nij,nj->ni", rot_mat_per, pos_centered)
    moved = rotated + centers[anchor_batch] + tr_update[anchor_batch]
    diff = moved - pep_orig_pos[idx]
    per_atom = diff.pow(2).sum(dim=-1)
    per_graph = scatter_mean(per_atom, anchor_batch, dim=0, dim_size=num_graphs)
    # 归一化到肽自身尺度，避免对齐损失量级爆炸
    orig_centers = scatter_mean(pep_orig_pos, pep_batch, dim=0, dim_size=num_graphs)
    orig_centered = pep_orig_pos - orig_centers[pep_batch]
    scale_per_atom = orig_centered.pow(2).sum(dim=-1)
    scale_per_graph = scatter_mean(scale_per_atom, pep_batch, dim=0, dim_size=num_graphs).clamp_min(1e-6)
    per_graph = per_graph / scale_per_graph
    return per_graph.mean()


def _clash_loss(
    pep_pos: torch.Tensor,
    pep_batch: torch.Tensor,
    rec_pos: torch.Tensor,
    rec_batch: torch.Tensor,
    pep_orig_pos: Optional[torch.Tensor],
    tr_pred: torch.Tensor,
    rot_pred: Optional[torch.Tensor],
    time_samples: torch.Tensor,
    target_mode: str,
    min_dist: float,
    unroll_steps: int = 0,
    soft_dist: Optional[float] = None,
    soft_weight: float = 0.0,
    density_cutoff: Optional[float] = None,
    density_weight: float = 0.0,
    density_allowance: float = 2.5,
    density_temperature: float = 0.5,
    local_rec_radius: Optional[float] = None,
    local_pep_radius: Optional[float] = None,
    local_min_rec_atoms: int = 8,
    local_min_pep_atoms: int = 4,
    local_fallback_global: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """受体-肽最小距离惩罚，避免穿模。"""
    if (
        pep_pos is None
        or pep_batch is None
        or rec_pos is None
        or rec_batch is None
        or tr_pred is None
        or time_samples is None
    ):
        zero_device = tr_pred.device if tr_pred is not None else "cpu"
        zero = torch.tensor(0.0, device=zero_device)
        return zero, zero, zero
    if pep_pos.numel() == 0 or rec_pos.numel() == 0:
        zero = torch.tensor(0.0, device=pep_pos.device)
        return zero, zero, zero
    time_scale = resolve_time_scale(time_samples, target_mode)
    if time_scale is None:
        zero = torch.tensor(0.0, device=pep_pos.device)
        return zero, zero, zero
    if unroll_steps and unroll_steps > 0:
        step_scale = 1.0 / float(unroll_steps)
        time_scale = torch.full_like(time_scale, step_scale)

    tr_update = tr_pred * time_scale[:, None]
    use_rot = rot_pred is not None and rot_pred.ndim >= 1 and rot_pred.shape[-1] == 3
    rot_update = rot_pred * time_scale[:, None] if use_rot else None
    moved, _centers = apply_rigid_body_updates(pep_pos, pep_batch, tr_update, rot_update)
    stats = steric_penalty_from_positions(
        moved,
        pep_batch,
        rec_pos,
        rec_batch,
        min_dist=min_dist,
        soft_dist=soft_dist,
        soft_weight=soft_weight,
        density_cutoff=density_cutoff,
        density_weight=density_weight,
        density_allowance=density_allowance,
        density_temperature=density_temperature,
        ref_pep_pos=pep_orig_pos,
        ref_pep_batch=pep_batch if pep_orig_pos is not None else None,
        local_rec_radius=local_rec_radius,
        local_pep_radius=local_pep_radius,
        local_min_rec_atoms=local_min_rec_atoms,
        local_min_pep_atoms=local_min_pep_atoms,
        local_fallback_global=local_fallback_global,
    )
    if not torch.isfinite(stats["loss"]):
        zero = torch.tensor(0.0, device=pep_pos.device)
        return zero, zero, zero
    return stats["loss"], stats["min_dist"], stats["collide_ratio"]


def _select_pep_ca_indices(
    pep_atom_idx: torch.Tensor,
    atom2res_index: torch.Tensor,
    atom2atomid_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """选出每个残基的 CA 原子索引（每残基取第一个），返回 (ca_idx, res_idx)。"""
    if pep_atom_idx.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=pep_atom_idx.device), torch.empty(
            0, dtype=torch.long, device=pep_atom_idx.device
        )
    ca_mask = atom2atomid_index[pep_atom_idx] == 1
    ca_idx = pep_atom_idx[ca_mask]
    if ca_idx.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=pep_atom_idx.device), torch.empty(
            0, dtype=torch.long, device=pep_atom_idx.device
        )
    res_idx = atom2res_index[ca_idx]
    order = torch.argsort(res_idx)
    ca_idx = ca_idx[order]
    res_idx = res_idx[order]
    # 由于已排序，相同 res_idx 连续，取首个作为代表
    keep = torch.ones_like(res_idx, dtype=torch.bool)
    keep[1:] = res_idx[1:] != res_idx[:-1]
    return ca_idx[keep], res_idx[keep]


def _contact_loss(
    pep_pos: torch.Tensor,
    pep_orig_pos: torch.Tensor,
    pep_batch: torch.Tensor,
    rec_pos: torch.Tensor,
    rec_batch: torch.Tensor,
    atom2res_index: torch.Tensor,
    atom2atomid_index: torch.Tensor,
    tr_pred: torch.Tensor,
    rot_pred: Optional[torch.Tensor],
    time_samples: torch.Tensor,
    target_mode: str,
    cutoff: float,
    top_k: int,
    true_max: float,
    t_min: float,
    noncontact_weight: float,
    noncontact_min: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """接触对齐损失：用真值接触对约束预测姿态，支持可选非接触惩罚。"""
    if (
        pep_pos is None
        or pep_orig_pos is None
        or pep_batch is None
        or rec_pos is None
        or rec_batch is None
        or atom2res_index is None
        or atom2atomid_index is None
        or tr_pred is None
        or time_samples is None
    ):
        zero_device = tr_pred.device if tr_pred is not None else "cpu"
        zero = torch.tensor(0.0, device=zero_device)
        return zero, zero, zero, zero
    if pep_pos.numel() == 0 or rec_pos.numel() == 0:
        zero = torch.tensor(0.0, device=pep_pos.device)
        return zero, zero, zero, zero
    if time_samples.ndim == 2 and time_samples.shape[1] == 1:
        time_samples = time_samples.squeeze(1)
    time_scale = time_samples.float().clamp_min(eps)
    if str(target_mode or "velocity").lower() != "velocity":
        time_scale = (1.0 - time_scale).clamp_min(eps)

    num_graphs = tr_pred.shape[0]
    if t_min is None:
        t_min = 0.0
    t_min = float(t_min)
    valid_graph = time_samples >= t_min if t_min > 0 else torch.ones_like(time_samples, dtype=torch.bool)

    tr_update = tr_pred * time_scale[:, None]
    use_rot = rot_pred is not None and rot_pred.ndim >= 1 and rot_pred.shape[-1] == 3
    rot_update = rot_pred * time_scale[:, None] if use_rot else None
    rot_mat = axis_angle_to_matrix(rot_update) if use_rot else None

    centers = scatter_mean(pep_pos, pep_batch, dim=0, dim_size=num_graphs)
    pos_centered = pep_pos - centers[pep_batch]
    if rot_mat is not None:
        rot_mat_per = rot_mat[pep_batch]
        rotated = torch.einsum("nij,nj->ni", rot_mat_per, pos_centered)
    else:
        rotated = pos_centered
    moved = rotated + centers[pep_batch] + tr_update[pep_batch]

    per_graph_contact = []
    per_graph_contact_pairs = []
    per_graph_noncontact = []
    per_graph_noncontact_pairs = []
    for g in range(num_graphs):
        if not valid_graph[g].item():
            continue
        pep_mask = pep_batch == g
        rec_mask = rec_batch == g
        if not pep_mask.any() or not rec_mask.any():
            continue
        pep_atom_idx = torch.nonzero(pep_mask, as_tuple=False).squeeze(-1)
        rec_idx = torch.nonzero(rec_mask, as_tuple=False).squeeze(-1)
        if pep_atom_idx.numel() == 0 or rec_idx.numel() == 0:
            continue
        ca_idx, _res_idx = _select_pep_ca_indices(pep_atom_idx, atom2res_index, atom2atomid_index)
        if ca_idx.numel() == 0:
            continue
        pep_ca_orig = pep_orig_pos[ca_idx]
        pep_ca_pred = moved[ca_idx]
        rec_pos_g = rec_pos[rec_idx]
        if pep_ca_orig.numel() == 0 or rec_pos_g.numel() == 0:
            continue
        # 真值接触对：每个肽残基取 top-k 近邻受体残基，再做阈值筛选
        k = int(top_k) if top_k is not None else 0
        if k <= 0:
            continue
        k = min(k, rec_pos_g.shape[0])
        dmat = torch.cdist(pep_ca_orig, rec_pos_g)
        min_d, _ = dmat.min(dim=1)
        interface_mask = min_d <= float(true_max)
        if interface_mask.sum().item() == 0:
            continue
        pep_ca_orig = pep_ca_orig[interface_mask]
        pep_ca_pred = pep_ca_pred[interface_mask]
        dmat = dmat[interface_mask]
        k = min(k, dmat.shape[1])
        topk_d, topk_idx = torch.topk(dmat, k=k, dim=1, largest=False)
        contact_mask = topk_d <= float(cutoff)
        pep_ids = torch.arange(pep_ca_orig.shape[0], device=pep_ca_orig.device).unsqueeze(1).expand(-1, k)

        if contact_mask.any():
            pep_sel = pep_ids[contact_mask]
            rec_sel = topk_idx[contact_mask]
            pred_d = (pep_ca_pred[pep_sel] - rec_pos_g[rec_sel]).norm(dim=-1)
            per_graph_contact.append(F.relu(pred_d - float(cutoff)).mean())
            per_graph_contact_pairs.append(torch.tensor(float(pred_d.numel()), device=pred_d.device))
        else:
            per_graph_contact.append(torch.tensor(0.0, device=pep_ca_pred.device))
            per_graph_contact_pairs.append(torch.tensor(0.0, device=pep_ca_pred.device))

        if noncontact_weight > 0 and (~contact_mask).any():
            pep_sel = pep_ids[~contact_mask]
            rec_sel = topk_idx[~contact_mask]
            pred_d = (pep_ca_pred[pep_sel] - rec_pos_g[rec_sel]).norm(dim=-1)
            per_graph_noncontact.append(F.relu(float(noncontact_min) - pred_d).mean())
            per_graph_noncontact_pairs.append(torch.tensor(float(pred_d.numel()), device=pred_d.device))
        elif noncontact_weight > 0:
            per_graph_noncontact.append(torch.tensor(0.0, device=pep_ca_pred.device))
            per_graph_noncontact_pairs.append(torch.tensor(0.0, device=pep_ca_pred.device))

    if not per_graph_contact:
        zero = torch.tensor(0.0, device=pep_pos.device)
        return zero, zero, zero, zero
    contact_loss = torch.stack(per_graph_contact, dim=0).mean()
    contact_pairs = torch.stack(per_graph_contact_pairs, dim=0).mean()
    if per_graph_noncontact:
        noncontact_loss = torch.stack(per_graph_noncontact, dim=0).mean()
        noncontact_pairs = torch.stack(per_graph_noncontact_pairs, dim=0).mean()
    else:
        noncontact_loss = torch.tensor(0.0, device=pep_pos.device)
        noncontact_pairs = torch.tensor(0.0, device=pep_pos.device)
    return contact_loss, contact_pairs, noncontact_loss, noncontact_pairs


def _interface_aux_loss(
    edge_index: Optional[torch.Tensor],
    contact_logits: Optional[torch.Tensor],
    pairdist_pred: Optional[torch.Tensor],
    pep_res_orig_pos: Optional[torch.Tensor],
    pep_res_batch: Optional[torch.Tensor],
    rec_pos: Optional[torch.Tensor],
    rec_batch: Optional[torch.Tensor],
    contact_cutoff: float = 8.0,
    pairdist_max: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    训练期接口辅助损失（不影响推理更新）：
    - contact: cross-edge 二分类（是否小于 cutoff）
    - pairdist: cross-edge 距离回归（SmoothL1）
    """
    # 兜底 device
    device = None
    for x in (contact_logits, pairdist_pred, pep_res_orig_pos, rec_pos):
        if isinstance(x, torch.Tensor):
            device = x.device
            break
    if device is None:
        device = torch.device("cpu")

    zero = torch.tensor(0.0, device=device)
    if (
        edge_index is None
        or pep_res_orig_pos is None
        or pep_res_batch is None
        or rec_pos is None
        or rec_batch is None
    ):
        return zero, zero, zero, zero
    if edge_index.numel() == 0:
        return zero, zero, zero, zero

    src = edge_index[0].long()
    dst = edge_index[1].long()
    valid = (
        (src >= 0)
        & (dst >= 0)
        & (src < pep_res_orig_pos.shape[0])
        & (dst < rec_pos.shape[0])
        & (pep_res_batch[src] == rec_batch[dst])
    )
    if valid.sum().item() == 0:
        return zero, zero, zero, zero
    src = src[valid]
    dst = dst[valid]
    true_dist = (pep_res_orig_pos[src] - rec_pos[dst]).norm(dim=-1)
    true_contact = (true_dist <= float(contact_cutoff)).float()

    contact_loss = zero
    contact_acc = zero
    if contact_logits is not None:
        logits = contact_logits[valid]
        if logits.numel() > 0:
            contact_loss = F.binary_cross_entropy_with_logits(logits, true_contact)
            pred_contact = (torch.sigmoid(logits) >= 0.5).float()
            contact_acc = (pred_contact == true_contact).float().mean()

    pairdist_loss = zero
    if pairdist_pred is not None:
        pred_dist = pairdist_pred[valid].clamp_min(0.0)
        if pred_dist.numel() > 0:
            pairdist_loss = F.smooth_l1_loss(
                pred_dist.clamp_max(float(pairdist_max)),
                true_dist.clamp_max(float(pairdist_max)),
            )

    num_pairs = torch.tensor(float(src.numel()), device=device)
    return contact_loss, contact_acc, pairdist_loss, num_pairs


def flow_matching_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, Optional[torch.Tensor]],
    loss_weights: Dict[str, float],
    time_samples: Optional[torch.Tensor] = None,
    time_weighting: Optional[Dict[str, str]] = None,
    target_mode: str = "velocity",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    标准RF loss = Σ w_i * ||v_hat_i - v*_i||^2。
    predictions: {'tr_pred': ..., 'rot_pred': ..., ...}
    targets: {'tr': ..., 'rot': ..., ...}
    loss_weights: {'tr': float, 'rot': float, 'tor_bb': float, 'tor_sc': float}
    """
    first_pred = next(iter(predictions.values()))
    total_loss = torch.tensor(0.0, device=first_pred.device)
    loss_dict = {}
    # 平移分量做尺度归一，降低量级以稳定训练
    tr_scale = loss_weights.get("tr_scale", 10.0)
    key_map = {
        "tr": "tr_pred",
        "rot": "rot_pred",
        "tor_bb": "tor_pred_backbone",
        "tor_sc": "tor_pred_sidechain",
    }

    time_weights = None
    if time_samples is not None:
        # 允许 [B] 或 [B,1]
        if time_samples.ndim == 2 and time_samples.shape[1] == 1:
            time_samples = time_samples.squeeze(1)
        if time_samples.ndim != 1:
            raise ValueError(f"[flow] time_samples 需为 [B] 或 [B,1]，收到 shape={tuple(time_samples.shape)}")
        time_weights = time_samples.to(device=first_pred.device, dtype=torch.float32).clamp_min(1e-6)

    def _mse_with_time_weight(
        pred: torch.Tensor,
        target: torch.Tensor,
        short_name: str,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if time_weights is None or time_weighting is None:
            if mask is None:
                return F.mse_loss(pred, target)
            if mask.numel() == 0 or mask.sum().item() == 0:
                return torch.tensor(0.0, device=pred.device)
            return F.mse_loss(pred[mask], target[mask])
        mode = (time_weighting.get(short_name) or "").lower()
        if not mode:
            if mask is None:
                return F.mse_loss(pred, target)
            if mask.numel() == 0 or mask.sum().item() == 0:
                return torch.tensor(0.0, device=pred.device)
            return F.mse_loss(pred[mask], target[mask])
        # 只对“按 batch 对齐”的张量加权；torsion 是 [sum T]，不在这里瞎加权
        if pred.shape[0] != time_weights.shape[0]:
            return F.mse_loss(pred, target)
        if mode == "t":
            w = time_weights
        elif mode in {"t2", "t^2"}:
            w = time_weights * time_weights
        elif mode in {"sqrt_t", "sqrt"}:
            w = torch.sqrt(time_weights)
        else:
            raise ValueError(f"[flow] 不支持的 time_weighting[{short_name}]={mode}（支持 t/t2/sqrt_t 或留空）")
        # per-sample mse: mean over non-batch dims
        per = (pred - target).pow(2)
        while per.ndim > 1:
            per = per.mean(dim=-1)
        if mask is not None:
            if mask.numel() == 0 or mask.sum().item() == 0:
                return torch.tensor(0.0, device=pred.device)
            per = per[mask]
            w = w[mask]
        return (per * w).mean()

    for short_name, pred_key in key_map.items():
        if short_name == "tor_bb":
            target_key = "tor_backbone"
        elif short_name == "tor_sc":
            target_key = "tor_sidechain"
        else:
            target_key = short_name
        target = targets.get(target_key)
        if target is None or pred_key not in predictions:
            continue
        pred = predictions[pred_key]
        weight = loss_weights.get(short_name, 1.0)
        if short_name == "tr" and tr_scale and tr_scale > 0:
            pred = pred / tr_scale
            target = target / tr_scale
        if short_name == "rot":
            rot_scale = loss_weights.get("rot_target_scale", 1.0)
            if rot_scale and rot_scale > 0:
                target = target * rot_scale
        mask = None
        if short_name == "rot" and time_weights is not None:
            rot_t_min = loss_weights.get("rot_loss_t_min", None)
            rot_t_max = loss_weights.get("rot_loss_t_max", None)
            if rot_t_min is not None or rot_t_max is not None:
                mask = torch.ones_like(time_weights, dtype=torch.bool)
                if rot_t_min is not None:
                    mask = mask & (time_weights >= float(rot_t_min))
                if rot_t_max is not None:
                    mask = mask & (time_weights <= float(rot_t_max))
                loss_dict["rot_mask_ratio"] = mask.float().mean().detach()
            rot_valid = targets.get("rot_valid")
            if rot_valid is not None:
                rot_valid = rot_valid.to(dtype=torch.bool)
                mask = rot_valid if mask is None else (mask & rot_valid)
        if short_name == "rot":
            rot_loss_mode = str(loss_weights.get("rot_loss_mode", "mse") or "mse").lower()
        else:
            rot_loss_mode = "mse"
        if short_name == "rot" and rot_loss_mode in {"geodesic", "so3"}:
            rot_target_mat = targets.get("rot_mat")
            loss_val = _geodesic_rot_loss(
                pred,
                target,
                rot_target_mat,
                time_weights if short_name == "rot" else None,
                (time_weighting.get(short_name) or "").lower()
                if time_weighting is not None
                else "",
                mask,
            )
        elif short_name == "rot" and rot_loss_mode in {"quat", "quaternion"}:
            rot_target_mat = targets.get("rot_mat")
            loss_val = _quat_rot_loss(
                pred,
                target,
                rot_target_mat,
                time_weights if short_name == "rot" else None,
                (time_weighting.get(short_name) or "").lower()
                if time_weighting is not None
                else "",
                mask,
            )
        elif short_name == "rot" and rot_loss_mode in {"axis_angle_sep", "axis_sep"}:
            loss_val = _axis_angle_sep_loss(
                pred,
                target,
                time_weights if short_name == "rot" else None,
                (time_weighting.get(short_name) or "").lower()
                if time_weighting is not None
                else "",
                mask,
            )
        elif short_name == "rot" and rot_loss_mode in {"dual_axis", "dual_vec"}:
            rot_target_mat = targets.get("rot_mat")
            loss_val = _dual_axis_loss(
                pred,
                target,
                rot_target_mat,
                time_weights if short_name == "rot" else None,
                (time_weighting.get(short_name) or "").lower()
                if time_weighting is not None
                else "",
                mask,
            )
        elif short_name == "rot" and rot_loss_mode in {"swing_twist", "swingtwist"}:
            rot_target_mat = targets.get("rot_mat")
            rot_axis_ref = targets.get("rot_axis_ref")
            loss_val = _swing_twist_loss(
                pred,
                target,
                rot_target_mat,
                rot_axis_ref,
                time_weights if short_name == "rot" else None,
                (time_weighting.get(short_name) or "").lower()
                if time_weighting is not None
                else "",
                mask,
            )
        elif short_name == "rot" and rot_loss_mode in {"frame_vector", "frame_vec"}:
            ortho_weight = float(loss_weights.get("rot_frame_ortho_weight", 0.1) or 0.0)
            axis_weights = None
            axis_spec = str(loss_weights.get("rot_frame_axis", "") or "").strip().lower()
            if axis_spec:
                axis_map = {"x": 0, "y": 1, "z": 2}
                axis_vec = torch.zeros(3, device=pred.device)
                for ch in axis_spec:
                    if ch in axis_map:
                        axis_vec[axis_map[ch]] = 1.0
                if axis_vec.sum().item() > 0:
                    axis_weights = axis_vec
            else:
                axis_list = loss_weights.get("rot_frame_axis_weights", None)
                if isinstance(axis_list, (list, tuple)) and len(axis_list) == 3:
                    axis_weights = torch.tensor(axis_list, device=pred.device)
            loss_val = _frame_vector_loss(
                pred,
                target,
                time_weights if short_name == "rot" else None,
                (time_weighting.get(short_name) or "").lower()
                if time_weighting is not None
                else "",
                mask,
                ortho_weight=ortho_weight,
                axis_weights=axis_weights,
            )
        elif short_name == "rot" and rot_loss_mode in {"vmf_axis", "axis_vmf"}:
            loss_val = _vmf_axis_loss(
                pred,
                target,
                time_weights if short_name == "rot" else None,
                (time_weighting.get(short_name) or "").lower()
                if time_weighting is not None
                else "",
                mask,
            )
        else:
            loss_val = _mse_with_time_weight(pred, target, short_name, mask=mask)
        total_loss = total_loss + weight * loss_val
        loss_dict[short_name] = loss_val.detach()

    rot_cls_weight = float(loss_weights.get("rot_cls_weight", 0.0) or 0.0)
    rot_cls_logits = predictions.get("rot_cls_logits") if isinstance(predictions, dict) else None
    rot_cls_target = targets.get("rot_cls_target") if isinstance(targets, dict) else None
    if rot_cls_logits is not None and rot_cls_target is not None:
        try:
            rot_cls_loss = F.cross_entropy(rot_cls_logits, rot_cls_target)
            loss_dict["rot_cls"] = rot_cls_loss.detach()
            rot_cls_acc = (
                rot_cls_logits.argmax(dim=-1) == rot_cls_target
            ).float().mean().detach()
            loss_dict["rot_cls_acc"] = rot_cls_acc
            if rot_cls_weight > 0:
                total_loss = total_loss + rot_cls_weight * rot_cls_loss
        except Exception:
            pass

    pose_weight = float(loss_weights.get("pose_align", 0.0) or 0.0)
    pose_loss = None
    if time_samples is not None:
        pep_pos = targets.get("pep_pos")
        pep_orig_pos = targets.get("pep_orig_pos")
        pep_batch = targets.get("pep_batch")
        tr_pred = predictions.get("tr_pred")
        rot_pred = predictions.get("rot_pred")
        rot_pred_ok = rot_pred is not None and rot_pred.ndim >= 1 and rot_pred.shape[-1] == 3
        if (
            pep_pos is not None
            and pep_orig_pos is not None
            and pep_batch is not None
            and tr_pred is not None
            and rot_pred_ok
        ):
            pose_loss = _pose_align_loss(
                pep_pos,
                pep_orig_pos,
                pep_batch,
                tr_pred,
                rot_pred,
                time_samples,
                target_mode=str(target_mode or "velocity").lower(),
            )
            loss_dict["pose_align"] = pose_loss.detach()
            if pose_weight > 0:
                total_loss = total_loss + pose_weight * pose_loss

    rot_kabsch_weight = float(loss_weights.get("rot_kabsch", 0.0) or 0.0)
    if rot_kabsch_weight > 0 and time_samples is not None:
        pep_pos = targets.get("pep_pos")
        pep_orig_pos = targets.get("pep_orig_pos")
        pep_batch = targets.get("pep_batch")
        atom2atomid_index = targets.get("pep_atom2atomid_index")
        rot_pred = predictions.get("rot_pred")
        rot_pred_ok = rot_pred is not None and rot_pred.ndim >= 1 and rot_pred.shape[-1] == 3
        if pep_pos is not None and pep_orig_pos is not None and pep_batch is not None and rot_pred_ok:
            rot_kabsch_loss = _kabsch_rot_loss(
                pep_pos,
                pep_orig_pos,
                pep_batch,
                atom2atomid_index,
                rot_pred,
                time_samples,
                target_mode=str(target_mode or "velocity").lower(),
            )
            loss_dict["rot_kabsch"] = rot_kabsch_loss.detach()
            total_loss = total_loss + rot_kabsch_weight * rot_kabsch_loss

    coord_weight = float(loss_weights.get("coord_align", 0.0) or 0.0)
    if coord_weight > 0:
        pep_pos = targets.get("pep_pos")
        pep_orig_pos = targets.get("pep_orig_pos")
        pep_batch = targets.get("pep_batch")
        atom2atomid_index = targets.get("pep_atom2atomid_index")
        tr_pred = predictions.get("tr_pred")
        rot_pred = predictions.get("rot_pred")
        rot_pred_ok = rot_pred is not None and rot_pred.ndim >= 1 and rot_pred.shape[-1] == 3
        if rot_pred_ok:
            coord_loss = _coord_align_loss(
                pep_pos,
                pep_orig_pos,
                pep_batch,
                atom2atomid_index,
                tr_pred,
                rot_pred,
                time_samples,
            )
            loss_dict["coord_align"] = coord_loss.detach()
            total_loss = total_loss + coord_weight * coord_loss

    ca_weight = float(loss_weights.get("ca_coord", 0.0) or 0.0)
    if ca_weight > 0:
        ca_pred = predictions.get("ca_pred") if isinstance(predictions, dict) else None
        pep_pos = targets.get("pep_pos")
        pep_orig_pos = targets.get("pep_orig_pos")
        atom2res_index = targets.get("pep_atom2res_index")
        atom2atomid_index = targets.get("pep_atom2atomid_index")
        pep_atom_batch = targets.get("pep_batch")
        pep_res_batch = targets.get("pep_res_batch")
        ca_loss, ca_ratio, ca_pred_norm, ca_tgt_norm = _ca_coord_loss(
            ca_pred,
            pep_pos,
            pep_orig_pos,
            atom2res_index,
            atom2atomid_index,
            pep_atom_batch,
            pep_res_batch,
            time_samples,
            target_mode,
        )
        loss_dict["ca_coord"] = ca_loss.detach()
        loss_dict["ca_coord_ratio"] = torch.tensor(ca_ratio, device=ca_loss.device)
        loss_dict["ca_pred_norm"] = torch.tensor(ca_pred_norm, device=ca_loss.device)
        loss_dict["ca_tgt_norm"] = torch.tensor(ca_tgt_norm, device=ca_loss.device)
        total_loss = total_loss + ca_weight * ca_loss
    triplet_weight = float(loss_weights.get("triplet_align", 0.0) or 0.0)
    triplet_loss = None
    if time_samples is not None:
        pep_pos = targets.get("pep_pos")
        pep_orig_pos = targets.get("pep_orig_pos")
        pep_batch = targets.get("pep_batch")
        atom2res_index = targets.get("pep_atom2res_index")
        atom2atomid_index = targets.get("pep_atom2atomid_index")
        tr_pred = predictions.get("tr_pred") if isinstance(predictions, dict) else None
        rot_pred = predictions.get("rot_pred") if isinstance(predictions, dict) else None
        rot_pred_ok = rot_pred is not None and rot_pred.ndim >= 1 and rot_pred.shape[-1] == 3
        if (
            pep_pos is not None
            and pep_orig_pos is not None
            and pep_batch is not None
            and atom2res_index is not None
            and atom2atomid_index is not None
            and tr_pred is not None
            and rot_pred_ok
        ):
            triplet_loss = _triplet_align_loss(
                pep_pos,
                pep_orig_pos,
                pep_batch,
                atom2res_index,
                atom2atomid_index,
                tr_pred,
                rot_pred,
                time_samples,
                target_mode=str(target_mode or "velocity").lower(),
            )
            loss_dict["triplet_align"] = triplet_loss.detach()
            if triplet_weight > 0:
                total_loss = total_loss + triplet_weight * triplet_loss

    clash_weight = float(loss_weights.get("clash", 0.0) or 0.0)
    if clash_weight > 0 and time_samples is not None:
        pep_pos = targets.get("pep_pos")
        pep_orig_pos = targets.get("pep_orig_pos")
        pep_batch = targets.get("pep_batch")
        rec_pos = targets.get("rec_pos")
        rec_batch = targets.get("rec_batch")
        tr_pred = predictions.get("tr_pred") if isinstance(predictions, dict) else None
        rot_pred = predictions.get("rot_pred") if isinstance(predictions, dict) else None
        min_dist = float(loss_weights.get("clash_min_dist", 2.0) or 2.0)
        soft_dist = float(loss_weights.get("clash_soft_dist", max(min_dist, 3.4)) or max(min_dist, 3.4))
        soft_weight = float(loss_weights.get("clash_soft_weight", 0.5) or 0.0)
        density_cutoff = float(loss_weights.get("clash_density_cutoff", max(soft_dist, 4.0)) or max(soft_dist, 4.0))
        density_weight = float(loss_weights.get("clash_density_weight", 0.25) or 0.0)
        density_allowance = float(loss_weights.get("clash_density_allowance", 2.5) or 0.0)
        density_temperature = float(loss_weights.get("clash_density_temperature", 0.4) or 0.4)
        local_rec_radius = float(loss_weights.get("clash_local_rec_radius", 0.0) or 0.0)
        local_pep_radius = float(loss_weights.get("clash_local_pep_radius", 0.0) or 0.0)
        local_min_rec_atoms = int(loss_weights.get("clash_local_min_rec_atoms", 8) or 8)
        local_min_pep_atoms = int(loss_weights.get("clash_local_min_pep_atoms", 4) or 4)
        local_fallback_global = bool(loss_weights.get("clash_local_fallback_global", True))
        adaptive_metric = str(loss_weights.get("clash_adaptive_metric", "collide_ratio") or "collide_ratio").lower()
        adaptive_center = float(loss_weights.get("clash_adaptive_center", 0.10) or 0.10)
        adaptive_temperature = float(loss_weights.get("clash_adaptive_temperature", 0.03) or 0.03)
        adaptive_min_factor = float(loss_weights.get("clash_adaptive_min_factor", 1.0) or 1.0)
        adaptive_enabled = adaptive_min_factor < 0.999
        hard_overlap_center = float(loss_weights.get("clash_hard_overlap_center", 0.0) or 0.0)
        hard_overlap_temperature = float(loss_weights.get("clash_hard_overlap_temperature", 0.12) or 0.12)
        hard_overlap_max_factor = float(loss_weights.get("clash_hard_overlap_max_factor", 1.0) or 1.0)
        hard_overlap_enabled = hard_overlap_center > 0 and hard_overlap_max_factor > 1.0
        unroll_steps = int(loss_weights.get("clash_unroll_steps", 0) or 0)
        time_scale = resolve_time_scale(time_samples, target_mode)
        if time_scale is not None and unroll_steps > 0:
            time_scale = torch.full_like(time_scale, 1.0 / float(unroll_steps))
        if (
            pep_pos is not None
            and pep_orig_pos is not None
            and pep_batch is not None
            and rec_pos is not None
            and rec_batch is not None
            and tr_pred is not None
            and time_scale is not None
        ):
            tr_update = tr_pred * time_scale[:, None]
            rot_update = (
                rot_pred * time_scale[:, None]
                if rot_pred is not None and rot_pred.ndim >= 1 and rot_pred.shape[-1] == 3
                else None
            )
            moved, _centers = apply_rigid_body_updates(pep_pos, pep_batch, tr_update, rot_update)
            steric_stats = steric_penalty_from_positions(
                moved,
                pep_batch,
                rec_pos,
                rec_batch,
                min_dist=min_dist,
                soft_dist=soft_dist,
                soft_weight=soft_weight,
                density_cutoff=density_cutoff,
                density_weight=density_weight,
                density_allowance=density_allowance,
                density_temperature=density_temperature,
                ref_pep_pos=pep_orig_pos,
                ref_pep_batch=pep_batch,
                local_rec_radius=local_rec_radius if local_rec_radius > 0 else None,
                local_pep_radius=local_pep_radius if local_pep_radius > 0 else None,
                local_min_rec_atoms=local_min_rec_atoms,
                local_min_pep_atoms=local_min_pep_atoms,
                local_fallback_global=local_fallback_global,
            )
            loss_dict["clash"] = steric_stats["loss"].detach()
            loss_dict["clash_min_dist"] = steric_stats["min_dist"].detach()
            loss_dict["clash_collide_ratio"] = steric_stats["collide_ratio"].detach()
            loss_dict["clash_shell_ratio"] = steric_stats["shell_ratio"].detach()
            loss_dict["clash_density"] = steric_stats["density"].detach()
            loss_dict["clash_local_rec_ratio"] = steric_stats["rec_local_ratio"].detach()
            loss_dict["clash_local_pep_ratio"] = steric_stats["pep_local_ratio"].detach()
            overall_factor = torch.ones_like(steric_stats["per_graph_loss"])
            if adaptive_enabled:
                metric_map = {
                    "collide_ratio": steric_stats["per_graph_collide_ratio"],
                    "shell_ratio": steric_stats["per_graph_shell_ratio"],
                    "min_dist": steric_stats["per_graph_min_dist"],
                    "density": steric_stats["per_graph_density"],
                }
                adaptive_source = metric_map.get(adaptive_metric, steric_stats["per_graph_collide_ratio"])
                adaptive_temperature = max(adaptive_temperature, 1e-4)
                adaptive_min_factor = min(max(adaptive_min_factor, 0.0), 1.0)
                if adaptive_metric == "min_dist":
                    gate = torch.sigmoid((adaptive_center - adaptive_source) / adaptive_temperature)
                else:
                    gate = torch.sigmoid((adaptive_source - adaptive_center) / adaptive_temperature)
                gate = adaptive_min_factor + (1.0 - adaptive_min_factor) * gate
                overall_factor = overall_factor * gate
                loss_dict["clash_adaptive_gate"] = gate.mean().detach()
            if hard_overlap_enabled:
                hard_overlap_temperature = max(hard_overlap_temperature, 1e-4)
                hard_gate = torch.sigmoid(
                    (hard_overlap_center - steric_stats["per_graph_global_min_dist"]) / hard_overlap_temperature
                )
                hard_factor = 1.0 + (hard_overlap_max_factor - 1.0) * hard_gate
                overall_factor = overall_factor * hard_factor
                loss_dict["clash_hard_overlap_gate"] = hard_factor.mean().detach()
            clash_loss = (steric_stats["per_graph_loss"] * overall_factor).mean()
            total_loss = total_loss + clash_weight * clash_loss

    contact_weight = float(loss_weights.get("contact", 0.0) or 0.0)
    if contact_weight > 0 and time_samples is not None:
        pep_pos = targets.get("pep_pos")
        pep_orig_pos = targets.get("pep_orig_pos")
        pep_batch = targets.get("pep_batch")
        rec_pos = targets.get("rec_pos")
        rec_batch = targets.get("rec_batch")
        atom2res_index = targets.get("pep_atom2res_index")
        atom2atomid_index = targets.get("pep_atom2atomid_index")
        tr_pred = predictions.get("tr_pred") if isinstance(predictions, dict) else None
        rot_pred = predictions.get("rot_pred") if isinstance(predictions, dict) else None
        cutoff = float(loss_weights.get("contact_cutoff", 8.0) or 8.0)
        top_k = int(loss_weights.get("contact_top_k", 1) or 1)
        true_max = float(loss_weights.get("contact_true_max", 6.0) or 6.0)
        t_min = float(loss_weights.get("contact_t_min", 0.0) or 0.0)
        noncontact_weight = float(loss_weights.get("contact_noncontact", 0.0) or 0.0)
        noncontact_min = float(loss_weights.get("contact_noncontact_min", 4.0) or 4.0)
        contact_loss, contact_pairs, noncontact_loss, noncontact_pairs = _contact_loss(
            pep_pos,
            pep_orig_pos,
            pep_batch,
            rec_pos,
            rec_batch,
            atom2res_index,
            atom2atomid_index,
            tr_pred,
            rot_pred,
            time_samples,
            target_mode=str(target_mode or "velocity").lower(),
            cutoff=cutoff,
            top_k=top_k,
            true_max=true_max,
            t_min=t_min,
            noncontact_weight=noncontact_weight,
            noncontact_min=noncontact_min,
        )
        loss_dict["contact"] = contact_loss.detach()
        loss_dict["contact_pairs"] = contact_pairs.detach()
        if noncontact_weight > 0:
            loss_dict["contact_noncontact"] = noncontact_loss.detach()
            loss_dict["contact_noncontact_pairs"] = noncontact_pairs.detach()
            total_loss = total_loss + noncontact_weight * noncontact_loss
        total_loss = total_loss + contact_weight * contact_loss

    # 训练期专用 interface 辅助头监督（推理路径不使用这些输出）
    iface_contact_weight = float(loss_weights.get("interface_contact", 0.0) or 0.0)
    iface_pairdist_weight = float(loss_weights.get("interface_pairdist", 0.0) or 0.0)
    iface_contact_logits = predictions.get("interface_contact_logits") if isinstance(predictions, dict) else None
    iface_pairdist_pred = predictions.get("interface_pairdist_pred") if isinstance(predictions, dict) else None
    iface_edge_index = predictions.get("interface_edge_index") if isinstance(predictions, dict) else None
    if (
        iface_edge_index is not None
        and (iface_contact_logits is not None or iface_pairdist_pred is not None)
    ):
        pep_res_orig_pos = targets.get("pep_res_orig_pos")
        pep_res_batch = targets.get("pep_res_batch")
        rec_pos = targets.get("rec_pos")
        rec_batch = targets.get("rec_batch")
        iface_cutoff = float(loss_weights.get("interface_contact_cutoff", 8.0) or 8.0)
        iface_pairdist_max = float(loss_weights.get("interface_pairdist_max", 20.0) or 20.0)
        (
            iface_contact_loss,
            iface_contact_acc,
            iface_pairdist_loss,
            iface_pairs,
        ) = _interface_aux_loss(
            iface_edge_index,
            iface_contact_logits,
            iface_pairdist_pred,
            pep_res_orig_pos,
            pep_res_batch,
            rec_pos,
            rec_batch,
            contact_cutoff=iface_cutoff,
            pairdist_max=iface_pairdist_max,
        )
        loss_dict["iface_contact"] = iface_contact_loss.detach()
        loss_dict["iface_contact_acc"] = iface_contact_acc.detach()
        loss_dict["iface_pairdist"] = iface_pairdist_loss.detach()
        loss_dict["iface_pairs"] = iface_pairs.detach()
        if iface_contact_weight > 0:
            total_loss = total_loss + iface_contact_weight * iface_contact_loss
        if iface_pairdist_weight > 0:
            total_loss = total_loss + iface_pairdist_weight * iface_pairdist_loss
    return total_loss, loss_dict
