import time

import torch
import numpy as np
from utils.geometry import kabsch_torch, axis_angle_to_matrix


def _mask_to_index_list(mask_rotate):
    return tuple(
        torch.nonzero(mask_rotate[idx_edge], as_tuple=False).view(-1)
        for idx_edge in range(mask_rotate.shape[0])
    )


def _coerce_tensor(value, device, dtype):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.device == device and value.dtype == dtype:
            return value
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _build_flat_rotate_index_cache(rotate_index_list, device):
    lengths = torch.tensor(
        [idx.numel() for idx in rotate_index_list], dtype=torch.long, device=device
    )
    offsets = torch.zeros(lengths.numel() + 1, dtype=torch.long, device=device)
    if lengths.numel() > 0:
        offsets[1:] = torch.cumsum(lengths, dim=0)
    total = int(offsets[-1].item())
    if total > 0:
        flat = torch.cat(
            [idx.to(device=device, dtype=torch.long) for idx in rotate_index_list if idx.numel() > 0],
            dim=0,
        )
    else:
        flat = torch.empty(0, dtype=torch.long, device=device)
    return flat, offsets


def _resolve_torsion_inputs(data):
    pep = data["pep_a"]
    pep_edge = data["pep_a", "pep_a"]
    cache = getattr(data, "_torsion_inputs_cache", None)
    cache_device = getattr(data, "_torsion_inputs_cache_device", None)
    current_device = pep_edge.edge_index.device
    if cache is not None and cache_device == current_device:
        return cache
    mask_edges_backbone = torch.as_tensor(pep.mask_edges_backbone, device=current_device).squeeze()
    mask_edges_sidechain = torch.as_tensor(pep.mask_edges_sidechain, device=current_device).squeeze()
    edge_backbone_index = pep_edge.edge_index.T[mask_edges_backbone]
    edge_sidechain_index = pep_edge.edge_index.T[mask_edges_sidechain]
    mask_rotate_backbone = torch.as_tensor(pep.mask_rotate_backbone, device=current_device)
    mask_rotate_sidechain = torch.as_tensor(pep.mask_rotate_sidechain, device=current_device)
    if mask_rotate_backbone.dim() != 2:
        mask_rotate_backbone = mask_rotate_backbone.squeeze(dim=0)
    if mask_rotate_sidechain.dim() != 2:
        mask_rotate_sidechain = mask_rotate_sidechain.squeeze(dim=0)
    rotate_backbone_index_list = _mask_to_index_list(mask_rotate_backbone)
    rotate_sidechain_index_list = _mask_to_index_list(mask_rotate_sidechain)
    cache = (
        edge_backbone_index,
        edge_sidechain_index,
        rotate_backbone_index_list,
        rotate_sidechain_index_list,
    )
    object.__setattr__(data, "_torsion_inputs_cache", cache)
    object.__setattr__(data, "_torsion_inputs_cache_device", current_device)
    return cache


def _resolve_torsion_compact_inputs(data):
    cache = getattr(data, "_torsion_compact_cache", None)
    cache_device = getattr(data, "_torsion_compact_cache_device", None)
    current_device = data["pep_a", "pep_a"].edge_index.device
    if cache is not None and cache_device == current_device:
        return cache
    edge_backbone_index, edge_sidechain_index, rotate_backbone_index_list, rotate_sidechain_index_list = _resolve_torsion_inputs(data)
    rotate_backbone_index_flat, rotate_backbone_offsets = _build_flat_rotate_index_cache(
        rotate_backbone_index_list, current_device
    )
    rotate_sidechain_index_flat, rotate_sidechain_offsets = _build_flat_rotate_index_cache(
        rotate_sidechain_index_list, current_device
    )
    edge_backbone_u = edge_backbone_index[:, 0].contiguous()
    edge_backbone_v = edge_backbone_index[:, 1].contiguous()
    edge_sidechain_u = edge_sidechain_index[:, 0].contiguous()
    edge_sidechain_v = edge_sidechain_index[:, 1].contiguous()
    cache = (
        edge_backbone_u,
        edge_backbone_v,
        edge_sidechain_u,
        edge_sidechain_v,
        rotate_backbone_index_flat,
        rotate_backbone_offsets,
        rotate_sidechain_index_flat,
        rotate_sidechain_offsets,
    )
    object.__setattr__(data, "_torsion_compact_cache", cache)
    object.__setattr__(data, "_torsion_compact_cache_device", current_device)
    return cache


def _get_torsion_edge_counts(data):
    cache = getattr(data, "_torsion_edge_count_cache", None)
    if cache is not None:
        return cache
    edge_backbone_index, edge_sidechain_index, _, _ = _resolve_torsion_inputs(data)
    cache = (int(edge_backbone_index.shape[0]), int(edge_sidechain_index.shape[0]))
    object.__setattr__(data, "_torsion_edge_count_cache", cache)
    return cache


def _concat_edge_indices_with_offsets(edge_index_list, atom_offsets, device):
    shifted = []
    for edge_index, atom_offset in zip(edge_index_list, atom_offsets):
        if edge_index is None or edge_index.numel() == 0:
            continue
        shifted.append(edge_index.to(device=device, dtype=torch.long) + int(atom_offset))
    if shifted:
        return torch.cat(shifted, dim=0)
    return torch.empty((0, 2), dtype=torch.long, device=device)


def _concat_rotate_cache_with_offsets(rotate_flat_list, rotate_offsets_list, atom_offsets, device):
    rotate_flat_parts = []
    rotate_offset_parts = [torch.zeros(1, dtype=torch.long, device=device)]
    rotate_total = 0
    for rotate_flat, rotate_offsets, atom_offset in zip(
        rotate_flat_list, rotate_offsets_list, atom_offsets
    ):
        local_flat = rotate_flat.to(device=device, dtype=torch.long)
        local_offsets = rotate_offsets.to(device=device, dtype=torch.long)
        if local_flat.numel() > 0:
            rotate_flat_parts.append(local_flat + int(atom_offset))
        if local_offsets.numel() > 1:
            rotate_offset_parts.append(local_offsets[1:] + rotate_total)
        rotate_total += int(local_flat.numel())
    rotate_flat = (
        torch.cat(rotate_flat_parts, dim=0)
        if rotate_flat_parts
        else torch.empty(0, dtype=torch.long, device=device)
    )
    rotate_offsets = torch.cat(rotate_offset_parts, dim=0)
    return rotate_flat, rotate_offsets


def _compress_compact_group(edge_index, rotate_index_flat, rotate_offsets, device):
    edge_index = edge_index.to(device=device, dtype=torch.long)
    rotate_index_flat = rotate_index_flat.to(device=device, dtype=torch.long)
    rotate_offsets = rotate_offsets.to(device=device, dtype=torch.long)
    edge_count = int(edge_index.size(0))
    if edge_count == 0 or rotate_offsets.numel() <= 1:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return {
            "edge_u": empty,
            "edge_v": empty,
            "rotate_index_flat": rotate_index_flat,
            "rotate_offsets": torch.zeros(1, dtype=torch.long, device=device),
            "active_update_index": empty,
        }
    lengths = rotate_offsets[1:] - rotate_offsets[:-1]
    active_update_index = torch.nonzero(lengths > 0, as_tuple=False).view(-1)
    if active_update_index.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return {
            "edge_u": empty,
            "edge_v": empty,
            "rotate_index_flat": torch.empty(0, dtype=torch.long, device=device),
            "rotate_offsets": torch.zeros(1, dtype=torch.long, device=device),
            "active_update_index": empty,
        }
    active_lengths = lengths.index_select(0, active_update_index)
    compact_offsets = torch.zeros(
        active_lengths.numel() + 1, dtype=torch.long, device=device
    )
    compact_offsets[1:] = torch.cumsum(active_lengths, dim=0)
    active_edges = edge_index.index_select(0, active_update_index)
    return {
        "edge_u": active_edges[:, 0].contiguous(),
        "edge_v": active_edges[:, 1].contiguous(),
        "rotate_index_flat": rotate_index_flat.contiguous(),
        "rotate_offsets": compact_offsets,
        "active_update_index": active_update_index.contiguous(),
    }


def build_batch_torsion_cache(data_list):
    if not data_list:
        return None
    pep_device = data_list[0]["pep_a"].pos.device
    atom_offsets = []
    atom_total = 0
    edge_backbone_index_list = []
    edge_sidechain_index_list = []
    rotate_backbone_flat_list = []
    rotate_backbone_offsets_list = []
    rotate_sidechain_flat_list = []
    rotate_sidechain_offsets_list = []
    backbone_edge_offsets = [0]
    sidechain_edge_offsets = [0]

    for graph in data_list:
        atom_offsets.append(atom_total)
        atom_total += int(graph["pep_a"].pos.shape[0])
        edge_backbone_index, edge_sidechain_index, _, _ = _resolve_torsion_inputs(graph)
        (
            rotate_backbone_index_flat,
            rotate_backbone_offsets,
            rotate_sidechain_index_flat,
            rotate_sidechain_offsets,
        ) = _resolve_torsion_compact_inputs(graph)
        edge_backbone_index_list.append(edge_backbone_index)
        edge_sidechain_index_list.append(edge_sidechain_index)
        rotate_backbone_flat_list.append(rotate_backbone_index_flat)
        rotate_backbone_offsets_list.append(rotate_backbone_offsets)
        rotate_sidechain_flat_list.append(rotate_sidechain_index_flat)
        rotate_sidechain_offsets_list.append(rotate_sidechain_offsets)
        backbone_edge_offsets.append(
            backbone_edge_offsets[-1] + max(int(rotate_backbone_offsets.numel()) - 1, 0)
        )
        sidechain_edge_offsets.append(
            sidechain_edge_offsets[-1] + max(int(rotate_sidechain_offsets.numel()) - 1, 0)
        )

    edge_backbone_index = _concat_edge_indices_with_offsets(
        edge_backbone_index_list, atom_offsets, pep_device
    )
    edge_sidechain_index = _concat_edge_indices_with_offsets(
        edge_sidechain_index_list, atom_offsets, pep_device
    )
    rotate_backbone_index_flat, rotate_backbone_offsets = _concat_rotate_cache_with_offsets(
        rotate_backbone_flat_list,
        rotate_backbone_offsets_list,
        atom_offsets,
        pep_device,
    )
    rotate_sidechain_index_flat, rotate_sidechain_offsets = _concat_rotate_cache_with_offsets(
        rotate_sidechain_flat_list,
        rotate_sidechain_offsets_list,
        atom_offsets,
        pep_device,
    )
    backbone_compact = _compress_compact_group(
        edge_backbone_index,
        rotate_backbone_index_flat,
        rotate_backbone_offsets,
        pep_device,
    )
    sidechain_compact = _compress_compact_group(
        edge_sidechain_index,
        rotate_sidechain_index_flat,
        rotate_sidechain_offsets,
        pep_device,
    )
    return {
        "edge_backbone_index": edge_backbone_index,
        "edge_sidechain_index": edge_sidechain_index,
        "rotate_backbone_index_flat": rotate_backbone_index_flat,
        "rotate_backbone_offsets": rotate_backbone_offsets,
        "rotate_sidechain_index_flat": rotate_sidechain_index_flat,
        "rotate_sidechain_offsets": rotate_sidechain_offsets,
        "backbone_edge_u": backbone_compact["edge_u"],
        "backbone_edge_v": backbone_compact["edge_v"],
        "backbone_active_update_index": backbone_compact["active_update_index"],
        "backbone_rotate_index_flat_compact": backbone_compact["rotate_index_flat"],
        "backbone_rotate_offsets_compact": backbone_compact["rotate_offsets"],
        "sidechain_edge_u": sidechain_compact["edge_u"],
        "sidechain_edge_v": sidechain_compact["edge_v"],
        "sidechain_active_update_index": sidechain_compact["active_update_index"],
        "sidechain_rotate_index_flat_compact": sidechain_compact["rotate_index_flat"],
        "sidechain_rotate_offsets_compact": sidechain_compact["rotate_offsets"],
        "backbone_edge_offsets": tuple(backbone_edge_offsets),
        "sidechain_edge_offsets": tuple(sidechain_edge_offsets),
    }


def _apply_single_torsion_group(pos, edge_index, rotate_index_list, torsion_updates):
    if torsion_updates is None:
        return pos
    max_edges = min(edge_index.shape[0], torsion_updates.shape[0], len(rotate_index_list))
    for idx_edge in range(max_edges):
        angle = torsion_updates[idx_edge]
        u = edge_index[idx_edge, 0]
        v = edge_index[idx_edge, 1]
        rotate_index = rotate_index_list[idx_edge]
        if rotate_index.numel() == 0:
            continue
        pivot = pos[v]
        rel = pos[rotate_index] - pivot
        axis = pos[u] - pivot
        norm = torch.norm(axis)
        valid_axis = (norm >= 1e-8).to(dtype=pos.dtype)
        axis = axis / norm.clamp_min(1e-8)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        axis_row = axis.unsqueeze(0)
        cross_term = torch.cross(axis_row, rel, dim=-1) * sin_angle
        proj = (rel * axis_row).sum(dim=-1, keepdim=True)
        rotated_rel = (
            rel * cos_angle
            + cross_term
            + axis_row * proj * (1.0 - cos_angle)
        )
        pos[rotate_index] = rotated_rel * valid_axis + rel * (1.0 - valid_axis) + pivot
    return pos


def _apply_single_torsion_group_compact_impl(
    pos: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    rotate_index_flat: torch.Tensor,
    rotate_offsets: torch.Tensor,
    torsion_updates: torch.Tensor,
) -> torch.Tensor:
    if torsion_updates.numel() == 0:
        return pos
    max_edges = edge_u.size(0)
    if torsion_updates.size(0) < max_edges:
        max_edges = torsion_updates.size(0)
    if rotate_offsets.size(0) - 1 < max_edges:
        max_edges = rotate_offsets.size(0) - 1
    for idx_edge in range(max_edges):
        start = int(rotate_offsets[idx_edge].item())
        end = int(rotate_offsets[idx_edge + 1].item())
        if end <= start:
            continue
        angle = torsion_updates[idx_edge]
        u = int(edge_u[idx_edge].item())
        v = int(edge_v[idx_edge].item())
        rotate_index = rotate_index_flat[start:end]
        pivot = pos[v]
        rel = pos.index_select(0, rotate_index) - pivot
        axis = pos[u] - pivot
        norm = torch.norm(axis)
        valid_axis = (norm >= 1e-8).to(dtype=pos.dtype)
        axis = axis / norm.clamp_min(1e-8)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        axis_row = axis.unsqueeze(0)
        cross_term = torch.cross(axis_row, rel, dim=-1) * sin_angle
        proj = (rel * axis_row).sum(dim=-1, keepdim=True)
        rotated_rel = (
            rel * cos_angle
            + cross_term
            + axis_row * proj * (1.0 - cos_angle)
        )
        pos.index_copy_(
            0,
            rotate_index,
            rotated_rel * valid_axis + rel * (1.0 - valid_axis) + pivot,
        )
    return pos


try:
    _apply_single_torsion_group_compact = torch.jit.script(
        _apply_single_torsion_group_compact_impl
    )
except Exception:
    _apply_single_torsion_group_compact = _apply_single_torsion_group_compact_impl


def peptide_updater(
    data,
    tr_update,
    rot_update,
    torsion_backbone_updates,
    torsion_sidechain_updates,
    torsion_device="cpu",
    torsion_debug=False,
    return_timings=False,
):
    """先整体刚体变换，再把骨架/侧链扭转量灌进去，最后重新对齐"""
    timings = {
        "rigid_seconds": 0.0,
        "resolve_inputs_seconds": 0.0,
        "torsion_seconds": 0.0,
        "kabsch_seconds": 0.0,
        "assign_seconds": 0.0,
    }
    pos_dtype = data['pep_a'].pos.dtype
    pos_device = data['pep_a'].pos.device
    t_rigid_start = time.perf_counter()
    tr_update = _coerce_tensor(tr_update, pos_device, pos_dtype)
    rot_update = _coerce_tensor(rot_update, pos_device, pos_dtype)
    pep_a_center = torch.mean(data['pep_a'].pos, dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze()).to(device=pos_device, dtype=pos_dtype)
    rigid_new_pos = (data['pep_a'].pos - pep_a_center) @ rot_mat.T + tr_update + pep_a_center
    timings["rigid_seconds"] += time.perf_counter() - t_rigid_start
    flexible_new_pos, torsion_timings = peptide_torsion_update_from_rigid(
        data,
        rigid_new_pos,
        torsion_backbone_updates,
        torsion_sidechain_updates,
        torsion_device=torsion_device,
        torsion_debug=torsion_debug,
        return_timings=True,
    )
    timings["resolve_inputs_seconds"] += torsion_timings["resolve_inputs_seconds"]
    timings["torsion_seconds"] += torsion_timings["torsion_seconds"]
    t_kabsch_start = time.perf_counter()
    R, t = kabsch_torch(flexible_new_pos.T, rigid_new_pos.T, check_det=False)
    aligned_flexible_pos = flexible_new_pos @ R.T + t.T
    timings["kabsch_seconds"] += time.perf_counter() - t_kabsch_start
    t_assign_start = time.perf_counter()
    data['pep_a'].pos = aligned_flexible_pos
    timings["assign_seconds"] += time.perf_counter() - t_assign_start
    if return_timings:
        return data, timings
    return data


def peptide_torsion_update_from_rigid(
    data,
    rigid_new_pos,
    torsion_backbone_updates,
    torsion_sidechain_updates,
    torsion_device="cpu",
    torsion_debug=False,
    return_timings=False,
):
    """给定刚体更新后的坐标，只做 torsion 形变，保持数学路径不变。"""
    timings = {
        "resolve_inputs_seconds": 0.0,
        "torsion_seconds": 0.0,
    }
    pos_device = data['pep_a'].pos.device
    pos_dtype = data['pep_a'].pos.dtype
    rigid_new_pos = _coerce_tensor(rigid_new_pos, pos_device, pos_dtype)
    t_resolve_start = time.perf_counter()
    (
        edge_backbone_index,
        edge_sidechain_index,
        rotate_backbone_index_list,
        rotate_sidechain_index_list,
    ) = _resolve_torsion_inputs(data)
    rotate_backbone_index_flat = None
    rotate_backbone_offsets = None
    rotate_sidechain_index_flat = None
    rotate_sidechain_offsets = None
    timings["resolve_inputs_seconds"] += time.perf_counter() - t_resolve_start
    has_backbone_updates = torsion_backbone_updates is not None and len(torsion_backbone_updates) > 0
    has_sidechain_updates = torsion_sidechain_updates is not None and len(torsion_sidechain_updates) > 0
    if not has_backbone_updates and not has_sidechain_updates:
        if return_timings:
            return rigid_new_pos, timings
        return rigid_new_pos

    torsion_device = str(torsion_device or "cpu").lower()
    use_gpu = torsion_device == "gpu" and rigid_new_pos.device.type == "cuda"
    if use_gpu:
        (
            edge_backbone_u,
            edge_backbone_v,
            edge_sidechain_u,
            edge_sidechain_v,
            rotate_backbone_index_flat,
            rotate_backbone_offsets,
            rotate_sidechain_index_flat,
            rotate_sidechain_offsets,
        ) = _resolve_torsion_compact_inputs(data)
    t_torsion_start = time.perf_counter()
    if torsion_debug and use_gpu:
        cpu_pos = rigid_new_pos.detach().cpu()
        cpu_backbone_updates = (
            None
            if torsion_backbone_updates is None
            else _coerce_tensor(torsion_backbone_updates, rigid_new_pos.device, rigid_new_pos.dtype).detach().cpu()
        )
        cpu_sidechain_updates = (
            None
            if torsion_sidechain_updates is None
            else _coerce_tensor(torsion_sidechain_updates, rigid_new_pos.device, rigid_new_pos.dtype).detach().cpu()
        )
        cpu_result = apply_torsion_updates(
            cpu_pos,
            edge_backbone_index.cpu(),
            edge_sidechain_index.cpu(),
            tuple(idx.cpu() for idx in rotate_backbone_index_list),
            tuple(idx.cpu() for idx in rotate_sidechain_index_list),
            cpu_backbone_updates,
            cpu_sidechain_updates,
        )
        gpu_result = apply_torsion_updates_gpu(
            rigid_new_pos,
            edge_backbone_u,
            edge_backbone_v,
            edge_sidechain_u,
            edge_sidechain_v,
            rotate_backbone_index_flat,
            rotate_backbone_offsets,
            rotate_sidechain_index_flat,
            rotate_sidechain_offsets,
            torsion_backbone_updates,
            torsion_sidechain_updates,
        )
        diff = cpu_result.to(device=gpu_result.device) - gpu_result
        rmsd = torch.sqrt(torch.mean(diff * diff)).item()
        print(f"[torsion_debug] cpu_vs_gpu_rmsd={rmsd:.6e}")
        flexible_new_pos = gpu_result
    elif use_gpu:
        flexible_new_pos = apply_torsion_updates_gpu(
            rigid_new_pos,
            edge_backbone_u,
            edge_backbone_v,
            edge_sidechain_u,
            edge_sidechain_v,
            rotate_backbone_index_flat,
            rotate_backbone_offsets,
            rotate_sidechain_index_flat,
            rotate_sidechain_offsets,
            torsion_backbone_updates,
            torsion_sidechain_updates,
        )
    else:
        flexible_new_pos = apply_torsion_updates(
            rigid_new_pos,
            edge_backbone_index,
            edge_sidechain_index,
            rotate_backbone_index_list,
            rotate_sidechain_index_list,
            torsion_backbone_updates,
            torsion_sidechain_updates,
        )
    timings["torsion_seconds"] += time.perf_counter() - t_torsion_start
    if return_timings:
        return flexible_new_pos, timings
    return flexible_new_pos

def apply_torsion_updates(pos, edge_backbone_index, edge_sidechain_index, rotate_backbone_index_list, rotate_sidechain_index_list, torsion_backbone_updates, torsion_sidechain_updates):
    """沿着标记好的边旋转指定残基集合，分别处理骨架和侧链"""
    pos = pos.clone()
    device = pos.device

    torsion_backbone_updates = _coerce_tensor(torsion_backbone_updates, device, pos.dtype)
    torsion_sidechain_updates = _coerce_tensor(torsion_sidechain_updates, device, pos.dtype)

    pos = _apply_single_torsion_group(
        pos, edge_backbone_index, rotate_backbone_index_list, torsion_backbone_updates
    )
    pos = _apply_single_torsion_group(
        pos, edge_sidechain_index, rotate_sidechain_index_list, torsion_sidechain_updates
    )
    return pos
def apply_torsion_updates_gpu(
    pos,
    edge_backbone_u,
    edge_backbone_v,
    edge_sidechain_u,
    edge_sidechain_v,
    rotate_backbone_index_flat,
    rotate_backbone_offsets,
    rotate_sidechain_index_flat,
    rotate_sidechain_offsets,
    torsion_backbone_updates,
    torsion_sidechain_updates,
):
    """GPU 版扭转更新：保持与 CPU 顺序一致，所有运算在 torch 上完成"""
    pos = pos.clone()
    device = pos.device

    torsion_backbone_updates = _coerce_tensor(torsion_backbone_updates, device, pos.dtype)
    torsion_sidechain_updates = _coerce_tensor(torsion_sidechain_updates, device, pos.dtype)

    pos = _apply_single_torsion_group_compact(
        pos,
        edge_backbone_u,
        edge_backbone_v,
        rotate_backbone_index_flat,
        rotate_backbone_offsets,
        torsion_backbone_updates
        if torsion_backbone_updates is not None
        else torch.empty(0, device=device, dtype=pos.dtype),
    )
    pos = _apply_single_torsion_group_compact(
        pos,
        edge_sidechain_u,
        edge_sidechain_v,
        rotate_sidechain_index_flat,
        rotate_sidechain_offsets,
        torsion_sidechain_updates
        if torsion_sidechain_updates is not None
        else torch.empty(0, device=device, dtype=pos.dtype),
    )
    return pos


def apply_batched_torsion_updates_gpu(
    pos,
    batch_torsion_cache,
    torsion_backbone_updates,
    torsion_sidechain_updates,
    return_timings=False,
):
    timings = {
        "resolve_inputs_seconds": 0.0,
        "torsion_seconds": 0.0,
    }
    if batch_torsion_cache is None:
        if return_timings:
            return pos, timings
        return pos
    pos = pos.clone()
    device = pos.device
    torsion_backbone_updates = _coerce_tensor(torsion_backbone_updates, device, pos.dtype)
    torsion_sidechain_updates = _coerce_tensor(torsion_sidechain_updates, device, pos.dtype)
    has_backbone_updates = (
        torsion_backbone_updates is not None and torsion_backbone_updates.numel() > 0
    )
    has_sidechain_updates = (
        torsion_sidechain_updates is not None and torsion_sidechain_updates.numel() > 0
    )
    if not has_backbone_updates and not has_sidechain_updates:
        if return_timings:
            return pos, timings
        return pos

    t_torsion_start = time.perf_counter()
    backbone_updates_compact = (
        torsion_backbone_updates.index_select(
            0, batch_torsion_cache["backbone_active_update_index"]
        )
        if has_backbone_updates
        and batch_torsion_cache["backbone_active_update_index"].numel() != torsion_backbone_updates.numel()
        else torsion_backbone_updates
    )
    sidechain_updates_compact = (
        torsion_sidechain_updates.index_select(
            0, batch_torsion_cache["sidechain_active_update_index"]
        )
        if has_sidechain_updates
        and batch_torsion_cache["sidechain_active_update_index"].numel() != torsion_sidechain_updates.numel()
        else torsion_sidechain_updates
    )
    pos = _apply_single_torsion_group_compact(
        pos,
        batch_torsion_cache["backbone_edge_u"],
        batch_torsion_cache["backbone_edge_v"],
        batch_torsion_cache["backbone_rotate_index_flat_compact"],
        batch_torsion_cache["backbone_rotate_offsets_compact"],
        backbone_updates_compact
        if has_backbone_updates
        else torch.empty(0, device=device, dtype=pos.dtype),
    )
    pos = _apply_single_torsion_group_compact(
        pos,
        batch_torsion_cache["sidechain_edge_u"],
        batch_torsion_cache["sidechain_edge_v"],
        batch_torsion_cache["sidechain_rotate_index_flat_compact"],
        batch_torsion_cache["sidechain_rotate_offsets_compact"],
        sidechain_updates_compact
        if has_sidechain_updates
        else torch.empty(0, device=device, dtype=pos.dtype),
    )
    timings["torsion_seconds"] += time.perf_counter() - t_torsion_start
    if return_timings:
        return pos, timings
    return pos

def randomize_position(data_list, no_random, tr_sigma_max):
    """扩散前做随机初始化，先乱扭角，再随机旋转/平移"""
    for complex_graph in data_list:
        torsion_updates_backbone = np.random.uniform(
            low=-np.pi,
            high=np.pi,
            size=int(complex_graph['pep_a'].mask_edges_backbone.sum().item()),
        )
        torsion_updates_sidechain = np.random.uniform(
            low=-np.pi,
            high=np.pi,
            size=int(complex_graph['pep_a'].mask_edges_sidechain.sum().item()),
        )
        edge_backbone_index, edge_sidechain_index, rotate_backbone_index_list, rotate_sidechain_index_list = _resolve_torsion_inputs(complex_graph)
        complex_graph['pep_a'].pos = \
            apply_torsion_updates(complex_graph['pep_a'].pos, edge_backbone_index, edge_sidechain_index, rotate_backbone_index_list, rotate_sidechain_index_list, torsion_updates_backbone, torsion_updates_sidechain)

    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['pep_a'].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(Rotation.random().as_matrix()).float()
        complex_graph['pep_a'].pos = (complex_graph['pep_a'].pos - molecule_center) @ random_rotation.T

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            complex_graph['pep_a'].pos += tr_update
            

def randomize_position_gaussian_shell(
    data_list,
    r_min,
    r_max,
    r_mu=None,
    r_sigma=None,
    center_mode="receptor_com",
    min_dist=0.0,
    max_tries=30,
):
    """在球壳上初始化肽位置，半径服从截断高斯。"""
    if r_mu is None:
        r_mu = 0.5 * (r_min + r_max)
    if r_sigma is None:
        r_sigma = max((r_max - r_min) / 4.0, 1e-3)
    if r_sigma <= 0:
        raise ValueError(f"r_sigma必须>0，当前={r_sigma}")
    max_tries = max(1, int(max_tries))

    for complex_graph in data_list:
        rec_pos = complex_graph["receptor"].pos if "receptor" in complex_graph else None
        pep_pos = complex_graph["pep"].pos if "pep" in complex_graph else None
        pep_a_pos = complex_graph["pep_a"].pos if "pep_a" in complex_graph else None
        if rec_pos is None or pep_a_pos is None:
            continue
        rec_com = rec_pos.mean(dim=0) if rec_pos.numel() else None
        pep_com = pep_pos.mean(dim=0) if pep_pos is not None and pep_pos.numel() else pep_a_pos.mean(dim=0)
        if rec_com is None or pep_com is None:
            continue
        center = rec_com if str(center_mode).lower() == "receptor_com" else pep_com

        last_update = None
        for _ in range(max_tries):
            direction = torch.randn(3)
            norm = direction.norm()
            if norm < 1e-6:
                direction = torch.tensor([1.0, 0.0, 0.0])
                norm = 1.0
            direction = direction / norm

            radius = None
            for _ in range(32):
                cand = float(torch.normal(mean=float(r_mu), std=float(r_sigma), size=(1,)).item())
                if r_min <= cand <= r_max:
                    radius = cand
                    break
            if radius is None:
                radius = float(torch.empty(1).uniform_(float(r_min), float(r_max)).item())
            target_com = center + direction * radius
            tr_update = (target_com - pep_com).unsqueeze(0)
            last_update = tr_update

            if min_dist <= 0 or rec_pos is None:
                break
            moved_pep = pep_a_pos + tr_update
            try:
                dist = torch.cdist(moved_pep, rec_pos).min().item()
            except Exception:
                break
            if dist >= float(min_dist):
                break

        if last_update is not None:
            complex_graph["pep_a"].pos += last_update
