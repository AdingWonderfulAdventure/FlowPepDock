##########################################################################
# File Name: steric.py
# Author: FlowPepDock contributors
# Description: Shared steric penalties and rigid-body guidance helpers
#########################################################################

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean

from utils.geometry import axis_angle_to_matrix


def resolve_time_scale(
    time_samples: Optional[torch.Tensor],
    target_mode: str,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    if time_samples is None:
        return None
    if time_samples.ndim == 2 and time_samples.shape[1] == 1:
        time_samples = time_samples.squeeze(1)
    time_scale = time_samples.float().clamp_min(eps)
    if str(target_mode or "velocity").lower() != "velocity":
        time_scale = (1.0 - time_scale).clamp_min(eps)
    return time_scale


def apply_rigid_body_updates(
    pep_pos: torch.Tensor,
    pep_batch: torch.Tensor,
    tr_update: torch.Tensor,
    rot_update: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_graphs = int(tr_update.shape[0])
    centers = scatter_mean(pep_pos, pep_batch, dim=0, dim_size=num_graphs)
    pos_centered = pep_pos - centers[pep_batch]
    if rot_update is not None and rot_update.ndim >= 1 and rot_update.shape[-1] == 3:
        rot_mat = axis_angle_to_matrix(rot_update)
        moved = torch.einsum("nij,nj->ni", rot_mat[pep_batch], pos_centered)
        moved = moved + centers[pep_batch] + tr_update[pep_batch]
    else:
        moved = pep_pos + tr_update[pep_batch]
    return moved, centers


def steric_penalty_from_positions(
    pep_pos: torch.Tensor,
    pep_batch: torch.Tensor,
    rec_pos: torch.Tensor,
    rec_batch: torch.Tensor,
    min_dist: float,
    soft_dist: Optional[float] = None,
    soft_weight: float = 0.0,
    density_cutoff: Optional[float] = None,
    density_weight: float = 0.0,
    density_allowance: float = 2.5,
    density_temperature: float = 0.5,
    ref_pep_pos: Optional[torch.Tensor] = None,
    ref_pep_batch: Optional[torch.Tensor] = None,
    local_rec_radius: Optional[float] = None,
    local_pep_radius: Optional[float] = None,
    local_min_rec_atoms: int = 8,
    local_min_pep_atoms: int = 4,
    local_fallback_global: bool = True,
) -> Dict[str, torch.Tensor]:
    if pep_pos.numel() == 0 or rec_pos.numel() == 0:
        zero = torch.tensor(0.0, device=pep_pos.device)
        return {
            "loss": zero,
            "per_graph_loss": zero.unsqueeze(0),
            "min_dist": zero,
            "per_graph_min_dist": zero.unsqueeze(0),
            "global_min_dist": zero,
            "per_graph_global_min_dist": zero.unsqueeze(0),
            "collide_ratio": zero,
            "per_graph_collide_ratio": zero.unsqueeze(0),
            "shell_ratio": zero,
            "per_graph_shell_ratio": zero.unsqueeze(0),
            "density": zero,
            "per_graph_density": zero.unsqueeze(0),
            "rec_local_ratio": zero,
            "per_graph_rec_local_ratio": zero.unsqueeze(0),
            "pep_local_ratio": zero,
            "per_graph_pep_local_ratio": zero.unsqueeze(0),
        }
    soft_dist = float(max(soft_dist if soft_dist is not None else min_dist, min_dist))
    density_cutoff = float(max(density_cutoff if density_cutoff is not None else soft_dist, min_dist))
    density_temperature = float(max(density_temperature, 1e-4))
    density_allowance = float(max(density_allowance, 0.0))
    num_graphs = int(max(int(pep_batch.max().item()) if pep_batch.numel() > 0 else -1,
                         int(rec_batch.max().item()) if rec_batch.numel() > 0 else -1) + 1)

    per_graph_loss = []
    per_graph_min_dist = []
    per_graph_global_min_dist = []
    per_graph_collide = []
    per_graph_shell = []
    per_graph_density = []
    per_graph_rec_local_ratio = []
    per_graph_pep_local_ratio = []
    for g in range(num_graphs):
        pep_mask = pep_batch == g
        rec_mask = rec_batch == g
        if not pep_mask.any() or not rec_mask.any():
            continue
        pep_g = pep_pos[pep_mask]
        rec_g = rec_pos[rec_mask]
        pep_local = pep_g
        rec_local = rec_g
        rec_local_ratio = torch.ones((), device=pep_pos.device, dtype=pep_pos.dtype)
        pep_local_ratio = torch.ones((), device=pep_pos.device, dtype=pep_pos.dtype)
        use_local = (
            ref_pep_pos is not None
            and ref_pep_batch is not None
            and local_rec_radius is not None
            and local_rec_radius > 0
        )
        if use_local:
            ref_mask = ref_pep_batch == g
            if ref_mask.any():
                ref_g = ref_pep_pos[ref_mask]
                ref_to_rec = torch.cdist(ref_g, rec_g)
                rec_keep = ref_to_rec.min(dim=0).values <= float(local_rec_radius)
                if int(rec_keep.sum().item()) >= int(max(local_min_rec_atoms, 1)):
                    rec_local = rec_g[rec_keep]
                    rec_local_ratio = rec_keep.float().mean()
                elif not local_fallback_global:
                    continue
                if local_pep_radius is not None and local_pep_radius > 0 and rec_local.numel() > 0:
                    pep_to_local_rec = torch.cdist(ref_g, rec_local)
                    pep_keep = pep_to_local_rec.min(dim=1).values <= float(local_pep_radius)
                    if int(pep_keep.sum().item()) >= int(max(local_min_pep_atoms, 1)):
                        pep_local = pep_g[pep_keep]
                        pep_local_ratio = pep_keep.float().mean()
                    elif not local_fallback_global:
                        continue
        if pep_local.numel() == 0 or rec_local.numel() == 0:
            if not local_fallback_global:
                continue
            pep_local = pep_g
            rec_local = rec_g
        local_dists = torch.cdist(pep_local, rec_local)
        min_d = local_dists.min(dim=1).values
        hard_penalty = F.relu(float(min_dist) - min_d).pow(2)
        total_penalty = hard_penalty
        if soft_weight > 0:
            soft_penalty = F.relu(soft_dist - min_d).pow(2)
            total_penalty = total_penalty + float(soft_weight) * soft_penalty
        if density_weight > 0:
            density = torch.sigmoid((density_cutoff - local_dists) / density_temperature).sum(dim=1)
            density_penalty = F.relu(density - density_allowance).pow(2)
            total_penalty = total_penalty + float(density_weight) * density_penalty
            per_graph_density.append(density.mean())
        else:
            per_graph_density.append(torch.zeros((), device=pep_pos.device, dtype=pep_pos.dtype))
        per_graph_loss.append(total_penalty.mean())
        per_graph_min_dist.append(min_d.mean())
        per_graph_global_min_dist.append(local_dists.min())
        per_graph_collide.append((min_d < float(min_dist)).float().mean())
        per_graph_shell.append((min_d < soft_dist).float().mean())
        per_graph_rec_local_ratio.append(rec_local_ratio)
        per_graph_pep_local_ratio.append(pep_local_ratio)
    if not per_graph_loss:
        zero = torch.tensor(0.0, device=pep_pos.device)
        return {
            "loss": zero,
            "per_graph_loss": zero.unsqueeze(0),
            "min_dist": zero,
            "per_graph_min_dist": zero.unsqueeze(0),
            "global_min_dist": zero,
            "per_graph_global_min_dist": zero.unsqueeze(0),
            "collide_ratio": zero,
            "per_graph_collide_ratio": zero.unsqueeze(0),
            "shell_ratio": zero,
            "per_graph_shell_ratio": zero.unsqueeze(0),
            "density": zero,
            "per_graph_density": zero.unsqueeze(0),
            "rec_local_ratio": zero,
            "per_graph_rec_local_ratio": zero.unsqueeze(0),
            "pep_local_ratio": zero,
            "per_graph_pep_local_ratio": zero.unsqueeze(0),
        }
    stacked_loss = torch.stack(per_graph_loss, dim=0)
    stacked_min_dist = torch.stack(per_graph_min_dist, dim=0)
    stacked_global_min_dist = torch.stack(per_graph_global_min_dist, dim=0)
    stacked_collide = torch.stack(per_graph_collide, dim=0)
    stacked_shell = torch.stack(per_graph_shell, dim=0)
    stacked_density = torch.stack(per_graph_density, dim=0)
    stacked_rec_local = torch.stack(per_graph_rec_local_ratio, dim=0)
    stacked_pep_local = torch.stack(per_graph_pep_local_ratio, dim=0)
    return {
        "loss": stacked_loss.mean(),
        "per_graph_loss": stacked_loss,
        "min_dist": stacked_min_dist.mean(),
        "per_graph_min_dist": stacked_min_dist,
        "global_min_dist": stacked_global_min_dist.mean(),
        "per_graph_global_min_dist": stacked_global_min_dist,
        "collide_ratio": stacked_collide.mean(),
        "per_graph_collide_ratio": stacked_collide,
        "shell_ratio": stacked_shell.mean(),
        "per_graph_shell_ratio": stacked_shell,
        "density": stacked_density.mean(),
        "per_graph_density": stacked_density,
        "rec_local_ratio": stacked_rec_local.mean(),
        "per_graph_rec_local_ratio": stacked_rec_local,
        "pep_local_ratio": stacked_pep_local.mean(),
        "per_graph_pep_local_ratio": stacked_pep_local,
    }


def steric_rigid_guidance(
    pep_pos: torch.Tensor,
    pep_batch: torch.Tensor,
    rec_pos: torch.Tensor,
    rec_batch: torch.Tensor,
    cutoff: float,
    temperature: float = 0.5,
    max_tr_norm: float = 2.0,
    max_rot_norm: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_graphs = int(max(int(pep_batch.max().item()) if pep_batch.numel() > 0 else -1,
                         int(rec_batch.max().item()) if rec_batch.numel() > 0 else -1) + 1)
    tr_guidance = torch.zeros((num_graphs, 3), device=pep_pos.device, dtype=pep_pos.dtype)
    rot_guidance = torch.zeros((num_graphs, 3), device=pep_pos.device, dtype=pep_pos.dtype)
    energy = torch.zeros((num_graphs,), device=pep_pos.device, dtype=pep_pos.dtype)
    cutoff = float(max(cutoff, 1e-3))
    temperature = float(max(temperature, 1e-4))
    max_tr_norm = float(max(max_tr_norm, 0.0))
    max_rot_norm = float(max(max_rot_norm, 0.0))

    for g in range(num_graphs):
        pep_mask = pep_batch == g
        rec_mask = rec_batch == g
        if not pep_mask.any() or not rec_mask.any():
            continue
        pep_g = pep_pos[pep_mask]
        rec_g = rec_pos[rec_mask]
        diff = pep_g[:, None, :] - rec_g[None, :, :]
        dists = diff.norm(dim=-1).clamp_min(1e-4)
        weights = torch.sigmoid((cutoff - dists) / temperature)
        force_atom = (weights.unsqueeze(-1) * (diff / dists.unsqueeze(-1))).sum(dim=1)
        tr_vec = force_atom.mean(dim=0)
        if max_tr_norm > 0:
            tr_norm = tr_vec.norm()
            if tr_norm > max_tr_norm:
                tr_vec = tr_vec * (max_tr_norm / tr_norm.clamp_min(1e-6))
        center = pep_g.mean(dim=0)
        torque_atom = torch.cross(pep_g - center, force_atom, dim=-1)
        rot_vec = torque_atom.mean(dim=0)
        if max_rot_norm > 0:
            rot_norm = rot_vec.norm()
            if rot_norm > max_rot_norm:
                rot_vec = rot_vec * (max_rot_norm / rot_norm.clamp_min(1e-6))
        tr_guidance[g] = tr_vec
        rot_guidance[g] = rot_vec
        energy[g] = weights.mean()
    return tr_guidance, rot_guidance, energy
