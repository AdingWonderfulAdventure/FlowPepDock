import time
import torch
import numpy as np
from typing import Optional
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from utils.flow_utils import set_time
from utils.peptide_updater import peptide_updater, _get_torsion_edge_counts
from utils.geometry import kabsch_torch, matrix_to_axis_angle, rot6d_to_matrix
from utils.steric import apply_rigid_body_updates, steric_rigid_guidance

def _cuda_sync(device, enabled=False):
    if enabled and isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def _empty_sampling_timing_stats():
    return {
        "forward_seconds": 0.0,
        "update_seconds": 0.0,
        "final_refine_seconds": 0.0,
    }


def _flow_overlap_guard(
    pep_pos: torch.Tensor,
    pep_batch: torch.Tensor,
    rec_pos: torch.Tensor,
    rec_batch: torch.Tensor,
    tr_update: torch.Tensor,
    rot_update: torch.Tensor,
    min_global_dist: float,
    backoff: float,
    max_backtracks: int,
    repair_backbone_flags: Optional[torch.Tensor] = None,
):
    if min_global_dist <= 0 or max_backtracks <= 0:
        return tr_update, rot_update, None
    safe_tr = tr_update.clone()
    safe_rot = rot_update.clone()
    num_graphs = int(tr_update.shape[0])
    final_min = torch.zeros((num_graphs,), device=tr_update.device, dtype=tr_update.dtype)
    backoff = float(min(max(backoff, 1e-3), 0.999))
    for g in range(num_graphs):
        pep_mask = pep_batch == g
        rec_mask = rec_batch == g
        if not pep_mask.any() or not rec_mask.any():
            continue
        pep_g = pep_pos[pep_mask]
        rec_g = rec_pos[rec_mask]
        cur_tr = safe_tr[g : g + 1]
        cur_rot = safe_rot[g : g + 1]
        current_dmat = torch.cdist(pep_g, rec_g)
        current_min = current_dmat.min()
        current_atom_min = current_dmat.min(dim=1).values
        current_collide = (current_atom_min < float(min_global_dist)).float().mean()
        is_repair = False
        if repair_backbone_flags is not None and g < int(repair_backbone_flags.shape[0]):
            is_repair = bool(repair_backbone_flags[g].item() > 0)
        min_drop_tol = 0.0 if is_repair else 0.05
        collide_increase_tol = 0.015 if is_repair else 0.04
        min_required = max(float(min_global_dist), float(current_min.item()) - min_drop_tol)
        min_d = None
        for _ in range(max_backtracks + 1):
            moved, _ = apply_rigid_body_updates(
                pep_g,
                torch.zeros((pep_g.shape[0],), device=pep_g.device, dtype=torch.long),
                cur_tr,
                cur_rot,
            )
            dmat = torch.cdist(moved, rec_g)
            min_d = dmat.min()
            atom_min = dmat.min(dim=1).values
            collide_ratio = (atom_min < float(min_global_dist)).float().mean()
            if float(min_d.item()) >= min_required and float(collide_ratio.item()) <= float(current_collide.item()) + collide_increase_tol:
                break
            cur_tr = cur_tr * backoff
            cur_rot = cur_rot * backoff
        safe_tr[g] = cur_tr[0]
        safe_rot[g] = cur_rot[0]
        final_min[g] = min_d if min_d is not None else torch.tensor(0.0, device=pep_g.device, dtype=pep_g.dtype)
    return safe_tr, safe_rot, final_min


def _safe_normalize(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = vec.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return vec / norm


def _resolve_inference_device(args) -> torch.device:
    dev = getattr(args, "inference_device", None)
    if dev is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(dev, str):
        return torch.device(dev)
    if isinstance(dev, torch.device):
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _frame_from_rec_pca(rec_pos: torch.Tensor, pep_pos: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if rec_pos is None or rec_pos.numel() == 0:
        return None
    if rec_pos.shape[0] < 3:
        return None
    rec_com = rec_pos.mean(dim=0)
    centered = rec_pos - rec_com
    cov = centered.T @ centered
    _evals, evecs = torch.linalg.eigh(cov)
    x = _safe_normalize(evecs[:, 2])
    y = _safe_normalize(evecs[:, 1])
    z = torch.cross(x, y)
    if z.norm().item() < 1e-6:
        return None
    z = _safe_normalize(z)
    y = _safe_normalize(torch.cross(z, x))
    if pep_pos is not None and pep_pos.numel() > 0:
        pep_center = pep_pos.mean(dim=0)
        if torch.dot(z, pep_center - rec_com) < 0:
            z = -z
            y = -y
    return torch.stack([x, y, z], dim=1)


def _frame_from_ncac(
    pos: torch.Tensor,
    atom2res_index: torch.Tensor,
    atom2atomid_index: torch.Tensor,
    n_res: int,
    rec_pos: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if n_res <= 0:
        return None
    device = pos.device
    has_n = torch.zeros(n_res, dtype=torch.bool, device=device)
    has_ca = torch.zeros(n_res, dtype=torch.bool, device=device)
    has_c = torch.zeros(n_res, dtype=torch.bool, device=device)
    n_idx = atom2res_index[atom2atomid_index == 0]
    ca_idx = atom2res_index[atom2atomid_index == 1]
    c_idx = atom2res_index[atom2atomid_index == 2]
    if n_idx.numel() > 0:
        has_n[n_idx.unique()] = True
    if ca_idx.numel() > 0:
        has_ca[ca_idx.unique()] = True
    if c_idx.numel() > 0:
        has_c[c_idx.unique()] = True
    valid = has_n & has_ca & has_c
    if valid.sum().item() == 0:
        return None
    anchor_list = valid.nonzero(as_tuple=False).view(-1)
    anchor_idx = anchor_list[len(anchor_list) // 2]

    def _pick(atom_id: int) -> torch.Tensor | None:
        mask = (atom2res_index == anchor_idx) & (atom2atomid_index == atom_id)
        if mask.sum().item() == 0:
            return None
        return pos[mask][0]

    n = _pick(0)
    ca = _pick(1)
    c = _pick(2)
    if n is None or ca is None or c is None:
        return None

    x = _safe_normalize(c - ca)
    y = _safe_normalize(n - ca)
    z = torch.cross(x, y)
    if z.norm().item() < 1e-8:
        return None
    z = _safe_normalize(z)
    y = _safe_normalize(torch.cross(z, x))

    if rec_pos is not None and rec_pos.numel() > 0:
        rec_com = rec_pos.mean(dim=0)
        if torch.dot(z, ca - rec_com) < 0:
            z = -z
            y = -y

    return torch.stack([x, y, z], dim=1)


def _select_rec_interface(
    rec_pos: Optional[torch.Tensor],
    pep_pos: Optional[torch.Tensor],
    cutoff: float,
    min_points: int,
) -> Optional[torch.Tensor]:
    if rec_pos is None or rec_pos.numel() == 0:
        return None
    if pep_pos is None or pep_pos.numel() == 0:
        return None
    if rec_pos.shape[0] < min_points:
        return None
    dmat = torch.cdist(rec_pos, pep_pos)
    min_d = dmat.min(dim=1).values
    mask = min_d <= float(cutoff)
    if mask.sum().item() < int(min_points):
        return None
    return rec_pos[mask]


def _flow_rot_update_frame_vector(
    graphs,
    rot_pred: torch.Tensor,
    current_t: float,
    t_min: float,
    dt: float,
    flow_cfg: Optional[dict] = None,
) -> torch.Tensor:
    """frame_vector 推理专用 rot 更新：
    - rot_pred 是预测的 R_hat（rec_frame @ pep_frame0^T）的 6D/9D 表示（batch维度）
    - 将其转成“当前 pep_frame -> 目标 pep_frame0_pred”的相对旋转，然后按 remaining 做 step_scale
    - 输出 axis-angle (B,3)，供 peptide_updater 使用
    """
    device = rot_pred.device if isinstance(rot_pred, torch.Tensor) else None
    b = len(graphs)
    if b == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=device)
    if rot_pred.ndim == 1:
        rot_pred = rot_pred.unsqueeze(0)
    if rot_pred.shape[0] != b:
        rot_pred = rot_pred[:b]

    if rot_pred.shape[-1] == 6:
        rot_hat = rot6d_to_matrix(rot_pred)  # (B,3,3)
    elif rot_pred.shape[-1] == 9:
        rot_hat = rot_pred.reshape(-1, 3, 3)
    else:
        return torch.zeros((b, 3), dtype=torch.float32, device=device)

    remaining = max(current_t - t_min, 1e-6)
    step_scale = dt / remaining
    flow_cfg = flow_cfg or {}
    contact_only = bool(flow_cfg.get("rot_frame_contact_only", False))
    contact_cutoff = float(flow_cfg.get("rot_frame_contact_cutoff", 8.0) or 8.0)
    contact_min_points = int(flow_cfg.get("rot_frame_contact_min_points", 6) or 6)
    contact_fallback = bool(flow_cfg.get("rot_frame_contact_fallback", True))
    updates = []
    for i, g in enumerate(graphs):
        try:
            rec_pos = g["receptor"].pos
            pep_pos = g["pep_a"].pos
            atom2res_index = g["pep_a"].atom2res_index
            atom2atomid_index = g["pep_a"].atom2atomid_index
            n_res = int(g["pep"].x.shape[0]) if "pep" in g.node_types else int(atom2res_index.max().item() + 1)
            rec_frame = None
            if contact_only:
                rec_iface = _select_rec_interface(rec_pos, pep_pos, contact_cutoff, contact_min_points)
                if rec_iface is None:
                    if contact_fallback:
                        rec_frame = _frame_from_rec_pca(rec_pos, pep_pos)
                    else:
                        updates.append(torch.zeros(3, dtype=torch.float32, device=device))
                        continue
                else:
                    rec_frame = _frame_from_rec_pca(rec_iface, pep_pos)
            else:
                rec_frame = _frame_from_rec_pca(rec_pos, pep_pos)
            pep_frame = _frame_from_ncac(pep_pos, atom2res_index, atom2atomid_index, n_res, rec_pos)
            if rec_frame is None or pep_frame is None:
                updates.append(torch.zeros(3, dtype=torch.float32, device=device))
                continue
            pep_frame_target = rec_frame.T @ rot_hat[i].to(dtype=torch.float32)
            delta_R = pep_frame_target @ pep_frame.T
            diff_vec = matrix_to_axis_angle(delta_R.to(dtype=torch.float32))
            updates.append(step_scale * diff_vec)
        except Exception:
            updates.append(torch.zeros(3, dtype=torch.float32, device=device))
    return torch.stack(updates, dim=0)


def _flow_rot_update_frame_local(
    graphs,
    rot_pred: torch.Tensor,
    current_t: float,
    t_min: float,
    dt: float,
    flow_cfg: Optional[dict] = None,
) -> torch.Tensor:
    """frame_local 推理专用 rot 更新：
    - rot_pred 是预测的 R_local（rec_frame^T @ pep_frame*）的 6D/9D 表示
    - 先还原到全局目标 pep_frame，再计算与当前 pep_frame 的相对旋转
    - 输出 axis-angle (B,3)
    """
    device = rot_pred.device if isinstance(rot_pred, torch.Tensor) else None
    b = len(graphs)
    if b == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=device)
    if rot_pred.ndim == 1:
        rot_pred = rot_pred.unsqueeze(0)
    if rot_pred.shape[0] != b:
        rot_pred = rot_pred[:b]

    if rot_pred.shape[-1] == 6:
        rot_hat = rot6d_to_matrix(rot_pred)  # (B,3,3)
    elif rot_pred.shape[-1] == 9:
        rot_hat = rot_pred.reshape(-1, 3, 3)
    else:
        return torch.zeros((b, 3), dtype=torch.float32, device=device)

    remaining = max(current_t - t_min, 1e-6)
    step_scale = dt / remaining
    flow_cfg = flow_cfg or {}
    contact_only = bool(flow_cfg.get("rot_frame_contact_only", False))
    contact_cutoff = float(flow_cfg.get("rot_frame_contact_cutoff", 8.0) or 8.0)
    contact_min_points = int(flow_cfg.get("rot_frame_contact_min_points", 6) or 6)
    contact_fallback = bool(flow_cfg.get("rot_frame_contact_fallback", True))
    updates = []
    for i, g in enumerate(graphs):
        try:
            rec_pos = g["receptor"].pos
            pep_pos = g["pep_a"].pos
            atom2res_index = g["pep_a"].atom2res_index
            atom2atomid_index = g["pep_a"].atom2atomid_index
            n_res = int(g["pep"].x.shape[0]) if "pep" in g.node_types else int(atom2res_index.max().item() + 1)
            rec_frame = None
            if contact_only:
                rec_iface = _select_rec_interface(rec_pos, pep_pos, contact_cutoff, contact_min_points)
                if rec_iface is None:
                    if contact_fallback:
                        rec_frame = _frame_from_rec_pca(rec_pos, pep_pos)
                    else:
                        updates.append(torch.zeros(3, dtype=torch.float32, device=device))
                        continue
                else:
                    rec_frame = _frame_from_rec_pca(rec_iface, pep_pos)
            else:
                rec_frame = _frame_from_rec_pca(rec_pos, pep_pos)
            pep_frame = _frame_from_ncac(pep_pos, atom2res_index, atom2atomid_index, n_res, rec_pos)
            if rec_frame is None or pep_frame is None:
                updates.append(torch.zeros(3, dtype=torch.float32, device=device))
                continue
            pep_frame_target = rec_frame @ rot_hat[i].to(dtype=torch.float32)
            delta_R = pep_frame_target @ pep_frame.T
            diff_vec = matrix_to_axis_angle(delta_R.to(dtype=torch.float32))
            updates.append(step_scale * diff_vec)
        except Exception:
            updates.append(torch.zeros(3, dtype=torch.float32, device=device))
    return torch.stack(updates, dim=0)


def _apply_updates_to_data_list(
    data_list,
    tr_update,
    rot_update,
    tor_backbone_update,
    tor_sidechain_update,
    torsion_device="cpu",
    torsion_debug=False,
):
    """将 tr/rot/tor 更新应用到一个 data_list 切片（原地更新）。"""
    bb_offset = 0
    sc_offset = 0
    for i, complex_graph in enumerate(data_list):
        n_bb, n_sc = _get_torsion_edge_counts(complex_graph)
        bb_slice = None
        if tor_backbone_update is not None and n_bb > 0:
            end = min(bb_offset + n_bb, len(tor_backbone_update))
            bb_slice = tor_backbone_update[bb_offset:end]
            bb_offset = end
        sc_slice = None
        if tor_sidechain_update is not None and n_sc > 0:
            end = min(sc_offset + n_sc, len(tor_sidechain_update))
            sc_slice = tor_sidechain_update[sc_offset:end]
            sc_offset = end
        tr_slice = torch.as_tensor(tr_update[i : i + 1])
        rot_slice = torch.as_tensor(rot_update[i : i + 1])
        pep_device = complex_graph["pep_a"].pos.device
        if isinstance(tr_slice, torch.Tensor) and tr_slice.device != pep_device:
            tr_slice = tr_slice.to(pep_device)
        if isinstance(rot_slice, torch.Tensor) and rot_slice.device != pep_device:
            rot_slice = rot_slice.to(pep_device)
        data_list[i] = peptide_updater(
            complex_graph,
            tr_slice,
            rot_slice.squeeze(0),
            bb_slice,
            sc_slice,
            torsion_device=torsion_device,
            torsion_debug=torsion_debug and i == 0,
        )
    return data_list


def sampling(
    data_list,
    model,
    args,
    inference_steps=20,
    no_random=False,
    ode=False,
    visualization_list=None,
    confidence_model=None,
    batch_size=32,
    no_final_step_noise=False,
    actual_steps=None,
    sampling_mode="flow",
    flow_num_steps=50,
    flow_solver="euler",
):
    """核心 flow 采样循环，仅支持 Rectified Flow ODE。"""
    if sampling_mode != "flow":
        raise ValueError(f"sampling_mode 只支持 flow，收到 {sampling_mode}")
    sampler = _flow_matching_sampling

    data_list = sampler(
        data_list=data_list,
        model=model,
        args=args,
        inference_steps=inference_steps,
        no_random=no_random,
        ode=ode,
        visualization_list=visualization_list,
        batch_size=batch_size,
        no_final_step_noise=no_final_step_noise,
        actual_steps=actual_steps,
        flow_num_steps=flow_num_steps,
        flow_solver=flow_solver,
    )
    data_list, visualization_list = data_list

    device = _resolve_inference_device(args)
    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence = []
            for complex_graph_batch in loader:
                b = complex_graph_batch.num_graphs
                set_time(complex_graph_batch, 0, 0, 0, 0, b, device)
                complex_graph_batch = complex_graph_batch.to(device)
                confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None
    return data_list, confidence, visualization_list


def _flow_matching_sampling(
    data_list,
    model,
    args,
    inference_steps,
    no_random,
    ode,
    visualization_list,
    batch_size,
    no_final_step_noise,
    actual_steps,
    flow_num_steps,
    flow_solver,
):
    del inference_steps, no_random, ode, no_final_step_noise, actual_steps  # unused in flow模式
    if flow_num_steps is None or flow_num_steps <= 0:
        raise ValueError("flow_num_steps必须是正整数。")
    timing_stats = _empty_sampling_timing_stats()
    device = _resolve_inference_device(args)
    flow_cfg = getattr(args, "flow", {}) or {}
    t_min = float(flow_cfg.get("t_min", 0.0) or 0.0)
    t_max = float(flow_cfg.get("t_max", 1.0) or 1.0)
    rot_target_mode = str(flow_cfg.get("rot_target_mode", "noise") or "noise").lower()
    if not (0.0 <= t_min <= t_max <= 1.0):
        raise ValueError(
            f"[flow] invalid t_min/t_max: t_min={t_min} t_max={t_max}, expected 0<=t_min<=t_max<=1"
        )
    span = t_max - t_min
    if span <= 0:
        # 固定t推理：保持原始步长，重复同一t
        span = 1.0
        fixed_t = t_min
    else:
        fixed_t = None
    dt = span / flow_num_steps
    flow_solver = flow_solver.lower()
    solver_supported = {"euler", "heun"}
    if flow_solver not in solver_supported:
        raise ValueError(f"flow_solver只支持{solver_supported}，收到{flow_solver}")
    rot_oracle = bool(getattr(args, "rot_oracle", False))
    if rot_target_mode in {"rel", "x1"} and flow_solver == "heun":
        print("[flow] rot_target_mode=rel/x1 与 heun 不兼容，回退到 euler")
        flow_solver = "euler"
    if rot_target_mode in {"frame_vector", "frame_vec", "frame_xt", "frame_local", "frame_local_xt"} and flow_solver == "heun":
        print("[flow] rot_target_mode=frame_vector/frame_xt/frame_local/frame_local_xt 与 heun 不兼容，回退到 euler")
        flow_solver = "euler"
    if rot_oracle and flow_solver == "heun":
        print("[flow] rot_oracle enabled; fallback to euler")
        flow_solver = "euler"
    steric_guidance = bool(
        getattr(args, "flow_steric_guidance", flow_cfg.get("steric_guidance", False))
    )
    if steric_guidance and flow_solver == "heun":
        print("[flow] steric_guidance enabled; fallback to euler")
        flow_solver = "euler"
    steric_guidance_scale = float(
        getattr(args, "flow_steric_guidance_scale", flow_cfg.get("steric_guidance_scale", 0.15))
        or 0.0
    )
    steric_guidance_cutoff = float(
        getattr(args, "flow_steric_guidance_cutoff", flow_cfg.get("steric_guidance_cutoff", 3.6))
        or 3.6
    )
    steric_guidance_temperature = float(
        getattr(args, "flow_steric_guidance_temperature", flow_cfg.get("steric_guidance_temperature", 0.35))
        or 0.35
    )
    steric_guidance_torque_scale = float(
        getattr(args, "flow_steric_guidance_torque_scale", flow_cfg.get("steric_guidance_torque_scale", 0.35))
        or 0.0
    )
    steric_guidance_max_tr = float(
        getattr(args, "flow_steric_guidance_max_tr", flow_cfg.get("steric_guidance_max_tr", 2.0))
        or 0.0
    )
    steric_guidance_max_rot = float(
        getattr(args, "flow_steric_guidance_max_rot", flow_cfg.get("steric_guidance_max_rot", 0.5))
        or 0.0
    )
    overlap_guard = bool(
        getattr(args, "flow_hard_overlap_guard", flow_cfg.get("hard_overlap_guard", False))
    )
    overlap_guard_min_dist = float(
        getattr(args, "flow_hard_overlap_guard_min_dist", flow_cfg.get("hard_overlap_guard_min_dist", 1.6))
        or 0.0
    )
    overlap_guard_backoff = float(
        getattr(args, "flow_hard_overlap_guard_backoff", flow_cfg.get("hard_overlap_guard_backoff", 0.5))
        or 0.5
    )
    overlap_guard_max_backtracks = int(
        getattr(args, "flow_hard_overlap_guard_max_backtracks", flow_cfg.get("hard_overlap_guard_max_backtracks", 4))
        or 0
    )
    overlap_guard_last_steps = int(
        getattr(args, "flow_hard_overlap_guard_last_steps", flow_cfg.get("hard_overlap_guard_last_steps", 0))
        or 0
    )

    use_amp = bool(getattr(args, "inference_amp", False))
    use_inference_mode = bool(getattr(args, "inference_mode", True))
    use_timing = bool(getattr(args, "inference_timing", False))
    force_cuda_sync = bool(getattr(args, "timing_force_cuda_sync", False))
    self_cond_infer = bool(
        getattr(args, "flow_self_condition_infer", flow_cfg.get("self_condition", False))
    )
    final_refine = bool(
        getattr(args, "flow_final_refine", flow_cfg.get("final_refine", False))
    )
    final_refine_scale = float(
        getattr(args, "flow_final_refine_scale", flow_cfg.get("final_refine_scale", 0.35))
        or 0.35
    )
    final_refine_tr_scale = float(
        getattr(args, "flow_final_refine_tr_scale", flow_cfg.get("final_refine_tr_scale", 0.0))
        or 0.0
    )
    final_refine_rot_scale = float(
        getattr(args, "flow_final_refine_rot_scale", flow_cfg.get("final_refine_rot_scale", 0.35))
        or 0.35
    )
    final_refine_tor_scale = float(
        getattr(args, "flow_final_refine_tor_scale", flow_cfg.get("final_refine_tor_scale", 0.35))
        or 0.35
    )
    torsion_device = str(getattr(args, "torsion_device", "cpu") or "cpu").lower()
    torsion_debug = bool(getattr(args, "torsion_debug", False))
    gpu_update_fastpath = bool(getattr(args, "gpu_update_fastpath", False)) and device.type == "cuda"
    if gpu_update_fastpath and flow_solver == "heun":
        print("[flow] gpu_update_fastpath 与 heun 不兼容，回退到 euler")
        flow_solver = "euler"
    if gpu_update_fastpath and torsion_device != "gpu":
        print("[flow] gpu_update_fastpath 强制启用 torsion_device=gpu")
        torsion_device = "gpu"
    chunk_indices = _build_index_chunks(len(data_list), batch_size)

    if gpu_update_fastpath or (torsion_device == "gpu" and device.type == "cuda"):
        data_list = [g.to(device) for g in data_list]
    self_cond_tr_buf = None
    self_cond_rot_buf = None
    if self_cond_infer:
        self_cond_tr_buf = torch.zeros((len(data_list), 3), dtype=torch.float32)
        self_cond_rot_buf = torch.zeros((len(data_list), 3), dtype=torch.float32)

    for step_idx in range(flow_num_steps):
        step_start = time.perf_counter()
        t_build = 0.0
        t_to_device = 0.0
        t_set_time = 0.0
        t_forward = 0.0
        t_update = 0.0
        if fixed_t is None:
            current_t = t_max - step_idx * dt
            next_t = t_min if step_idx == flow_num_steps - 1 else t_max - (step_idx + 1) * dt
        else:
            current_t = fixed_t
            next_t = fixed_t
        for start, end in chunk_indices:
            build_start = time.perf_counter()
            complex_graph_batch = Batch.from_data_list(data_list[start:end])
            t_build += time.perf_counter() - build_start
            b = complex_graph_batch.num_graphs
            set_start = time.perf_counter()
            set_time(
                complex_graph_batch,
                current_t,
                current_t,
                current_t,
                current_t,
                b,
                device,
            )
            t_set_time += time.perf_counter() - set_start
            to_dev_start = time.perf_counter()
            if gpu_update_fastpath and complex_graph_batch["pep_a"].pos.device == device:
                batch_on_device = complex_graph_batch
            else:
                batch_on_device = complex_graph_batch.to(device)
            if self_cond_infer and self_cond_tr_buf is not None and self_cond_rot_buf is not None:
                batch_on_device.self_cond_tr = self_cond_tr_buf[start:end].to(
                    device=batch_on_device["pep_a"].pos.device, dtype=torch.float32
                )
                batch_on_device.self_cond_rot = self_cond_rot_buf[start:end].to(
                    device=batch_on_device["pep_a"].pos.device, dtype=torch.float32
                )
            t_to_device += time.perf_counter() - to_dev_start
            _cuda_sync(device, enabled=force_cuda_sync)
            fwd_start = time.perf_counter()
            with torch.inference_mode() if use_inference_mode else torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                    outputs = model(batch_on_device)
            _cuda_sync(device, enabled=force_cuda_sync)
            t_forward += time.perf_counter() - fwd_start
            if self_cond_infer and self_cond_tr_buf is not None and self_cond_rot_buf is not None:
                tr_sc = outputs.get("tr_pred") if isinstance(outputs, dict) else None
                rot_sc = outputs.get("rot_pred") if isinstance(outputs, dict) else None
                if tr_sc is not None and tr_sc.ndim == 2 and tr_sc.shape[-1] == 3:
                    self_cond_tr_buf[start:end] = tr_sc.detach().to("cpu", dtype=torch.float32)
                if rot_sc is not None and rot_sc.ndim == 2 and rot_sc.shape[-1] == 3:
                    self_cond_rot_buf[start:end] = rot_sc.detach().to("cpu", dtype=torch.float32)
            use_gpu_torsion = torsion_device == "gpu" and device.type == "cuda"
            if use_gpu_torsion:
                tr_vel = outputs["tr_pred"].detach()
                rot_vel = outputs["rot_pred"].detach()
                tor_backbone_vel = outputs["tor_pred_backbone"]
                tor_sidechain_vel = outputs["tor_pred_sidechain"]
                tor_backbone_vel = (
                    tor_backbone_vel.detach()
                    if tor_backbone_vel is not None and tor_backbone_vel.numel() > 0
                    else None
                )
                tor_sidechain_vel = (
                    tor_sidechain_vel.detach()
                    if tor_sidechain_vel is not None and tor_sidechain_vel.numel() > 0
                    else None
                )
            else:
                tr_vel = outputs["tr_pred"].detach().cpu()
                rot_vel = outputs["rot_pred"].detach().cpu()
                tor_backbone_vel = outputs["tor_pred_backbone"]
                tor_sidechain_vel = outputs["tor_pred_sidechain"]
                tor_backbone_vel = (
                    tor_backbone_vel.detach().cpu()
                    if tor_backbone_vel is not None and tor_backbone_vel.numel() > 0
                    else None
                )
                tor_sidechain_vel = (
                    tor_sidechain_vel.detach().cpu()
                    if tor_sidechain_vel is not None and tor_sidechain_vel.numel() > 0
                    else None
                )
            if steric_guidance and steric_guidance_scale > 0:
                guide_tr, guide_rot, _guide_energy = steric_rigid_guidance(
                    batch_on_device["pep_a"].pos.detach(),
                    batch_on_device["pep_a"].batch.detach(),
                    batch_on_device["receptor"].pos.detach(),
                    batch_on_device["receptor"].batch.detach(),
                    cutoff=steric_guidance_cutoff,
                    temperature=steric_guidance_temperature,
                    max_tr_norm=steric_guidance_max_tr,
                    max_rot_norm=steric_guidance_max_rot,
                )
                guide_tr = guide_tr.to(device=tr_vel.device, dtype=tr_vel.dtype)
                tr_vel = tr_vel - steric_guidance_scale * guide_tr
                if rot_vel is not None and rot_vel.ndim >= 1 and rot_vel.shape[-1] == 3:
                    guide_rot = guide_rot.to(device=rot_vel.device, dtype=rot_vel.dtype)
                    rot_vel = rot_vel - (steric_guidance_scale * steric_guidance_torque_scale) * guide_rot
            torsions_backbone_per_molecule = (
                tor_backbone_vel.shape[0] // b if tor_backbone_vel is not None else None
            )
            torsions_sidechain_per_molecule = (
                tor_sidechain_vel.shape[0] // b if tor_sidechain_vel is not None else None
            )

            update_start = time.perf_counter()
            if flow_solver == "heun":
                batch_for_update = batch_on_device if use_gpu_torsion else complex_graph_batch
                data_list[start:end] = _heun_flow_step(
                    batch_for_update,
                    model,
                    device,
                    dt,
                    next_t,
                    tr_vel,
                    rot_vel,
                    tor_backbone_vel,
                    tor_sidechain_vel,
                    torsions_backbone_per_molecule,
                    torsions_sidechain_per_molecule,
                    use_amp,
                    torsion_device=torsion_device,
                    torsion_debug=torsion_debug and step_idx == 0,
                )
            else:
                tr_update = -dt * tr_vel
                if rot_oracle:
                    rot_update = _rot_oracle_update(complex_graph_batch)
                else:
                    if rot_target_mode in {"rel", "x1"}:
                        remaining = max(current_t - t_min, 1e-6)
                        step_scale = dt / remaining
                        if rot_target_mode == "x1":
                            rot_mat = rot6d_to_matrix(rot_vel)
                            diff_vec = matrix_to_axis_angle(rot_mat)
                        else:
                            diff_vec = rot_vel
                        rot_update = step_scale * diff_vec
                    elif rot_target_mode in {"frame_vector", "frame_vec", "frame_xt"}:
                        rot_update = _flow_rot_update_frame_vector(
                            data_list[start:end], rot_vel, current_t, t_min, dt, flow_cfg
                        )
                    elif rot_target_mode in {"frame_local", "frame_local_xt"}:
                        rot_update = _flow_rot_update_frame_local(
                            data_list[start:end], rot_vel, current_t, t_min, dt, flow_cfg
                        )
                    else:
                        rot_update = -dt * rot_vel
                guard_active = overlap_guard and overlap_guard_min_dist > 0
                if guard_active and overlap_guard_last_steps > 0:
                    guard_active = step_idx >= max(0, flow_num_steps - overlap_guard_last_steps)
                if guard_active:
                    ref_batch = batch_on_device if use_gpu_torsion else batch_on_device
                    repair_backbone_flags = torch.tensor(
                        [
                            1 if bool(getattr(graph, "repair_backbone_used", False)) else 0
                            for graph in data_list[start:end]
                        ],
                        device=ref_batch["pep_a"].pos.device,
                        dtype=ref_batch["pep_a"].pos.dtype,
                    )
                    tr_update, rot_update, guarded_min = _flow_overlap_guard(
                        ref_batch["pep_a"].pos.detach(),
                        ref_batch["pep_a"].batch.detach(),
                        ref_batch["receptor"].pos.detach(),
                        ref_batch["receptor"].batch.detach(),
                        tr_update.to(device=ref_batch["pep_a"].pos.device, dtype=ref_batch["pep_a"].pos.dtype),
                        rot_update.to(device=ref_batch["pep_a"].pos.device, dtype=ref_batch["pep_a"].pos.dtype),
                        min_global_dist=overlap_guard_min_dist,
                        backoff=overlap_guard_backoff,
                        max_backtracks=overlap_guard_max_backtracks,
                        repair_backbone_flags=repair_backbone_flags,
                    )
                    if not use_gpu_torsion:
                        tr_update = tr_update.detach().cpu()
                        rot_update = rot_update.detach().cpu()
                if use_gpu_torsion:
                    tor_bb_update = (-dt * tor_backbone_vel) if tor_backbone_vel is not None else None
                    tor_sc_update = (-dt * tor_sidechain_vel) if tor_sidechain_vel is not None else None
                else:
                    tor_bb_update = (
                        (-dt * tor_backbone_vel.numpy()) if tor_backbone_vel is not None else None
                    )
                    tor_sc_update = (
                        (-dt * tor_sidechain_vel.numpy()) if tor_sidechain_vel is not None else None
                    )
                data_list[start:end] = _apply_updates_to_data_list(
                    data_list[start:end],
                    tr_update,
                    rot_update,
                    tor_bb_update,
                    tor_sc_update,
                    torsion_device=torsion_device,
                    torsion_debug=torsion_debug and step_idx == 0,
                )
            t_update += time.perf_counter() - update_start

        if visualization_list is not None:
            visualization_list.append(
                np.asarray(
                    [
                        complex_graph["pep_a"].pos.cpu().numpy()
                        + complex_graph.original_center.cpu().numpy()
                        for complex_graph in data_list
                    ]
                )
            )
        if use_timing:
            step_total = time.perf_counter() - step_start
            print(
                f"[timing] step {step_idx + 1}/{flow_num_steps} "
                f"build={t_build:.3f}s to_device={t_to_device:.3f}s set_time={t_set_time:.3f}s "
                f"forward={t_forward:.3f}s update={t_update:.3f}s total={step_total:.3f}s"
            )
        timing_stats["forward_seconds"] += float(t_forward)
        timing_stats["update_seconds"] += float(t_update)
    if final_refine and len(data_list) > 0:
        refine_start = time.perf_counter()
        dt_ref = dt if dt > 0 else (1.0 / max(1, flow_num_steps))
        ref_scale = dt_ref * final_refine_scale
        for start, end in chunk_indices:
            batch = Batch.from_data_list(data_list[start:end])
            set_time(batch, t_min, t_min, t_min, t_min, batch.num_graphs, device)
            batch = batch.to(device)
            if self_cond_infer and self_cond_tr_buf is not None and self_cond_rot_buf is not None:
                batch.self_cond_tr = self_cond_tr_buf[start:end].to(device=device, dtype=torch.float32)
                batch.self_cond_rot = self_cond_rot_buf[start:end].to(device=device, dtype=torch.float32)
            with torch.inference_mode() if use_inference_mode else torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                    outputs = model(batch)
            tr_vel = outputs["tr_pred"].detach()
            rot_vel = outputs["rot_pred"].detach()
            tor_backbone_vel = outputs["tor_pred_backbone"]
            tor_sidechain_vel = outputs["tor_pred_sidechain"]
            tr_update = -ref_scale * final_refine_tr_scale * tr_vel
            rot_update = -ref_scale * final_refine_rot_scale * rot_vel
            if torsion_device == "gpu" and device.type == "cuda":
                tor_bb_update = (
                    -ref_scale * final_refine_tor_scale * tor_backbone_vel.detach()
                    if tor_backbone_vel is not None and tor_backbone_vel.numel() > 0
                    else None
                )
                tor_sc_update = (
                    -ref_scale * final_refine_tor_scale * tor_sidechain_vel.detach()
                    if tor_sidechain_vel is not None and tor_sidechain_vel.numel() > 0
                    else None
                )
            else:
                tor_bb_update = (
                    (-ref_scale * final_refine_tor_scale * tor_backbone_vel.detach().cpu().numpy())
                    if tor_backbone_vel is not None and tor_backbone_vel.numel() > 0
                    else None
                )
                tor_sc_update = (
                    (-ref_scale * final_refine_tor_scale * tor_sidechain_vel.detach().cpu().numpy())
                    if tor_sidechain_vel is not None and tor_sidechain_vel.numel() > 0
                    else None
                )
            data_list[start:end] = _apply_updates_to_data_list(
                data_list[start:end],
                tr_update,
                rot_update,
                tor_bb_update,
                tor_sc_update,
                torsion_device=torsion_device,
                torsion_debug=torsion_debug,
            )
            if self_cond_infer and self_cond_tr_buf is not None and self_cond_rot_buf is not None:
                if tr_vel.ndim == 2 and tr_vel.shape[-1] == 3:
                    self_cond_tr_buf[start:end] = tr_vel.to("cpu", dtype=torch.float32)
                if rot_vel.ndim == 2 and rot_vel.shape[-1] == 3:
                    self_cond_rot_buf[start:end] = rot_vel.to("cpu", dtype=torch.float32)
        refine_elapsed = time.perf_counter() - refine_start
        if use_timing:
            print(f"[timing] final_refine total={refine_elapsed:.3f}s")
        timing_stats["final_refine_seconds"] += float(refine_elapsed)
    setattr(args, "_last_sampling_timing", timing_stats)
    return data_list, visualization_list


def _apply_perturbations_to_batch(
    batch,
    tr_update,
    rot_update,
    tor_backbone_update,
    tor_sidechain_update,
    torsions_backbone_per_molecule,
    torsions_sidechain_per_molecule,
    torsion_device="cpu",
    torsion_debug=False,
):
    """把给定扰动应用到一个batch的所有complex上。"""
    new_graphs = []
    pep_device = batch["pep_a"].pos.device
    use_gpu_torsion = torsion_device == "gpu" and pep_device.type == "cuda"
    batch_for_update = batch if use_gpu_torsion else (batch if pep_device.type == "cpu" else batch.to("cpu"))
    data_list = batch_for_update.to_data_list()
    bb_offset = 0
    sc_offset = 0
    for i, complex_graph in enumerate(data_list):
        n_bb, n_sc = _get_torsion_edge_counts(complex_graph)
        bb_slice = None
        if tor_backbone_update is not None and n_bb > 0:
            end = min(bb_offset + n_bb, len(tor_backbone_update))
            bb_slice = tor_backbone_update[bb_offset:end]
            bb_offset = end
        sc_slice = None
        if tor_sidechain_update is not None and n_sc > 0:
            end = min(sc_offset + n_sc, len(tor_sidechain_update))
            sc_slice = tor_sidechain_update[sc_offset:end]
            sc_offset = end
        tr_slice = tr_update[i : i + 1]
        rot_slice = rot_update[i : i + 1]
        if isinstance(tr_slice, torch.Tensor) and tr_slice.device != pep_device:
            tr_slice = tr_slice.to(pep_device)
        if isinstance(rot_slice, torch.Tensor) and rot_slice.device != pep_device:
            rot_slice = rot_slice.to(pep_device)
        new_graphs.append(
            peptide_updater(
                complex_graph,
                torch.as_tensor(tr_slice),
                torch.as_tensor(rot_slice).squeeze(0),
                bb_slice,
                sc_slice,
                torsion_device=torsion_device,
                torsion_debug=torsion_debug and i == 0,
            )
        )
    return new_graphs


def _build_index_chunks(total, batch_size):
    if total <= 0:
        return []
    if batch_size is None or batch_size <= 0:
        return [(0, total)]
    return [(i, min(i + batch_size, total)) for i in range(0, total, batch_size)]


def _rot_oracle_update(batch):
    pep = batch["pep_a"]
    if not hasattr(pep, "orig_pos"):
        return torch.zeros((batch.num_graphs, 3), dtype=torch.float32)
    updates = []
    for i in range(batch.num_graphs):
        mask = pep.batch == i
        if mask.sum().item() == 0:
            updates.append(torch.zeros(3, dtype=torch.float32))
            continue
        pos = pep.pos[mask].to(dtype=torch.float32)
        orig = pep.orig_pos[mask].to(dtype=torch.float32)
        atom_ids = pep.atom2atomid_index[mask]
        ca_mask = atom_ids == 1
        if ca_mask.sum().item() >= 3:
            cur = pos[ca_mask]
            ref = orig[ca_mask]
        else:
            cur = pos
            ref = orig
        if cur.shape[0] < 3 or ref.shape[0] < 3:
            updates.append(torch.zeros(3, dtype=torch.float32))
            continue
        try:
            R_hat, _t_hat = kabsch_torch(cur.T, ref.T)
            rot_vec = matrix_to_axis_angle(R_hat)
            if rot_vec.ndim == 2:
                rot_vec = rot_vec.squeeze(0)
            updates.append(rot_vec.to(dtype=torch.float32))
        except Exception:
            updates.append(torch.zeros(3, dtype=torch.float32))
    return torch.stack(updates, dim=0)


def _heun_flow_step(
    batch,
    model,
    device,
    dt,
    next_t,
    tr_vel,
    rot_vel,
    tor_backbone_vel,
    tor_sidechain_vel,
    torsions_backbone_per_molecule,
    torsions_sidechain_per_molecule,
    use_amp,
    torsion_device="cpu",
    torsion_debug=False,
):
    """Heun (RK2) solver for flow matching ODE."""
    use_gpu_torsion = torsion_device == "gpu" and device.type == "cuda"
    tr_delta1 = -dt * tr_vel
    rot_delta1 = -dt * rot_vel
    if use_gpu_torsion:
        tor_bb_delta1 = (-dt * tor_backbone_vel) if tor_backbone_vel is not None else None
        tor_sc_delta1 = (-dt * tor_sidechain_vel) if tor_sidechain_vel is not None else None
    else:
        tor_bb_delta1 = (
            (-dt * tor_backbone_vel.numpy()) if tor_backbone_vel is not None else None
        )
        tor_sc_delta1 = (
            (-dt * tor_sidechain_vel.numpy()) if tor_sidechain_vel is not None else None
        )

    mid_graphs = _apply_perturbations_to_batch(
        batch,
        tr_delta1,
        rot_delta1,
        tor_bb_delta1,
        tor_sc_delta1,
        torsions_backbone_per_molecule,
        torsions_sidechain_per_molecule,
        torsion_device=torsion_device,
        torsion_debug=torsion_debug,
    )
    mid_batch = Batch.from_data_list(mid_graphs).to(device)
    set_time(mid_batch, next_t, next_t, next_t, next_t, mid_batch.num_graphs, device)

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            mid_outputs = model(mid_batch)
    if use_gpu_torsion:
        tr_delta2 = -dt * mid_outputs["tr_pred"].detach()
        rot_delta2 = -dt * mid_outputs["rot_pred"].detach()
        tor_bb_delta2 = (
            (-dt * mid_outputs["tor_pred_backbone"].detach())
            if tor_backbone_vel is not None
            else None
        )
        tor_sc_delta2 = (
            (-dt * mid_outputs["tor_pred_sidechain"].detach())
            if tor_sidechain_vel is not None
            else None
        )
    else:
        tr_delta2 = -dt * mid_outputs["tr_pred"].detach().cpu()
        rot_delta2 = -dt * mid_outputs["rot_pred"].detach().cpu()
        tor_bb_delta2 = (
            (-dt * mid_outputs["tor_pred_backbone"].detach().cpu().numpy())
            if tor_backbone_vel is not None
            else None
        )
        tor_sc_delta2 = (
            (-dt * mid_outputs["tor_pred_sidechain"].detach().cpu().numpy())
            if tor_sidechain_vel is not None
            else None
        )

    avg_tr = 0.5 * (tr_delta1 + tr_delta2)
    avg_rot = 0.5 * (rot_delta1 + rot_delta2)
    avg_bb = None
    if tor_bb_delta1 is not None and tor_bb_delta2 is not None:
        avg_bb = 0.5 * (tor_bb_delta1 + tor_bb_delta2)
    avg_sc = None
    if tor_sc_delta1 is not None and tor_sc_delta2 is not None:
        avg_sc = 0.5 * (tor_sc_delta1 + tor_sc_delta2)

    return _apply_perturbations_to_batch(
        batch,
        avg_tr,
        avg_rot,
        avg_bb,
        avg_sc,
        torsions_backbone_per_molecule,
        torsions_sidechain_per_molecule,
        torsion_device=torsion_device,
        torsion_debug=torsion_debug,
    )
