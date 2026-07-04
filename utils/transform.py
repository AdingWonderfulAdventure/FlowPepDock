import os
import zlib
from typing import Optional

import torch
import numpy as np
from torch_geometric.transforms import BaseTransform
from utils.so3 import sample_vec, sample
from utils.peptide_updater import peptide_updater
from utils.flow_utils import set_time
from utils.flow_matching import wrap_to_pi
from utils.geometry import (
    axis_angle_to_matrix,
    kabsch_torch,
    matrix_to_axis_angle,
    matrix_to_rot6d,
    get_so3_codebook,
    nearest_rotmat_bin,
)

class FlowMatchingTransform(BaseTransform):
    """Rectified Flow 噪声采样，把x_t和速度真值都写进data"""

    def __init__(self, args):
        self.args = args
        self.flow_cfg = getattr(args, "flow", {}) or {}
        self.sigma_max = self._resolve_sigma_max()
        self.fixed_t = self.flow_cfg.get("fixed_t", None)
        self.deterministic_seed = self.flow_cfg.get("deterministic_seed", None)

    def _sample_time(self) -> float:
        """
        采样 t∈(0,1]。
        经验结论：纯 uniform 会让大量小 t 样本主导训练，tr/rot 很容易塌成“预测 0”的基线；
        所以这里做成可配置（t_min + 偏向大 t 的分布）。
        """
        cfg = self.flow_cfg or {}
        if self.fixed_t is not None:
            t = float(self.fixed_t)
            if not (0.0 <= t <= 1.0):
                raise ValueError(f"[flow] 非法 fixed_t={t}，需满足 0<=t<=1")
            return t
        t_min = float(cfg.get("t_min", 0.0) or 0.0)
        t_max = float(cfg.get("t_max", 1.0) or 1.0)
        # 允许 t_min==t_max（固定 t 调试/overfit 用），例如 t_min=t_max=1.0
        if not (0.0 <= t_min <= t_max <= 1.0):
            raise ValueError(
                f"[flow] 非法 t_min/t_max：t_min={t_min} t_max={t_max}，需满足 0<=t_min<=t_max<=1"
            )

        mode = str(cfg.get("time_sampling", "uniform") or "uniform").lower()
        already_scaled = False
        if mode == "uniform":
            t = float(np.random.rand())
        elif mode in {"sqrt", "sqrt_uniform"}:
            # 更偏向 1：t = sqrt(u)
            t = float(np.sqrt(np.random.rand()))
        elif mode == "beta":
            alpha = float(cfg.get("beta_alpha", 2.0) or 2.0)
            beta = float(cfg.get("beta_beta", 1.0) or 1.0)
            if alpha <= 0 or beta <= 0:
                raise ValueError(f"[flow] 非法 beta 参数：alpha={alpha} beta={beta}（需>0）")
            t = float(np.random.beta(alpha, beta))
        elif mode == "mixed":
            def _rand_uniform(lo, hi):
                return float(lo + (hi - lo) * np.random.rand())

            mix_fixed_prob = float(cfg.get("mix_fixed_prob", 0.0) or 0.0)
            mix_fixed_t = float(cfg.get("mix_fixed_t", 0.6) or 0.6)
            mix_beta_prob = float(cfg.get("mix_beta_prob", 0.0) or 0.0)
            mix_beta_alpha = float(cfg.get("mix_beta_alpha", 2.0) or 2.0)
            mix_beta_beta = float(cfg.get("mix_beta_beta", 1.0) or 1.0)
            mix_beta_min = float(cfg.get("mix_beta_min", t_min) or t_min)
            mix_beta_max = float(cfg.get("mix_beta_max", t_max) or t_max)
            mix_small_prob = float(cfg.get("mix_small_prob", 0.0) or 0.0)
            mix_small_min = float(cfg.get("mix_small_min", t_min) or t_min)
            mix_small_max = float(cfg.get("mix_small_max", t_max) or t_max)

            if not (0.0 <= mix_fixed_t <= 1.0):
                raise ValueError(f"[flow] 非法 mix_fixed_t={mix_fixed_t}（需在0~1）")
            if not (t_min <= mix_fixed_t <= t_max):
                raise ValueError(
                    f"[flow] mix_fixed_t={mix_fixed_t} 需在 t_min/t_max 范围内（{t_min}~{t_max}）"
                )
            def _check_range(name, lo, hi):
                if not (0.0 <= lo <= hi <= 1.0):
                    raise ValueError(f"[flow] 非法 {name} 范围：{lo}~{hi}（需在0~1）")
                if lo < t_min or hi > t_max:
                    raise ValueError(
                        f"[flow] {name} 范围需在 t_min/t_max 内：{lo}~{hi} vs {t_min}~{t_max}"
                    )

            _check_range("mix_beta", mix_beta_min, mix_beta_max)
            _check_range("mix_small", mix_small_min, mix_small_max)
            if mix_beta_alpha <= 0 or mix_beta_beta <= 0:
                raise ValueError(
                    f"[flow] 非法 mix_beta 参数：alpha={mix_beta_alpha} beta={mix_beta_beta}（需>0）"
                )
            total = mix_fixed_prob + mix_beta_prob + mix_small_prob
            if total <= 0:
                raise ValueError("[flow] mixed 采样权重和必须>0")
            r = np.random.rand() * total
            if r < mix_fixed_prob:
                t = mix_fixed_t
            elif r < mix_fixed_prob + mix_beta_prob:
                u = np.random.beta(mix_beta_alpha, mix_beta_beta)
                t = mix_beta_min + (mix_beta_max - mix_beta_min) * float(u)
            else:
                t = _rand_uniform(mix_small_min, mix_small_max)
            already_scaled = True
        elif mode in {"clipped", "clamp"}:
            # 只做截断：t∈[t_min,t_max]
            t = float(np.random.rand())
        else:
            raise ValueError(f"[flow] 不支持的 time_sampling={mode}（支持 uniform/sqrt/beta/clipped/mixed）")

        # 截断避免 t≈0 引起 tr/rot 监督不可辨识
        if not already_scaled:
            t = t_min + (t_max - t_min) * t
        eps = float(cfg.get("t_eps", 1e-6) or 1e-6)
        if t <= eps:
            t = eps
        return t

    def _sample_seed(self, data):
        if self.deterministic_seed is None:
            return None
        base_seed = int(self.deterministic_seed)
        name = getattr(data, "complex_name", None)
        if not name:
            name = getattr(data, "idx", None)
        key = str(name) if name is not None else "0"
        hashed = zlib.adler32(key.encode("utf-8")) & 0xFFFFFFFF
        return (base_seed + hashed) % (2**32)

    def _resolve_sigma_max(self):
        cfg = self.flow_cfg
        names = ["tr", "rot", "tor_backbone", "tor_sidechain"]
        values = [
            cfg.get("sigma_tr_max"),
            cfg.get("sigma_rot_max"),
            cfg.get("sigma_tor_bb_max"),
            cfg.get("sigma_tor_sc_max"),
        ]
        sigma = {}
        for name, value in zip(names, values):
            if value is None:
                raise ValueError(f"Flow config缺少{name}对应的sigma_max")
            sigma[name] = float(value)
        return sigma

    def _get_com(self, pos):
        if pos is None or not torch.is_tensor(pos) or pos.numel() == 0:
            return None
        return pos.mean(dim=0)

    def _get_local_receptor_anchor(self, rec_pos: Optional[torch.Tensor], ref_point: Optional[torch.Tensor], k_neighbors: int):
        if rec_pos is None or ref_point is None or not torch.is_tensor(rec_pos) or rec_pos.numel() == 0:
            return None
        if not torch.is_tensor(ref_point) or ref_point.numel() != 3:
            return None
        dists = (rec_pos - ref_point).norm(dim=1)
        nearest_idx = int(torch.argmin(dists).item())
        nearest_atom = rec_pos[nearest_idx]
        k = max(1, min(int(k_neighbors), int(rec_pos.shape[0])))
        idx = torch.topk(dists, k=k, largest=False).indices
        anchor = rec_pos[idx].mean(dim=0)
        if (anchor - ref_point).norm().item() < 1e-6:
            anchor = nearest_atom
        return anchor

    def _sample_shell_direction(self, rec_pos: Optional[torch.Tensor], pep_com: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if pep_com is None or not torch.is_tensor(pep_com):
            return None
        direction = torch.randn(3, dtype=pep_com.dtype, device=pep_com.device)
        norm = direction.norm()
        if norm < 1e-6:
            direction = pep_com.new_tensor([1.0, 0.0, 0.0])
            norm = direction.norm()
        direction = direction / norm

        direction_mode = str((self.flow_cfg or {}).get("tr_direction_mode", "isotropic") or "isotropic").lower()
        if direction_mode in {"isotropic", "none"}:
            return direction
        if direction_mode != "local_hemisphere":
            raise ValueError(f"[flow] 不支持的 tr_direction_mode={direction_mode}")

        local_k = int((self.flow_cfg or {}).get("tr_direction_local_k", 16) or 16)
        local_anchor = self._get_local_receptor_anchor(rec_pos, pep_com, local_k)
        if local_anchor is None:
            return direction
        inward_vec = local_anchor - pep_com
        inward_norm = inward_vec.norm()
        if inward_norm < 1e-6:
            return direction
        inward_unit = inward_vec / inward_norm
        if torch.dot(direction, inward_unit) < 0:
            direction = -direction
        return direction

    def _sample_tr_update_shell(self, data):
        cfg = self.flow_cfg or {}
        r_min = float(cfg.get("tr_r_min", 0.0) or 0.0)
        r_max = cfg.get("tr_r_max", None)
        if r_max is None:
            r_max = r_min
        r_max = float(r_max)
        if r_min < 0 or r_max <= 0 or r_max < r_min:
            raise ValueError(f"[flow] 非法 tr_r_min/tr_r_max：r_min={r_min} r_max={r_max}")
        r_mu = cfg.get("tr_r_mu", None)
        r_sigma = cfg.get("tr_r_sigma", None)
        if r_mu is None:
            r_mu = 0.5 * (r_min + r_max)
        if r_sigma is None:
            r_sigma = max((r_max - r_min) / 4.0, 1e-3)
        r_mu = float(r_mu)
        r_sigma = float(r_sigma)
        if r_sigma <= 0:
            raise ValueError(f"[flow] 非法 tr_r_sigma={r_sigma}（需>0）")

        has_receptor = hasattr(data, "node_types") and "receptor" in data.node_types
        has_pep = hasattr(data, "node_types") and "pep" in data.node_types
        has_pep_a = hasattr(data, "node_types") and "pep_a" in data.node_types
        rec_pos = data["receptor"].pos if (has_receptor and hasattr(data["receptor"], "pos")) else None
        pep_pos = data["pep"].pos if (has_pep and hasattr(data["pep"], "pos")) else None
        pep_a_pos = data["pep_a"].pos if (has_pep_a and hasattr(data["pep_a"], "pos")) else None
        rec_com = self._get_com(rec_pos)
        pep_com = self._get_com(pep_pos)
        if pep_com is None:
            pep_com = self._get_com(pep_a_pos)

        center_mode = str(cfg.get("tr_center_mode", "receptor_com") or "receptor_com").lower()
        if center_mode == "receptor_com":
            center = rec_com if rec_com is not None else pep_com
        elif center_mode == "pep_com":
            center = pep_com if pep_com is not None else rec_com
        else:
            raise ValueError(f"[flow] 不支持的 tr_center_mode={center_mode}（支持 receptor_com/pep_com）")

        if center is None or pep_com is None:
            return None

        min_dist = float(cfg.get("tr_min_dist", 0.0) or 0.0)
        max_tries = int(cfg.get("tr_reject_max_tries", 30) or 30)
        max_tries = max(1, max_tries)

        last_update = None
        for _ in range(max_tries):
            direction = self._sample_shell_direction(rec_pos, pep_com)
            if direction is None:
                direction = pep_com.new_tensor([1.0, 0.0, 0.0])
            # Gaussian_shell: 半径服从截断高斯，集中在 r_mu 附近
            radius = None
            for _ in range(32):
                cand = float(torch.normal(mean=r_mu, std=r_sigma, size=(1,)).item())
                if r_min <= cand <= r_max:
                    radius = cand
                    break
            if radius is None:
                radius = float(torch.empty(1).uniform_(r_min, r_max).item())
            target_com = center + direction * radius
            tr_update = (target_com - pep_com).unsqueeze(0)
            last_update = tr_update

            if min_dist <= 0 or rec_pos is None or pep_a_pos is None:
                return tr_update

            moved_pep = pep_a_pos + tr_update
            try:
                dist = torch.cdist(moved_pep, rec_pos).min().item()
            except Exception:
                return tr_update
            if dist >= min_dist:
                return tr_update

        return last_update

    def __call__(self, data):
        seed = self._sample_seed(data)
        np_state = None
        torch_state = None
        if seed is not None:
            np_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            np.random.seed(seed)
            torch.manual_seed(seed)
        try:
            # 可选核查：验证 tr/rot 的 target 与实际坐标更新一致（用于定位 tr/rot 学不动的根因）
            debug_pose = bool(int(os.environ.get("RAPIDOCK_FLOW_DEBUG_POSE", "0")))
            if debug_pose:
                try:
                    pep_a_before = data["pep_a"].pos.detach().clone()
                except Exception:
                    pep_a_before = None

            t = self._sample_time()
            target_mode = str((self.flow_cfg or {}).get("target_mode", "velocity") or "velocity").lower()
            if target_mode not in {"velocity", "noise"}:
                raise ValueError(f"[flow] 不支持的 target_mode={target_mode}（支持 velocity/noise）")
            tr_sigma = self.sigma_max["tr"]
            rot_sigma = self.sigma_max["rot"]
            tor_bb_sigma = self.sigma_max["tor_backbone"]
            tor_sc_sigma = self.sigma_max["tor_sidechain"]

            rot_target_mode = str((self.flow_cfg or {}).get("rot_target_mode", "noise") or "noise").lower()
            if rot_target_mode == "frame_vec":
                rot_target_mode = "frame_vector"
            if rot_target_mode not in {
                "noise",
                "rel",
                "x1",
                "kabsch",
                "kabsch_rigid",
                "frame",
                "frame_vector",
                "frame_xt",
                "frame_unique",
                "frame_local",
                "frame_local_xt",
                "frame_rel_iface",
                "frame_world",
                "const_frame",
                "anchor",
                "anchor_ref",
                "anchor_seq_cb2",
                "anchor_ref_seq_cb2",
                "anchor_ref_iface_sc",
                "anchor_ref_iface_sc_twist",
                "anchor_ref_seq_weighted_twist",
                "contact_near2",
                "contact_nearfar",
                "contact_pca",
                "contact_normal",
            }:
                raise ValueError(
                    "[flow] 不支持的 rot_target_mode="
                    f"{rot_target_mode}（支持 noise/rel/x1/kabsch/kabsch_rigid/frame/frame_vector/frame_xt/frame_unique/frame_local/frame_local_xt/frame_rel_iface/frame_world/const_frame/anchor/anchor_ref/anchor_seq_cb2/anchor_ref_seq_cb2/anchor_ref_iface_sc/anchor_ref_iface_sc_twist/anchor_ref_seq_weighted_twist/contact_*）"
                )
            rot_noise_mode = str((self.flow_cfg or {}).get("rot_noise_mode", "isotropic") or "isotropic").lower()
            if rot_noise_mode not in {"isotropic", "anchor"}:
                raise ValueError(
                    f"[flow] 不支持的 rot_noise_mode={rot_noise_mode}（支持 isotropic/anchor）"
                )

            def _safe_normalize(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
                norm = vec.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
                return vec / norm

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

                def _pick(atom_id: int) -> Optional[torch.Tensor]:
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

            def _frame_from_seq_cb(
                pos: torch.Tensor,
                atom2res_index: torch.Tensor,
                atom2atomid_index: torch.Tensor,
            ) -> Optional[torch.Tensor]:
                if pos.numel() == 0:
                    return None
                if atom2res_index is None or atom2atomid_index is None:
                    return None
                n_mask = atom2atomid_index == 0
                c_mask = atom2atomid_index == 2
                if not (n_mask.any() and c_mask.any()):
                    return None
                n_idx = atom2res_index[n_mask]
                c_idx = atom2res_index[c_mask]
                n_first = n_mask.clone()
                n_first[n_mask] = n_idx == n_idx.min()
                c_last = c_mask.clone()
                c_last[c_mask] = c_idx == c_idx.max()
                n_pos = pos[n_first].mean(dim=0)
                c_pos = pos[c_last].mean(dim=0)
                x = c_pos - n_pos
                if x.norm().item() < 1e-6:
                    return None
                x = _safe_normalize(x)
                ca_mask = atom2atomid_index == 1
                if not ca_mask.any():
                    return None
                ca_res = atom2res_index[ca_mask]
                if ca_res.numel() == 0:
                    return None
                mid_res = int(ca_res.median().item())
                sc_mask = (~torch.isin(atom2atomid_index, torch.tensor([0, 1, 2, 3], device=atom2atomid_index.device))) & (
                    atom2res_index == mid_res
                )
                if not sc_mask.any():
                    return None
                sc_center = pos[sc_mask].mean(dim=0)
                ca_mid_mask = ca_mask & (atom2res_index == mid_res)
                if not ca_mid_mask.any():
                    return None
                ca_mid = pos[ca_mid_mask].mean(dim=0)
                sc_vec = sc_center - ca_mid
                if sc_vec.norm().item() < 1e-6:
                    return None
                y = sc_vec - (sc_vec * x).sum() * x
                if y.norm().item() < 1e-6:
                    return None
                y = _safe_normalize(y)
                z = _safe_normalize(torch.cross(x, y))
                if z.norm().item() < 1e-6:
                    return None
                return torch.stack([x, y, z], dim=1)

            def _frame_from_seq_cb2(
                pos: torch.Tensor,
                atom2res_index: torch.Tensor,
                atom2atomid_index: torch.Tensor,
            ) -> Optional[torch.Tensor]:
                if pos.numel() == 0:
                    return None
                if atom2res_index is None or atom2atomid_index is None:
                    return None
                n_mask = atom2atomid_index == 0
                c_mask = atom2atomid_index == 2
                if not (n_mask.any() and c_mask.any()):
                    return None
                n_idx = atom2res_index[n_mask]
                c_idx = atom2res_index[c_mask]
                n_first = n_mask.clone()
                n_first[n_mask] = n_idx == n_idx.min()
                c_last = c_mask.clone()
                c_last[c_mask] = c_idx == c_idx.max()
                n_pos = pos[n_first].mean(dim=0)
                c_pos = pos[c_last].mean(dim=0)
                x = c_pos - n_pos
                if x.norm().item() < 1e-6:
                    return None
                x = _safe_normalize(x)
                sc_mask = ~torch.isin(atom2atomid_index, torch.tensor([0, 1, 2, 3], device=atom2atomid_index.device))
                if not sc_mask.any():
                    return None
                sc_res = atom2res_index[sc_mask]
                if sc_res.numel() == 0:
                    return None
                res_min = int(sc_res.min().item())
                res_max = int(sc_res.max().item())
                if res_max - res_min < 2:
                    return None
                res_i = res_min + 1
                res_j = res_max - 1
                sc_i = sc_mask & (atom2res_index == res_i)
                sc_j = sc_mask & (atom2res_index == res_j)
                if not (sc_i.any() and sc_j.any()):
                    return None
                sc_i_center = pos[sc_i].mean(dim=0)
                sc_j_center = pos[sc_j].mean(dim=0)
                y_raw = sc_j_center - sc_i_center
                if y_raw.norm().item() < 1e-6:
                    return None
                y = y_raw - (y_raw * x).sum() * x
                if y.norm().item() < 1e-6:
                    return None
                y = _safe_normalize(y)
                z = _safe_normalize(torch.cross(x, y))
                if z.norm().item() < 1e-6:
                    return None
                return torch.stack([x, y, z], dim=1)

            def _frame_from_rec_anchor(
                rec_pos: Optional[torch.Tensor],
                pep_center: Optional[torch.Tensor],
            ) -> Optional[torch.Tensor]:
                if rec_pos is None or rec_pos.numel() == 0 or pep_center is None:
                    return None
                if rec_pos.shape[0] < 3:
                    return None
                dists = (rec_pos - pep_center).norm(dim=1)
                idx = torch.argsort(dists)
                p0 = rec_pos[idx[0]]
                p1 = rec_pos[idx[1]]
                p2 = rec_pos[idx[2]]
                x = _safe_normalize(p1 - p0)
                y_raw = _safe_normalize(p2 - p0)
                z = torch.cross(x, y_raw)
                if z.norm().item() < 1e-6:
                    centered = rec_pos - rec_pos.mean(dim=0, keepdim=True)
                    cov = centered.T @ centered
                    _evals, evecs = torch.linalg.eigh(cov)
                    x = _safe_normalize(evecs[:, 2])
                    y = _safe_normalize(evecs[:, 1])
                    z = torch.cross(x, y)
                    if z.norm().item() < 1e-6:
                        return None
                z = _safe_normalize(z)
                y = _safe_normalize(torch.cross(z, x))
                rec_com = rec_pos.mean(dim=0)
                if torch.dot(z, pep_center - rec_com) < 0:
                    z = -z
                    y = -y
                return torch.stack([x, y, z], dim=1)

            def _frame_from_rec_iface(
                rec_pos: Optional[torch.Tensor],
                pep_pos: Optional[torch.Tensor],
                contact_cutoff: float,
                contact_min_points: int,
            ) -> Optional[torch.Tensor]:
                if rec_pos is None or pep_pos is None:
                    return None
                if rec_pos.numel() == 0 or pep_pos.numel() == 0:
                    return None
                dists = torch.cdist(rec_pos, pep_pos)
                mask = dists.min(dim=1).values <= contact_cutoff
                rec_iface = rec_pos[mask] if mask.any() else None
                if rec_iface is None or rec_iface.shape[0] < contact_min_points:
                    rec_iface = rec_pos if rec_pos.shape[0] >= 3 else None
                if rec_iface is None or rec_iface.shape[0] < 3:
                    return None
                rec_center = rec_iface.mean(dim=0)
                centered = rec_iface - rec_center
                cov = centered.T @ centered
                _evals, evecs = torch.linalg.eigh(cov)
                z = _safe_normalize(evecs[:, 0])
                pep_center = pep_pos.mean(dim=0)
                if torch.dot(z, pep_center - rec_center) < 0:
                    z = -z
                return z

            def _frame_from_iface_sc(
                rec_pos: Optional[torch.Tensor],
                pep_pos: Optional[torch.Tensor],
                atom2res_index: torch.Tensor,
                atom2atomid_index: torch.Tensor,
            ) -> Optional[torch.Tensor]:
                if rec_pos is None or pep_pos is None:
                    return None
                if rec_pos.numel() == 0 or pep_pos.numel() == 0:
                    return None
                if atom2res_index is None or atom2atomid_index is None:
                    return None
                contact_cutoff = float(self.flow_cfg.get("rot_frame_contact_cutoff", 8.0) or 8.0)
                contact_min_points = int(self.flow_cfg.get("rot_frame_contact_min_points", 6) or 6)
                z_rec = _frame_from_rec_iface(rec_pos, pep_pos, contact_cutoff, contact_min_points)
                if z_rec is None:
                    return None
                n_mask = atom2atomid_index == 0
                c_mask = atom2atomid_index == 2
                if not (n_mask.any() and c_mask.any()):
                    return None
                n_idx = atom2res_index[n_mask]
                c_idx = atom2res_index[c_mask]
                n_first = n_mask.clone()
                n_first[n_mask] = n_idx == n_idx.min()
                c_last = c_mask.clone()
                c_last[c_mask] = c_idx == c_idx.max()
                n_pos = pep_pos[n_first].mean(dim=0)
                c_pos = pep_pos[c_last].mean(dim=0)
                x = c_pos - n_pos
                if x.norm().item() < 1e-6:
                    return None
                x = _safe_normalize(x)
                # use interface residue sidechain to resolve sign flip
                ca_mask = atom2atomid_index == 1
                if not ca_mask.any():
                    return None
                ca_pos = pep_pos[ca_mask]
                ca_res = atom2res_index[ca_mask]
                dists = torch.cdist(ca_pos, rec_pos)
                closest_idx = int(torch.argmin(dists.min(dim=1).values))
                anchor_res = ca_res[closest_idx].item()
                sc_mask = (~torch.isin(atom2atomid_index, torch.tensor([0, 1, 2, 3], device=atom2atomid_index.device))) & (
                    atom2res_index == anchor_res
                )
                if not sc_mask.any():
                    return None
                sc_center = pep_pos[sc_mask].mean(dim=0)
                ca_center = ca_pos[closest_idx]
                sc_vec = sc_center - ca_center
                if sc_vec.norm().item() < 1e-6:
                    return None
                z_proj = z_rec - (z_rec * x).sum() * x
                if z_proj.norm().item() < 1e-6:
                    return None
                z = _safe_normalize(z_proj)
                y = _safe_normalize(torch.cross(z, x))
                if y.norm().item() < 1e-6:
                    return None
                if torch.dot(y, sc_vec) < 0:
                    y = -y
                    z = -z
                z = _safe_normalize(torch.cross(x, y))
                return torch.stack([x, y, z], dim=1)

            def _frame_from_seq_weighted(
                pep_pos: torch.Tensor,
                atom2res_index: torch.Tensor,
                atom2atomid_index: torch.Tensor,
                rec_pos: Optional[torch.Tensor],
            ) -> Optional[torch.Tensor]:
                if pep_pos is None or pep_pos.numel() == 0:
                    return None
                if atom2res_index is None or atom2atomid_index is None:
                    return None
                ca_mask = atom2atomid_index == 1
                if ca_mask.sum().item() < 3:
                    return None
                ca_pos = pep_pos[ca_mask]
                ca_res = atom2res_index[ca_mask]
                order = torch.argsort(ca_res)
                ca_pos = ca_pos[order]
                x = ca_pos[-1] - ca_pos[0]
                if x.norm().item() < 1e-6:
                    return None
                x = _safe_normalize(x)
                n_ca = ca_pos.shape[0]
                weights = torch.linspace(-1.0, 1.0, n_ca, device=ca_pos.device)
                w_norm = weights.abs().sum().clamp_min(1e-6)
                v = (ca_pos * weights[:, None]).sum(dim=0) / w_norm
                y_raw = v - (v * x).sum() * x
                if y_raw.norm().item() < 1e-6:
                    centered = ca_pos - ca_pos.mean(dim=0, keepdim=True)
                    cov = centered.T @ centered
                    _evals, evecs = torch.linalg.eigh(cov)
                    y_raw = evecs[:, 1]
                    if torch.cross(x, y_raw).norm().item() < 1e-6:
                        y_raw = evecs[:, 0]
                if y_raw.norm().item() < 1e-6:
                    return None
                y = _safe_normalize(y_raw)
                z = _safe_normalize(torch.cross(x, y))
                if z.norm().item() < 1e-6:
                    return None
                if rec_pos is not None and rec_pos.numel() > 0:
                    rec_center = rec_pos.mean(dim=0)
                    pep_center = ca_pos.mean(dim=0)
                    if torch.dot(z, pep_center - rec_center) < 0:
                        z = -z
                        y = -y
                return torch.stack([x, y, z], dim=1)

            def _frame_from_pep_ca(
                pos: torch.Tensor,
                atom2res_index: torch.Tensor,
                atom2atomid_index: torch.Tensor,
                rec_pos: Optional[torch.Tensor],
            ) -> Optional[torch.Tensor]:
                ca_mask = atom2atomid_index == 1
                if ca_mask.sum().item() < 3:
                    return None
                ca_pos = pos[ca_mask]
                ca_res = atom2res_index[ca_mask]
                order = torch.argsort(ca_res)
                ca_pos = ca_pos[order]
                axis1 = ca_pos[-1] - ca_pos[0]
                if axis1.norm().item() < 1e-6 and ca_pos.shape[0] >= 2:
                    axis1 = ca_pos[1] - ca_pos[0]
                if axis1.norm().item() < 1e-6:
                    return None
                x = _safe_normalize(axis1)
                centered = ca_pos - ca_pos.mean(dim=0, keepdim=True)
                cov = centered.T @ centered
                _evals, evecs = torch.linalg.eigh(cov)
                y_raw = evecs[:, 1]
                if torch.cross(x, y_raw).norm().item() < 1e-6:
                    y_raw = evecs[:, 0]
                z = torch.cross(x, y_raw)
                if z.norm().item() < 1e-6:
                    return None
                z = _safe_normalize(z)
                y = _safe_normalize(torch.cross(z, x))
                if rec_pos is not None and rec_pos.numel() > 0:
                    rec_com = rec_pos.mean(dim=0)
                    pep_com = ca_pos.mean(dim=0)
                    if torch.dot(z, pep_com - rec_com) < 0:
                        z = -z
                        y = -y
                return torch.stack([x, y, z], dim=1)

            def _frame_from_rec_contact(
                rec_pos: Optional[torch.Tensor],
                pep_center: Optional[torch.Tensor],
                mode: str,
                radius: float,
                min_points: int,
            ) -> Optional[torch.Tensor]:
                if rec_pos is None or rec_pos.numel() == 0 or pep_center is None:
                    return None
                if rec_pos.shape[0] < 3:
                    return None
                rec_com = rec_pos.mean(dim=0)
                dists = (rec_pos - pep_center).norm(dim=1)
                if mode == "contact_near2":
                    idx = torch.argsort(dists)
                    p0 = rec_pos[idx[0]]
                    p1 = rec_pos[idx[1]]
                    v1 = p1 - p0
                    v2 = pep_center - rec_com
                elif mode == "contact_nearfar":
                    idx = torch.argsort(dists)
                    p0 = rec_pos[idx[0]]
                    p1 = rec_pos[idx[-1]]
                    v1 = p1 - p0
                    v2 = pep_center - rec_com
                else:
                    local_mask = dists <= radius
                    if local_mask.sum().item() < min_points:
                        local_mask = torch.ones_like(dists, dtype=torch.bool)
                    local = rec_pos[local_mask]
                    centered = local - local.mean(dim=0, keepdim=True)
                    cov = centered.T @ centered
                    _evals, evecs = torch.linalg.eigh(cov)
                    if mode == "contact_pca":
                        v1 = evecs[:, 2]
                        v2 = evecs[:, 1]
                    else:
                        v1 = evecs[:, 0]
                        v2 = evecs[:, 2]
                if v1.norm().item() < 1e-6 or v2.norm().item() < 1e-6:
                    return None
                x = _safe_normalize(v1)
                z = torch.cross(x, v2)
                if z.norm().item() < 1e-6:
                    return None
                z = _safe_normalize(z)
                y = _safe_normalize(torch.cross(z, x))
                if torch.dot(z, pep_center - rec_com) < 0:
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

            def _frame_from_rec_pca(
                rec_pos: Optional[torch.Tensor],
                pep_pos: Optional[torch.Tensor],
            ) -> Optional[torch.Tensor]:
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

            def _frame_from_rec_pca_unique(
                rec_pos: Optional[torch.Tensor],
                pep_pos: Optional[torch.Tensor],
            ) -> Optional[torch.Tensor]:
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
                pep_center = None
                if pep_pos is not None and pep_pos.numel() > 0:
                    pep_center = pep_pos.mean(dim=0)
                    if torch.dot(z, pep_center - rec_com) < 0:
                        z = -z
                        y = -y
                if pep_center is not None:
                    dists = (rec_pos - pep_center).norm(dim=1)
                    idx = torch.argsort(dists)
                    v1 = rec_pos[idx[0]] - rec_com
                    v2 = rec_pos[idx[-1]] - rec_com
                    if torch.dot(x, v1) < 0:
                        x = -x
                    if torch.dot(y, v2) < 0:
                        y = -y
                    z = torch.cross(x, y)
                    if z.norm().item() < 1e-6:
                        return None
                    z = _safe_normalize(z)
                    y = _safe_normalize(torch.cross(z, x))
                return torch.stack([x, y, z], dim=1)

            def _frame_from_rec_iface_pca(
                rec_pos: Optional[torch.Tensor],
                pep_pos: Optional[torch.Tensor],
                cutoff: float,
                min_points: int,
            ) -> Optional[torch.Tensor]:
                if rec_pos is None or rec_pos.numel() == 0:
                    return None
                if pep_pos is None or pep_pos.numel() == 0:
                    return None
                rec_iface = _select_rec_interface(rec_pos, pep_pos, cutoff, min_points)
                if rec_iface is None or rec_iface.shape[0] < 3:
                    rec_iface = rec_pos
                if rec_iface.shape[0] < 3:
                    return None
                rec_center = rec_iface.mean(dim=0)
                centered = rec_iface - rec_center
                cov = centered.T @ centered
                _evals, evecs = torch.linalg.eigh(cov)
                z = _safe_normalize(evecs[:, 0])
                pep_center = pep_pos.mean(dim=0)
                if torch.dot(z, pep_center - rec_center) < 0:
                    z = -z
                x_raw = pep_center - rec_center
                x = x_raw - (x_raw * z).sum() * z
                if x.norm().item() < 1e-6:
                    x = evecs[:, 2]
                if x.norm().item() < 1e-6:
                    return None
                x = _safe_normalize(x)
                y = torch.cross(z, x)
                if y.norm().item() < 1e-6:
                    return None
                y = _safe_normalize(y)
                x = _safe_normalize(torch.cross(y, z))
                return torch.stack([x, y, z], dim=1)

            def _frame_from_rec_iface_unique(
                rec_pos: Optional[torch.Tensor],
                pep_pos: Optional[torch.Tensor],
                cutoff: float,
                min_points: int,
            ) -> Optional[torch.Tensor]:
                if rec_pos is None or pep_pos is None:
                    return None
                if rec_pos.numel() == 0 or pep_pos.numel() == 0:
                    return None
                rec_iface = _select_rec_interface(rec_pos, pep_pos, cutoff, min_points)
                if rec_iface is None or rec_iface.shape[0] < 3:
                    rec_iface = rec_pos
                if rec_iface.shape[0] < 3:
                    return None
                rec_center = rec_iface.mean(dim=0)
                pep_center = pep_pos.mean(dim=0)
                dists = (rec_iface - rec_center).norm(dim=1)
                idx = torch.argsort(dists)
                if idx.numel() >= 3:
                    p0 = rec_iface[idx[0]] - rec_center
                    p1 = rec_iface[idx[1]] - rec_center
                    p2 = rec_iface[idx[2]] - rec_center
                    x = p1 - p0
                    y_raw = p2 - p0
                    if x.norm().item() >= 1e-6 and y_raw.norm().item() >= 1e-6:
                        x = _safe_normalize(x)
                        z = torch.cross(x, y_raw)
                        if z.norm().item() >= 1e-6:
                            z = _safe_normalize(z)
                            if torch.dot(z, pep_center - rec_center) < 0:
                                z = -z
                            y = _safe_normalize(torch.cross(z, x))
                            return torch.stack([x, y, z], dim=1)
                centered = rec_iface - rec_center
                cov = centered.T @ centered
                _evals, evecs = torch.linalg.eigh(cov)
                z = _safe_normalize(evecs[:, 0])
                if torch.dot(z, pep_center - rec_center) < 0:
                    z = -z
                anchor = rec_iface[idx[0]] - rec_center
                x_raw = anchor - (anchor * z).sum() * z
                if x_raw.norm().item() < 1e-6 and idx.numel() > 1:
                    anchor = rec_iface[idx[1]] - rec_center
                    x_raw = anchor - (anchor * z).sum() * z
                if x_raw.norm().item() < 1e-6:
                    x_raw = evecs[:, 2]
                if x_raw.norm().item() < 1e-6:
                    return None
                x = _safe_normalize(x_raw)
                y_raw = None
                if idx.numel() > 1:
                    anchor2 = rec_iface[idx[1]] - rec_center
                    y_raw = anchor2 - (anchor2 * z).sum() * z - (anchor2 * x).sum() * x
                if y_raw is None or y_raw.norm().item() < 1e-6:
                    y_raw = evecs[:, 1]
                if y_raw.norm().item() < 1e-6:
                    return None
                y = _safe_normalize(y_raw)
                if torch.dot(y, pep_center - rec_center) < 0:
                    y = -y
                z = _safe_normalize(torch.cross(x, y))
                if z.norm().item() < 1e-6:
                    return None
                y = _safe_normalize(torch.cross(z, x))
                return torch.stack([x, y, z], dim=1)

            def _frame_from_rec_dirfield(
                rec_pos: Optional[torch.Tensor],
                pep_pos: Optional[torch.Tensor],
                rec_normal: Optional[torch.Tensor],
                rec_tangent_u: Optional[torch.Tensor],
                rec_tangent_v: Optional[torch.Tensor],
                cutoff: float,
                min_points: int,
            ) -> Optional[torch.Tensor]:
                if rec_pos is None or pep_pos is None:
                    return None
                if rec_normal is None or rec_tangent_u is None:
                    return None
                if rec_pos.numel() == 0 or pep_pos.numel() == 0:
                    return None
                if rec_normal.shape[0] != rec_pos.shape[0]:
                    return None
                dmat = torch.cdist(rec_pos, pep_pos)
                min_d = dmat.min(dim=1).values
                mask = min_d <= float(cutoff)
                if mask.sum().item() < int(min_points):
                    return None
                normals = rec_normal[mask]
                tan_u = rec_tangent_u[mask]
                tan_v = rec_tangent_v[mask] if rec_tangent_v is not None else None
                z = _safe_normalize(normals.mean(dim=0))
                if z is None:
                    return None
                x_raw = tan_u.mean(dim=0)
                x_raw = x_raw - (x_raw * z).sum() * z
                x = _safe_normalize(x_raw)
                if x is None and tan_v is not None:
                    x_raw = tan_v.mean(dim=0)
                    x_raw = x_raw - (x_raw * z).sum() * z
                    x = _safe_normalize(x_raw)
                if x is None:
                    return None
                y = _safe_normalize(torch.cross(z, x))
                if y is None:
                    return None
                x = _safe_normalize(torch.cross(y, z))
                if x is None:
                    return None
                rec_center = rec_pos[mask].mean(dim=0)
                pep_center = pep_pos.mean(dim=0)
                if torch.dot(z, pep_center - rec_center) < 0:
                    z = -z
                    y = -y
                return torch.stack([x, y, z], dim=1)

            def _frame_from_pep_unique(
                pep_pos: torch.Tensor,
                atom2res_index: torch.Tensor,
                atom2atomid_index: torch.Tensor,
                rec_pos: Optional[torch.Tensor],
            ) -> Optional[torch.Tensor]:
                if pep_pos is None or pep_pos.numel() == 0:
                    return None
                if atom2res_index is None or atom2atomid_index is None:
                    return None
                n_mask = atom2atomid_index == 0
                c_mask = atom2atomid_index == 2
                if not (n_mask.any() and c_mask.any()):
                    return None
                n_idx = atom2res_index[n_mask]
                c_idx = atom2res_index[c_mask]
                n_first = n_mask.clone()
                n_first[n_mask] = n_idx == n_idx.min()
                c_last = c_mask.clone()
                c_last[c_mask] = c_idx == c_idx.max()
                n_pos = pep_pos[n_first].mean(dim=0)
                c_pos = pep_pos[c_last].mean(dim=0)
                x = c_pos - n_pos
                if x.norm().item() < 1e-6:
                    return None
                x = _safe_normalize(x)
                ca_mask = atom2atomid_index == 1
                if not ca_mask.any():
                    return None
                ca_pos = pep_pos[ca_mask]
                ca_res = atom2res_index[ca_mask]
                rec_center = rec_pos.mean(dim=0) if rec_pos is not None and rec_pos.numel() > 0 else ca_pos.mean(dim=0)
                dists = torch.cdist(ca_pos, rec_center.unsqueeze(0)).squeeze(1)
                order = torch.argsort(dists)
                anchor_res = int(ca_res[order[0]].item())
                sc_mask = (~torch.isin(atom2atomid_index, torch.tensor([0, 1, 2, 3], device=atom2atomid_index.device))) & (
                    atom2res_index == anchor_res
                )
                ca_anchor_mask = ca_mask & (atom2res_index == anchor_res)
                if ca_anchor_mask.any():
                    ca_anchor = pep_pos[ca_anchor_mask].mean(dim=0)
                else:
                    ca_anchor = ca_pos[order[0]]
                y_raw = None
                if sc_mask.any():
                    sc_center = pep_pos[sc_mask].mean(dim=0)
                    y_raw = sc_center - ca_anchor
                if y_raw is None or y_raw.norm().item() < 1e-6:
                    if order.numel() > 1:
                        ca_alt = ca_pos[order[1]]
                        y_raw = ca_alt - ca_anchor
                if y_raw is None or y_raw.norm().item() < 1e-6:
                    return None
                y = y_raw - (y_raw * x).sum() * x
                if y.norm().item() < 1e-6:
                    return None
                y = _safe_normalize(y)
                z = torch.cross(x, y)
                if z.norm().item() < 1e-6:
                    return None
                z = _safe_normalize(z)
                if order.numel() > 1:
                    ca_alt = ca_pos[order[1]]
                    if torch.dot(y, ca_alt - ca_anchor) < 0:
                        y = -y
                        z = -z
                elif torch.dot(y, rec_center - ca_anchor) < 0:
                    y = -y
                    z = -z
                z = _safe_normalize(torch.cross(x, y))
                y = _safe_normalize(torch.cross(z, x))
                return torch.stack([x, y, z], dim=1)

            def _frame_from_anchor_idx3(
                pos: Optional[torch.Tensor],
                idx3: Optional[torch.Tensor],
                ref_dir: Optional[torch.Tensor],
            ) -> Optional[torch.Tensor]:
                if pos is None or idx3 is None:
                    return None
                if pos.numel() == 0:
                    return None
                if idx3.numel() != 3:
                    return None
                idx = idx3.to(device=pos.device).long()
                if idx.min().item() < 0 or idx.max().item() >= pos.shape[0]:
                    return None
                p0 = pos[idx[0]]
                p1 = pos[idx[1]]
                p2 = pos[idx[2]]
                x = p1 - p0
                if x.norm().item() < 1e-6:
                    return None
                y_raw = p2 - p0
                if y_raw.norm().item() < 1e-6:
                    return None
                x = _safe_normalize(x)
                z = torch.cross(x, y_raw)
                if z.norm().item() < 1e-6:
                    return None
                z = _safe_normalize(z)
                if ref_dir is not None and ref_dir.norm().item() >= 1e-6:
                    if torch.dot(z, ref_dir) < 0:
                        z = -z
                y = _safe_normalize(torch.cross(z, x))
                return torch.stack([x, y, z], dim=1)

            def _frame_from_pep_endpoints(
                endpoints: Optional[torch.Tensor],
                aux_point: Optional[torch.Tensor],
            ) -> Optional[torch.Tensor]:
                if endpoints is None:
                    return None
                if endpoints.numel() == 0 or endpoints.shape[0] < 2:
                    return None
                p0 = endpoints[0]
                p1 = endpoints[1]
                x = p1 - p0
                if x.norm().item() < 1e-6:
                    return None
                x = _safe_normalize(x)
                if aux_point is None:
                    return None
                mid = 0.5 * (p0 + p1)
                y_raw = aux_point - mid
                y_raw = y_raw - (y_raw * x).sum() * x
                if y_raw.norm().item() < 1e-6:
                    return None
                y = _safe_normalize(y_raw)
                z = torch.cross(x, y)
                if z.norm().item() < 1e-6:
                    return None
                z = _safe_normalize(z)
                y = _safe_normalize(torch.cross(z, x))
                return torch.stack([x, y, z], dim=1)

            def _flip_frame_by_rec(
                frame: Optional[torch.Tensor],
                rec_pos: Optional[torch.Tensor],
                pep_pos: Optional[torch.Tensor],
            ) -> Optional[torch.Tensor]:
                if frame is None:
                    return None
                if rec_pos is None or pep_pos is None:
                    return frame
                if rec_pos.numel() == 0 or pep_pos.numel() == 0:
                    return frame
                rec_center = rec_pos.mean(dim=0)
                pep_center = pep_pos.mean(dim=0)
                if torch.dot(frame[:, 2], pep_center - rec_center) < 0:
                    frame = frame.clone()
                    frame[:, 1] = -frame[:, 1]
                    frame[:, 2] = -frame[:, 2]
                return frame

            def _sample_rot_target() -> torch.Tensor:
                if rot_noise_mode == "anchor":
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    pep_pos = data["pep_a"].pos if "pep_a" in data.node_types else None
                    if rec_pos is not None and pep_pos is not None:
                        pep_center = pep_pos.mean(dim=0)
                        rec_frame = _frame_from_rec_anchor(rec_pos, pep_center)
                        if rec_frame is not None:
                            axis = _safe_normalize(rec_frame[:, 2])
                            omega = float(sample(rot_sigma))
                            return axis * omega
                return torch.from_numpy(sample_vec(rot_sigma)).float()

            rot_axis_ref_mode = str(self.flow_cfg.get("rot_axis_ref_mode", "") or "").lower()
            if rot_axis_ref_mode in {"none", "off", "false", "0"}:
                rot_axis_ref_mode = ""
            rot_ref_input = bool(self.flow_cfg.get("rot_ref_input", False))
            rot_ref_mode = str(self.flow_cfg.get("rot_ref_mode", "rec_pca_z") or "rec_pca_z").lower()
            rot_ref_dist = float(self.flow_cfg.get("rot_ref_dist", 10.0) or 10.0)

            def _get_rot_axis_ref() -> Optional[torch.Tensor]:
                if not rot_axis_ref_mode:
                    return None
                rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                if rot_axis_ref_mode in {"rec_pca", "rec_pca_z"}:
                    if rec_pos is None or rec_pos.numel() == 0:
                        return None
                    centered = rec_pos - rec_pos.mean(dim=0, keepdim=True)
                    cov = centered.T @ centered
                    _evals, evecs = torch.linalg.eigh(cov)
                    axis = _safe_normalize(evecs[:, 2])
                    far_idx = centered.norm(dim=1).argmax()
                    v = centered[far_idx]
                    if torch.dot(axis, v) < 0:
                        axis = -axis
                    return axis
                pep_pos = None
                if "pep_a" in data.node_types and hasattr(data["pep_a"], "orig_pos"):
                    pep_pos = data["pep_a"].orig_pos
                elif "pep_a" in data.node_types and hasattr(data["pep_a"], "pos"):
                    pep_pos = data["pep_a"].pos
                if rec_pos is None or pep_pos is None:
                    return None
                rec_center = rec_pos.mean(dim=0)
                pep_center = pep_pos.mean(dim=0)
                ref_axis = pep_center - rec_center
                if ref_axis.norm().item() < 1e-6:
                    return None
                return _safe_normalize(ref_axis)

            def _get_ref_axis_input(
                rec_pos_g: Optional[torch.Tensor],
                pep_pos_g: Optional[torch.Tensor],
            ) -> Optional[torch.Tensor]:
                if rot_ref_mode in {"world_x", "world_y", "world_z"}:
                    if rot_ref_mode == "world_x":
                        return torch.tensor([1.0, 0.0, 0.0], device=rec_pos_g.device if rec_pos_g is not None else None)
                    if rot_ref_mode == "world_y":
                        return torch.tensor([0.0, 1.0, 0.0], device=rec_pos_g.device if rec_pos_g is not None else None)
                    return torch.tensor([0.0, 0.0, 1.0], device=rec_pos_g.device if rec_pos_g is not None else None)
                if rec_pos_g is None or rec_pos_g.numel() == 0:
                    return None
                if rot_ref_mode in {"rec_pca", "rec_pca_z"}:
                    centered = rec_pos_g - rec_pos_g.mean(dim=0, keepdim=True)
                    cov = centered.T @ centered
                    _evals, evecs = torch.linalg.eigh(cov)
                    axis = _safe_normalize(evecs[:, 2])
                    far_idx = centered.norm(dim=1).argmax()
                    v = centered[far_idx]
                    if torch.dot(axis, v) < 0:
                        axis = -axis
                    return axis
                if pep_pos_g is None or pep_pos_g.numel() == 0:
                    return None
                rec_center = rec_pos_g.mean(dim=0)
                pep_center = pep_pos_g.mean(dim=0)
                axis = pep_center - rec_center
                if axis.norm().item() < 1e-6:
                    return None
                return _safe_normalize(axis)

            def _get_rot_axis_for_loss() -> Optional[torch.Tensor]:
                rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                pep_pos = None
                if "pep_a" in data and hasattr(data["pep_a"], "orig_pos"):
                    pep_pos = data["pep_a"].orig_pos
                elif "pep_a" in data and hasattr(data["pep_a"], "pos"):
                    pep_pos = data["pep_a"].pos
                if rec_pos is None or pep_pos is None:
                    return None
                rec_center = rec_pos.mean(dim=0)
                pep_center = pep_pos.mean(dim=0)
                axis = pep_center - rec_center
                if axis.norm().item() < 1e-6:
                    return None
                return _safe_normalize(axis)

            def _apply_axis_ref(rot_vec: torch.Tensor, ref_axis: Optional[torch.Tensor]) -> torch.Tensor:
                if ref_axis is None:
                    return rot_vec
                if rot_vec.ndim == 1:
                    return rot_vec if torch.dot(rot_vec, ref_axis) >= 0 else -rot_vec
                dot = (rot_vec * ref_axis).sum(dim=-1, keepdim=True)
                return torch.where(dot < 0, -rot_vec, rot_vec)

            # velocity: x_t = x0 + t*v，监督 v（旧逻辑，t=1 时 v 不可观测，tr/rot 很容易贴 0 基线）
            # noise:    x_t = x1 + (1-t)*eps，监督 eps（更接近 diffusion 的噪声回归，t→1 时噪声自然→0）
            if target_mode == "velocity":
                rot_target = _sample_rot_target().unsqueeze(0)
                time_scale = t
            else:
                rot_target = _sample_rot_target().unsqueeze(0)
                time_scale = 1.0 - t

            tr_sampling = str((self.flow_cfg or {}).get("tr_sampling", "gaussian") or "gaussian").lower()
            tr_update = None
            if tr_sampling in {"shell", "ring", "outside", "gaussian_shell"}:
                tr_update = self._sample_tr_update_shell(data)
            if tr_update is None:
                tr_target = torch.normal(0.0, tr_sigma, size=(1, 3))
                tr_update = tr_target * float(time_scale)
            else:
                denom = float(time_scale)
                if denom <= 1e-6:
                    denom = 1e-6
                tr_target = tr_update / denom

            bb_count = int(data['pep_a'].mask_edges_backbone.sum())
            sc_count = int(data['pep_a'].mask_edges_sidechain.sum())
            if bb_count > 0:
                tor_bb_target = torch.from_numpy(
                    np.random.uniform(-tor_bb_sigma, tor_bb_sigma, size=bb_count)
                ).float()
                tor_bb_target = wrap_to_pi(tor_bb_target)
            else:
                tor_bb_target = torch.empty(0)

            if sc_count > 0:
                tor_sc_target = torch.from_numpy(
                    np.random.uniform(-tor_sc_sigma, tor_sc_sigma, size=sc_count)
                ).float()
                tor_sc_target = wrap_to_pi(tor_sc_target)
            else:
                tor_sc_target = torch.empty(0)

            time_scale = float(time_scale)
            if time_scale < 1e-6:
                time_scale = 1e-6
            rot_update = rot_target.squeeze(0) * time_scale
            tor_bb_update = wrap_to_pi(tor_bb_target * time_scale).cpu().numpy() if bb_count > 0 else None
            tor_sc_update = wrap_to_pi(tor_sc_target * time_scale).cpu().numpy() if sc_count > 0 else None

            set_time(data, t, t, t, t, 1, device=None)
            try:
                pep_pos = data["pep_a"].pos
                pep_center = pep_pos.mean(dim=0, keepdim=True)
                rot_mat = axis_angle_to_matrix(rot_update.squeeze())
                rigid_pos = (pep_pos - pep_center) @ rot_mat.T + tr_update + pep_center
                data["pep_a"].rigid_pos = rigid_pos.to(dtype=torch.float32)
            except Exception:
                data["pep_a"].rigid_pos = data["pep_a"].pos

            peptide_updater(
                data,
                tr_update,
                rot_update,
                tor_bb_update,
                tor_sc_update,
            )

            data.flow_tr_target = tr_target

            rot_target_mat = None
            rot_valid = True
            rot_error = None
            if rot_target_mode in {"rel", "x1"}:
                try:
                    pep_pos = data["pep_a"].pos
                    pep_orig = data["pep_a"].orig_pos
                    atom_ids = data["pep_a"].atom2atomid_index
                    ca_mask = atom_ids == 1
                    if ca_mask.sum().item() >= 3:
                        cur_ca = pep_pos[ca_mask]
                        orig_ca = pep_orig[ca_mask]
                    else:
                        cur_ca = pep_pos
                        orig_ca = pep_orig
                    if cur_ca.shape[0] >= 3 and orig_ca.shape[0] >= 3:
                        R_hat, _t_hat = kabsch_torch(cur_ca.T, orig_ca.T)
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        if rot_target_mode == "rel":
                            rot_update = matrix_to_axis_angle(R_hat).to(dtype=torch.float32)
                            if rot_update.ndim == 2:
                                rot_update = rot_update.squeeze(0)
                            rot_target = rot_update.unsqueeze(0)
                        else:
                            rot_target = matrix_to_rot6d(R_hat).to(dtype=torch.float32)
                            if rot_target.ndim == 1:
                                rot_target = rot_target.unsqueeze(0)
                    else:
                        rot_valid = False
                except Exception:
                    rot_valid = False
            elif rot_target_mode == "kabsch":
                try:
                    pep_pos = data["pep_a"].pos
                    pep_orig = data["pep_a"].orig_pos
                    atom_ids = data["pep_a"].atom2atomid_index
                    ca_mask = atom_ids == 1
                    if ca_mask.sum().item() >= 3:
                        cur_ca = pep_pos[ca_mask]
                        orig_ca = pep_orig[ca_mask]
                    else:
                        cur_ca = pep_pos
                        orig_ca = pep_orig
                    if cur_ca.shape[0] >= 3 and orig_ca.shape[0] >= 3:
                        R_hat, _t_hat = kabsch_torch(cur_ca.T, orig_ca.T)
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_update = matrix_to_axis_angle(R_hat).to(dtype=torch.float32)
                        if rot_update.ndim == 2:
                            rot_update = rot_update.squeeze(0)
                        rot_target = (rot_update / time_scale).unsqueeze(0)
                except Exception:
                    pass
            elif rot_target_mode == "kabsch_rigid":
                try:
                    pep_pos = (
                        data["pep_a"].rigid_pos
                        if hasattr(data["pep_a"], "rigid_pos")
                        else data["pep_a"].pos
                    )
                    pep_orig = data["pep_a"].orig_pos
                    atom_ids = data["pep_a"].atom2atomid_index
                    ca_mask = atom_ids == 1
                    if ca_mask.sum().item() >= 3:
                        cur_ca = pep_pos[ca_mask]
                        orig_ca = pep_orig[ca_mask]
                    else:
                        cur_ca = pep_pos
                        orig_ca = pep_orig
                    if cur_ca.shape[0] >= 3 and orig_ca.shape[0] >= 3:
                        R_hat, _t_hat = kabsch_torch(cur_ca.T, orig_ca.T)
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_update = matrix_to_axis_angle(R_hat).to(dtype=torch.float32)
                        if rot_update.ndim == 2:
                            rot_update = rot_update.squeeze(0)
                        rot_target = (rot_update / time_scale).unsqueeze(0)
                except Exception:
                    pass
            elif rot_target_mode == "frame":
                try:
                    pep_pos = data["pep_a"].pos
                    pep_orig = data["pep_a"].orig_pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    n_res = (
                        int(data["pep"].x.shape[0])
                        if "pep" in data.node_types
                        else int(atom2res_index.max().item() + 1)
                    )
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    frame_t = _frame_from_ncac(
                        pep_pos, atom2res_index, atom2atomid_index, n_res, rec_pos
                    )
                    frame_0 = _frame_from_ncac(
                        pep_orig, atom2res_index, atom2atomid_index, n_res, rec_pos
                    )
                    if frame_t is not None and frame_0 is not None:
                        R_hat = frame_0 @ frame_t.T
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_update = matrix_to_axis_angle(R_hat).to(dtype=torch.float32)
                        if rot_update.ndim == 2:
                            rot_update = rot_update.squeeze(0)
                        rot_target = (rot_update / time_scale).unsqueeze(0)
                except Exception:
                    pass
            elif rot_target_mode == "frame_vector":
                try:
                    pep_orig = data["pep_a"].orig_pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    n_res = (
                        int(data["pep"].x.shape[0])
                        if "pep" in data.node_types
                        else int(atom2res_index.max().item() + 1)
                    )
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    contact_only = bool(self.flow_cfg.get("rot_frame_contact_only", False))
                    contact_cutoff = float(self.flow_cfg.get("rot_frame_contact_cutoff", 8.0) or 8.0)
                    contact_min_points = int(self.flow_cfg.get("rot_frame_contact_min_points", 6) or 6)
                    contact_fallback = bool(self.flow_cfg.get("rot_frame_contact_fallback", True))
                    rec_frame = None
                    if contact_only:
                        rec_iface = _select_rec_interface(
                            rec_pos, pep_orig, contact_cutoff, contact_min_points
                        )
                        if rec_iface is None:
                            if contact_fallback:
                                rec_frame = _frame_from_rec_pca(rec_pos, pep_orig)
                            else:
                                rot_valid = False
                                rot_error = "frame_vector:contact_insufficient"
                        else:
                            rec_frame = _frame_from_rec_pca(rec_iface, pep_orig)
                    else:
                        rec_frame = _frame_from_rec_pca(rec_pos, pep_orig)
                    pep_frame = _frame_from_ncac(
                        pep_orig, atom2res_index, atom2atomid_index, n_res, rec_pos
                    )
                    if pep_frame is None:
                        pep_frame = _frame_from_pep_ca(
                            pep_orig, atom2res_index, atom2atomid_index, rec_pos
                        )
                    if rec_frame is not None and pep_frame is not None:
                        R_hat = rec_frame @ pep_frame.T
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_target = R_hat.to(dtype=torch.float32).reshape(1, 9)
                    else:
                        rot_valid = False
                        rot_error = "frame_vector:frame_none"
                except Exception:
                    rot_valid = False
                    rot_error = "frame_vector:exception"
            elif rot_target_mode == "frame_rel_iface":
                try:
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    pep_pos = data["pep_a"].pos if "pep_a" in data.node_types else None
                    rec_normal = None
                    rec_tangent_u = None
                    rec_tangent_v = None
                    if "receptor" in data.node_types:
                        rec_normal = getattr(data["receptor"], "normal", None)
                        rec_tangent_u = getattr(data["receptor"], "tangent_u", None)
                        rec_tangent_v = getattr(data["receptor"], "tangent_v", None)
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    n_res = (
                        int(data["pep"].x.shape[0])
                        if "pep" in data.node_types
                        else int(atom2res_index.max().item() + 1)
                    )
                    contact_only = bool(self.flow_cfg.get("rot_frame_contact_only", False))
                    contact_cutoff = float(self.flow_cfg.get("rot_frame_contact_cutoff", 8.0) or 8.0)
                    contact_min_points = int(self.flow_cfg.get("rot_frame_contact_min_points", 6) or 6)
                    rec_idx3 = getattr(data, "rec_anchor_idx3", None)
                    if rec_idx3 is None and "rec_anchor_idx3" in data:
                        rec_idx3 = data["rec_anchor_idx3"]
                    pep_idx3 = getattr(data, "pep_anchor_idx3", None)
                    if pep_idx3 is None and "pep_anchor_idx3" in data:
                        pep_idx3 = data["pep_anchor_idx3"]
                    pep_endpoints = getattr(data, "pep_axis_endpoints", None)
                    if pep_endpoints is None and "pep_axis_endpoints" in data:
                        pep_endpoints = data["pep_axis_endpoints"]
                    rec_center = rec_pos.mean(dim=0) if rec_pos is not None and rec_pos.numel() > 0 else None
                    pep_center = pep_pos.mean(dim=0) if pep_pos is not None and pep_pos.numel() > 0 else None
                    ref_dir = None
                    if rec_center is not None and pep_center is not None:
                        ref_dir = pep_center - rec_center
                    rec_frame = None
                    if rec_normal is not None and rec_tangent_u is not None:
                        rec_frame = _frame_from_rec_dirfield(
                            rec_pos,
                            pep_pos,
                            rec_normal,
                            rec_tangent_u,
                            rec_tangent_v,
                            contact_cutoff,
                            contact_min_points,
                        )
                    if contact_only:
                        if rec_frame is None:
                            rec_frame = _frame_from_anchor_idx3(rec_pos, rec_idx3, ref_dir)
                            if rec_frame is None:
                                rec_frame = _frame_from_rec_iface_unique(
                                    rec_pos, pep_pos, contact_cutoff, contact_min_points
                                )
                    else:
                        if rec_frame is None:
                            rec_frame = _frame_from_rec_pca(rec_pos, pep_pos)
                    pep_aux = None
                    if pep_idx3 is not None and pep_pos is not None:
                        pep_idx = pep_idx3.to(device=pep_pos.device).long()
                        if pep_idx.numel() >= 3 and pep_idx.max().item() < pep_pos.shape[0]:
                            pep_aux = pep_pos[pep_idx[2]]
                    if pep_aux is None:
                        pep_aux = rec_center
                    pep_frame = _frame_from_pep_endpoints(pep_endpoints, pep_aux)
                    if pep_frame is None:
                        pep_frame = _frame_from_anchor_idx3(pep_pos, pep_idx3, None)
                    if pep_frame is None:
                        pep_frame = _frame_from_pep_unique(
                            pep_pos, atom2res_index, atom2atomid_index, rec_pos
                        )
                    if pep_frame is None:
                        pep_frame = _frame_from_seq_weighted(
                            pep_pos, atom2res_index, atom2atomid_index, rec_pos
                        )
                    if pep_frame is None:
                        pep_frame = _frame_from_seq_cb2(
                            pep_pos, atom2res_index, atom2atomid_index
                        )
                    if pep_frame is None:
                        pep_frame = _frame_from_ncac(
                            pep_pos, atom2res_index, atom2atomid_index, n_res, rec_pos
                        )
                    pep_frame = _flip_frame_by_rec(pep_frame, rec_pos, pep_pos)
                    if rec_frame is not None and pep_frame is not None:
                        R_hat = rec_frame.T @ pep_frame
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_target = R_hat.to(dtype=torch.float32).reshape(1, 9)
                    else:
                        rot_valid = False
                        rot_error = "frame_rel_iface:frame_none"
                except Exception:
                    rot_valid = False
                    rot_error = "frame_rel_iface:exception"
            elif rot_target_mode == "frame_world":
                try:
                    pep_pos = data["pep_a"].pos if "pep_a" in data.node_types else None
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    n_res = (
                        int(data["pep"].x.shape[0])
                        if "pep" in data.node_types
                        else int(atom2res_index.max().item() + 1)
                    )
                    pep_idx3 = getattr(data, "pep_anchor_idx3", None)
                    if pep_idx3 is None and "pep_anchor_idx3" in data:
                        pep_idx3 = data["pep_anchor_idx3"]
                    pep_endpoints = getattr(data, "pep_axis_endpoints", None)
                    if pep_endpoints is None and "pep_axis_endpoints" in data:
                        pep_endpoints = data["pep_axis_endpoints"]
                    pep_aux = None
                    if pep_idx3 is not None and pep_pos is not None:
                        pep_idx = pep_idx3.to(device=pep_pos.device).long()
                        if pep_idx.numel() >= 3 and pep_idx.max().item() < pep_pos.shape[0]:
                            pep_aux = pep_pos[pep_idx[2]]
                    if pep_aux is None and rec_pos is not None and rec_pos.numel() > 0:
                        pep_aux = rec_pos.mean(dim=0)
                    pep_frame = _frame_from_pep_endpoints(pep_endpoints, pep_aux)
                    if pep_frame is None:
                        pep_frame = _frame_from_anchor_idx3(pep_pos, pep_idx3, None)
                    if pep_frame is None:
                        pep_frame = _frame_from_pep_unique(
                            pep_pos, atom2res_index, atom2atomid_index, rec_pos
                        )
                    if pep_frame is None:
                        pep_frame = _frame_from_seq_weighted(
                            pep_pos, atom2res_index, atom2atomid_index, rec_pos
                        )
                    if pep_frame is None:
                        pep_frame = _frame_from_seq_cb2(
                            pep_pos, atom2res_index, atom2atomid_index
                        )
                    if pep_frame is None:
                        pep_frame = _frame_from_ncac(
                            pep_pos, atom2res_index, atom2atomid_index, n_res, rec_pos
                        )
                    if pep_frame is not None:
                        R_hat = pep_frame
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_target = R_hat.to(dtype=torch.float32).reshape(1, 9)
                    else:
                        rot_valid = False
                        rot_error = "frame_world:frame_none"
                except Exception:
                    rot_valid = False
                    rot_error = "frame_world:exception"
            elif rot_target_mode == "const_frame":
                try:
                    axis = self.flow_cfg.get("rot_const_axis", [0.0, 0.0, 1.0])
                    angle = float(self.flow_cfg.get("rot_const_angle", 1.0) or 1.0)
                    axis = torch.tensor(axis, dtype=torch.float32, device=rot_target.device if rot_target is not None else None)
                    if axis.norm().item() < 1e-6:
                        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=axis.device)
                    axis = axis / axis.norm()
                    rot_update = (axis * angle).to(dtype=torch.float32)
                    R_hat = axis_angle_to_matrix(rot_update)
                    rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                    rot_target = R_hat.to(dtype=torch.float32).reshape(1, 9)
                except Exception:
                    rot_valid = False
                    rot_error = "const_frame:exception"
            elif rot_target_mode == "frame_xt":
                try:
                    pep_pos = data["pep_a"].pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    n_res = (
                        int(data["pep"].x.shape[0])
                        if "pep" in data.node_types
                        else int(atom2res_index.max().item() + 1)
                    )
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    contact_only = bool(self.flow_cfg.get("rot_frame_contact_only", False))
                    contact_cutoff = float(self.flow_cfg.get("rot_frame_contact_cutoff", 8.0) or 8.0)
                    contact_min_points = int(self.flow_cfg.get("rot_frame_contact_min_points", 6) or 6)
                    contact_fallback = bool(self.flow_cfg.get("rot_frame_contact_fallback", True))
                    rec_frame = None
                    if contact_only:
                        rec_iface = _select_rec_interface(
                            rec_pos, pep_pos, contact_cutoff, contact_min_points
                        )
                        if rec_iface is None:
                            if contact_fallback:
                                rec_frame = _frame_from_rec_pca(rec_pos, pep_pos)
                            else:
                                rot_valid = False
                                rot_error = "frame_xt:contact_insufficient"
                        else:
                            rec_frame = _frame_from_rec_pca(rec_iface, pep_pos)
                    else:
                        rec_frame = _frame_from_rec_pca(rec_pos, pep_pos)
                    pep_frame = _frame_from_ncac(
                        pep_pos, atom2res_index, atom2atomid_index, n_res, rec_pos
                    )
                    if pep_frame is None:
                        pep_frame = _frame_from_pep_ca(
                            pep_pos, atom2res_index, atom2atomid_index, rec_pos
                        )
                    if rec_frame is not None and pep_frame is not None:
                        R_hat = rec_frame @ pep_frame.T
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_target = R_hat.to(dtype=torch.float32).reshape(1, 9)
                    else:
                        rot_valid = False
                        rot_error = "frame_xt:frame_none"
                except Exception:
                    rot_valid = False
                    rot_error = "frame_xt:exception"
            elif rot_target_mode == "frame_unique":
                try:
                    pep_orig = data["pep_a"].orig_pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    n_res = (
                        int(data["pep"].x.shape[0])
                        if "pep" in data.node_types
                        else int(atom2res_index.max().item() + 1)
                    )
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    rec_frame = _frame_from_rec_pca_unique(rec_pos, pep_orig)
                    pep_frame = _frame_from_ncac(
                        pep_orig, atom2res_index, atom2atomid_index, n_res, rec_pos
                    )
                    if pep_frame is None:
                        pep_frame = _frame_from_pep_ca(
                            pep_orig, atom2res_index, atom2atomid_index, rec_pos
                        )
                    if rec_frame is not None and pep_frame is not None:
                        R_hat = rec_frame @ pep_frame.T
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_target = R_hat.to(dtype=torch.float32).reshape(1, 9)
                    else:
                        rot_valid = False
                        rot_error = "frame_unique:frame_none"
                except Exception:
                    rot_valid = False
                    rot_error = "frame_unique:exception"
            elif rot_target_mode == "frame_local":
                try:
                    pep_orig = data["pep_a"].orig_pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    n_res = (
                        int(data["pep"].x.shape[0])
                        if "pep" in data.node_types
                        else int(atom2res_index.max().item() + 1)
                    )
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    contact_only = bool(self.flow_cfg.get("rot_frame_contact_only", False))
                    contact_cutoff = float(self.flow_cfg.get("rot_frame_contact_cutoff", 8.0) or 8.0)
                    contact_min_points = int(self.flow_cfg.get("rot_frame_contact_min_points", 6) or 6)
                    contact_fallback = bool(self.flow_cfg.get("rot_frame_contact_fallback", True))
                    rec_frame = None
                    if contact_only:
                        rec_iface = _select_rec_interface(
                            rec_pos, pep_orig, contact_cutoff, contact_min_points
                        )
                        if rec_iface is None:
                            if contact_fallback:
                                rec_frame = _frame_from_rec_pca(rec_pos, pep_orig)
                            else:
                                rot_valid = False
                                rot_error = "frame_local:contact_insufficient"
                        else:
                            rec_frame = _frame_from_rec_pca(rec_iface, pep_orig)
                    else:
                        rec_frame = _frame_from_rec_pca(rec_pos, pep_orig)
                    pep_frame = _frame_from_ncac(
                        pep_orig, atom2res_index, atom2atomid_index, n_res, rec_pos
                    )
                    if pep_frame is None:
                        pep_frame = _frame_from_pep_ca(
                            pep_orig, atom2res_index, atom2atomid_index, rec_pos
                        )
                    if rec_frame is not None and pep_frame is not None:
                        R_local = rec_frame.T @ pep_frame
                        rot_target_mat = R_local.to(dtype=torch.float32).unsqueeze(0)
                        rot_target = R_local.to(dtype=torch.float32).reshape(1, 9)
                    else:
                        rot_valid = False
                        rot_error = "frame_local:frame_none"
                except Exception:
                    rot_valid = False
                    rot_error = "frame_local:exception"
            elif rot_target_mode == "frame_local_xt":
                try:
                    pep_pos = data["pep_a"].pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    n_res = (
                        int(data["pep"].x.shape[0])
                        if "pep" in data.node_types
                        else int(atom2res_index.max().item() + 1)
                    )
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    contact_only = bool(self.flow_cfg.get("rot_frame_contact_only", False))
                    contact_cutoff = float(self.flow_cfg.get("rot_frame_contact_cutoff", 8.0) or 8.0)
                    contact_min_points = int(self.flow_cfg.get("rot_frame_contact_min_points", 6) or 6)
                    contact_fallback = bool(self.flow_cfg.get("rot_frame_contact_fallback", True))
                    rec_frame = None
                    if contact_only:
                        rec_iface = _select_rec_interface(
                            rec_pos, pep_pos, contact_cutoff, contact_min_points
                        )
                        if rec_iface is None:
                            if contact_fallback:
                                rec_frame = _frame_from_rec_pca(rec_pos, pep_pos)
                            else:
                                rot_valid = False
                                rot_error = "frame_local_xt:contact_insufficient"
                        else:
                            rec_frame = _frame_from_rec_pca(rec_iface, pep_pos)
                    else:
                        rec_frame = _frame_from_rec_pca(rec_pos, pep_pos)
                    pep_frame = _frame_from_ncac(
                        pep_pos, atom2res_index, atom2atomid_index, n_res, rec_pos
                    )
                    if pep_frame is None:
                        pep_frame = _frame_from_pep_ca(
                            pep_pos, atom2res_index, atom2atomid_index, rec_pos
                        )
                    if rec_frame is not None and pep_frame is not None:
                        R_local = rec_frame.T @ pep_frame
                        rot_target_mat = R_local.to(dtype=torch.float32).unsqueeze(0)
                        rot_target = R_local.to(dtype=torch.float32).reshape(1, 9)
                    else:
                        rot_valid = False
                        rot_error = "frame_local_xt:frame_none"
                except Exception:
                    rot_valid = False
                    rot_error = "frame_local_xt:exception"
            elif rot_target_mode == "anchor":
                try:
                    pep_pos = data["pep_a"].pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    n_res = (
                        int(data["pep"].x.shape[0])
                        if "pep" in data.node_types
                        else int(atom2res_index.max().item() + 1)
                    )
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    pep_center = pep_pos.mean(dim=0)
                    rec_frame = _frame_from_rec_anchor(rec_pos, pep_center)
                    frame_t = _frame_from_ncac(
                        pep_pos, atom2res_index, atom2atomid_index, n_res, rec_pos
                    )
                    if rec_frame is not None and frame_t is not None:
                        R_hat = rec_frame @ frame_t.T
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_update = matrix_to_axis_angle(R_hat).to(dtype=torch.float32)
                        if rot_update.ndim == 2:
                            rot_update = rot_update.squeeze(0)
                        rot_target = (rot_update / time_scale).unsqueeze(0)
                except Exception:
                    pass
            elif rot_target_mode == "anchor_ref":
                try:
                    pep_orig = data["pep_a"].orig_pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    n_res = (
                        int(data["pep"].x.shape[0])
                        if "pep" in data.node_types
                        else int(atom2res_index.max().item() + 1)
                    )
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    pep_center = pep_orig.mean(dim=0)
                    rec_frame = _frame_from_rec_anchor(rec_pos, pep_center)
                    frame_ref = _frame_from_ncac(
                        pep_orig, atom2res_index, atom2atomid_index, n_res, rec_pos
                    )
                    if rec_frame is not None and frame_ref is not None:
                        R_hat = rec_frame @ frame_ref.T
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_update = matrix_to_axis_angle(R_hat).to(dtype=torch.float32)
                        if rot_update.ndim == 2:
                            rot_update = rot_update.squeeze(0)
                        rot_target = (rot_update / time_scale).unsqueeze(0)
                except Exception:
                    pass
            elif rot_target_mode == "anchor_seq_cb2":
                try:
                    pep_pos = data["pep_a"].pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    pep_center = pep_pos.mean(dim=0)
                    rec_frame = _frame_from_rec_anchor(rec_pos, pep_center)
                    frame_t = _frame_from_seq_cb2(pep_pos, atom2res_index, atom2atomid_index)
                    if frame_t is None:
                        frame_t = _frame_from_seq_cb(pep_pos, atom2res_index, atom2atomid_index)
                    if frame_t is None:
                        n_res = (
                            int(data["pep"].x.shape[0])
                            if "pep" in data.node_types
                            else int(atom2res_index.max().item() + 1)
                        )
                        frame_t = _frame_from_ncac(
                            pep_pos, atom2res_index, atom2atomid_index, n_res, rec_pos
                        )
                    if rec_frame is not None and frame_t is not None:
                        R_hat = rec_frame @ frame_t.T
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_update = matrix_to_axis_angle(R_hat).to(dtype=torch.float32)
                        if rot_update.ndim == 2:
                            rot_update = rot_update.squeeze(0)
                        rot_target = (rot_update / time_scale).unsqueeze(0)
                except Exception:
                    pass
            elif rot_target_mode == "anchor_ref_seq_cb2":
                try:
                    pep_orig = data["pep_a"].orig_pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    pep_center = pep_orig.mean(dim=0)
                    rec_frame = _frame_from_rec_anchor(rec_pos, pep_center)
                    frame_ref = _frame_from_seq_cb2(pep_orig, atom2res_index, atom2atomid_index)
                    if frame_ref is None:
                        frame_ref = _frame_from_seq_cb(pep_orig, atom2res_index, atom2atomid_index)
                    if frame_ref is None:
                        n_res = (
                            int(data["pep"].x.shape[0])
                            if "pep" in data.node_types
                            else int(atom2res_index.max().item() + 1)
                        )
                        frame_ref = _frame_from_ncac(
                            pep_orig, atom2res_index, atom2atomid_index, n_res, rec_pos
                        )
                    if rec_frame is not None and frame_ref is not None:
                        R_hat = rec_frame @ frame_ref.T
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_update = matrix_to_axis_angle(R_hat).to(dtype=torch.float32)
                        if rot_update.ndim == 2:
                            rot_update = rot_update.squeeze(0)
                        rot_target = (rot_update / time_scale).unsqueeze(0)
                except Exception:
                    pass
            elif rot_target_mode == "anchor_ref_iface_sc":
                try:
                    pep_orig = data["pep_a"].orig_pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    pep_center = pep_orig.mean(dim=0)
                    rec_frame = _frame_from_rec_anchor(rec_pos, pep_center)
                    frame_ref = _frame_from_iface_sc(
                        rec_pos, pep_orig, atom2res_index, atom2atomid_index
                    )
                    if frame_ref is None:
                        n_res = (
                            int(data["pep"].x.shape[0])
                            if "pep" in data.node_types
                            else int(atom2res_index.max().item() + 1)
                        )
                        frame_ref = _frame_from_ncac(
                            pep_orig, atom2res_index, atom2atomid_index, n_res, rec_pos
                        )
                    if rec_frame is not None and frame_ref is not None:
                        R_hat = rec_frame @ frame_ref.T
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_update = matrix_to_axis_angle(R_hat).to(dtype=torch.float32)
                        if rot_update.ndim == 2:
                            rot_update = rot_update.squeeze(0)
                        rot_target = (rot_update / time_scale).unsqueeze(0)
                except Exception:
                    pass
            elif rot_target_mode == "anchor_ref_iface_sc_twist":
                try:
                    pep_orig = data["pep_a"].orig_pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    pep_center = pep_orig.mean(dim=0)
                    rec_frame = _frame_from_rec_anchor(rec_pos, pep_center)
                    frame_ref = _frame_from_iface_sc(
                        rec_pos, pep_orig, atom2res_index, atom2atomid_index
                    )
                    if frame_ref is None:
                        n_res = (
                            int(data["pep"].x.shape[0])
                            if "pep" in data.node_types
                            else int(atom2res_index.max().item() + 1)
                        )
                        frame_ref = _frame_from_ncac(
                            pep_orig, atom2res_index, atom2atomid_index, n_res, rec_pos
                        )
                    if rec_frame is not None and frame_ref is not None:
                        R_hat = rec_frame @ frame_ref.T
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_update = matrix_to_axis_angle(R_hat).to(dtype=torch.float32)
                        if rot_update.ndim == 2:
                            rot_update = rot_update.squeeze(0)
                        rot_target = (rot_update / time_scale).unsqueeze(0)
                        axis_ref = _safe_normalize(frame_ref[:, 0])
                        data.flow_rot_axis_ref = axis_ref.to(dtype=torch.float32).unsqueeze(0)
                except Exception:
                    pass
            elif rot_target_mode == "anchor_ref_seq_weighted_twist":
                try:
                    pep_orig = data["pep_a"].orig_pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    pep_center = pep_orig.mean(dim=0)
                    rec_frame = _frame_from_rec_anchor(rec_pos, pep_center)
                    frame_ref = _frame_from_seq_weighted(
                        pep_orig, atom2res_index, atom2atomid_index, rec_pos
                    )
                    if frame_ref is None:
                        n_res = (
                            int(data["pep"].x.shape[0])
                            if "pep" in data.node_types
                            else int(atom2res_index.max().item() + 1)
                        )
                        frame_ref = _frame_from_ncac(
                            pep_orig, atom2res_index, atom2atomid_index, n_res, rec_pos
                        )
                    if rec_frame is not None and frame_ref is not None:
                        R_hat = rec_frame @ frame_ref.T
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_update = matrix_to_axis_angle(R_hat).to(dtype=torch.float32)
                        if rot_update.ndim == 2:
                            rot_update = rot_update.squeeze(0)
                        rot_target = (rot_update / time_scale).unsqueeze(0)
                        axis_ref = _safe_normalize(frame_ref[:, 0])
                        data.flow_rot_axis_ref = axis_ref.to(dtype=torch.float32).unsqueeze(0)
                except Exception:
                    pass
            elif rot_target_mode in {
                "contact_near2",
                "contact_nearfar",
                "contact_pca",
                "contact_normal",
            }:
                try:
                    pep_pos = data["pep_a"].pos
                    atom2res_index = data["pep_a"].atom2res_index
                    atom2atomid_index = data["pep_a"].atom2atomid_index
                    rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                    pep_center = pep_pos.mean(dim=0) if pep_pos is not None else None
                    radius = float(self.flow_cfg.get("rot_contact_radius", 8.0) or 8.0)
                    min_points = int(self.flow_cfg.get("rot_contact_min_points", 6) or 6)
                    rec_frame = _frame_from_rec_contact(
                        rec_pos, pep_center, rot_target_mode, radius, min_points
                    )
                    pep_frame = _frame_from_pep_ca(
                        pep_pos, atom2res_index, atom2atomid_index, rec_pos
                    )
                    if pep_frame is None:
                        n_res = (
                            int(data["pep"].x.shape[0])
                            if "pep" in data.node_types
                            else int(atom2res_index.max().item() + 1)
                        )
                        pep_frame = _frame_from_ncac(
                            pep_pos, atom2res_index, atom2atomid_index, n_res, rec_pos
                        )
                    if rec_frame is not None and pep_frame is not None:
                        R_hat = rec_frame @ pep_frame.T
                        rot_target_mat = R_hat.to(dtype=torch.float32).unsqueeze(0)
                        rot_update = matrix_to_axis_angle(R_hat).to(dtype=torch.float32)
                        if rot_update.ndim == 2:
                            rot_update = rot_update.squeeze(0)
                        rot_target = (rot_update / time_scale).unsqueeze(0)
                    else:
                        rot_valid = False
                except Exception:
                    rot_valid = False

            ref_axis = _get_rot_axis_ref()
            if rot_target is not None and rot_target_mode not in {"x1", "frame_vector", "frame_unique", "frame_local", "frame_rel_iface"}:
                rot_target = _apply_axis_ref(rot_target, ref_axis)

            data.flow_rot_target = rot_target
            if rot_target_mat is not None:
                data.flow_rot_target_mat = rot_target_mat
            data.flow_rot_valid = torch.tensor([1 if rot_valid else 0], dtype=torch.long)
            if rot_error is not None:
                data.flow_rot_error = rot_error
            axis_loss_ref = _get_rot_axis_for_loss()
            if axis_loss_ref is not None and not hasattr(data, "flow_rot_axis_ref"):
                data.flow_rot_axis_ref = axis_loss_ref.to(dtype=torch.float32).unsqueeze(0)

            if rot_ref_input:
                rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                pep_pos = None
                if "pep_a" in data.node_types and hasattr(data["pep_a"], "orig_pos"):
                    pep_pos = data["pep_a"].orig_pos
                elif "pep_a" in data.node_types and hasattr(data["pep_a"], "pos"):
                    pep_pos = data["pep_a"].pos
                num_graphs = getattr(data, "num_graphs", 1)
                device = rec_pos.device if rec_pos is not None else (pep_pos.device if pep_pos is not None else None)
                if device is not None:
                    ref_axis = torch.zeros((num_graphs, 3), dtype=torch.float32, device=device)
                    ref_point = torch.zeros((num_graphs, 3), dtype=torch.float32, device=device)
                    rec_batch = getattr(data["receptor"], "batch", None) if "receptor" in data.node_types else None
                    pep_batch = getattr(data["pep_a"], "batch", None) if "pep_a" in data.node_types else None
                    if rec_batch is None and rec_pos is not None:
                        rec_batch = torch.zeros(rec_pos.shape[0], dtype=torch.long, device=device)
                    if pep_batch is None and pep_pos is not None:
                        pep_batch = torch.zeros(pep_pos.shape[0], dtype=torch.long, device=device)
                    for g in range(num_graphs):
                        rec_pos_g = rec_pos[rec_batch == g] if rec_pos is not None else None
                        pep_pos_g = pep_pos[pep_batch == g] if pep_pos is not None else None
                        axis = _get_ref_axis_input(rec_pos_g, pep_pos_g)
                        if axis is None:
                            axis = torch.tensor([1.0, 0.0, 0.0], device=device)
                        ref_axis[g] = axis
                        if pep_pos_g is not None and pep_pos_g.numel() > 0:
                            pep_center = pep_pos_g.mean(dim=0)
                        elif rec_pos_g is not None and rec_pos_g.numel() > 0:
                            pep_center = rec_pos_g.mean(dim=0)
                        else:
                            pep_center = torch.zeros(3, device=device)
                        ref_point[g] = pep_center + axis * rot_ref_dist
                    data.ref_axis = ref_axis
                    data.ref_point = ref_point

            rot_cls_bins = int(self.flow_cfg.get("rot_cls_bins", 0) or 0)
            if rot_cls_bins > 0:
                rot_cls_target_mode = str(
                    self.flow_cfg.get("rot_cls_target_mode", "kabsch") or "kabsch"
                ).lower()
                rot_cls_mat = None
                if rot_cls_target_mode == rot_target_mode and rot_target_mat is not None:
                    rot_cls_mat = rot_target_mat
                elif rot_cls_target_mode == "kabsch":
                    try:
                        pep_pos = data["pep_a"].pos
                        pep_orig = data["pep_a"].orig_pos
                        atom_ids = data["pep_a"].atom2atomid_index
                        ca_mask = atom_ids == 1
                        if ca_mask.sum().item() >= 3:
                            cur_ca = pep_pos[ca_mask]
                            orig_ca = pep_orig[ca_mask]
                        else:
                            cur_ca = pep_pos
                            orig_ca = pep_orig
                        if cur_ca.shape[0] >= 3 and orig_ca.shape[0] >= 3:
                            R_hat, _t_hat = kabsch_torch(cur_ca.T, orig_ca.T)
                            rot_cls_mat = R_hat.to(dtype=torch.float32)
                    except Exception:
                        rot_cls_mat = None
                elif rot_cls_target_mode == "frame":
                    try:
                        pep_pos = data["pep_a"].pos
                        pep_orig = data["pep_a"].orig_pos
                        atom2res_index = data["pep_a"].atom2res_index
                        atom2atomid_index = data["pep_a"].atom2atomid_index
                        n_res = (
                            int(data["pep"].x.shape[0])
                            if "pep" in data.node_types
                            else int(atom2res_index.max().item() + 1)
                        )
                        rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                        frame_t = _frame_from_ncac(
                            pep_pos, atom2res_index, atom2atomid_index, n_res, rec_pos
                        )
                        frame_0 = _frame_from_ncac(
                            pep_orig, atom2res_index, atom2atomid_index, n_res, rec_pos
                        )
                        if frame_t is not None and frame_0 is not None:
                            rot_cls_mat = (frame_0 @ frame_t.T).to(dtype=torch.float32)
                    except Exception:
                        rot_cls_mat = None
                elif rot_cls_target_mode == "anchor":
                    try:
                        pep_pos = data["pep_a"].pos
                        atom2res_index = data["pep_a"].atom2res_index
                        atom2atomid_index = data["pep_a"].atom2atomid_index
                        n_res = (
                            int(data["pep"].x.shape[0])
                            if "pep" in data.node_types
                            else int(atom2res_index.max().item() + 1)
                        )
                        rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                        pep_center = pep_pos.mean(dim=0)
                        rec_frame = _frame_from_rec_anchor(rec_pos, pep_center)
                        frame_t = _frame_from_ncac(
                            pep_pos, atom2res_index, atom2atomid_index, n_res, rec_pos
                        )
                        if rec_frame is not None and frame_t is not None:
                            rot_cls_mat = (rec_frame @ frame_t.T).to(dtype=torch.float32)
                    except Exception:
                        rot_cls_mat = None
                elif rot_cls_target_mode == "anchor_ref":
                    try:
                        pep_orig = data["pep_a"].orig_pos
                        atom2res_index = data["pep_a"].atom2res_index
                        atom2atomid_index = data["pep_a"].atom2atomid_index
                        n_res = (
                            int(data["pep"].x.shape[0])
                            if "pep" in data.node_types
                            else int(atom2res_index.max().item() + 1)
                        )
                        rec_pos = data["receptor"].pos if "receptor" in data.node_types else None
                        pep_center = pep_orig.mean(dim=0)
                        rec_frame = _frame_from_rec_anchor(rec_pos, pep_center)
                        frame_ref = _frame_from_ncac(
                            pep_orig, atom2res_index, atom2atomid_index, n_res, rec_pos
                        )
                        if rec_frame is not None and frame_ref is not None:
                            rot_cls_mat = (rec_frame @ frame_ref.T).to(dtype=torch.float32)
                    except Exception:
                        rot_cls_mat = None

                if rot_cls_mat is None and rot_target is not None:
                    rot_cls_mat = axis_angle_to_matrix(rot_target.squeeze(0))
                if rot_cls_mat is not None:
                    codebook = get_so3_codebook(rot_cls_bins, device=rot_cls_mat.device)
                    rot_cls_target = nearest_rotmat_bin(rot_cls_mat, codebook).to(dtype=torch.long)
                    data.flow_rot_cls_target = rot_cls_target
            data.flow_tor_backbone_target = tor_bb_target
            data.flow_tor_sidechain_target = tor_sc_target
            data.flow_time = torch.tensor([t], dtype=torch.float32)

            if debug_pose and pep_a_before is not None:
                try:
                    pep_a_after = data["pep_a"].pos.detach()
                    com_before = pep_a_before.mean(dim=0)
                    com_after = pep_a_after.mean(dim=0)
                    delta_com = com_after - com_before
                    tr_expected = tr_update.squeeze(0).detach()
                    tr_err = (delta_com - tr_expected).norm().item()

                    # 估计整体刚体旋转：Kabsch(before -> after)
                    R_hat, _t_hat = kabsch_torch(pep_a_before.T, pep_a_after.T)
                    R_exp = axis_angle_to_matrix(rot_update.detach().squeeze())
                    R_diff = R_hat @ R_exp.T
                    trace = torch.clamp((torch.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
                    rot_err = torch.acos(trace).item()

                    data.flow_debug_tr_err = torch.tensor([tr_err], dtype=torch.float32)
                    data.flow_debug_rot_err = torch.tensor([rot_err], dtype=torch.float32)
                    data.flow_debug_t = torch.tensor([t], dtype=torch.float32)
                    data.flow_debug_tr_mag = torch.tensor([tr_expected.norm().item()], dtype=torch.float32)
                    data.flow_debug_rot_mag = torch.tensor([rot_update.norm().item()], dtype=torch.float32)
                except Exception as e:
                    data.flow_debug_error = str(e)

            return data
        finally:
            if seed is not None:
                np.random.set_state(np_state)
                torch.random.set_rng_state(torch_state)
