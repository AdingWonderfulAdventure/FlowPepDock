import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn import functional as F

class NoiseSchedule:
    """把归一化时间t映射成各自由度对应的sigma，控制扩散幅度"""
    def __init__(self, args):
        self.tr_sigma_min = args.tr_sigma_min # 0.1
        self.tr_sigma_max = args.tr_sigma_max # 
        self.rot_sigma_min = args.rot_sigma_min # 0.1
        self.rot_sigma_max = args.rot_sigma_max # 1.65
        self.tor_backbone_sigma_min = args.tor_backbone_sigma_min # 0.0314
        self.tor_backbone_sigma_max = args.tor_backbone_sigma_max # 3.14 
        self.tor_sidechain_sigma_min = args.tor_sidechain_sigma_min # 0.0314
        self.tor_sidechain_sigma_max = args.tor_sidechain_sigma_max # 3.14

    def __call__(self, t_tr, t_rot, t_tor_backbone, t_tor_sidechain):
        """指数插值出目标sigma，保持噪声随时间单调衰减"""
        tr_sigma = self.tr_sigma_min ** (1 - t_tr) * self.tr_sigma_max**t_tr
        rot_sigma = self.rot_sigma_min ** (1 - t_rot) * self.rot_sigma_max**t_rot
        tor_backbone_sigma = self.tor_backbone_sigma_min ** (1 - t_tor_backbone) * self.tor_backbone_sigma_max**t_tor_backbone
        tor_sidechain_sigma = self.tor_sidechain_sigma_min ** (1 - t_tor_sidechain) * self.tor_sidechain_sigma_max**t_tor_sidechain
        return tr_sigma, rot_sigma, tor_backbone_sigma, tor_sidechain_sigma
    
class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_size, scale):
        super().__init__()
        self.embed_dim = embedding_size
        self.scale = scale
        self.max_positions = 1e4

    def forward(self, x):
        """参考DDPM实现，用正余弦对时间位置编码"""
        assert len(x.shape) == 1
        x = self.scale * x

        half_dim = self.embed_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim,
                dtype=torch.float32,
                device=x.device) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode="constant")
        assert emb.shape == (x.shape[0], self.embed_dim)
        return emb
    
class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embedding_size // 2) * scale,
            requires_grad=False
        )

    def forward(self, x):
        """以高斯初始化的频率对噪声级别做Fourier编码"""
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)],
                        dim=-1)
        return emb

def get_t_schedule(inference_steps):
    """推理阶段把时间从1线性扫到0"""
    return np.linspace(1, 0, inference_steps + 1)[:-1]

def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    """根据配置选择时间嵌入函数，默认走正弦位置编码"""
    if embedding_type == "sinusoidal":
        emb_func = SinusoidalEmbedding(embedding_size = embedding_dim, scale = embedding_scale)
    elif embedding_type == "fourier":
        emb_func = GaussianFourierProjection(embedding_size = embedding_dim, scale = embedding_scale)
    else:
        raise NotImplemented
    return emb_func

def set_time(complex_graphs, t_tr, t_rot, t_tor_backbone, t_tor_sidechain, batch_size: int, device=None):
    """把采样的四种时间戳写回图结构，后续网络就能查到对应sigma"""
    target_device = None if device is None else torch.device(device)

    def _assign_time_dict(holder, attr_name, size):
        current = getattr(holder, attr_name, None)
        tr = current.get("tr") if isinstance(current, dict) else None
        rot = current.get("rot") if isinstance(current, dict) else None
        tor_backbone = current.get("tor_backbone") if isinstance(current, dict) else None
        tor_sidechain = current.get("tor_sidechain") if isinstance(current, dict) else None
        need_new = (
            tr is None
            or rot is None
            or tor_backbone is None
            or tor_sidechain is None
            or tr.numel() != size
            or rot.numel() != size
            or tor_backbone.numel() != size
            or tor_sidechain.numel() != size
            or (target_device is not None and tr.device != target_device)
        )
        if need_new:
            current = {
                "tr": torch.empty(size, device=target_device),
                "rot": torch.empty(size, device=target_device),
                "tor_backbone": torch.empty(size, device=target_device),
                "tor_sidechain": torch.empty(size, device=target_device),
            }
            setattr(holder, attr_name, current)
        current["tr"].fill_(t_tr)
        current["rot"].fill_(t_rot)
        current["tor_backbone"].fill_(t_tor_backbone)
        current["tor_sidechain"].fill_(t_tor_sidechain)

    _assign_time_dict(complex_graphs["pep"], "node_t", complex_graphs["pep"].num_nodes)
    _assign_time_dict(complex_graphs["receptor"], "node_t", complex_graphs["receptor"].num_nodes)
    _assign_time_dict(complex_graphs["pep_a"], "node_t", complex_graphs["pep_a"].num_nodes)
    _assign_time_dict(complex_graphs, "complex_t", batch_size)
