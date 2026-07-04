import copy
import time
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_cluster import radius
from torch_scatter import scatter, scatter_mean
from e3nn import o3
from e3nn.nn import BatchNorm
from utils.flow_utils import get_timestep_embedding
from utils.geometry import matrix_to_axis_angle, rot6d_to_matrix
from utils.sampling import _frame_from_rec_pca, _frame_from_ncac, _safe_normalize
from dataset.peptide_feature import three2idx, allowable_features, atomname2idx, get_updated_peptide_feature

# 这个SB列表把氨基酸类型/原子类型/原子特征维度都罗列好，后面嵌入层统一复用
feature_dims = [
            len(three2idx), max([len(res) for res in atomname2idx]) # Amino_idx_dim/ Atom_idx_dim
        ] + \
        [
            len(value) for key, value in allowable_features.items() # Atom_features_dim
        ] + [4] # Atom_charity_center_dim

class GaussianSmearing(nn.Module):
    # used to embed the edge dists
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        # 距离高斯展开，保证后面卷积拿到平滑连续的特征
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
class AminoEmbedding(nn.Module):
    """
        Embeddings for atom identity only (as of now)
    """
    def __init__(self, num_amino, emb_dim, sigma_embed_dim, intra_dim, dihediral_dim, lm_embedding_type = None):
        super(AminoEmbedding, self).__init__()
        self.amino_type_dim = 1
        self.sigma_embed_dim = sigma_embed_dim
        self.intra_dim = intra_dim
        self.dihediral_dim = dihediral_dim
        self.lm_embedding_type = lm_embedding_type
        
        self.amino_ebd = nn.Embedding(num_amino+1, emb_dim) # add 1 for padding
        self.sigma_ebd = nn.Linear(sigma_embed_dim, emb_dim)
        self.intra_dis_ebd = nn.Linear(intra_dim, emb_dim)
        self.dihediral_angles_ebd = nn.Linear(dihediral_dim, emb_dim)
        # onehot 尾巴投影（104 -> emb_dim）。提前建好，避免 forward 里动态创建导致 ckpt resume 直接炸。
        self.tail_proj = nn.Linear(104, emb_dim)
        # rot_ref_input 额外参照轴（3 -> emb_dim）
        self.ref_proj = nn.Linear(3, emb_dim)
        
        # LM embedding (ESM2)
        if self.lm_embedding_type is not None:
            if self.lm_embedding_type == 'esm':
                self.lm_embedding_dim = 1280
                self.lm_embedding_layer = nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)
            else: raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ', self.lm_embedding_type)

    def forward(self, x):
        proj_dtype = self.sigma_ebd.weight.dtype
        base_expected = self.amino_type_dim + self.sigma_embed_dim + self.intra_dim + self.dihediral_dim
        if self.lm_embedding_type is not None:
            base_expected += self.lm_embedding_dim

        tail_dim = x.shape[1] - base_expected
        # 允许 tail_dim 为 0/104，或带 ref_axis 的 3/107
        if tail_dim not in (0, 104, 3, 107):
            raise AssertionError(
                f"Input dim mismatch: expected {base_expected} or {base_expected+104} (+3 ref), got {x.shape[1]}"
            )
        self.tail_dim = tail_dim

        x_embedding = 0
        # 氨基酸种类嵌入+噪声嵌入+局部构象特征统统叠加，搞个全量表示
        x_embedding += self.amino_ebd(x[:,:self.amino_type_dim].long()).squeeze() # amino_ebd
        # sigma noise embedding
        x_embedding += self.sigma_ebd(x[:,-self.sigma_embed_dim:].to(dtype=proj_dtype)) # sigma_ebd
        x_embedding += self.intra_dis_ebd(x[:,self.amino_type_dim:self.amino_type_dim+self.intra_dim].to(dtype=proj_dtype)) # intra_dis_ebd
        x_embedding += self.dihediral_angles_ebd(x[:,self.amino_type_dim+self.intra_dim:self.amino_type_dim+self.intra_dim+self.dihediral_dim].to(dtype=proj_dtype)) # dihediral_angles_ebd
        # 尾部 onehot（若存在 104 维）映射到 emb_dim
        if getattr(self, "tail_dim", 0) in (104, 107):
            tail_start = self.amino_type_dim + self.intra_dim + self.dihediral_dim
            x_embedding += self.tail_proj(x[:, tail_start:tail_start + 104].to(dtype=self.tail_proj.weight.dtype))
        # 参照轴（若存在 3 维），位于 sigma 前 3 维
        if getattr(self, "tail_dim", 0) in (3, 107):
            ref_end = x.shape[1] - self.sigma_embed_dim
            ref_start = ref_end - 3
            if ref_start >= 0:
                x_embedding += self.ref_proj(x[:, ref_start:ref_end].to(dtype=self.ref_proj.weight.dtype))
        # # consider LM embedding here
        if self.lm_embedding_type is not None:
            lm_dtype = self.lm_embedding_layer.weight.dtype
            x_embedding = self.lm_embedding_layer(
                torch.cat(
                    [
                        x_embedding.to(dtype=lm_dtype),
                        x[
                            :,
                            self.amino_type_dim + self.intra_dim + self.dihediral_dim:
                            self.amino_type_dim + self.intra_dim + self.dihediral_dim + self.lm_embedding_dim,
                        ].to(dtype=lm_dtype),
                    ],
                    dim=1,
                )
            )
        return x_embedding

class AtomEncoder(nn.Module):

    def __init__(self, emb_dim, sigma_embed_dim, pep_attr_dim):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims)
        self.num_scalar_features = sigma_embed_dim
        self.emb_dim = emb_dim
        self.pep_attr_dim = pep_attr_dim
        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
        self.final_layer = torch.nn.Linear(emb_dim + pep_attr_dim, emb_dim)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.pep_attr_dim
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
        # 拼上肽残基附加属性，再走一层线性整形
        x_embedding = self.final_layer(torch.cat([x_embedding, x[:, -self.pep_attr_dim:]], axis=1))
        return x_embedding
    
class EdgeEmbedding(nn.Module):
    """
        Embeddings for edge feature (as of now)
    """
    def __init__(self, ns,sigma_embed_dim=32,feature_embed_dim=103, dropout=0.0):
        self.connect_dim = 1
        self.sigma_embed_dim = sigma_embed_dim
        self.feature_embed_dim = feature_embed_dim
        
        super(EdgeEmbedding, self).__init__()
        self.connect_ebd = nn.Embedding(2, ns) # 0,1
        self.sigma_ebd = nn.Linear(self.sigma_embed_dim, ns)
        self.feature_ebd = nn.Linear(self.feature_embed_dim, ns)
        self.edge_embedding = nn.Sequential(nn.Linear(ns*3, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        # _init(self)

    def forward(self, x):
        proj_dtype = self.sigma_ebd.weight.dtype
        # connect
        connect_ebd = self.connect_ebd(x[:,self.sigma_embed_dim:self.sigma_embed_dim+self.connect_dim].long()).squeeze()
        # sigma noise embedding
        sigma_ebd = self.sigma_ebd(x[:,:self.sigma_embed_dim].to(dtype=proj_dtype))
        # 
        feature_ebd = self.feature_ebd(x[:,-self.feature_embed_dim:].to(dtype=self.feature_ebd.weight.dtype))
        # # add together
        cat_dtype = self.edge_embedding[0].weight.dtype
        x_embedding = torch.cat(
            [
                connect_ebd.to(dtype=cat_dtype),
                sigma_ebd.to(dtype=cat_dtype),
                feature_ebd.to(dtype=cat_dtype),
            ],
            1,
        )
        
        x_embedding = self.edge_embedding(x_embedding)
        return x_embedding
    
class CGTPEL(nn.Module):
    """
        Clebsch-Gordan tensor product equivariant layer
    """
    def __init__(self, 
                 in_irreps, sh_irreps, out_irreps, n_edge_features,
                 residual=True, batch_norm=True,hidden_features=None, is_last_layer=False,dropout=0.0):
        super(CGTPEL, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tensor_prod = o3.FullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, shared_weights=False
        )

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, self.tensor_prod.weight_numel),
        )
        self.batch_norm = BatchNorm(out_irreps) if (batch_norm and not is_last_layer) else None

    def forward(
        self,
        node_attr,
        edge_index,
        edge_attr,
        edge_sh,
        out_nodes=None,
        reduction="mean",
    ):
        """
        @param edge_index  [2, E]
        @param edge_sh  edge spherical harmonics
        """
        if not torch.is_floating_point(node_attr):
            node_attr = node_attr.float()
        if not torch.is_floating_point(edge_attr):
            edge_attr = edge_attr.float()
        if not torch.is_floating_point(edge_sh):
            edge_sh = edge_sh.float()
        edge_src, edge_dst = edge_index
        tp = self.tensor_prod(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduction)
        if not torch.is_floating_point(out):
            out = out.float()

        if self.residual:
            new_shape = (0, out.shape[-1] - node_attr.shape[-1])
            padded = F.pad(node_attr, new_shape)
            out = out + padded  # 保持维度匹配的残差，别把梯度饿死

        if self.batch_norm:
            out = self.batch_norm(out)
        return out
    
class CGTensorProductEquivariantModel(nn.Module):
    """
        Clebsch-Gordan tensor product-based equivariant model
    """
    def __init__(self, args,
            confidence_mode=False,
            confidence_dropout=0,
            confidence_no_batchnorm=False,
            num_confidence_outputs=1,
        ):
        super(CGTensorProductEquivariantModel, self).__init__()
        
        self.dropout = args.dropout
        self.cross_cutoff_weight = args.cross_cutoff_weight
        self.cross_cutoff_bias = args.cross_cutoff_bias
        self.cross_max_dist = args.cross_max_distance
        self.cross_max_neighbors = getattr(args, "cross_max_neighbors", 256)
        self.dynamic_max_cross=args.dynamic_max_cross
        self.center_max_dist = args.center_max_distance
        
        training_mode = str(getattr(args, "training_mode", "flow") or "flow").lower()
        if training_mode != "flow":
            raise ValueError(f"training_mode 只支持 flow，收到 {training_mode}")
        self.training_mode = "flow"
        self.flow_cfg = getattr(args, "flow", {}) or {}
        self.rot_repr = str(self.flow_cfg.get("rot_repr", "axis_angle") or "axis_angle").lower()
        self.rot_target_mode = str(self.flow_cfg.get("rot_target_mode", "noise") or "noise").lower()
        self.rot_head_mode = str(self.flow_cfg.get("rot_head_mode", "equivariant") or "equivariant").lower()
        if self.rot_head_mode not in {"equivariant", "absolute"}:
            raise ValueError(f"flow.rot_head_mode 仅支持 equivariant/absolute，收到 {self.rot_head_mode}")
        # 已废弃但保留兼容：
        # 1) rot-only 诊断未见验证收益；
        # 2) subset512/e8 复核下 multi-anchor 方向未优于 baseline。
        # 训练入口会拒绝继续启用这些分支，这里仅为兼容旧 ckpt 字段保留解析逻辑。
        self.rot_frame_input = bool(self.flow_cfg.get("rot_frame_input", False))
        self.rot_frame_input_mode = str(self.flow_cfg.get("rot_frame_input_mode", "r_local") or "r_local").lower()
        self.rot_frame_input_scale = float(self.flow_cfg.get("rot_frame_input_scale", 1.0) or 1.0)
        self.rot_frame_multi_anchor = bool(self.flow_cfg.get("rot_frame_multi_anchor", False))
        self.rot_frame_multi_anchor_k = int(self.flow_cfg.get("rot_frame_multi_anchor_k", 3) or 3)
        self.rot_frame_multi_anchor_neighbors = int(
            self.flow_cfg.get("rot_frame_multi_anchor_neighbors", 16) or 16
        )
        # 已废弃但保留兼容：锚点输入历史诊断未带来明显提升。
        self.rot_anchor_input = bool(self.flow_cfg.get("rot_anchor_input", False))
        self.rot_anchor_input_mode = str(
            self.flow_cfg.get("rot_anchor_input_mode", "rec_pca") or "rec_pca"
        ).lower()
        if self.rot_anchor_input_mode not in {"rec_pca", "iface_ncac", "iface_ncac_sc", "seq_ncac_cb", "seq_ncac_cb2", "seq_weighted"}:
            raise ValueError(
                f"flow.rot_anchor_input_mode 仅支持 rec_pca/iface_ncac/iface_ncac_sc/seq_ncac_cb/seq_ncac_cb2/seq_weighted，收到 {self.rot_anchor_input_mode}"
            )
        self.rot_anchor_input_scale = float(self.flow_cfg.get("rot_anchor_input_scale", 1.0) or 1.0)
        # 已废弃但保留兼容：参考点输入重跑后仍未提升验证指标。
        self.rot_ref_input = bool(self.flow_cfg.get("rot_ref_input", False))
        self.ns, self.nv = args.ns, args.nv
        ns, nv = self.ns, self.nv
        self.num_conv_layers = args.num_conv_layers
        self.lig_max_radius = args.max_radius
        self.batch_norm = not args.no_batch_norm
        self.confidence_mode = confidence_mode
        self.scale_by_sigma = False
        self.rec_amino_dim = args.rec_amino_dim
        self.pep_amino_dim = args.pep_amino_dim 
        self.sigma_embed_dim = args.sigma_embed_dim
        self.intra_dim = args.intra_dim
        self.dihediral_dim = args.dihediral_dim
        self.edge_feature_dim = args.edge_feature_dim
        self.embedding_type = args.embedding_type
        self.embedding_scale = args.embedding_scale
        self.cross_dist_embed_dim = args.cross_distance_embed_dim
        self.use_second_order_repr = args.use_second_order_repr
        self.dist_embed_dim = args.distance_embed_dim
        self.esm_embeddings_receptor = args.esm_embeddings_path_train is not None
        self.esm_embeddings_peptide = args.esm_embeddings_peptide_train is not None
        # 保留实验项：组合实验可运行，但同预算明显不如 baseline，尚未形成独立正收益。
        self.rot_tor_refiner = bool(self.flow_cfg.get("rot_tor_refiner", False))
        self.rot_tor_refiner_rot_scale = float(
            self.flow_cfg.get("rot_tor_refiner_rot_scale", 1.0) or 1.0
        )
        self.rot_tor_refiner_tor_scale = float(
            self.flow_cfg.get("rot_tor_refiner_tor_scale", 1.0) or 1.0
        )
        # 保留实验项：与 rot_tor_refiner 一起测试时可运行，但未证明有净收益。
        self.tor_pairdist_inject = bool(self.flow_cfg.get("tor_pairdist_inject", False))
        # 保留实验项：seed1/seed2 对照方向相反，收益不稳定，默认关闭。
        self.inter_edge_update = bool(self.flow_cfg.get("inter_edge_update", False))
        self.inter_edge_update_scale = float(
            self.flow_cfg.get("inter_edge_update_scale", 1.0) or 1.0
        )
        # 保留实验项：light-enhancement 组合未证明稳定正收益，默认关闭。
        self.self_condition = bool(self.flow_cfg.get("self_condition", False))
        self.self_condition_scale = float(
            self.flow_cfg.get("self_condition_scale", 1.0) or 1.0
        )
        # 保留实验项：仅在 light-enhancement 组合中验证过，默认不进入主线。
        self.sparse_interface = bool(self.flow_cfg.get("sparse_interface", False))
        self.sparse_interface_topk = int(
            self.flow_cfg.get("sparse_interface_topk", 12) or 12
        )
        self.noise_schedule = None
        self.timestep_emb_func = get_timestep_embedding(self.embedding_type,self.sigma_embed_dim,self.embedding_scale)
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=2)
        # 受体/肽图用两套嵌入，必要时还能接ESM语义，别问为啥这么花哨
        self.rec_node_embedding = AminoEmbedding(
            self.rec_amino_dim,
            ns,
            self.sigma_embed_dim,
            self.intra_dim,
            self.dihediral_dim,
            'esm' if self.esm_embeddings_receptor else None,
        )
        self.rec_edge_embedding = EdgeEmbedding(ns,self.sigma_embed_dim, self.edge_feature_dim, self.dropout)
        self.pep_node_embedding = AminoEmbedding(self.pep_amino_dim,ns,self.sigma_embed_dim,self.intra_dim, 3, 'esm') if self.esm_embeddings_peptide else AminoEmbedding(self.pep_amino_dim,ns,self.sigma_embed_dim,self.intra_dim, 3)
        self.pep_edge_embedding = EdgeEmbedding(ns,self.sigma_embed_dim, self.edge_feature_dim, self.dropout)
        self.top_k = args.top_k
        self.cross_edge_embedding = nn.Sequential(
            nn.Linear(self.sigma_embed_dim + self.cross_dist_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(ns, ns),
        )
        # 4 种 cross edge（ca/ tips 组合）类型嵌入；不改变 cross_edge_embedding 的输入维度：把它加到 length_emb 上
        self.cross_edge_type = nn.Embedding(4, self.cross_dist_embed_dim)
        # 蛋白-肽之间的交互距离走高斯展开，方便后面的CG层吃
        self.cross_distance_expansion = GaussianSmearing(
                0.0, self.cross_max_dist, self.cross_dist_embed_dim)
        if self.inter_edge_update:
            self.inter_edge_geom_embed = nn.Sequential(
                nn.Linear(4, ns),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(ns, ns),
            )

            self.inter_edge_update_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(4 * ns, ns),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(ns, ns),
                    )
                    for _ in range(self.num_conv_layers)
                ]
            )
            self.inter_edge_update_gates = nn.ModuleList(
                [nn.Linear(4 * ns, ns) for _ in range(self.num_conv_layers)]
            )
        
        if self.use_second_order_repr:
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o + {nv}x2e",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o",
            ]
        else:
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o",
                f"{ns}x0e + {nv}x1o + {nv}x1e",
                f"{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o",
            ]
        
        intra_convs = []
        cross_convs = []
        for i in range(self.num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            params = {
                "in_irreps": in_irreps,
                "sh_irreps": self.sh_irreps,
                "out_irreps": out_irreps,
                "n_edge_features": 3 * ns,
                "hidden_features": 3 * ns,
                "residual": False,
                'batch_norm': self.batch_norm,
                'dropout': self.dropout
            }
            intra_convs.append(CGTPEL(**params))
            cross_convs.append(CGTPEL(**params))
            
        self.intra_convs = nn.ModuleList(intra_convs)
        self.cross_convs = nn.ModuleList(cross_convs)
        
        # compute confidence score
        if self.confidence_mode:
            self.confidence_predictor = nn.Sequential(
                nn.Linear(2 * self.ns if self.num_conv_layers >= 3 else self.ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, num_confidence_outputs),
            )
        else:
            # center of mass translation and rotation components
            # 不是置信模式时才需要预测平移/旋转/扭转向量
            self.center_distance_expansion = GaussianSmearing(
                0.0, self.center_max_dist, self.dist_embed_dim
            )
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(self.dist_embed_dim + self.sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(ns, ns)
            )
            
            self.final_conv = CGTPEL(
                in_irreps=self.intra_convs[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps='2x1o + 2x1e + 1x0o',
                n_edge_features=2 * ns,
                residual=False,
                dropout=self.dropout,
                batch_norm= self.batch_norm,
                is_last_layer=True,
            )
            # 方案A：为肽残基级 CA 回归准备的向量 head（不走 rot 向量监督）
            self.pep_final_irreps = o3.Irreps(self.intra_convs[-1].out_irreps)
            self.pep_ca_head = o3.Linear(self.pep_final_irreps, "1x1o")
            self.rot_cls_bins = int(self.flow_cfg.get("rot_cls_bins", 0) or 0)
            self.rot_cls_head = None
            if self.rot_cls_bins > 0:
                self.rot_cls_head = nn.Sequential(
                    nn.Linear(self.ns, self.ns),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.ns, self.rot_cls_bins),
                )
            # 保留实验项：训练期辅助监督本身可学习，但未稳定转化为 infer_mix 主指标提升。
            self.interface_contact_head = bool(self.flow_cfg.get("interface_contact_head", False))
            self.interface_pairdist_head = bool(self.flow_cfg.get("interface_pairdist_head", False))
            self.interface_head_dim = int(self.flow_cfg.get("interface_head_dim", self.ns) or self.ns)
            self.interface_contact_predictor = None
            self.interface_pairdist_predictor = None
            if self.interface_contact_head:
                self.interface_contact_predictor = nn.Sequential(
                    nn.Linear(3 * self.ns, self.interface_head_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.interface_head_dim, 1),
                )
            if self.interface_pairdist_head:
                self.interface_pairdist_predictor = nn.Sequential(
                    nn.Linear(3 * self.ns, self.interface_head_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.interface_head_dim, 1),
                )
            if self.rot_head_mode == "absolute":
                # 非等变旋转头：强行输出绝对旋转（axis-angle），用于打破对称性
                # 输入：pep graph 标量 + rec/pep COM + rec 局部轴均值 + pep 主轴
                abs_in_dim = self.ns + 18
                self.rot_abs_head = nn.Sequential(
                    nn.Linear(abs_in_dim, self.ns),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.ns, 3),
                )
            if self.rot_frame_input:
                if self.rot_frame_input_mode not in {"r_local"}:
                    raise ValueError(
                        f"flow.rot_frame_input_mode 仅支持 r_local，收到 {self.rot_frame_input_mode}"
                    )
                self.rot_frame_mlp = nn.Sequential(
                    nn.Linear(9, self.ns),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.ns, 3),
                )
                if self.rot_frame_multi_anchor:
                    self.rot_frame_score_mlp = nn.Sequential(
                        nn.Linear(9, self.ns),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.ns, 1),
                    )
            if self.rot_anchor_input:
                self.anchor_frame_mlp = nn.Sequential(
                    nn.Linear(9, self.ns),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.ns, self.dist_embed_dim),
                )
            
            # Build Pep_atom_ebd
            # 扭转载荷依赖肽原子表示，重新编码一遍以对齐CG输出维度
            self.pep_a_node_embedding = AtomEncoder(o3.Irreps(self.intra_convs[-1].out_irreps).dim,self.sigma_embed_dim, o3.Irreps(self.intra_convs[-1].out_irreps).dim)
            
            self.lig_distance_expansion = GaussianSmearing(
                0.0, self.lig_max_radius, self.dist_embed_dim)
            
            # torsion angles components
            self.final_edge_embedding = nn.Sequential(
                    nn.Linear(self.dist_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(ns, ns),
                )
            self.final_tp_tor_bb = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.final_tp_tor_sc = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.tor_bb_bond_conv = CGTPEL(
                    in_irreps=self.intra_convs[-1].out_irreps,
                    sh_irreps=self.final_tp_tor_bb.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                )
            self.tor_sc_bond_conv = CGTPEL(
                    in_irreps=self.intra_convs[-1].out_irreps,
                    sh_irreps=self.final_tp_tor_sc.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                )
            self.tor_bb_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(self.dropout),
                    nn.Linear(ns, 1, bias=False)
                )
            if self.rot_tor_refiner:
                self.refiner_ctx = nn.Sequential(
                    nn.Linear(2 * self.ns + 1, self.ns),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.ns, self.ns),
                    nn.ReLU(),
                )
                self.refiner_gate = nn.Linear(self.ns, 3)
                self.refiner_rot = nn.Linear(self.ns, 3)
                self.refiner_tor_bb = nn.Linear(self.ns, 1)
                self.refiner_tor_sc = nn.Linear(self.ns, 1)
            if self.tor_pairdist_inject:
                self.iface_pair_to_tor = nn.Sequential(
                    nn.Linear(1, 3 * self.ns),
                    nn.Tanh(),
                )
            self.tor_sc_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(self.dropout),
                    nn.Linear(ns, 1, bias=False)
                )
            if self.self_condition:
                # 仅在 tr/rot 上做轻量残差，不改主干消息传递路径
                self.self_cond_mlp = nn.Sequential(
                    nn.Linear(self.ns + 6, self.ns),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.ns, 6),
                )
                self.self_cond_gate = nn.Sequential(
                    nn.Linear(self.ns + 6, self.ns),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.ns, 2),
                )
            
    def forward(self, _data):
        data = copy.copy(_data)
        collect_forward_timing = bool(getattr(self, "collect_timing_stats", False))
        forward_timings = {
            "receptor_graph_seconds": 0.0,
            "peptide_feature_seconds": 0.0,
            "peptide_graph_seconds": 0.0,
            "cross_graph_seconds": 0.0,
            "message_passing_seconds": 0.0,
            "global_head_seconds": 0.0,
            "torsion_head_seconds": 0.0,
        }
        # get noise schedule
        tr_t = data.complex_t["tr"]
        rot_t = data.complex_t["rot"]
        tor_backbone_t = data.complex_t["tor_backbone"]
        tor_sidechain_t = data.complex_t["tor_sidechain"]
        if not self.confidence_mode:
            # flow 数据增强用 flow_cfg 里的 sigma_max（线性随 t 放大）。
            if isinstance(self.flow_cfg, dict) and self.flow_cfg.get("sigma_tr_max") is not None:
                tr_sigma = tr_t * float(self.flow_cfg["sigma_tr_max"])
                rot_sigma = rot_t * float(self.flow_cfg.get("sigma_rot_max", 0.0) or 0.0)
                tor_backbone_sigma = tor_backbone_t * float(self.flow_cfg.get("sigma_tor_bb_max", 0.0) or 0.0)
                tor_sidechain_sigma = tor_sidechain_t * float(self.flow_cfg.get("sigma_tor_sc_max", 0.0) or 0.0)
            else:
                raise ValueError("[flow] 缺少 sigma_*_max 配置，请检查 flow 配置。")
        else:
            tr_sigma, rot_sigma, tor_backbone_sigma,tor_sidechain_sigma = tr_t, rot_t, tor_backbone_t,tor_sidechain_t
            
        self.device = data['pep'].x.device

        # build receptor graph
        t0 = time.perf_counter()
        receptor_graph = self.build_rec_conv_graph(data)
        rec_node_attr = self.rec_node_embedding(receptor_graph[0])
        rec_src, rec_dst = rec_edge_index = receptor_graph[1]
        rec_edge_attr = self.rec_edge_embedding(receptor_graph[2])
        rec_edge_sh = receptor_graph[3]
        if collect_forward_timing:
            forward_timings["receptor_graph_seconds"] += time.perf_counter() - t0
        
        # build pep graph：动态补齐肽残基特征和边信息
        t0 = time.perf_counter()
        node_s_pep, ca_pep, tips_pep, edge_index_pep, edge_s_pep, edge_v_pep = get_updated_peptide_feature(data, self.device, self.top_k)
        # 训练/推理都要求把“序列特征尾巴”拼回去：
        # - ESM: 1280 维
        # - onehot: pep_amino_dim 维（默认 104）
        pep_tail_dim = 1280 if self.esm_embeddings_peptide else int(self.pep_amino_dim)
        if data['pep'].x.shape[1] >= 1 + pep_tail_dim:
            pep_tail = data['pep'].x[:, -pep_tail_dim:]
        else:
            pep_tail = torch.empty((data['pep'].x.shape[0], 0), device=data['pep'].x.device, dtype=data['pep'].x.dtype)
        # [num_res, 1 + node_s(8) + tail]
        data['pep'].x = torch.cat([data['pep'].x[:, :1], node_s_pep, pep_tail], dim=1)
        data['pep'].pos = ca_pep.to(dtype=torch.float)
        data['pep'].tips = tips_pep.to(dtype=torch.float)
        data['pep', 'pep_contact', 'pep'].edge_index = edge_index_pep
        data['pep', 'pep_contact', 'pep'].edge_s = edge_s_pep.to(dtype=torch.float)
        data['pep', 'pep_contact', 'pep'].edge_v = edge_v_pep.to(dtype=torch.float)
        if collect_forward_timing:
            forward_timings["peptide_feature_seconds"] += time.perf_counter() - t0
        
        t0 = time.perf_counter()
        pep_graph = self.build_pep_conv_graph(data)
        pep_node_attr = self.pep_node_embedding(pep_graph[0])
        pep_src, pep_dst = pep_edge_index = pep_graph[1]
        pep_edge_attr = self.pep_edge_embedding(pep_graph[2])
        pep_edge_sh = pep_graph[3]
        if collect_forward_timing:
            forward_timings["peptide_graph_seconds"] += time.perf_counter() - t0
        
        # build cross graph
        t0 = time.perf_counter()
        if self.dynamic_max_cross:
            cross_cutoff = (tr_sigma * self.cross_cutoff_weight + self.cross_cutoff_bias).unsqueeze(1)
        else:
            cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(
            data, cross_cutoff
        )
        cross_pep, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)
        cross_edge_vec = (
            data["receptor"].pos[cross_rec.long()] - data["pep"].pos[cross_pep.long()]
            if cross_edge_index.numel() > 0
            else None
        )
        if collect_forward_timing:
            forward_timings["cross_graph_seconds"] += time.perf_counter() - t0
        
        t0 = time.perf_counter()
        for idx in range(len(self.intra_convs)):
            if self.inter_edge_update and cross_edge_index.numel() > 0:
                cross_edge_attr = self._apply_inter_edge_update(
                    cross_edge_attr,
                    pep_node_attr[cross_pep, : self.ns],
                    rec_node_attr[cross_rec, : self.ns],
                    cross_edge_vec,
                    idx,
                )
            # message passing within pep graph (intra)
            pep_edge_attr_ = torch.cat([pep_edge_attr, pep_node_attr[pep_src, :self.ns], pep_node_attr[pep_dst, :self.ns]], -1)
            pep_intra_update = self.intra_convs[idx](pep_node_attr, pep_edge_index, pep_edge_attr_, pep_edge_sh)
            
            # message passing between two graphs (inter)
            rec_to_pep_edge_attr_ = torch.cat([cross_edge_attr, pep_node_attr[cross_pep, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            cross_reduction = "sum"
            pep_inter_update = self.cross_convs[idx](
                rec_node_attr,
                cross_edge_index,
                rec_to_pep_edge_attr_,
                cross_edge_sh,
                out_nodes=pep_node_attr.shape[0],
                reduction=cross_reduction,
            )
            
            # message passing within receptor graph (intra)
            if idx != len(self.intra_convs) - 1:
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
                rec_intra_update = self.intra_convs[idx](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)
                
                pep_to_rec_edge_attr_ = torch.cat([cross_edge_attr, pep_node_attr[cross_pep, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
                rec_inter_update = self.cross_convs[idx](
                    pep_node_attr,
                    torch.flip(cross_edge_index, dims=[0]),
                    pep_to_rec_edge_attr_,
                    cross_edge_sh,
                    out_nodes=rec_node_attr.shape[0],
                    reduction=cross_reduction,
                )
            
            # padding original features
            pep_node_attr = F.pad(
                pep_node_attr,
                (0, pep_intra_update.shape[-1] - pep_node_attr.shape[-1]))
            # update features with residual updates
            pep_node_attr = pep_node_attr + pep_intra_update + pep_inter_update
            
            if idx != len(self.intra_convs) - 1:
                rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update
        if collect_forward_timing:
            forward_timings["message_passing_seconds"] += time.perf_counter() - t0
                
        # 训练期接口辅助头：使用 cross 边特征预测接触概率/距离，推理可完全忽略
        t0 = time.perf_counter()
        iface_contact_logits = None
        iface_pairdist_pred = None
        iface_edge_index = None
        pair_graph_feat = pep_node_attr.new_zeros((data.num_graphs, 1))
        if (not self.confidence_mode) and (self.interface_contact_predictor is not None or self.interface_pairdist_predictor is not None):
            iface_edge_index = cross_edge_index
            if cross_edge_index.numel() > 0:
                iface_edge_feat = torch.cat(
                    [
                        pep_node_attr[cross_pep, : self.ns],
                        rec_node_attr[cross_rec, : self.ns],
                        cross_edge_attr,
                    ],
                    dim=-1,
                )
                if self.interface_contact_predictor is not None:
                    iface_contact_logits = self.interface_contact_predictor(iface_edge_feat).squeeze(-1)
                if self.interface_pairdist_predictor is not None:
                    # 距离监督输出限定为非负，数值更稳
                    iface_pairdist_pred = F.softplus(
                        self.interface_pairdist_predictor(iface_edge_feat).squeeze(-1)
                    ) + 1e-6
                    # 接口距离信号压成每图标量，供 rot/tor 精修使用（不反传回 pairdist head）
                    cross_graph = data["pep"].batch[cross_pep]
                    pair_graph_feat = scatter_mean(
                        torch.exp(-iface_pairdist_pred.detach()).unsqueeze(-1),
                        cross_graph,
                        dim=0,
                        dim_size=data.num_graphs,
                    )
            else:
                if self.interface_contact_predictor is not None:
                    iface_contact_logits = pep_node_attr.new_zeros((0,))
                if self.interface_pairdist_predictor is not None:
                    iface_pairdist_pred = pep_node_attr.new_zeros((0,))
        if self.tor_pairdist_inject:
            pair_graph_embed = self.iface_pair_to_tor(pair_graph_feat)
        else:
            pair_graph_embed = pep_node_attr.new_zeros((data.num_graphs, 3 * self.ns))

        # compute confidence score
        if self.confidence_mode:
            scalar_pep_attr = (
                torch.cat(
                    [pep_node_attr[:, : self.ns], pep_node_attr[:, -self.ns :]], dim=1
                )
                if self.num_conv_layers >= 3
                else pep_node_attr[:, : self.ns]
            )
            confidence = self.confidence_predictor(scatter_mean(scalar_pep_attr, data['pep'].batch, dim=0)).squeeze(dim=-1)
            return confidence
        
        # compute translational and rotational vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, pep_node_attr[center_edge_index[1], :self.ns]], -1)
        global_pred = self.final_conv(pep_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)
        # 方案A：肽残基级 CA 向量预测（用于坐标回归 + Kabsch 刚体还原）
        ca_pred = self.pep_ca_head(pep_node_attr)
        ca_pred = torch.nan_to_num(ca_pred)
        rot_cls_logits = None
        if self.rot_cls_head is not None:
            graph_scalar = scatter_mean(pep_node_attr[:, : self.ns], data["pep"].batch, dim=0)
            rot_cls_logits = self.rot_cls_head(graph_scalar)
        
        # e3nn irreps: final_conv 输出为 '2x1o + 2x1e + 1x0o'（13维）。
        # 物理含义：
        # - 平移是极矢量（polar vector）=> 1o
        # - 旋转轴角是赝矢量（axial vector）=> 1e
        tr_pred = global_pred[:, 0:3] + global_pred[:, 3:6]   # 2x1o
        if self.rot_head_mode == "absolute":
            graph_scalar = scatter_mean(pep_node_attr[:, : self.ns], data["pep"].batch, dim=0)
            rec_pos = data["receptor"].pos
            rec_batch = getattr(data["receptor"], "batch", None)
            if rec_batch is None:
                rec_batch = torch.zeros(rec_pos.shape[0], dtype=torch.long, device=rec_pos.device)
            pep_pos = data["pep"].pos
            pep_batch = getattr(data["pep"], "batch", None)
            if pep_batch is None:
                pep_batch = torch.zeros(pep_pos.shape[0], dtype=torch.long, device=pep_pos.device)
            rec_com = scatter_mean(rec_pos, rec_batch, dim=0)
            pep_com = scatter_mean(pep_pos, pep_batch, dim=0)
            rec_axes = getattr(data["receptor"], "node_v", None)
            if rec_axes is None:
                rec_axes_feat = torch.zeros((rec_com.shape[0], 9), device=rec_com.device)
            else:
                axis_x = scatter_mean(rec_axes[:, :, 0], rec_batch, dim=0)
                axis_y = scatter_mean(rec_axes[:, :, 1], rec_batch, dim=0)
                axis_z = scatter_mean(rec_axes[:, :, 2], rec_batch, dim=0)
                rec_axes_feat = torch.cat([axis_x, axis_y, axis_z], dim=1)
            pep_axis = torch.zeros((pep_com.shape[0], 3), device=pep_com.device)
            num_graphs = pep_com.shape[0]
            for g in range(num_graphs):
                pos_g = pep_pos[pep_batch == g]
                if pos_g.shape[0] >= 2:
                    axis = pos_g[-1] - pos_g[0]
                    if axis.norm().item() < 1e-6 and pos_g.shape[0] >= 3:
                        axis = pos_g[-2] - pos_g[0]
                    pep_axis[g] = axis
            pep_axis = pep_axis / pep_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            abs_feat = torch.cat([graph_scalar, rec_com, pep_com, rec_axes_feat, pep_axis], dim=1)
            rot_pred = self.rot_abs_head(abs_feat)
        else:
            rot_vec_a = global_pred[:, 6:9]
            rot_vec_b = global_pred[:, 9:12]
            rot_pseudoscalar = global_pred[:, 12:13]              # 1x0o
            if self.rot_repr in {"rot6d", "rot6d_x1"}:
                rot_6d = torch.cat([rot_vec_a, rot_vec_b], dim=1)
                if self.rot_repr == "rot6d_x1":
                    rot_pred = rot_6d
                else:
                    rot_mat_pred = rot6d_to_matrix(rot_6d)
                    rot_pred = matrix_to_axis_angle(rot_mat_pred)
            else:
                rot_pred = rot_vec_a + rot_vec_b  # 2x1e
            if self.rot_frame_input and rot_pred.shape[-1] == 3:
                if self.rot_frame_multi_anchor:
                    rec_frames, frame_local, frame_valid = self._compute_multi_anchor_frame_feat(data)
                    rot_local = torch.matmul(
                        rec_frames.transpose(-1, -2),
                        rot_pred[:, None, :].unsqueeze(-1),
                    ).squeeze(-1)
                    rot_local_delta = self.rot_frame_mlp(frame_local) * self.rot_frame_input_scale
                    rot_local = rot_local + rot_local_delta * frame_valid
                    rot_world = torch.matmul(rec_frames, rot_local.unsqueeze(-1)).squeeze(-1)
                    frame_scores = self.rot_frame_score_mlp(frame_local).squeeze(-1)
                    frame_mask = frame_valid.squeeze(-1) > 0.5
                    frame_scores = frame_scores.masked_fill(~frame_mask, -1e4)
                    frame_weights = torch.softmax(frame_scores, dim=1)
                    frame_weights = frame_weights * frame_mask.float()
                    frame_weights = frame_weights / frame_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    rot_pred = (rot_world * frame_weights.unsqueeze(-1)).sum(dim=1)
                else:
                    rec_frames, frame_local, frame_valid = self._compute_frame_local_feat(data)
                    rot_local = torch.matmul(rec_frames.transpose(1, 2), rot_pred.unsqueeze(-1)).squeeze(-1)
                    rot_local_delta = self.rot_frame_mlp(frame_local) * self.rot_frame_input_scale
                    rot_local = rot_local + rot_local_delta * frame_valid
                    rot_pred = torch.matmul(rec_frames, rot_local.unsqueeze(-1)).squeeze(-1)
        if (
            self.self_condition
            and tr_pred.shape[-1] == 3
            and rot_pred.shape[-1] == 3
            and hasattr(self, "self_cond_mlp")
        ):
            graph_scalar = scatter_mean(
                pep_node_attr[:, : self.ns],
                data["pep"].batch,
                dim=0,
                dim_size=data.num_graphs,
            )
            sc_tr, sc_rot = self._resolve_self_condition(
                data,
                data.num_graphs,
                graph_scalar.device,
                graph_scalar.dtype,
            )
            sc_feat = torch.cat([graph_scalar, sc_tr, sc_rot], dim=-1)
            sc_delta = self.self_cond_mlp(sc_feat) * self.self_condition_scale
            sc_gate = torch.sigmoid(self.self_cond_gate(sc_feat))
            tr_pred = tr_pred + sc_gate[:, 0:1] * sc_delta[:, 0:3]
            rot_pred = rot_pred + sc_gate[:, 1:2] * sc_delta[:, 3:6]
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])

        tr_pred = torch.nan_to_num(tr_pred)
        rot_pred = torch.nan_to_num(rot_pred)
        refiner_ctx = None
        refiner_gate = None
        if self.rot_tor_refiner:
            rec_batch = getattr(data["receptor"], "batch", None)
            if rec_batch is None:
                rec_batch = torch.zeros(
                    data["receptor"].pos.shape[0], dtype=torch.long, device=data["receptor"].pos.device
                )
            pep_graph = scatter_mean(
                pep_node_attr[:, : self.ns], data["pep"].batch, dim=0, dim_size=data.num_graphs
            )
            rec_graph = scatter_mean(
                rec_node_attr[:, : self.ns], rec_batch, dim=0, dim_size=data.num_graphs
            )
            refiner_ctx = self.refiner_ctx(torch.cat([pep_graph, rec_graph, pair_graph_feat], dim=-1))
            refiner_gate = torch.sigmoid(self.refiner_gate(refiner_ctx))
            rot_pred = rot_pred + refiner_gate[:, 0:1] * (
                self.refiner_rot(refiner_ctx) * self.rot_tor_refiner_rot_scale
            )
            rot_pred = torch.nan_to_num(rot_pred)
        if collect_forward_timing:
            forward_timings["global_head_seconds"] += time.perf_counter() - t0

        # torsional components
        t0 = time.perf_counter()
        # torsion 分支的梯度很容易“反向拖垮”全局 pose 表示（尤其 rot），因此在 flow 训练下把
        # pep_node_attr 当作只读上下文（detach），避免 torsion loss 直接改写 residue-level 表示。
        pep_node_attr_for_atom = pep_node_attr.detach()
        pep_global_res_index = (
            data["pep_a"].atom2res_index + data["pep"].ptr[data["pep_a"].batch]
        )
        pep_a_node_attr_from_pep = pep_node_attr_for_atom[pep_global_res_index]
        data['pep_a'].node_sigma_emb = self.timestep_emb_func(data['pep_a'].node_t['tr']) # tr rot and tor noise is all the same
        pep_a_node_attr = torch.cat(
                [data['pep_a'].atom2resid_index.unsqueeze(-1), data['pep_a'].atom2atomid_index.unsqueeze(-1), data['pep_a'].x, data['pep_a'].node_sigma_emb,pep_a_node_attr_from_pep], 1
            ) # 1+1+19+32+??84
        pep_a_node_attr = self.pep_a_node_embedding(pep_a_node_attr)
        
        
        if not data['pep_a'].mask_edges_backbone.squeeze().sum() == 0:
            tor_bonds_backbone, tor_edge_index_backbone, tor_edge_attr_backbone, tor_edge_sh_backbone = self.build_backbone_bond_conv_graph(data)
            tor_bond_vec_backbone = data['pep_a'].pos[tor_bonds_backbone[1]] - data['pep_a'].pos[tor_bonds_backbone[0]]
            tor_bond_attr_backbone = pep_a_node_attr[tor_bonds_backbone[0]] + pep_a_node_attr[tor_bonds_backbone[1]]
            tor_bonds_sh_backbone = o3.spherical_harmonics(
                "2e", tor_bond_vec_backbone, normalize=True, normalization="component"
            )
            tor_edge_sh_backbone = self.final_tp_tor_bb(tor_edge_sh_backbone, tor_bonds_sh_backbone[tor_edge_index_backbone[0]])
            tor_edge_attr_backbone = torch.cat(
                [
                    tor_edge_attr_backbone,
                    pep_a_node_attr[tor_edge_index_backbone[1], : self.ns],
                    tor_bond_attr_backbone[tor_edge_index_backbone[0], : self.ns],
                ],
                -1,
            )
            if self.tor_pairdist_inject:
                edge_graph_backbone = data["pep_a"].batch[tor_edge_index_backbone[1]]
                tor_edge_attr_backbone = tor_edge_attr_backbone + pair_graph_embed[edge_graph_backbone]
            tor_pred_backbone = self.tor_bb_bond_conv(
                pep_a_node_attr,
                tor_edge_index_backbone,
                tor_edge_attr_backbone,
                tor_edge_sh_backbone,
                out_nodes=data['pep_a'].mask_edges_backbone.sum(),
                reduction="mean",
            )
            tor_pred_backbone = self.tor_bb_final_layer(tor_pred_backbone).squeeze(1)
            if self.rot_tor_refiner and refiner_ctx is not None and tor_pred_backbone.numel() > 0:
                graph_idx_backbone = data["pep_a"].batch[tor_bonds_backbone[0]]
                tor_pred_backbone = tor_pred_backbone + refiner_gate[graph_idx_backbone, 1] * (
                    self.refiner_tor_bb(refiner_ctx[graph_idx_backbone]).squeeze(-1)
                    * self.rot_tor_refiner_tor_scale
                )
            edge_sigma_backbone = tor_backbone_sigma[data["pep_a"].batch][
                data["pep_a", "pep_a"].edge_index[0]
            ][data['pep_a'].mask_edges_backbone.squeeze()]
        else: tor_pred_backbone = torch.empty(0, device=self.device)
        
        if not data['pep_a'].mask_edges_sidechain.squeeze().sum() == 0:
            tor_bonds_sidechain, tor_edge_index_sidechain, tor_edge_attr_sidechain, tor_edge_sh_sidechain = self.build_sidechain_bond_conv_graph(data)
            tor_bond_vec_sidechain = data['pep_a'].pos[tor_bonds_sidechain[1]] - data['pep_a'].pos[tor_bonds_sidechain[0]]
            tor_bond_attr_sidechain = pep_a_node_attr[tor_bonds_sidechain[0]] + pep_a_node_attr[tor_bonds_sidechain[1]]
            tor_bonds_sh_sidechain = o3.spherical_harmonics(
                "2e", tor_bond_vec_sidechain, normalize=True, normalization="component"
            )
            tor_edge_sh_sidechain = self.final_tp_tor_sc(tor_edge_sh_sidechain, tor_bonds_sh_sidechain[tor_edge_index_sidechain[0]])
            tor_edge_attr_sidechain = torch.cat(
                [
                    tor_edge_attr_sidechain,
                    pep_a_node_attr[tor_edge_index_sidechain[1], : self.ns],
                    tor_bond_attr_sidechain[tor_edge_index_sidechain[0], : self.ns],
                ],
                -1,
            )
            if self.tor_pairdist_inject:
                edge_graph_sidechain = data["pep_a"].batch[tor_edge_index_sidechain[1]]
                tor_edge_attr_sidechain = tor_edge_attr_sidechain + pair_graph_embed[edge_graph_sidechain]
            tor_pred_sidechain = self.tor_sc_bond_conv(
                pep_a_node_attr,
                tor_edge_index_sidechain,
                tor_edge_attr_sidechain,
                tor_edge_sh_sidechain,
                out_nodes=data['pep_a'].mask_edges_sidechain.sum(),
                reduction="mean",
            )
            tor_pred_sidechain = self.tor_sc_final_layer(tor_pred_sidechain).squeeze(1)
            if self.rot_tor_refiner and refiner_ctx is not None and tor_pred_sidechain.numel() > 0:
                graph_idx_sidechain = data["pep_a"].batch[tor_bonds_sidechain[0]]
                tor_pred_sidechain = tor_pred_sidechain + refiner_gate[graph_idx_sidechain, 2] * (
                    self.refiner_tor_sc(refiner_ctx[graph_idx_sidechain]).squeeze(-1)
                    * self.rot_tor_refiner_tor_scale
                )
            edge_sigma_sidechain = tor_sidechain_sigma[data["pep_a"].batch][
                data["pep_a", "pep_a"].edge_index[0]
            ][data['pep_a'].mask_edges_sidechain.squeeze()]
        else: tor_pred_sidechain = torch.empty(0, device=self.device)
        if collect_forward_timing:
            forward_timings["torsion_head_seconds"] += time.perf_counter() - t0
            object.__setattr__(self, "_last_forward_timing", forward_timings)
        if self.rot_cls_head is not None:
            return (
                tr_pred,
                rot_pred,
                tor_pred_backbone,
                tor_pred_sidechain,
                rot_cls_logits,
                ca_pred,
                iface_contact_logits,
                iface_pairdist_pred,
                iface_edge_index,
            )
        return (
            tr_pred,
            rot_pred,
            tor_pred_backbone,
            tor_pred_sidechain,
            ca_pred,
            iface_contact_logits,
            iface_pairdist_pred,
            iface_edge_index,
        )

    def _apply_inter_edge_update(
        self,
        cross_edge_attr: torch.Tensor,
        pep_edge_scalar: torch.Tensor,
        rec_edge_scalar: torch.Tensor,
        edge_vec: Optional[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        if (
            (not self.inter_edge_update)
            or edge_vec is None
            or edge_vec.numel() == 0
            or cross_edge_attr.numel() == 0
        ):
            return cross_edge_attr
        edge_dist = edge_vec.norm(dim=-1, keepdim=True)
        edge_dir = edge_vec / edge_dist.clamp_min(1e-6)
        dist_scaled = edge_dist / max(float(self.cross_max_dist), 1.0)
        geom_raw = torch.cat([dist_scaled, edge_dir], dim=-1)
        geom_feat = self.inter_edge_geom_embed(torch.nan_to_num(geom_raw))
        edge_ctx = torch.cat(
            [cross_edge_attr, pep_edge_scalar, rec_edge_scalar, geom_feat], dim=-1
        )
        delta = self.inter_edge_update_mlps[layer_idx](edge_ctx)
        gate = torch.sigmoid(self.inter_edge_update_gates[layer_idx](edge_ctx))
        return torch.nan_to_num(
            cross_edge_attr + self.inter_edge_update_scale * gate * delta
        )

    def _resolve_self_condition(
        self,
        data,
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        zeros = torch.zeros((num_graphs, 3), device=device, dtype=dtype)
        sc_tr = getattr(data, "self_cond_tr", None)
        sc_rot = getattr(data, "self_cond_rot", None)
        if not torch.is_tensor(sc_tr) or sc_tr.ndim != 2 or sc_tr.shape[0] != num_graphs or sc_tr.shape[1] != 3:
            sc_tr = zeros
        else:
            sc_tr = sc_tr.to(device=device, dtype=dtype)
        if not torch.is_tensor(sc_rot) or sc_rot.ndim != 2 or sc_rot.shape[0] != num_graphs or sc_rot.shape[1] != 3:
            sc_rot = zeros
        else:
            sc_rot = sc_rot.to(device=device, dtype=dtype)
        return sc_tr, sc_rot

    def _sparsify_cross_edges(
        self,
        edge_index: torch.Tensor,
        edge_vec_ca_ca: torch.Tensor,
        edge_min_dist: torch.Tensor,
        edge_type_idx: torch.Tensor,
    ):
        if (
            (not self.sparse_interface)
            or self.sparse_interface_topk <= 0
            or edge_index.numel() == 0
            or edge_min_dist.numel() == 0
        ):
            return edge_index, edge_vec_ca_ca, edge_min_dist, edge_type_idx
        src = edge_index[0]
        # 先按距离升序，再按 src 分组（稳定排序），保证每组前 top-k 就是最近边
        order_dist = torch.argsort(edge_min_dist, descending=False, stable=True)
        src_after_dist = src[order_dist]
        order_src = torch.argsort(src_after_dist, stable=True)
        order = order_dist[order_src]
        src_sorted = src[order]
        keep_mask = torch.zeros(edge_min_dist.shape[0], dtype=torch.bool, device=edge_min_dist.device)
        topk = int(self.sparse_interface_topk)
        start = 0
        total = int(order.shape[0])
        while start < total:
            end = start + 1
            cur_src = int(src_sorted[start].item())
            while end < total and int(src_sorted[end].item()) == cur_src:
                end += 1
            keep_end = min(start + topk, end)
            keep_mask[order[start:keep_end]] = True
            start = end
        edge_index = edge_index[:, keep_mask]
        edge_vec_ca_ca = edge_vec_ca_ca[keep_mask]
        edge_min_dist = edge_min_dist[keep_mask]
        edge_type_idx = edge_type_idx[keep_mask]
        return edge_index, edge_vec_ca_ca, edge_min_dist, edge_type_idx

    def _frame_from_pep_ca(
        self,
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

    def _compute_frame_local_feat(self, data):
        num_graphs = getattr(data, "num_graphs", 1)
        device = data["pep"].x.device
        rec_pos = data["receptor"].pos
        pep_pos = data["pep_a"].pos
        rec_batch = getattr(data["receptor"], "batch", None)
        pep_batch = getattr(data["pep_a"], "batch", None)
        pep_res_batch = getattr(data["pep"], "batch", None)
        if rec_batch is None:
            rec_batch = torch.zeros(rec_pos.shape[0], dtype=torch.long, device=device)
        if pep_batch is None:
            pep_batch = torch.zeros(pep_pos.shape[0], dtype=torch.long, device=device)
        if pep_res_batch is None:
            pep_res_batch = torch.zeros(data["pep"].x.shape[0], dtype=torch.long, device=device)

        n_res_per_graph = torch.bincount(pep_res_batch, minlength=num_graphs)
        frame_local = torch.zeros((num_graphs, 9), dtype=torch.float32, device=device)
        rec_frames = torch.zeros((num_graphs, 3, 3), dtype=torch.float32, device=device)
        frame_valid = torch.zeros((num_graphs, 1), dtype=torch.float32, device=device)

        for g in range(num_graphs):
            rec_pos_g = rec_pos[rec_batch == g]
            pep_pos_g = pep_pos[pep_batch == g]
            atom2res_g = data["pep_a"].atom2res_index[pep_batch == g]
            atom2atomid_g = data["pep_a"].atom2atomid_index[pep_batch == g]
            n_res = int(n_res_per_graph[g].item())
            rec_frame = _frame_from_rec_pca(rec_pos_g, pep_pos_g)
            pep_frame = _frame_from_ncac(pep_pos_g, atom2res_g, atom2atomid_g, n_res, rec_pos_g)
            if pep_frame is None:
                pep_frame = self._frame_from_pep_ca(
                    pep_pos_g, atom2res_g, atom2atomid_g, rec_pos_g
                )
            if rec_frame is None or pep_frame is None:
                rec_frames[g] = torch.eye(3, device=device)
                frame_local[g] = torch.zeros(9, device=device)
                continue
            R_local = rec_frame.T @ pep_frame
            rec_frames[g] = rec_frame.to(dtype=torch.float32)
            frame_local[g] = R_local.reshape(-1).to(dtype=torch.float32)
            frame_valid[g, 0] = 1.0
        return rec_frames, frame_local, frame_valid

    def _compute_multi_anchor_frame_feat(self, data):
        num_graphs = getattr(data, "num_graphs", 1)
        device = data["pep"].x.device
        k = max(1, int(self.rot_frame_multi_anchor_k))
        n_local = max(3, int(self.rot_frame_multi_anchor_neighbors))
        rec_pos = data["receptor"].pos
        pep_pos = data["pep_a"].pos
        rec_batch = getattr(data["receptor"], "batch", None)
        pep_batch = getattr(data["pep_a"], "batch", None)
        pep_res_batch = getattr(data["pep"], "batch", None)
        if rec_batch is None:
            rec_batch = torch.zeros(rec_pos.shape[0], dtype=torch.long, device=device)
        if pep_batch is None:
            pep_batch = torch.zeros(pep_pos.shape[0], dtype=torch.long, device=device)
        if pep_res_batch is None:
            pep_res_batch = torch.zeros(data["pep"].x.shape[0], dtype=torch.long, device=device)

        n_res_per_graph = torch.bincount(pep_res_batch, minlength=num_graphs)
        frame_local = torch.zeros((num_graphs, k, 9), dtype=torch.float32, device=device)
        rec_frames = torch.zeros((num_graphs, k, 3, 3), dtype=torch.float32, device=device)
        frame_valid = torch.zeros((num_graphs, k, 1), dtype=torch.float32, device=device)

        for g in range(num_graphs):
            rec_pos_g = rec_pos[rec_batch == g]
            pep_pos_g = pep_pos[pep_batch == g]
            atom2res_g = data["pep_a"].atom2res_index[pep_batch == g]
            atom2atomid_g = data["pep_a"].atom2atomid_index[pep_batch == g]
            n_res = int(n_res_per_graph[g].item())
            if rec_pos_g.numel() == 0 or pep_pos_g.numel() == 0:
                continue

            pep_frame = _frame_from_ncac(pep_pos_g, atom2res_g, atom2atomid_g, n_res, rec_pos_g)
            if pep_frame is None:
                pep_frame = self._frame_from_pep_ca(
                    pep_pos_g, atom2res_g, atom2atomid_g, rec_pos_g
                )
            if pep_frame is None:
                continue

            pep_com = pep_pos_g.mean(dim=0)
            dists = (rec_pos_g - pep_com).norm(dim=1)
            anchor_count = min(k, rec_pos_g.shape[0])
            anchor_ids = torch.topk(dists, k=anchor_count, largest=False).indices

            valid_cnt = 0
            for j, ridx in enumerate(anchor_ids.tolist()):
                rec_center = rec_pos_g[ridx]
                local_d = (rec_pos_g - rec_center).norm(dim=1)
                local_k = min(n_local, rec_pos_g.shape[0])
                local_ids = torch.topk(local_d, k=local_k, largest=False).indices
                rec_local = rec_pos_g[local_ids]
                rec_frame = _frame_from_rec_pca(rec_local, pep_pos_g)
                if rec_frame is None:
                    continue
                r_local = rec_frame.T @ pep_frame
                rec_frames[g, j] = rec_frame.to(dtype=torch.float32)
                frame_local[g, j] = r_local.reshape(-1).to(dtype=torch.float32)
                frame_valid[g, j, 0] = 1.0
                valid_cnt += 1

            if valid_cnt == 0:
                rec_frame = _frame_from_rec_pca(rec_pos_g, pep_pos_g)
                if rec_frame is not None:
                    r_local = rec_frame.T @ pep_frame
                    rec_frames[g, 0] = rec_frame.to(dtype=torch.float32)
                    frame_local[g, 0] = r_local.reshape(-1).to(dtype=torch.float32)
                    frame_valid[g, 0, 0] = 1.0

        return rec_frames, frame_local, frame_valid

    def _compute_anchor_frame(self, data) -> torch.Tensor:
        num_graphs = getattr(data, "num_graphs", 1)
        device = data["receptor"].pos.device
        rec_pos = data["receptor"].pos
        pep_pos = data["pep_a"].pos if "pep_a" in data.node_types else None
        rec_batch = getattr(data["receptor"], "batch", None)
        if rec_batch is None:
            rec_batch = torch.zeros(rec_pos.shape[0], dtype=torch.long, device=device)
        pep_batch = None
        atom2res_index = None
        atom2atomid_index = None
        if "pep_a" in data.node_types:
            pep_batch = getattr(data["pep_a"], "batch", None)
            if pep_batch is None:
                pep_batch = torch.zeros(pep_pos.shape[0], dtype=torch.long, device=device)
            atom2res_index = data["pep_a"].atom2res_index
            atom2atomid_index = data["pep_a"].atom2atomid_index
        anchor_frame = torch.zeros((num_graphs, 9), dtype=torch.float32, device=device)

        def _safe_normalize(vec, eps=1e-8):
            norm = vec.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
            return vec / norm

        def _frame_from_iface_ncac(rec_pos_g, pep_pos_g, atom2res_g, atom2atomid_g):
            if rec_pos_g is None or pep_pos_g is None or rec_pos_g.numel() == 0 or pep_pos_g.numel() == 0:
                return None
            contact_cutoff = float(self.flow_cfg.get("rot_frame_contact_cutoff", 8.0) or 8.0)
            contact_min_points = int(self.flow_cfg.get("rot_frame_contact_min_points", 6) or 6)
            dists = torch.cdist(rec_pos_g, pep_pos_g)
            mask = (dists.min(dim=1).values <= contact_cutoff)
            rec_iface = rec_pos_g[mask] if mask.any() else None
            if rec_iface is None or rec_iface.shape[0] < contact_min_points:
                rec_iface = rec_pos_g if rec_pos_g.shape[0] >= 3 else None
            if rec_iface is None or rec_iface.shape[0] < 3:
                return None
            rec_center = rec_iface.mean(dim=0)
            centered = rec_iface - rec_center
            cov = centered.T @ centered
            _evals, evecs = torch.linalg.eigh(cov)
            z = _safe_normalize(evecs[:, 0])
            pep_center = pep_pos_g.mean(dim=0)
            if torch.dot(z, pep_center - rec_center) < 0:
                z = -z
            if atom2res_g is None or atom2atomid_g is None:
                return None
            n_mask = atom2atomid_g == 0
            c_mask = atom2atomid_g == 2
            if n_mask.any() and c_mask.any():
                n_idx = atom2res_g[n_mask]
                c_idx = atom2res_g[c_mask]
                n_first = n_mask.clone()
                n_first[n_mask] = n_idx == n_idx.min()
                c_last = c_mask.clone()
                c_last[c_mask] = c_idx == c_idx.max()
                n_pos = pep_pos_g[n_first].mean(dim=0)
                c_pos = pep_pos_g[c_last].mean(dim=0)
            else:
                return None
            x = c_pos - n_pos
            if x.norm().item() < 1e-6:
                return None
            x = _safe_normalize(x)
            z_proj = z - (z * x).sum() * x
            if z_proj.norm().item() < 1e-6:
                return None
            z = _safe_normalize(z_proj)
            y = _safe_normalize(torch.cross(z, x))
            if y.norm().item() < 1e-6:
                return None
            z = _safe_normalize(torch.cross(x, y))
            return torch.stack([x, y, z], dim=1)

        def _frame_from_iface_ncac_sc(rec_pos_g, pep_pos_g, atom2res_g, atom2atomid_g):
            if rec_pos_g is None or pep_pos_g is None or rec_pos_g.numel() == 0 or pep_pos_g.numel() == 0:
                return None
            if atom2res_g is None or atom2atomid_g is None:
                return None
            # N->C axis
            n_mask = atom2atomid_g == 0
            c_mask = atom2atomid_g == 2
            if not (n_mask.any() and c_mask.any()):
                return None
            n_idx = atom2res_g[n_mask]
            c_idx = atom2res_g[c_mask]
            n_first = n_mask.clone()
            n_first[n_mask] = n_idx == n_idx.min()
            c_last = c_mask.clone()
            c_last[c_mask] = c_idx == c_idx.max()
            n_pos = pep_pos_g[n_first].mean(dim=0)
            c_pos = pep_pos_g[c_last].mean(dim=0)
            x = c_pos - n_pos
            if x.norm().item() < 1e-6:
                return None
            x = _safe_normalize(x)
            # pick closest residue to receptor as sidechain anchor
            ca_mask = atom2atomid_g == 1
            if not ca_mask.any():
                return None
            ca_pos = pep_pos_g[ca_mask]
            ca_res = atom2res_g[ca_mask]
            dists = torch.cdist(ca_pos, rec_pos_g)
            closest_idx = int(torch.argmin(dists.min(dim=1).values))
            anchor_res = ca_res[closest_idx].item()
            # sidechain atoms exclude N/CA/C/O
            sc_mask = (~torch.isin(atom2atomid_g, torch.tensor([0, 1, 2, 3], device=atom2atomid_g.device))) & (
                atom2res_g == anchor_res
            )
            if not sc_mask.any():
                return None
            sc_center = pep_pos_g[sc_mask].mean(dim=0)
            ca_center = ca_pos[closest_idx]
            sc_vec = sc_center - ca_center
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

        def _frame_from_seq_cb(pep_pos_g, atom2res_g, atom2atomid_g):
            if pep_pos_g is None or pep_pos_g.numel() == 0:
                return None
            if atom2res_g is None or atom2atomid_g is None:
                return None
            n_mask = atom2atomid_g == 0
            c_mask = atom2atomid_g == 2
            if not (n_mask.any() and c_mask.any()):
                return None
            n_idx = atom2res_g[n_mask]
            c_idx = atom2res_g[c_mask]
            n_first = n_mask.clone()
            n_first[n_mask] = n_idx == n_idx.min()
            c_last = c_mask.clone()
            c_last[c_mask] = c_idx == c_idx.max()
            n_pos = pep_pos_g[n_first].mean(dim=0)
            c_pos = pep_pos_g[c_last].mean(dim=0)
            x = c_pos - n_pos
            if x.norm().item() < 1e-6:
                return None
            x = _safe_normalize(x)
            ca_mask = atom2atomid_g == 1
            if not ca_mask.any():
                return None
            ca_pos = pep_pos_g[ca_mask]
            ca_res = atom2res_g[ca_mask]
            if ca_res.numel() == 0:
                return None
            mid_res = int(ca_res.median().item())
            ca_mid_mask = ca_mask & (atom2res_g == mid_res)
            if not ca_mid_mask.any():
                return None
            ca_mid = pep_pos_g[ca_mid_mask].mean(dim=0)
            sc_mask = (~torch.isin(atom2atomid_g, torch.tensor([0, 1, 2, 3], device=atom2atomid_g.device))) & (
                atom2res_g == mid_res
            )
            if sc_mask.any():
                sc_center = pep_pos_g[sc_mask].mean(dim=0)
            else:
                return None
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

        def _frame_from_seq_cb2(pep_pos_g, atom2res_g, atom2atomid_g):
            if pep_pos_g is None or pep_pos_g.numel() == 0:
                return None
            if atom2res_g is None or atom2atomid_g is None:
                return None
            n_mask = atom2atomid_g == 0
            c_mask = atom2atomid_g == 2
            if not (n_mask.any() and c_mask.any()):
                return None
            n_idx = atom2res_g[n_mask]
            c_idx = atom2res_g[c_mask]
            n_first = n_mask.clone()
            n_first[n_mask] = n_idx == n_idx.min()
            c_last = c_mask.clone()
            c_last[c_mask] = c_idx == c_idx.max()
            n_pos = pep_pos_g[n_first].mean(dim=0)
            c_pos = pep_pos_g[c_last].mean(dim=0)
            x = c_pos - n_pos
            if x.norm().item() < 1e-6:
                return None
            x = _safe_normalize(x)
            sc_mask = (~torch.isin(atom2atomid_g, torch.tensor([0, 1, 2, 3], device=atom2atomid_g.device)))
            if not sc_mask.any():
                return None
            sc_res = atom2res_g[sc_mask]
            if sc_res.numel() == 0:
                return None
            res_min = int(sc_res.min().item())
            res_max = int(sc_res.max().item())
            if res_max - res_min < 2:
                return None
            res_i = res_min + 1
            res_j = res_max - 1
            sc_i = sc_mask & (atom2res_g == res_i)
            sc_j = sc_mask & (atom2res_g == res_j)
            if not (sc_i.any() and sc_j.any()):
                return None
            sc_i_center = pep_pos_g[sc_i].mean(dim=0)
            sc_j_center = pep_pos_g[sc_j].mean(dim=0)
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

        def _frame_from_seq_weighted(pep_pos_g, atom2res_g, atom2atomid_g, rec_pos_g):
            if pep_pos_g is None or pep_pos_g.numel() == 0:
                return None
            if atom2res_g is None or atom2atomid_g is None:
                return None
            ca_mask = atom2atomid_g == 1
            if ca_mask.sum().item() < 3:
                return None
            ca_pos = pep_pos_g[ca_mask]
            ca_res = atom2res_g[ca_mask]
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
            if rec_pos_g is not None and rec_pos_g.numel() > 0:
                rec_center = rec_pos_g.mean(dim=0)
                pep_center = ca_pos.mean(dim=0)
                if torch.dot(z, pep_center - rec_center) < 0:
                    z = -z
                    y = -y
            return torch.stack([x, y, z], dim=1)
        for g in range(num_graphs):
            rec_pos_g = rec_pos[rec_batch == g]
            if rec_pos_g.numel() == 0:
                anchor_frame[g] = torch.eye(3, device=device).reshape(-1)
                continue
            pep_pos_g = pep_pos[pep_batch == g] if pep_pos is not None else None
            if self.rot_anchor_input_mode == "seq_weighted":
                frame = _frame_from_seq_weighted(
                    pep_pos_g,
                    atom2res_index[pep_batch == g] if atom2res_index is not None else None,
                    atom2atomid_index[pep_batch == g] if atom2atomid_index is not None else None,
                    rec_pos_g,
                )
                if frame is None:
                    frame = _frame_from_seq_cb2(
                        pep_pos_g,
                        atom2res_index[pep_batch == g] if atom2res_index is not None else None,
                        atom2atomid_index[pep_batch == g] if atom2atomid_index is not None else None,
                    )
            elif self.rot_anchor_input_mode == "seq_ncac_cb2":
                frame = _frame_from_seq_cb2(
                    pep_pos_g,
                    atom2res_index[pep_batch == g] if atom2res_index is not None else None,
                    atom2atomid_index[pep_batch == g] if atom2atomid_index is not None else None,
                )
                if frame is None:
                    frame = _frame_from_seq_cb(
                        pep_pos_g,
                        atom2res_index[pep_batch == g] if atom2res_index is not None else None,
                        atom2atomid_index[pep_batch == g] if atom2atomid_index is not None else None,
                    )
            elif self.rot_anchor_input_mode == "seq_ncac_cb":
                frame = _frame_from_seq_cb(
                    pep_pos_g,
                    atom2res_index[pep_batch == g] if atom2res_index is not None else None,
                    atom2atomid_index[pep_batch == g] if atom2atomid_index is not None else None,
                )
                if frame is None:
                    frame = _frame_from_iface_ncac_sc(
                        rec_pos_g,
                        pep_pos_g,
                        atom2res_index[pep_batch == g] if atom2res_index is not None else None,
                        atom2atomid_index[pep_batch == g] if atom2atomid_index is not None else None,
                    )
            elif self.rot_anchor_input_mode == "iface_ncac_sc":
                frame = _frame_from_iface_ncac_sc(
                    rec_pos_g,
                    pep_pos_g,
                    atom2res_index[pep_batch == g] if atom2res_index is not None else None,
                    atom2atomid_index[pep_batch == g] if atom2atomid_index is not None else None,
                )
                if frame is None:
                    frame = _frame_from_iface_ncac(
                        rec_pos_g,
                        pep_pos_g,
                        atom2res_index[pep_batch == g] if atom2res_index is not None else None,
                        atom2atomid_index[pep_batch == g] if atom2atomid_index is not None else None,
                    )
            elif self.rot_anchor_input_mode == "iface_ncac":
                frame = _frame_from_iface_ncac(
                    rec_pos_g,
                    pep_pos_g,
                    atom2res_index[pep_batch == g] if atom2res_index is not None else None,
                    atom2atomid_index[pep_batch == g] if atom2atomid_index is not None else None,
                )
                if frame is None:
                    frame = _frame_from_rec_pca(rec_pos_g, pep_pos_g)
            else:
                frame = _frame_from_rec_pca(rec_pos_g, pep_pos_g)
            if frame is None:
                frame = torch.eye(3, device=device)
            anchor_frame[g] = frame.reshape(-1).to(dtype=torch.float32)
        return anchor_frame
            
    def build_rec_conv_graph(self,data):
        data['receptor'].node_sigma_emb = self.timestep_emb_func(data['receptor'].node_t['tr']) # tr rot and tor noise is all the same
        if len(data['receptor'].x.shape) == 1:
            data['receptor'].x = data['receptor'].x[:,None]
        node_attr = torch.cat(
                [data['receptor'].x, data['receptor'].node_sigma_emb], 1
            ) # 1+5+4+1280+32
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        edge_vec = data['receptor', 'receptor'].edge_v.squeeze()
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0]]
        edge_attr = torch.cat([edge_sigma_emb, data['receptor', 'receptor'].edge_s], 1)
        edge_sh = getattr(data, "_cached_receptor_edge_sh", None)
        if edge_sh is None or edge_sh.device != edge_vec.device:
            edge_sh = o3.spherical_harmonics(
                self.sh_irreps, edge_vec, normalize=True, normalization='component'
            )
            object.__setattr__(data, "_cached_receptor_edge_sh", edge_sh)
        return node_attr, edge_index, edge_attr, edge_sh

    def build_pep_conv_graph(self,data):
        data['pep'].node_sigma_emb = self.timestep_emb_func(data['pep'].node_t['tr']) # tr rot and tor noise is all the same
        if len(data['pep'].x.shape) == 1:
            data['pep'].x = data['pep'].x[:,None]
        if self.rot_ref_input and hasattr(data, "ref_point"):
            pep_batch = getattr(data["pep"], "batch", None)
            if pep_batch is None:
                pep_batch = torch.zeros(data["pep"].x.shape[0], dtype=torch.long, device=data["pep"].x.device)
            ref_point = data.ref_point.to(dtype=data["pep"].x.dtype)
            ref_vec = ref_point[pep_batch] - data["pep"].pos
            node_attr = torch.cat([data['pep'].x, ref_vec, data['pep'].node_sigma_emb], 1)
        else:
            node_attr = torch.cat(
                    [data['pep'].x, data['pep'].node_sigma_emb], 1
                ) # 1+5+4+32
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['pep', 'pep'].edge_index
        edge_vec = data['pep', 'pep'].edge_v.squeeze()
        edge_sigma_emb = data['pep'].node_sigma_emb[edge_index[0]]
        edge_attr = torch.cat([edge_sigma_emb, data['pep', 'pep'].edge_s], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between pep and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index_ca_ca = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['pep'].pos / cross_distance_cutoff[data['pep'].batch], 1,
                                data['receptor'].batch, data['pep'].batch, max_num_neighbors=self.cross_max_neighbors)
            edge_index_tips_ca = radius(data['receptor'].tips / cross_distance_cutoff[data['receptor'].batch],
                                data['pep'].pos / cross_distance_cutoff[data['pep'].batch], 1,
                                data['receptor'].batch, data['pep'].batch, max_num_neighbors=self.cross_max_neighbors)
            edge_index_ca_tips = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['pep'].tips / cross_distance_cutoff[data['pep'].batch], 1,
                                data['receptor'].batch, data['pep'].batch, max_num_neighbors=self.cross_max_neighbors)
            edge_index_tips_tips = radius(data['receptor'].tips / cross_distance_cutoff[data['receptor'].batch],
                                data['pep'].tips / cross_distance_cutoff[data['pep'].batch], 1,
                                data['receptor'].batch, data['pep'].batch, max_num_neighbors=self.cross_max_neighbors)
        else:
            edge_index_ca_ca = radius(data['receptor'].pos, data['pep'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['pep'].batch, max_num_neighbors=self.cross_max_neighbors)
            edge_index_tips_ca = radius(data['receptor'].tips, data['pep'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['pep'].batch, max_num_neighbors=self.cross_max_neighbors)
            edge_index_ca_tips = radius(data['receptor'].pos, data['pep'].tips, cross_distance_cutoff,
                            data['receptor'].batch, data['pep'].batch, max_num_neighbors=self.cross_max_neighbors)
            edge_index_tips_tips = radius(data['receptor'].tips, data['pep'].tips, cross_distance_cutoff,
                            data['receptor'].batch, data['pep'].batch, max_num_neighbors=self.cross_max_neighbors)
        
        edge_index = torch.cat((edge_index_ca_ca, edge_index_tips_ca, edge_index_ca_tips, edge_index_tips_tips), dim=1)
        edge_index = torch.unique(edge_index,dim=1).reshape(2,-1)
        
        src, dst = edge_index
        edge_vec_ca_ca = data['receptor'].pos[dst.long()] - data['pep'].pos[src.long()]
        edge_vec_tips_ca = data['receptor'].tips[dst.long()] - data['pep'].pos[src.long()]
        edge_vec_ca_tips = data['receptor'].pos[dst.long()] - data['pep'].tips[src.long()]
        edge_vec_tips_tips = data['receptor'].tips[dst.long()] - data['pep'].tips[src.long()]
        
        edge_vec_min, indices = torch.cat(
            (
                edge_vec_ca_ca.norm(dim=-1)[:, None],
                edge_vec_tips_ca.norm(dim=-1)[:, None],
                edge_vec_ca_tips.norm(dim=-1)[:, None],
                edge_vec_tips_tips.norm(dim=-1)[:, None],
            ),
            dim=1,
        ).min(dim=1)
        edge_index, edge_vec_ca_ca, edge_vec_min, indices = self._sparsify_cross_edges(
            edge_index,
            edge_vec_ca_ca,
            edge_vec_min,
            indices,
        )
        src, dst = edge_index

        edge_length_emb = self.cross_distance_expansion(edge_vec_min)
        cross_edge_type_emb = self.cross_edge_type(indices)
        edge_length_emb = edge_length_emb + cross_edge_type_emb
        edge_sigma_emb = data['pep'].node_sigma_emb[src.long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        # 方向特征保持使用 ca-ca（更稳定），而 “tips 触发的近距离” 由 type_emb + min_len 交给网络学
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec_ca_ca, normalize=True, normalization='component'
        )

        return edge_index, edge_attr, edge_sh

    def build_center_conv_graph(self, data):
        # 全局 head 的“罗盘”必须绑定到 receptor 坐标系，否则平移噪声会被 pep 自己的 COM 抵消掉：
        # - 若用 pep COM：edge_vec = pep_pos - pep_COM，对“整体平移”严格不敏感（不可观测）。
        # - 改成 receptor COM：edge_vec = pep_pos - rec_COM，整体平移/旋转噪声会直接体现在 edge_vec 上，tr/rot 才有学习信号。
        edge_index = getattr(data, "_cached_center_edge_index", None)
        center_pos = getattr(data, "_cached_receptor_center_pos", None)
        if edge_index is None or edge_index.device != data["pep"].x.device:
            edge_index = torch.cat(
                [
                    data["pep"].batch.unsqueeze(0),
                    torch.arange(len(data["pep"].batch), device=data["pep"].x.device).unsqueeze(0),
                ],
                dim=0,
            )
            object.__setattr__(data, "_cached_center_edge_index", edge_index)
        if center_pos is None or center_pos.device != data["receptor"].pos.device:
            center_pos = torch.zeros((data.num_graphs, 3), device=data["receptor"].pos.device)
            center_pos.index_add_(0, index=data["receptor"].batch, source=data["receptor"].pos)
            center_pos = center_pos / torch.bincount(data["receptor"].batch).unsqueeze(1)
            object.__setattr__(data, "_cached_receptor_center_pos", center_pos)

        edge_vec = data["pep"].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        if self.rot_anchor_input:
            anchor_frame = self._compute_anchor_frame(data)
            anchor_emb = self.anchor_frame_mlp(anchor_frame) * self.rot_anchor_input_scale
            edge_attr = edge_attr + anchor_emb[edge_index[0]]
        edge_sigma_emb = data['pep'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh
    
    def build_backbone_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['pep_a', 'pep_a'].edge_index[:, data['pep_a'].mask_edges_backbone.squeeze()].long()
        bond_pos = (data['pep_a'].pos[bonds[0]] + data['pep_a'].pos[bonds[1]]) / 2
        bond_batch = data['pep_a'].batch[bonds[0]]
        edge_index = radius(data['pep_a'].pos, bond_pos, self.lig_max_radius, batch_x=data['pep_a'].batch, batch_y=bond_batch)

        edge_vec = data['pep_a'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh
    
    def build_sidechain_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['pep_a', 'pep_a'].edge_index[:, data['pep_a'].mask_edges_sidechain.squeeze()].long()
        bond_pos = (data['pep_a'].pos[bonds[0]] + data['pep_a'].pos[bonds[1]]) / 2
        bond_batch = data['pep_a'].batch[bonds[0]]
        edge_index = radius(data['pep_a'].pos, bond_pos, self.lig_max_radius, batch_x=data['pep_a'].batch, batch_y=bond_batch)

        edge_vec = data['pep_a'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh
