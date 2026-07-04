import torch.nn as nn
from .flow_model import CGTensorProductEquivariantModel

class BaseModel(nn.Module):
    """
        enc(receptor) -> R^(dxL)
        enc(ligand)  -> R^(dxL)
    """
    def __init__(self, args, 
                 confidence_mode=False,num_confidence_outputs=1):
        super(BaseModel, self).__init__()

        # 这个SB基类只干一件事：初始化真正干活的CG等变编码器
        self.encoder = CGTensorProductEquivariantModel(args, confidence_mode=confidence_mode, num_confidence_outputs = num_confidence_outputs)

class ScoreModel(BaseModel):
    def __init__(self, args):
        super(ScoreModel, self).__init__(args)

    def forward(self, batch):
        # 生成平移/旋转/扭转载荷，扩散采样全靠这些梯度信息
        outputs_raw = self.encoder(batch)
        if isinstance(outputs_raw, dict):
            return outputs_raw
        rot_cls_logits = None
        ca_pred = None
        iface_contact_logits = None
        iface_pairdist_pred = None
        iface_edge_index = None
        if isinstance(outputs_raw, tuple):
            if len(outputs_raw) < 4:
                raise ValueError(f"encoder 输出长度异常: {len(outputs_raw)}")
            tr_pred, rot_pred, tor_pred_backbone, tor_pred_sidechain = outputs_raw[:4]
            extra = list(outputs_raw[4:])
            if getattr(self.encoder, "rot_cls_head", None) is not None and extra:
                rot_cls_logits = extra.pop(0)
            if extra:
                ca_pred = extra.pop(0)
            if extra:
                iface_contact_logits = extra.pop(0)
            if extra:
                iface_pairdist_pred = extra.pop(0)
            if extra:
                iface_edge_index = extra.pop(0)
        else:
            tr_pred, rot_pred, tor_pred_backbone, tor_pred_sidechain = outputs_raw
        outputs = {}
        outputs["tr_pred"] = tr_pred
        outputs["rot_pred"] = rot_pred
        outputs["tor_pred_backbone"] = tor_pred_backbone
        outputs["tor_pred_sidechain"] = tor_pred_sidechain
        if rot_cls_logits is not None:
            outputs["rot_cls_logits"] = rot_cls_logits
        if ca_pred is not None:
            outputs["ca_pred"] = ca_pred
        if iface_contact_logits is not None:
            outputs["interface_contact_logits"] = iface_contact_logits
        if iface_pairdist_pred is not None:
            outputs["interface_pairdist_pred"] = iface_pairdist_pred
        if iface_edge_index is not None:
            outputs["interface_edge_index"] = iface_edge_index

        return outputs

class ConfidenceModel(BaseModel):
    def __init__(self, args):
        super(ConfidenceModel, self).__init__(args, confidence_mode=True, num_confidence_outputs=len(
                            args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                            args.rmsd_classification_cutoff, list) else 1)

    def forward(self, batch):
        # 信心模式下直接预测RMSD分档，帮你筛掉不靠谱的构象
        logits = self.encoder(batch)

        return logits
    
