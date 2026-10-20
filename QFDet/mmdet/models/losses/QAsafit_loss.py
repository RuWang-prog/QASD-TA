import torch
import torch.nn as nn
from mmdet.core import bbox_overlaps
from mmdet.models.builder import LOSSES
from .utils import weight_reduce_loss


def qasafit_loss(pred,
                 target,
                 weight=None,
                 C_min=16.0,
                 boundary=32.0,
                 tau=4.0,
                 reduction='mean',
                 avg_factor=None):
    """QAsafit Loss 的核心张量计算函数"""

    # # 1. 宏观惩罚: 计算配对的 GIoU (is_aligned=True 极其重要，防止显存爆炸)
    giou = bbox_overlaps(pred, target, is_aligned=True, mode='giou')
    loss_giou = 1.0 - giou

  #wr20260713 GIoU->CIoU
    # ciou = bbox_overlaps(pred, target, is_aligned=True, mode='ciou')
    # loss_ciou = 1.0 - ciou

    # 2. 获取 GT 尺度 (注意防除零)
    gt_w = target[:, 2] - target[:, 0]
    gt_h = target[:, 3] - target[:, 1]
    gt_area = torch.clamp(gt_w * gt_h, min=1e-6)
    sqrtA = torch.sqrt(gt_area)

    # 3. 微观惩罚: 计算 QA-NWD
    pred_cx = (pred[:, 0] + pred[:, 2]) / 2.0
    pred_cy = (pred[:, 1] + pred[:, 3]) / 2.0
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]

    gt_cx = (target[:, 0] + target[:, 2]) / 2.0
    gt_cy = (target[:, 1] + target[:, 3]) / 2.0

    dx = pred_cx - gt_cx
    dy = pred_cy - gt_cy
    dw = pred_w - gt_w
    dh = pred_h - gt_h

    w2_sq = dx ** 2 + dy ** 2 + (dw ** 2 + dh ** 2) / 4.0
    w2 = torch.sqrt(torch.clamp(w2_sq, min=1e-12))

    C_dyn = torch.clamp(sqrtA, min=C_min)
    qa_nwd = torch.exp(-w2 / C_dyn)
    loss_nwd = 1.0 - qa_nwd

    # 4. 尺度路由融合 (以 IoU 为主视角)
    w_iou = 1.0 / (1.0 + torch.exp(-(sqrtA - boundary) / tau))

    # 组合得到最终一维 loss (Shape: [N])
    loss = w_iou * loss_giou + (1.0 - w_iou) * loss_nwd

    # # wr20260713 GIoU->CIoU
    # loss = w_iou * loss_ciou + (1.0 - w_iou) * loss_nwd

    # 5. 按照框架标准应用权重和规约
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@LOSSES.register_module()
class QAsafitLoss(nn.Module):
    def __init__(self,
                 C_min=16.0,
                 boundary=32.0,
                 tau=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(QAsafitLoss, self).__init__()
        self.C_min = C_min
        self.boundary = boundary
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        # 1. 安全保护：如果没有正样本，直接返回 0 的梯度占位，防止 NaN
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 返回梯度为 0 的标量

        # 2. 规约方式覆盖
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # 3. 权重对齐降维 (N, 4) -> (N,)
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)

        # 4. 调用核心计算函数
        loss = self.loss_weight * qasafit_loss(
            pred,
            target,
            weight,
            C_min=self.C_min,
            boundary=self.boundary,
            tau=self.tau,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss