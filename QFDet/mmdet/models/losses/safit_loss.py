import torch
import torch.nn as nn
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss   # 导入装饰器

# 如果 sigmoid 是你自定义的，也需要实现或导入；此处假设它是一个简单的函数
def sigmoid(x, k):
    return 1/(1+torch.exp(-(1/k) * x))


# 将 xyxy 格式转换为 (x, y, w, h) 格式，这里假设你已有实现
def xyxy2xyw2h2(bbox):
    _bbox = bbox.clone()
    _bbox[:,0] = bbox[:,0]
    _bbox[:,1] = bbox[:,1]
    _bbox[:,2] = (bbox[:,2] - bbox[:,0])/2  # w/2
    _bbox[:,3] = (bbox[:,3] - bbox[:,1])/2  # h/2
    return _bbox
#wr20260404
def xyxy2cxcyw2h2(bbox):
    _bbox = bbox.clone()
    _bbox[:,0] = (bbox[..., 0] + bbox[..., 2]) / 2   # cx
    _bbox[:,1] = (bbox[..., 1] + bbox[..., 3]) / 2   # cy
    _bbox[:,2] = (bbox[:,2] - bbox[:,0])/2  # w/2
    _bbox[:,3] = (bbox[:,3] - bbox[:,1])/2  # h/2
    return _bbox

@weighted_loss
def safit_loss(pred, target, eps=1e-7, C=32):
    """SAFit 损失函数实现"""
    # IOU
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps) #.clamp(min=eps) 的作用是将张量中所有小于 eps 的值截断为 eps

    # NWD (使用你的公式)
    # pred_xyhw = xyxy2xyw2h2(pred)
    # target_xyhw = xyxy2xyw2h2(target)
    #wr20260403
    pred_xyhw = xyxy2cxcyw2h2(pred)
    target_xyhw = xyxy2cxcyw2h2(target)

    nwd = torch.exp(-torch.sqrt(torch.sum((pred_xyhw - target_xyhw) ** 2, dim=1)) / C)

    # SAFit
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    area = ag
    safit = sigmoid(torch.sqrt(area) - C, C) * ious + (1 - sigmoid(torch.sqrt(area) - C, C)) * nwd
    loss = 1 - safit

    return loss


@LOSSES.register_module()
class SafitLoss(nn.Module):
    """SAFit Loss 包装类"""

    def __init__(self, eps=1e-7, C=32, reduction='mean', loss_weight=1.0):
        super(SafitLoss, self).__init__()
        self.eps = eps
        self.C = C
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
                # 如果 weight 全为零，直接返回零（避免梯度问题）
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()

        reduction = reduction_override if reduction_override else self.reduction

        # 处理 weight 维度：如果 weight 形状与 pred 相同 (n,4)，则取平均变为 (n,)
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape, f"weight shape {weight.shape} vs pred {pred.shape}"
            weight = weight.mean(-1)

        loss = self.loss_weight * safit_loss(
            pred,  #预测边界框 xyxy
            target, #真实边界框xyxy
            weight=weight,
            eps=self.eps,
            C=self.C,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss