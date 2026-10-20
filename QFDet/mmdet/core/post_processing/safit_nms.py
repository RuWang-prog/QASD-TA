import math

from mmdet.core.bbox.iou_calculators import bbox_overlaps
import torch

def xyxy2xyw2h2(bbox):
    _bbox = bbox.clone()
    _bbox[:,0] = bbox[:,0]
    _bbox[:,1] = bbox[:,1]
    _bbox[:,2] = (bbox[:,2] - bbox[:,0])/2
    _bbox[:,3] = (bbox[:,3] - bbox[:,1])/2
    return _bbox

def xyxy2cxcyw2h2(bbox):
    _bbox = bbox.clone()
    _bbox[:,0] = (bbox[:,0] + bbox[:,2])/2
    _bbox[:,1] = (bbox[:,1] + bbox[:,3])/2
    _bbox[:,2] = (bbox[:,2] - bbox[:,0])/2
    _bbox[:,3] = (bbox[:,3] - bbox[:,1])/2
    return _bbox

def sigmoid(x, k):
    return 1/(1+torch.exp(-(1/k) * x))

def compute_nwd(pred, target, C=32):
    """计算 NWD (Normalized Wasserstein Distance)"""
    pred_xyhw = xyxy2cxcyw2h2(pred)
    target_xyhw = xyxy2cxcyw2h2(target)
    dist_sq = torch.sum((pred_xyhw - target_xyhw) ** 2, dim=1)
    nwd = torch.exp(-torch.sqrt(dist_sq) / C)
    return nwd

def safit_overlaps(pred, target, eps=1e-7, C=32,
               reduction=None, avg_factor=None):
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # IOU
    ious = bbox_overlaps(pred, target, is_aligned=False).clamp(min=eps)



    # NWD
    # pred_xyhw = xyxy2xyw2h2(pred)
    # target_xyhw = xyxy2xyw2h2(target)
    pred_xyhw = xyxy2cxcyw2h2(pred)
    target_xyhw = xyxy2cxcyw2h2(target)
    nwd = torch.exp(-torch.sqrt(torch.sum((pred_xyhw - target_xyhw) * (pred_xyhw - target_xyhw), dim=1)) / C)

    # # SAFit
    # ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    # area = ag
    # safit = sigmoid(torch.sqrt(area) - C, C) * ious + \
    #         (1 - sigmoid(torch.sqrt(area) - C, C)) * nwd
    # return safit

    #wr20260420 max(IoU NWD)
    return torch.max(ious, nwd)

#wr20260425 max(IoU,NWD),IoU=0.3, 同等严格程度下的NWD阈值，同尺度抑制
# def safit_nms(boxes, scores, iou_threshold=0.3, C=32, **kwargs):
#     """
#     基于 max(IoU, NWD) 的非极大值抑制，使用尺度自适应的动态阈值，并增加同尺度空间抑制。
#     只有两个框属于相同的面积区间时才进行抑制判断，否则保留。
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#
#     # 面积区间边界 [0,8²], [8²,16²], [16²,32²], [32²,96²], [96²,∞)
#     area_bins = [0, 64, 256, 1024, 9216, 1e10]
#
#     def get_area_bin(area):
#         """返回面积所属的区间索引 (0~4)"""
#         for i, (low, high) in enumerate(zip(area_bins[:-1], area_bins[1:])):
#             if low <= area < high:
#                 return i
#         return 4  # 最大区间
#
#     # 按得分降序排序
#     order = scores.sort(0, descending=True)[1]
#     keep = []
#
#     # 预计算常数：与 IoU=0.3 等效的中心偏移比例 d/s = 1 - sqrt(0.3)
#     k = 1 - math.sqrt(0.3)   # 约 0.452277
#
#     while order.numel() > 0:
#         i = order[0]
#         keep.append(i)
#         if order.numel() == 1:
#             break
#
#         current_box = boxes[i].unsqueeze(0)          # (1,4)
#         rest_boxes = boxes[order[1:]]                # (M,4)
#
#         # 当前框的面积与区间
#         cur_area = (current_box[0,2] - current_box[0,0]) * (current_box[0,3] - current_box[0,1])
#         cur_sqrt_area = torch.sqrt(cur_area + 1e-7)
#         cur_bin = get_area_bin(cur_area.item())
#
#         # 计算所有 rest_boxes 的面积和区间
#         rest_areas = (rest_boxes[:,2] - rest_boxes[:,0]) * (rest_boxes[:,3] - rest_boxes[:,1])
#         rest_bins = torch.tensor([get_area_bin(a.item()) for a in rest_areas], device=boxes.device)
#         same_bin_mask = (rest_bins == cur_bin)
#
#         # 只对同区间的框进行抑制判断
#         if same_bin_mask.any():
#             # 提取同区间的框
#             same_bin_boxes = rest_boxes[same_bin_mask]
#
#             # 计算 IoU 和 NWD
#             ious = bbox_overlaps(current_box, same_bin_boxes, is_aligned=False).clamp(min=1e-7).squeeze(0)
#             nwd = compute_nwd(current_box, same_bin_boxes, C).squeeze(0)
#
#             # 取最大值作为相似度度量
#             metric = torch.max(ious, nwd)
#
#             # 基于当前框（高分框）计算动态阈值
#             nwd_thresh = torch.exp(-k * cur_sqrt_area / C)
#             use_iou = ious >= nwd
#             thresh = torch.where(use_iou,
#                                  torch.tensor(iou_threshold, device=ious.device),
#                                  nwd_thresh)
#
#             # 抑制条件：相似度大于阈值则抑制
#             sup_mask_same = metric > thresh
#
#             # 构建完整的抑制掩码（长度与 rest_boxes 相同）
#             sup_mask_full = torch.zeros(rest_boxes.size(0), dtype=torch.bool, device=boxes.device)
#             sup_mask_full[same_bin_mask] = sup_mask_same
#         else:
#             sup_mask_full = torch.zeros(rest_boxes.size(0), dtype=torch.bool, device=boxes.device)
#
#         # 保留未被抑制的框
#         order = order[1:][~sup_mask_full]
#
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)

# #wr20260422同尺度空间抑制+自适应动态阈值
# def safit_nms(boxes, scores, iou_threshold=0.3, adaptive=True, base_thresh=0.3, scale=0.3, C=32, **kwargs):
#     """
#     基于 SAFit 的非极大值抑制，增加尺度空间判断：
#     只有两个框属于相同的面积区间时才进行抑制，否则保留。
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#
#     # 定义面积区间边界（五个区间）
#     area_bins = [0, 64, 256, 1024, 9216, 1e10]  # 对应 [0,8²], [8²,16²], [16²,32²], [32²,96²], [96²,∞)
#
#     def get_area_bin(area):
#         """返回面积所属的区间索引 (0~4)"""
#         for i, (low, high) in enumerate(zip(area_bins[:-1], area_bins[1:])):
#             if low <= area < high:
#                 return i
#         return 4  # 最大区间
#
#     # 按得分降序排序
#     order = scores.sort(0, descending=True)[1]
#     keep = []
#
#     while order.numel() > 0:
#         i = order[0]
#         keep.append(i)
#         if order.numel() == 1:
#             break
#
#         current_box = boxes[i].unsqueeze(0)          # (1,4)
#         rest_boxes = boxes[order[1:]]                # (M,4)
#
#         # 计算当前框的面积及区间索引
#         cur_area = (current_box[0,2] - current_box[0,0]) * (current_box[0,3] - current_box[0,1])
#         cur_bin = get_area_bin(cur_area.item())
#
#         # 计算所有 rest_boxes 的面积和区间索引
#         areas = (rest_boxes[:,2] - rest_boxes[:,0]) * (rest_boxes[:,3] - rest_boxes[:,1])
#         rest_bins = torch.tensor([get_area_bin(a.item()) for a in areas], device=boxes.device)
#
#         # 只对同区间的框进行抑制判断
#         same_bin_mask = (rest_bins == cur_bin)
#
#         if same_bin_mask.any():
#             # 提取同区间的框
#             same_bin_boxes = rest_boxes[same_bin_mask]
#             # 计算 IoU 和 NWD
#             ious = bbox_overlaps(current_box, same_bin_boxes, is_aligned=False).clamp(min=1e-7).squeeze(0)
#             nwd = compute_nwd(current_box, same_bin_boxes, C).squeeze(0)  # 需要实现 compute_nwd
#
#             # 计算 SAFit 加权和（权重基于被抑制框面积）
#             sqrt_area = torch.sqrt(areas[same_bin_mask] + 1e-7)
#             w = sigmoid(sqrt_area - C, C)          # 权重，小目标 w 小
#             safit_vals = w * ious + (1 - w) * nwd
#
#             # 自适应阈值（若 adaptive=True）
#             if adaptive:
#                 thresh = base_thresh + scale * (1 - w)
#                 sup_mask_same = safit_vals > thresh
#             else:
#                 sup_mask_same = safit_vals > iou_threshold
#
#             # 构建完整的抑制掩码（长度与 rest_boxes 相同）
#             sup_mask_full = torch.zeros(rest_boxes.size(0), dtype=torch.bool, device=boxes.device)
#             sup_mask_full[same_bin_mask] = sup_mask_same
#         else:
#             sup_mask_full = torch.zeros(rest_boxes.size(0), dtype=torch.bool, device=boxes.device)
#
#         # 保留未被抑制的框
#         order = order[1:][~sup_mask_full]
#
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)

# #wr20260424同尺度空间抑制+加权和IoU阈值和NWD阈值，基于高分框统一权重和阈值
# def safit_nms(boxes, scores, iou_threshold=0.3, C=32, **kwargs):
#     """
#     基于 SAFit 的非极大值抑制，同尺度空间抑制 + 统一权重与阈值（基于高分框）
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#
#     # 面积区间边界 [0,8²], [8²,16²], [16²,32²], [32²,96²], [96²,∞)
#     area_bins = [0, 64, 256, 1024, 9216, 1e10]
#
#     def get_area_bin(area):
#         for i, (low, high) in enumerate(zip(area_bins[:-1], area_bins[1:])):
#             if low <= area < high:
#                 return i
#         return 4
#
#     k = 1 - math.sqrt(0.3)  # 与 IoU=0.3 等效的偏移比例
#
#     order = scores.sort(0, descending=True)[1]
#     keep = []
#
#     while order.numel() > 0:
#         i = order[0]
#         keep.append(i)
#         if order.numel() == 1:
#             break
#
#         current_box = boxes[i].unsqueeze(0)          # (1,4)
#         rest_boxes = boxes[order[1:]]                # (M,4)
#
#         # 当前框的面积与区间
#         cur_area = (current_box[0,2] - current_box[0,0]) * (current_box[0,3] - current_box[0,1])
#         cur_bin = get_area_bin(cur_area.item())
#         cur_sqrt_area = torch.sqrt(cur_area + 1e-7)
#
#         # 基于当前框计算权重 w_cur 和动态阈值
#         w_cur = sigmoid(cur_sqrt_area - C, C)
#         nwd_thresh_cur = torch.exp(-k * cur_sqrt_area / C)
#         thresh = w_cur * iou_threshold + (1 - w_cur) * nwd_thresh_cur
#
#         # 计算所有 rest_boxes 的面积和区间
#         areas = (rest_boxes[:,2] - rest_boxes[:,0]) * (rest_boxes[:,3] - rest_boxes[:,1])
#         rest_bins = torch.tensor([get_area_bin(a.item()) for a in areas], device=boxes.device)
#         same_bin_mask = (rest_bins == cur_bin)
#
#         if same_bin_mask.any():
#             same_bin_boxes = rest_boxes[same_bin_mask]
#
#             # 计算 IoU 和 NWD
#             ious = bbox_overlaps(current_box, same_bin_boxes, is_aligned=False).clamp(min=1e-7).squeeze(0)
#             nwd = compute_nwd(current_box, same_bin_boxes, C).squeeze(0)
#
#             # SAFit 相似度：使用基于当前框的权重 w_cur（统一）
#             safit_vals = w_cur * ious + (1 - w_cur) * nwd
#
#             # 抑制条件：SAFit > 阈值
#             sup_mask_same = safit_vals > thresh
#
#             # 构建完整抑制掩码
#             sup_mask_full = torch.zeros(rest_boxes.size(0), dtype=torch.bool, device=boxes.device)
#             sup_mask_full[same_bin_mask] = sup_mask_same
#         else:
#             sup_mask_full = torch.zeros(rest_boxes.size(0), dtype=torch.bool, device=boxes.device)
#
#         order = order[1:][~sup_mask_full]
#
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)
#wr20260427 尺度自适应融合IoU+真正NWD的NMS
import torch
import torchvision.ops as ops

# def compute_nwd_matrix(pred_boxes, C_nwd=20.0):
#     """
#     计算预测框两两之间的 NWD 相似度（真正的 Wasserstein 距离）
#     pred_boxes: (N,4) [x1,y1,x2,y2]
#     """
#     N = pred_boxes.shape[0]
#     if N <= 1:
#         return pred_boxes.new_zeros((N, N))
#
#     cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
#     cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
#     w  = pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0
#     h  = pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0
#
#     # 中心距离项 (N,N)
#     cx_diff = cx[:, None] - cx[None, :]  #利用广播机制，一次性计算出所有框对之间的中心点坐标差
#     cy_diff = cy[:, None] - cy[None, :]
#     center_dist_sq = cx_diff**2 + cy_diff**2  #中心距离平方
#
#     # 尺度差异项 (N,N)
#     sigma_w1 = w[:, None] / 2.0
#     sigma_h1 = h[:, None] / 2.0
#     sigma_w2 = w[None, :] / 2.0
#     sigma_h2 = h[None, :] / 2.0
#     scale_dist_sq = (sigma_w1 - sigma_w2)**2 + (sigma_h1 - sigma_h2)**2  #半宽高差值的平方和
#
#     wasserstein_sq = center_dist_sq + scale_dist_sq  #2-Wasserstein距离的平方
#     wasserstein_dist = torch.sqrt(torch.clamp(wasserstein_sq, min=0.0))  #开方得到真正的Wasserstein距离
#     nwd_sim = torch.exp(-wasserstein_dist / C_nwd) #指数归一化为相似度
#     return nwd_sim


# def safit_nms(pred_boxes, pred_scores, iou_threshold=0.5, C=32.0, C_nwd=32,
#               soft_mode=False, sigma=0.5, score_thr=0.001):
#     """
#     safitNMS: 尺度自适应融合 IoU + NWD
#     pred_boxes: (N,4) [x1,y1,x2,y2]   (已包含类别偏移也没问题)
#     pred_scores: (N,) 置信度
#     iou_threshold: 统一抑制阈值（内部会按尺度微调，初始可设 0.5）
#     """
#     N = pred_boxes.shape[0]
#     if N == 0:
#         return torch.tensor([], dtype=torch.long, device=pred_boxes.device)
#
#     # 计算面积与自适应权重
#     w = pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0
#     h = pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0
#     areas = w * h
#     sqrt_areas = torch.sqrt(areas)
#     # 权重：小框 → 0 (NWD)，大框 → 1 (IoU)
#     weights = torch.sigmoid((sqrt_areas - C) / C)   # (N,)
#
#     # 预计算 IoU 和 NWD 矩阵
#     iou_matrix = ops.box_iou(pred_boxes, pred_boxes) #计算的是pred_boxes自身与自身的两两IoU矩阵
#     nwd_matrix = compute_nwd_matrix(pred_boxes, C_nwd=C_nwd)
#
#     # 成对权重取平均
#     w_pair = (weights[:, None] + weights[None, :]) / 2.0
#     safit_matrix = w_pair * iou_matrix + (1.0 - w_pair) * nwd_matrix
#     safit_matrix.fill_diagonal_(0.0)  #对角置零，避免一个框抑制自己
#
#     # ---------- NMS 循环 ----------
#     order = pred_scores.argsort(descending=True)
#     keep = []
#     scores = pred_scores.clone()
#
#     while order.numel() > 0:
#         i = order[0]
#         keep.append(i.item())
#
#         if order.numel() == 1:
#             break
#
#         sim = safit_matrix[i][order[1:]] #提取相似度向量，sim是框i与所有剩余框（order[1:]）的safit值
#
#         # 可选：根据当前框尺度微调阈值，小框稍严，大框稍松
#         # 这里以传入的 iou_threshold 为基准，±0.05 调整
#         current_sqrt = sqrt_areas[i]  #根据当前框的边长，动态调整抑制阈值
#         if current_sqrt < 16.0:          # ex_tiny / tiny
#             thr = iou_threshold - 0.05
#         elif current_sqrt < 32.0:        # small
#             thr = iou_threshold
#         else:                           # medium / large
#             thr = iou_threshold + 0.05
#         thr = max(0.2, min(0.8, thr))
#
#         if soft_mode:
#             penalty = torch.exp(-(sim ** 2) / sigma)
#             scores[order[1:]] *= penalty
#             mask = scores[order[1:]] >= score_thr
#             order = order[1:][mask]
#         else:
#             mask = sim <= thr
#             order = order[1:][mask]
#
#     return torch.tensor(keep, dtype=torch.long, device=pred_boxes.device)
#wr20260427 分尺度阈值
# def safit_nms(pred_boxes, pred_scores,
#               thr_small=0.65,   # ex_tiny / tiny（边长 <16）  # small（边长 16~32）
#               thr_large=0.3,   # medium / large（边长 ≥32）
#               C=32.0,
#               C_nwd=12.8,
#               soft_mode=False,
#               sigma=0.5,
#               score_thr=0.001,
#               # 向下兼容参数（若调用时未提供新阈值，可通过 iou_threshold 传递，但不推荐）
#               iou_threshold=None):
#     """
#     safitNMS: 尺度自适应融合 IoU + NWD，分尺度独立阈值版本
#     pred_boxes: (N,4) [x1,y1,x2,y2]   (已包含类别偏移也没问题)
#     pred_scores: (N,) 置信度
#     thr_small : 小目标阈值（边长 <16）
#     thr_medium: 中等目标阈值（边长 16~32）
#     thr_large : 大目标阈值（边长 ≥32）
#     """
#     N = pred_boxes.shape[0]
#     if N == 0:
#         return torch.tensor([], dtype=torch.long, device=pred_boxes.device)
#
#     # 如果用户通过旧接口传入 iou_threshold，将其转换为分尺度阈值（兼容过渡）
#     if iou_threshold is not None:
#         # 此时仍使用原来的动态调整逻辑，保持兼容性
#         use_old = True
#     else:
#         use_old = False
#
#     # 计算面积与自适应权重
#     w = pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0
#     h = pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0
#     areas = w * h
#     sqrt_areas = torch.sqrt(areas)
#     weights = torch.sigmoid((sqrt_areas - C) / C)   # (N,)
#
#     # 预计算 IoU 和 NWD 矩阵
#     iou_matrix = ops.box_iou(pred_boxes, pred_boxes)
#     nwd_matrix = compute_nwd_matrix(pred_boxes, C_nwd=C_nwd)
#
#     # 成对权重取平均
#     w_pair = (weights[:, None] + weights[None, :]) / 2.0
#     safit_matrix = w_pair * iou_matrix + (1.0 - w_pair) * nwd_matrix
#     safit_matrix.fill_diagonal_(0.0)
#
#     # ---------- NMS 循环 ----------
#     order = pred_scores.argsort(descending=True)
#     keep = []
#     scores = pred_scores.clone()
#
#     while order.numel() > 0:
#         i = order[0]
#         keep.append(i.item())
#
#         if order.numel() == 1:
#             break
#
#         sim = safit_matrix[i][order[1:]]
#         current_sqrt = sqrt_areas[i]
#
#         # 分尺度独立阈值选择
#         if not use_old:
#             if current_sqrt < 16.0:
#                 thr = thr_small
#             elif current_sqrt < 32.0:
#                 # # 在 small 区间线性过渡
#                 frac = (current_sqrt - 16.0) / 16.0  # 0~1
#                 thr = thr_small + (thr_large - thr_small) * frac
#                 # thr = thr_medium
#             else:
#                 thr = thr_large
#             thr = max(0.1, min(0.95, thr))   # 放宽钳位范围
#         else:
#             # 旧的动态微调逻辑（兼容）
#             if current_sqrt < 16.0:
#                 thr = iou_threshold - 0.05
#             elif current_sqrt < 32.0:
#                 thr = iou_threshold
#             else:
#                 thr = iou_threshold + 0.05
#             thr = max(0.2, min(0.8, thr))
#
#         if soft_mode:
#             penalty = torch.exp(-(sim ** 2) / sigma)
#             scores[order[1:]] *= penalty
#             mask = scores[order[1:]] >= score_thr
#             order = order[1:][mask]
#         else:
#             mask = sim <= thr
#             order = order[1:][mask]
#
#     return torch.tensor(keep, dtype=torch.long, device=pred_boxes.device)

# #wr20260429 高分框面积计算IoU和NWD的加权权重
# def safit_nms(pred_boxes, pred_scores,
#                        thr_small=0.7,   # 对于 ex_tiny/tiny 更加宽容
#                        thr_large=0.35,  # 对于大目标维持常规阈值
#                        C=32.0,
#                        C_nwd=16.0,
#                        soft_mode=False,
#                        sigma_base=0.5,
#                        score_thr=0.005):
#     """
#     SAfitNMS 完整实现
#     pred_boxes: (N,4) [x1,y1,x2,y2]
#     pred_scores: (N,)
#     """
#     N = pred_boxes.shape[0]
#     if N == 0:
#         return torch.tensor([], dtype=torch.long, device=pred_boxes.device)
#     # 1. 预计算所有框的面积和对应的混合权重
#     w = pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0
#     h = pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0
#     sqrt_areas = torch.sqrt(torch.clamp(w * h, min=0.1))
#     # 权重 weights 越接近 1，表示目标越大，越倾向于使用 IoU
#     weights = torch.sigmoid((sqrt_areas - C) / C)
#     # 2. 预计算矩阵（注意内存占用，若 N巨大建议在循环内动态计算单行）
#     iou_matrix = ops.box_iou(pred_boxes, pred_boxes)
#     nwd_matrix = compute_nwd_matrix(pred_boxes, C_nwd=C_nwd)
#     # 3. NMS 主循环
#     curr_scores = pred_scores.clone()
#     order = curr_scores.argsort(descending=True)
#     keep = []
#     while order.numel() > 0:
#         i = order[0]
#         keep.append(i.item())
#         if order.numel() == 1:
#             break
#         # 获取剩余框的索引
#         others = order[1:]
#         # --- 核心改进：非对称权重逻辑 ---
#         # 以当前最高分框 i 的属性作为判断基准
#         w_i = weights[i]
#
#         # 计算混合相似度 sim (N-1,)
#         # 如果 i 是小目标，w_i 趋于 0，sim 趋于 nwd；反之趋于 iou
#         sim = w_i * iou_matrix[i, others] + (1.0 - w_i) * nwd_matrix[i,others]
#         # --- 动态阈值计算 ---
#         if sqrt_areas[i] < 16.0:
#             current_thr = thr_small
#         elif sqrt_areas[i] < 32.0:
#             # 在 small 尺度执行二次函数平滑过渡
#             t = (sqrt_areas[i] - 16.0) / 16.0
#             current_thr = thr_small - (thr_small - thr_large) * (t ** 2)
#         else:
#             current_thr = thr_large
#
#         # --- 抑制执行 ---
#         if not soft_mode:
#             # 硬删除模式
#             mask = sim <= current_thr
#             order = others[mask]
#         else:
#          """
#         软抑制模式 (Soft-SAfit)
#         根据尺度微调抑制强度 sigma
#         小目标对位置偏移敏感，惩罚应稍微温和点 (sigma 稍大)
#         """
#         sigma = sigma_base if w_i > 0.5 else (sigma_base * 0.8)
#         penalty = torch.exp(-(sim ** 2) / sigma)
#
#         curr_scores[others] *= penalty
#
#         # 剔除得分过低的框
#         mask = curr_scores[others] > score_thr
#         new_others = others[mask]
#
#         # Soft-NMS 需要重新对剩余框按得分排序
#         if new_others.numel() > 0:
#             sub_order = curr_scores[new_others].argsort(descending=True)
#             order = new_others[sub_order]
#         else:
#             break
#     return torch.tensor(keep, dtype=torch.long, device=pred_boxes.device)

# def compute_nwd_matrix_single(current_box, rest_boxes, C_nwd_base=20.0):
#     """
#     计算当前最高分框与剩余所有框的 NWD 相似度
#     current_box: (1, 4) [x1, y1, x2, y2]
#     rest_boxes: (M, 4) [x1, y1, x2, y2]
#     """
#     M = rest_boxes.shape[0]
#     if M == 0:
#         return rest_boxes.new_zeros((1, 0))
#     # 计算中心点
#     cx_i = (current_box[:, 0] + current_box[:, 2]) / 2.0
#     cy_i = (current_box[:, 1] + current_box[:, 3]) / 2.0
#     cx_j = (rest_boxes[:, 0] + rest_boxes[:, 2]) / 2.0
#     cy_j = (rest_boxes[:, 1] + rest_boxes[:, 3]) / 2.0
#     # 计算宽高的一半 (sigma)
#     w_i, h_i = (current_box[:, 2] - current_box[:, 0] + 1.0), (current_box[:, 3] - current_box[:, 1] + 1.0)
#     w_j, h_j = (rest_boxes[:, 2] - rest_boxes[:, 0] + 1.0), (rest_boxes[:, 3] - rest_boxes[:, 1] + 1.0)
#
#     sigma_wi, sigma_hi = w_i / 2.0, h_i / 2.0
#     sigma_wj, sigma_hj = w_j / 2.0, h_j / 2.0
#
#     #动态调整C_nwd:如果当前目标很小，适当调大平滑参数以容忍定位误差
#     #这里的8.0是ex_tiny的边界
#     sqrt_area_i = torch.sqrt(w_i * h_i)
#     C_nwd = C_nwd_base * (1.5 if sqrt_area_i < 8.0 else 1.0)
#
#     # 2-Wasserstein 距离平方
#     center_dist_sq = (cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2
#     scale_dist_sq = (sigma_wi - sigma_wj) ** 2 + (sigma_hi - sigma_hj) ** 2
#
#     wasserstein_dist = torch.sqrt(torch.clamp(center_dist_sq + scale_dist_sq, min=0.0))
#     return torch.exp(-wasserstein_dist / C_nwd)
#
#
# def safit_nms(boxes, scores, iou_threshold=0.3, C=32.0, C_nwd_base=22.0):
#     """
#     SAfitNMS 最终修正版：Max相似度 + 几何等效动态阈值
#     Args:
#         boxes: (N, 4) [x1, y1, x2, y2]
#         scores: (N,)
#         iou_threshold: 基准 IoU 阈值 (如 0.3)
#         C: 尺度自适应切换中心 (建议 32)
#         C_nwd: NWD 尺度敏感度 (建议 20.0 以增加对小目标位移的容忍度)
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#     # 1. 预计算所有框的尺度信息
#     w = boxes[:, 2] - boxes[:, 0] + 1.0
#     h = boxes[:, 3] - boxes[:, 1] + 1.0
#     areas = w * h
#     sqrt_areas = torch.sqrt(torch.clamp(areas, min=1e-6))
#
#     # 2. 初始排序
#     curr_scores = scores.clone()
#     order = curr_scores.argsort(descending=True)
#     keep = []
#
#     # 增加宽容度因子 gamma (针对极小目标)
#     # gamma = 0.8 意味着我们只要求它达到理论重叠度的 80%即可判定为冗余
#     gamma_tiny = 0.85
#
#     # 常数 k: 与 IoU=0.3 等效的中心偏移比例 (1 - sqrt(0.3))
#     k = 1.0 - math.sqrt(iou_threshold)  # 约 0.4523
#     while order.numel() > 0:
#         i = order[0]
#         keep.append(i.item())
#         if order.numel() == 1:
#             break
#         indices = order[1:]
#
#         # --- 步骤 1:
#         # 采用 max 逻辑：只要有一种度量认为重叠严重，就考虑抑制
#         ious = ops.box_iou(boxes[i:i + 1], boxes[indices]).squeeze(0)
#         nwds = compute_nwd_matrix_single(boxes[i:i + 1], boxes[indices], C_nwd_base=C_nwd_base).squeeze(0)
#         sim = torch.max(ious, nwds)
#         # --- 步骤 2: 动态阈值融合 ---
#         # 基于当前最高分框 i 的尺度决定“尺子”的严苛程度
#
#         # 核心改进：带修正因子的动态阈值
#         #如果是极小目标，阈值应该设得更高（更难被抑制）
#         if sqrt_areas[i] < 12.0:  # 覆盖 ex_tiny 和部分 tiny
#             k *= gamma_tiny
#         # 计算该尺度下 NWD 的等效阈值
#         # sqrt_area 越小，nwd_thr 越高（例如 4x4 目标阈值可能在 0.5 以上）
#         nwd_thr = torch.exp(-k * sqrt_areas[i] / C_nwd_base)
#
#         # 计算当前框的尺度权重 (Sigmoid 软切换)
#         w_i = torch.sigmoid((sqrt_areas[i] - C) / C)
#
#         # 融合阈值：大目标倾向 iou_threshold，小目标倾向 nwd_thr
#         combined_thr = w_i * iou_threshold + (1.0 - w_i) * nwd_thr
#         # --- 步骤 3: 抑制 ---
#         mask = sim <= combined_thr
#         order = indices[mask]
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# def compute_nwd_single(current_box, rest_boxes, C_nwd=32.0):
#     """
#     计算 NWD 相似度矩阵
#     """
#     if rest_boxes.shape[0] == 0:
#         return rest_boxes.new_zeros((0,))
#     # 中心点
#     cx_i = (current_box[:, 0] + current_box[:, 2]) / 2.0
#     cy_i = (current_box[:, 1] + current_box[:, 3]) / 2.0
#     cx_j = (rest_boxes[:, 0] + rest_boxes[:, 2]) / 2.0
#     cy_j = (rest_boxes[:, 1] + rest_boxes[:, 3]) / 2.0
#     # 尺度项 (sigma)
#     wi, hi = (current_box[:, 2] - current_box[:, 0] + 1.0), (current_box[:, 3] - current_box[:, 1] + 1.0)
#     wj, hj = (rest_boxes[:, 2] - rest_boxes[:, 0] + 1.0), (rest_boxes[:, 3] - rest_boxes[:, 1] + 1.0)
#
#     # 2-Wasserstein 距离
#     dist_sq = (cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2 + \
#               ((wi - wj) / 2.0) ** 2 + ((hi - hj) / 2.0) ** 2
#
#     return torch.exp(-torch.sqrt(torch.clamp(dist_sq, min=0.0)) / C_nwd)
#
#
# def safit_nms(boxes, scores, iou_threshold=0.3, C=32.0, C_nwd=32.0):
#     """
#     对齐 SAfit 评估体系的 NMS
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#     # 1. 预计算尺度权重
#     # 使用预测框面积模拟 GT 面积逻辑
#     # 1. 预计算并对尺度进行轻微平滑，降低 Pred 面积抖动的影响
#     w = boxes[:, 2] - boxes[:, 0] + 1.0
#     h = boxes[:, 3] - boxes[:, 1] + 1.0
#     sqrt_areas = torch.sqrt(torch.clamp(w * h, min=0.1))
#
#     #排序
#     curr_scores = scores.clone()
#     order = curr_scores.argsort(descending=True)
#     keep = []
#
#     # 几何常数 k
#     k_base = 1.0 - math.sqrt(iou_threshold)
#     while order.numel() > 0:
#         i = order[0]
#         keep.append(i.item())
#         if order.numel() == 1: break
#         indices = order[1:]
#
#         # 2. 计算 IoU 和 NWD
#         ious = ops.box_iou(boxes[i:i+1], boxes[indices]).squeeze(0)
#         nwds = compute_nwd_single(boxes[i:i+1], boxes[indices],C_nwd=C_nwd)
#
#         # 策略：取 max逻辑，确保不同尺度的相似度都能被捕捉
#         sim = torch.max(ious, nwds)
#
#         # 3. 核心改进：非线性宽容增益 (Recall Guard)
#         w_i = torch.sigmoid((sqrt_areas[i] - C) / C)
#
#         # 计算该尺度下的 NWD 等效阈值
#         nwd_thr = torch.exp(-k_base * sqrt_areas[i] / C_nwd)
#
#         # 【关键】增益因子：目标越小，增益越高。
#         # 给 ex_tiny 额外的 0.2宽容偏置，因为评估函数对它们的中心偏移极其宽容
#         # 即使 sim 达到 0.7，只要 sqrt_area 小，我们也尽量保留
#         gain = 0.20 * (1.0 - w_i)
#
#         # 融合阈值：小目标阈值将接近 nwd_thr + 0.2 (约为 0.7 左右)
#         combined_thr = w_i * iou_threshold + (1.0 - w_i) * (nwd_thr + gain)
#
#         # 限制上限，防止完全不抑制
#         combined_thr = min(0.80, combined_thr)
#
#
#
#
#         mask = sim <= combined_thr
#         order = indices[mask]
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)

#wr20260503
import torch
from torchvision import ops


# def safit_nms(pred_boxes, pred_scores, iou_threshold=0.3, nwd_threshold=0.4, C=16.0, C_nwd=12.0):
#     """
#     C: 尺度切换的中心点。当 sqrt(area) == C 时，IoU 和 NWD 权重各占 50%。
#     C_nwd: NWD 内部的高斯核宽度，控制对位置偏差的敏感度。
#     nwd_threshold: 针对小目标的抑制阈值，通常由于 NWD 值域较宽，该值应略高于 iou_threshold。
#     """
#     N = pred_boxes.shape[0]
#     if N == 0: return torch.tensor([], dtype=torch.long, device=pred_boxes.device)
#
#     # 1. 基础属性计算
#     w = pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0
#     h = pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0
#     sqrt_areas = torch.sqrt(w * h)
#
#     # 2. 预计算相似度矩阵
#     iou_matrix = ops.box_iou(pred_boxes, pred_boxes)
#     nwd_matrix = compute_nwd_matrix(pred_boxes, C_nwd=C_nwd)
#
#     # 3. 尺度权重计算 (重新引入参数 C)
#     # 使用 C/2 作为平滑因子，确保在 C 附近的过渡是平滑的
#     node_weights = torch.sigmoid((sqrt_areas - C) / (C / 2.0))
#     # 采用最小权重主导，确保小目标受 NWD 保护
#     w_pair = torch.minimum(node_weights[:, None], node_weights[None, :])
#
#     safit_matrix = w_pair * iou_matrix + (1.0 - w_pair) * nwd_matrix
#     safit_matrix.fill_diagonal_(0.0)
#
#     # ---------- NMS 循环 ----------
#     order = pred_scores.argsort(descending=True)
#     keep = []
#
#     while order.numel() > 0:
#         i = order[0]
#         keep.append(i.item())
#         if order.numel() == 1: break
#
#         sim = safit_matrix[i][order[1:]]
#
#         # 4. 动态阈值逻辑优化
#         # 既然 APex_tiny 下降是因为过度抑制，我们针对性地放宽小目标的 dynamic_thr
#         current_sqrt = sqrt_areas[i]
#
#         # 阈值随尺度动态演变：小目标偏向 nwd_threshold，大目标偏向 iou_threshold
#         # 权重 factor 反映当前框的尺度倾向
#         factor = torch.sigmoid((current_sqrt - C) / (C / 2.0))
#         dynamic_thr = factor * iou_threshold + (1.0 - factor) * nwd_threshold
#
#         mask = sim <= dynamic_thr
#         order = order[1:][mask]
#
#     return torch.tensor(keep, dtype=torch.long, device=pred_boxes.device)

# #wr20260506-1
# def safit_nms(pred_boxes, pred_scores, iou_threshold=0.3, nwd_threshold=0.45, C=12.0):
#     """
#     V3 优化版：
#     1. 最小阈值保护：如果参与对比的两个框中有小目标，采用宽松阈值。
#     2. 平滑因子调整：让 NWD 到 IoU 的过渡更丝滑。
#     3. C 值微调：更聚焦于 ex_tiny 区间。
#     """
#     N = pred_boxes.shape[0]
#     if N == 0: return torch.tensor([], dtype=torch.long, device=pred_boxes.device)
#
#     w = pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0
#     h = pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0
#     sqrt_areas = torch.sqrt(w * h)
#
#     iou_matrix = ops.box_iou(pred_boxes, pred_boxes)
#     # C_nwd 设为 12，保持对微小位移的敏感
#     nwd_matrix = compute_nwd_matrix(pred_boxes, C_nwd=12.0)
#
#     # 计算每个框的尺度倾向属性 [0, 1]
#     # 增大分母(从 C/2 变为 C)，使过渡变平缓，防止权重跳变
#     node_weights = torch.sigmoid((sqrt_areas - C) / C)
#
#     # 相似度矩阵：最小面积权重主导
#     w_pair = torch.minimum(node_weights[:, None], node_weights[None, :])
#     safit_matrix = w_pair * iou_matrix + (1.0 - w_pair) * nwd_matrix
#     safit_matrix.fill_diagonal_(0.0)
#
#     # 预计算每个框对应的自适应阈值
#     # 尺度越小，阈值越趋向于 nwd_threshold (宽松)
#     node_thrs = node_weights * iou_threshold + (1.0 - node_weights) * nwd_threshold
#
#     order = pred_scores.argsort(descending=True)
#     keep = []
#
#     while order.numel() > 0:
#         i = order[0]
#         keep.append(i.item())
#         if order.numel() == 1: break
#
#         sim = safit_matrix[i][order[1:]]
#
#         # 核心优化：双向阈值保护
#         # 抑制阈值取当前高分框和待处理框中“最宽松”的那一个
#         pair_thrs = torch.maximum(node_thrs[i], node_thrs[order[1:]])
#
#         mask = sim <= pair_thrs
#         order = order[1:][mask]
#
#     return torch.tensor(keep, dtype=torch.long, device=pred_boxes.device)

#wr20260506-2
#
# def safit_nms(pred_boxes, pred_scores, iou_threshold=0.3, nwd_threshold=0.48, C=16.0):
#     """
#     V4 最终优化版：Soft-SAfit
#     1. 引入 Soft-NMS 线性衰减逻辑。
#     2. 进一步放宽 nwd_threshold 至 0.48 保护召回率。
#     3. 纯化极小目标的指标成分。
#     """
#     if pred_boxes.numel() == 0:
#         return torch.tensor([], dtype=torch.long, device=pred_boxes.device)
#
#     w = pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0
#     h = pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0
#     sqrt_areas = torch.sqrt(w * h)
#
#     # 预计算相似度矩阵
#     iou_matrix = ops.box_iou(pred_boxes, pred_boxes)
#     # C_nwd 保持 12.0，确保对极小目标的中心偏移敏感
#     nwd_matrix = compute_nwd_matrix(pred_boxes, C_nwd=12.0)
#
#     # 1. 尺度权重计算 (收紧缩放因子至 6.0，加快极小目标的纯化)
#     node_weights = torch.sigmoid((sqrt_areas - C) / 6.0)
#     w_pair = torch.minimum(node_weights[:, None], node_weights[None, :])
#     safit_matrix = w_pair * iou_matrix + (1.0 - w_pair) * nwd_matrix
#
#     # 2. 动态阈值预计算
#     node_thrs = node_weights * iou_threshold + (1.0 - node_weights) * nwd_threshold
#
#     scores = pred_scores.clone()
#     keep = []
#     # 只要分数值大于极小阈值就继续
#     score_keep_threshold = 0.001
#
#     # 记录哪些框已经被处理过（不再作为最高分框选取）
#     # 使用一个 mask 或者直接将已选框分数置为 -1
#
#     for _ in range(scores.numel()):
#         # 1. 寻找当前剩余框中分数最高的
#         max_idx = torch.argmax(scores)
#         max_val = scores[max_idx]
#
#         if max_val < score_keep_threshold:
#             break
#
#         # 2. 记录该框
#         keep.append(max_idx.item())
#
#         # 3. 提取当前框与其他所有框的相似度 (整行提取，无需 cat)
#         sims = safit_matrix[max_idx]
#
#         # 4. 获取双向保护阈值 (整行提取)
#         pair_thrs = torch.maximum(node_thrs[max_idx], node_thrs)
#
#         # 5. 计算衰减掩码
#         # 排除自身（防止自己衰减自己）以及分数已经很低的框
#         decay_mask = (sims > pair_thrs)
#
#         # 6. 线性衰减：分数 = 分数 * (1 - 相似度)
#         scores[decay_mask] *= (1.0 - sims[decay_mask])
#
#         # 7. 关键：将当前处理完的最高分框分数置为 -1，确保下次不会再选到它
#         # 这样就不需要对矩阵和张量执行 torch.cat 了
#         scores[max_idx] = -1.0
#
#     return torch.tensor(keep, dtype=torch.long, device=pred_boxes.device)

# #wr20260506-3
# def safit_nms(pred_boxes, pred_scores, iou_threshold=0.3, nwd_threshold=0.45, C=16.0):
#     """
#     V5 混合优化版：
#     1. 中大目标回归 Hard-NMS 保证 Precision。
#     2. 小目标采用两段式高斯衰减：极高相似度硬删，中等相似度软衰减。
#     """
#     if pred_boxes.numel() == 0:
#         return torch.tensor([], dtype=torch.long, device=pred_boxes.device)
#
#     w = pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0
#     h = pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0
#     sqrt_areas = torch.sqrt(w * h)
#
#     iou_matrix = ops.box_iou(pred_boxes, pred_boxes)
#     nwd_matrix = compute_nwd_matrix(pred_boxes, C_nwd=12.0)
#
#     # 尺度权重与相似度矩阵
#     node_weights = torch.sigmoid((sqrt_areas - C) / 6.0)
#     w_pair = torch.minimum(node_weights[:, None], node_weights[None, :])
#     safit_matrix = w_pair * iou_matrix + (1.0 - w_pair) * nwd_matrix
#
#     # 动态阈值
#     node_thrs = node_weights * iou_threshold + (1.0 - node_weights) * nwd_threshold
#
#     scores = pred_scores.clone()
#     keep = []
#     # 增加过滤阈值，过滤掉衰减后变得极弱的框
#     score_keep_threshold = 0.05
#     # 高斯衰减方差
#     sigma = 0.5
#
#     for _ in range(scores.numel()):
#         max_idx = torch.argmax(scores)
#         max_val = scores[max_idx]
#
#         if max_val < score_keep_threshold:
#             break
#
#         keep.append(max_idx.item())
#
#         sims = safit_matrix[max_idx]
#         pair_thrs = torch.maximum(node_thrs[max_idx], node_thrs)
#
#         # --- 混合抑制逻辑 ---
#         # 1. 找到需要抑制的框
#         suppress_mask = sims > pair_thrs
#
#         # 2. 区分尺度逻辑
#         # 如果当前高分框是中大目标 (node_weight > 0.5)，采用 Hard 模式
#         if node_weights[max_idx] > 0.5:
#             scores[suppress_mask] = 0  # 直接杀死
#         else:
#             # 针对小目标，采用两段式
#             # 极高相似度 (e.g. > 0.85) 的小框依然硬删
#             hard_kill_mask = suppress_mask & (sims > 0.85)
#             soft_decay_mask = suppress_mask & (~hard_kill_mask)
#
#             scores[hard_kill_mask] = 0
#             # 高斯衰减比线性衰减抑制力更强
#             scores[soft_decay_mask] *= torch.exp(-(sims[soft_decay_mask] ** 2) / sigma)
#
#         # 标记已选
#         scores[max_idx] = -1.0
#
#     return torch.tensor(keep, dtype=torch.long, device=pred_boxes.device)

# #wr20260506-4
# def safit_nms(pred_boxes, pred_scores, iou_threshold=0.3, nwd_threshold=0.48, C=16.0):
#     """
#     V7 尺度隔离版：
#     1. 引入跨尺度不抑制逻辑：尺寸差异巨大的框互不干扰。
#     2. 尺度内采用精细化抑制。
#     """
#     if pred_boxes.numel() == 0:
#         return torch.tensor([], dtype=torch.long, device=pred_boxes.device)
#
#     w = pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0
#     h = pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0
#     sqrt_areas = torch.sqrt(w * h)
#
#     # 预计算
#     iou_matrix = ops.box_iou(pred_boxes, pred_boxes)
#     nwd_matrix = compute_nwd_matrix(pred_boxes, C_nwd=12.0)
#
#     # 尺度权重
#     node_weights = torch.sigmoid((sqrt_areas - C) / 6.0)
#     w_pair = torch.minimum(node_weights[:, None], node_weights[None, :])
#     safit_matrix = w_pair * iou_matrix + (1.0 - w_pair) * nwd_matrix
#
#     # 尺度差异矩阵：计算所有框对之间的边长比值
#     # ratio > 1.0，表示大框与小框的比例
#     area_ratios = sqrt_areas[:, None] / (sqrt_areas[None, :] + 1e-6)
#     area_ratios = torch.max(area_ratios, 1.0 / (area_ratios + 1e-6))
#
#     # --- 核心改进：尺度隔离 ---
#     # 如果两个框尺度差异超过 3 倍（可调），相似度直接清零，不相互抑制
#     isolation_mask = area_ratios > 3.0
#     safit_matrix[isolation_mask] = 0.0
#
#     scores = pred_scores.clone()
#     keep = []
#
#     # 既然尺度已经隔离，我们可以把阈值稍微收紧，提升 Precision
#     node_thrs = node_weights * iou_threshold + (1.0 - node_weights) * nwd_threshold
#
#     for _ in range(scores.numel()):
#         max_idx = torch.argmax(scores)
#         max_val = scores[max_idx]
#         if max_val < 0.01: break
#
#         keep.append(max_idx.item())
#
#         sims = safit_matrix[max_idx]
#         pair_thrs = torch.maximum(node_thrs[max_idx], node_thrs)
#
#         # 此时的 sims 已经过隔离处理，跨尺度的 sim 均为 0
#         suppress_mask = sims > pair_thrs
#
#         # 尺度内仍然建议使用高斯衰减而非硬删，以保住召回率
#         if suppress_mask.any():
#             # 仅对相似度高的同尺度目标进行衰减
#             sigma = 0.5
#             scores[suppress_mask] *= torch.exp(-(sims[suppress_mask] ** 2) / sigma)
#
#         scores[max_idx] = -1.0
#
#     return torch.tensor(keep, dtype=torch.long, device=pred_boxes.device)

#wr20260506-5
# def safit_nms(pred_boxes, pred_scores, iou_threshold=0.3, nwd_threshold=0.45, C=16.0):
#     """
#     V8 尺度感知加权版：
#     1. 动态衰减：相似度受尺度比值影响，尺度差异越大，抑制力越弱，但不为零。
#     2. 混合阈值：综合考虑尺度、NWD 和 IoU。
#     """
#     if pred_boxes.numel() == 0:
#         return torch.tensor([], dtype=torch.long, device=pred_boxes.device)
#
#     w = pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0
#     h = pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0
#     sqrt_areas = torch.sqrt(w * h)
#
#     iou_matrix = ops.box_iou(pred_boxes, pred_boxes)
#     nwd_matrix = compute_nwd_matrix(pred_boxes, C_nwd=12.0)
#
#     # 计算尺度权重
#     node_weights = torch.sigmoid((sqrt_areas - C) / 6.0)
#     w_pair = torch.minimum(node_weights[:, None], node_weights[None, :])
#     safit_matrix = w_pair * iou_matrix + (1.0 - w_pair) * nwd_matrix
#
#     # --- 改进：引入尺度差异权重因子 ---
#     # ratio 越大（尺度差异大），penalty 越小
#     area_ratios = sqrt_areas[:, None] / (sqrt_areas[None, :] + 1e-6)
#     area_ratios = torch.max(area_ratios, 1.0 / (area_ratios + 1e-6))
#     # 使用指数衰减降低跨尺度抑制强度：ratio=1时权重为1，ratio=3时权重约0.3
#     scale_penalty_weight = torch.exp(-(area_ratios - 1.0) / 2.0)
#
#     # 修正相似度：跨尺度的相似度被“打折”，但不直接清零
#     safit_matrix = safit_matrix * scale_penalty_weight
#
#     scores = pred_scores.clone()
#     keep = []
#
#     # 动态阈值预计算
#     node_thrs = node_weights * iou_threshold + (1.0 - node_weights) * nwd_threshold
#
#     for _ in range(scores.numel()):
#         max_idx = torch.argmax(scores)
#         max_val = scores[max_idx]
#         if max_val < 0.05: break  # 提高过滤阈值，清理低分误检
#
#         keep.append(max_idx.item())
#
#         sims = safit_matrix[max_idx]
#         pair_thrs = torch.maximum(node_thrs[max_idx], node_thrs)
#
#         suppress_mask = sims > pair_thrs
#
#         if suppress_mask.any():
#             # 回归 Hard-NMS 以确保 Precision，仅对极小目标保留微小的 Soft 空间
#             if node_weights[max_idx] < 0.5:  # 仅当高分框是小目标时使用衰减
#                 scores[suppress_mask] *= (1.0 - sims[suppress_mask])
#             else:
#                 scores[suppress_mask] = 0  # 大目标回归硬删，清理冗余
#
#         scores[max_idx] = -1.0
#
#     return torch.tensor(keep, dtype=torch.long, device=pred_boxes.device)



# #20260507-1
# def safit_nms(pred_boxes, pred_scores, iou_threshold=0.3, nwd_threshold=0.45, C=32.0):
#     """
#     修正后的 V8 尺度感知 NMS：
#     1. 严格对齐评估指标的超参数（C=32）。
#     2. 采用"伪 GT"逻辑：融合权重由当前得分最高的抑制框决定，而非双向 minimum。
#     3. 移除错误的跨尺度打折惩罚。
#     """
#     if pred_boxes.numel() == 0:
#         return torch.tensor([], dtype=torch.long, device=pred_boxes.device)
#
#     w = pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0
#     # 注意: 某些 bbox 格式下无需 +1.0，请与你的 eval 代码 xyhw2xxyy 对齐
#     h = pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0
#     sqrt_areas = torch.sqrt(w * h)
#
#     # 1. 基础矩阵计算
#     iou_matrix = ops.box_iou(pred_boxes, pred_boxes)
#     # C_nwd 必须和评估代码中的 C 对齐（因为评估代码里 NWD 传入的是 C=32）
#     nwd_matrix = compute_nwd_matrix_optimized(pred_boxes, C_nwd=C)
#
#     # 2. 计算节点尺度权重 (对齐 Eval 中的 Sigmoid 斜率 1/C)
#     # Eval: 1/(1+np.exp(-(1/C) * (sqrt(Area)-C)))
#     node_weights = torch.sigmoid((sqrt_areas - C) / C)
#
#     scores = pred_scores.clone()
#     keep = []
#
#     for _ in range(scores.numel()):
#         max_idx = torch.argmax(scores)
#         max_val = scores[max_idx]
#
#         # 提高过滤阈值，清理低分误检（保持你的原始设定，但建议测试时先设为极低值如0.001对齐标准评测）
#         if max_val < 0.05:
#             break
#
#         keep.append(max_idx.item())
#
#         # 3. 核心修正："伪 GT" 定律
#         # 当前 max_idx 就相当于评估代码中的 GT。
#         # 因此，这单趟循环的权重由 max_idx 的尺度单方面决定！
#         w_max = node_weights[max_idx]
#
#         # 融合这一行的相似度
#         sims = w_max * iou_matrix[max_idx] + (1.0 - w_max) * nwd_matrix[max_idx]
#
#         # 融合阈值也由当前 max_idx 决定
#         current_thr = w_max * iou_threshold + (1.0 - w_max) * nwd_threshold
#
#         suppress_mask = sims > current_thr
#
#         if suppress_mask.any():
#             # 暂时回归标准的 Hard-NMS，为了确保基线正确，建议先不用 Soft-NMS 逻辑
#             # 等 Hard-NMS 涨点了，再去加 Soft 逻辑
#             scores[suppress_mask] = 0.0
#
#         scores[max_idx] = -1.0
#
#     return torch.tensor(keep, dtype=torch.long, device=pred_boxes.device)
#
#
# def compute_nwd_matrix_optimized(pred_boxes, C_nwd=32.0):
#     cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
#     cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
#     w = pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0
#     h = pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0
#
#     center_dist_sq = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
#     # 尺度距离 (严格按你的 Eval 代码计算)
#     scale_dist_sq = ((w[:, None] - w[None, :]) / 2.0) ** 2 + ((h[:, None] - h[None, :]) / 2.0) ** 2
#
#     wasserstein_dist = torch.sqrt(torch.clamp(center_dist_sq + scale_dist_sq, min=0.0))
#     # 对齐 Eval 代码的高斯核：np.exp(-dist / C)
#     return torch.exp(-wasserstein_dist / C_nwd)

#wr20260508-1


# def safit_nms(pred_boxes, pred_scores, iou_threshold=0.3, nwd_threshold=0.85, C=32.0):
#     """
#     Safit NMS V3 (超越基线版)
#     改进点：
#     1. 移除 +1.0 的浮点数计算误差。
#     2. NWD 阈值大幅提升至 0.85，对齐 IoU=0.3 的实际抑制物理距离，防止密集小目标被错杀。
#     3. 移除 0.05 强截断，保护召回率 (AR)。
#     4. 使用标准的 argsort 向量化掩码操作，提升执行稳定性和速度。
#     """
#     if pred_boxes.numel() == 0:
#         return torch.tensor([], dtype=torch.long, device=pred_boxes.device)
#
#     # 移除 +1.0，直接计算浮点宽和高
#     w = pred_boxes[:, 2] - pred_boxes[:, 0]
#     h = pred_boxes[:, 3] - pred_boxes[:, 1]
#
#     # 防止极小值导致的 nan
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#     sqrt_areas = torch.sqrt(w * h)
#
#     # 计算节点尺度权重 (对齐 Eval)
#     node_weights = torch.sigmoid((sqrt_areas - C) / C)
#
#     # 预计算相似度矩阵 (仅计算一次，避免循环内重复开销)
#     iou_matrix = ops.box_iou(pred_boxes, pred_boxes)
#     nwd_matrix = compute_nwd_matrix_v3(pred_boxes, C_nwd=C)
#
#     # 按置信度降序排序
#     order = torch.argsort(pred_scores, descending=True)
#     keep = []
#
#     while order.numel() > 0:
#         # 获取当前最高分框的索引（老大哥/伪GT）
#         max_idx = order[0]
#         keep.append(max_idx.item())
#
#         if order.numel() == 1:
#             break
#
#         # 获取剩余框的索引
#         rest_idx = order[1:]
#
#         # "伪 GT 定律": 权重完全由当前 max_idx (老大哥) 决定
#         w_max = node_weights[max_idx]
#
#         # 提取当前最高分框与剩余所有框的相似度
#         ious = iou_matrix[max_idx, rest_idx]
#         nwds = nwd_matrix[max_idx, rest_idx]
#
#         # 动态融合相似度
#         sims = w_max * ious + (1.0 - w_max) * nwds
#
#         # 动态融合阈值
#         current_thr = w_max * iou_threshold + (1.0 - w_max) * nwd_threshold
#
#         # 找到需要保留的框 (相似度 <= 阈值的保留)
#         keep_mask = sims <= current_thr
#
#         # 更新 order，进入下一轮迭代
#         order = rest_idx[keep_mask]
#
#     return torch.tensor(keep, dtype=torch.long, device=pred_boxes.device)
# #
# #
# def compute_nwd_matrix_v3(pred_boxes, C_nwd=32.0):
#     # 同样移除 +1.0
#     cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
#     cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
#     w = pred_boxes[:, 2] - pred_boxes[:, 0]
#     h = pred_boxes[:, 3] - pred_boxes[:, 1]
#
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#
#     center_dist_sq = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
#     scale_dist_sq = ((w[:, None] - w[None, :]) / 2.0) ** 2 + ((h[:, None] - h[None, :]) / 2.0) ** 2
#
#     wasserstein_dist = torch.sqrt(torch.clamp(center_dist_sq + scale_dist_sq, min=0.0))
#     return torch.exp(-wasserstein_dist / C_nwd)
#



#wr20260508-2
# def compute_nwd_matrix(boxes, C_nwd=32.0):
#     """向量化计算全体 boxes 之间的 NWD 相似度矩阵"""
#     cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
#     cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#
#     # 避免除 0
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#
#     center_dist_sq = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
#     scale_dist_sq = ((w[:, None] - w[None, :]) / 2.0) ** 2 + ((h[:, None] - h[None, :]) / 2.0) ** 2
#
#     dist = torch.sqrt(torch.clamp(center_dist_sq + scale_dist_sq, min=0.0))
#     return torch.exp(-dist / C_nwd)
#
#
# def safit_nms(boxes, scores, C=32.0, sigma=0.5, min_score=0.01):
#     """
#     Safit-Soft-NMS: 融合评价体系与高斯平滑惩罚，全面保护 AR 与全尺度 AP。
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#
#     # 1. 计算对齐 GT 的尺度权重
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#     sqrt_areas = torch.sqrt(w * h)
#
#     # Eval 标准权重
#     weights = torch.sigmoid((sqrt_areas - C) / C)
#
#     # 2. 预计算完整的相似度矩阵（用空间换时间，避免在循环中重复计算）
#     iou_matrix = ops.box_iou(boxes, boxes)
#     nwd_matrix = compute_nwd_matrix(boxes, C_nwd=C)
#
#     # 3. 核心 Soft-NMS 循环
#     scores_cloned = scores.clone()
#     keep = []
#
#     # 存活框的索引集合
#     idxs = torch.arange(scores.numel(), device=scores.device)
#
#     while idxs.numel() > 0:
#         # 找出当前存活框中分数最高的
#         max_idx_local = torch.argmax(scores_cloned[idxs])
#         max_idx = idxs[max_idx_local]
#
#         # 如果最高分已经低于我们设定的极小阈值，直接结束（节省计算）
#         if scores_cloned[max_idx] < min_score:
#             break
#
#         keep.append(max_idx.item())
#
#         if idxs.numel() == 1:
#             break
#
#         # 从存活池中移除当前 max_idx
#         idxs = torch.cat([idxs[:max_idx_local], idxs[max_idx_local + 1:]])
#
#         # 获取 max_idx 对于其余存活框的 IoU 和 NWD
#         ious = iou_matrix[max_idx, idxs]
#         nwds = nwd_matrix[max_idx, idxs]
#
#         # 伪 GT 逻辑：权重完全由当前的 max_idx 决定
#         w_max = weights[max_idx]
#
#         # 计算 Safit 相似度
#         sims = w_max * ious + (1.0 - w_max) * nwds
#
#         # 高斯惩罚核心：相似度越高，分数衰减越厉害
#         # 相比于 Hard-NMS 的强硬删除，这里给所有框留了一线生机
#         decay = torch.exp(-(sims ** 2) / sigma)
#
#         # 更新剩余框的分数
#         scores_cloned[idxs] *= decay
#
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)


#wr20260508-3


# def compute_nwd_matrix(boxes, C_nwd=32.0):
#     cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
#     cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#
#     center_dist_sq = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
#     scale_dist_sq = ((w[:, None] - w[None, :]) / 2.0) ** 2 + ((h[:, None] - h[None, :]) / 2.0) ** 2
#
#     dist = torch.sqrt(torch.clamp(center_dist_sq + scale_dist_sq, min=0.0))
#     return torch.exp(-dist / C_nwd)
#
#
# def safit_nms(boxes, scores, iou_thr=0.3, nwd_thr=0.85, C_scale=32.0, min_score=0.05):
#     """
#     Scale-Decoupled NMS (SD-NMS) - 终极跨越版
#     1. 不再强行融合 IoU 和 NWD 分数，避免中大目标阈值失调。
#     2. 基于尺度条件使用双重验证逻辑，精准互补。
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#
#     # 预计算边长，用于判断是否为小目标
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#     sqrt_areas = torch.sqrt(torch.clamp(w * h, min=1e-3))
#
#     # 预计算矩阵
#     iou_matrix = ops.box_iou(boxes, boxes)
#     nwd_matrix = compute_nwd_matrix(boxes, C_nwd=C_scale)
#
#     scores_cloned = scores.clone()
#     keep = []
#
#     idxs = torch.arange(scores.numel(), device=scores.device)
#
#     while idxs.numel() > 0:
#         # 获取当前最高分
#         max_idx_local = torch.argmax(scores_cloned[idxs])
#         max_idx = idxs[max_idx_local]
#
#         # 过滤低分框，防止长尾噪声
#         if scores_cloned[max_idx] < min_score:
#             break
#
#         keep.append(max_idx.item())
#
#         if idxs.numel() == 1:
#             break
#
#         idxs = torch.cat([idxs[:max_idx_local], idxs[max_idx_local + 1:]])
#
#         ious = iou_matrix[max_idx, idxs]
#         nwds = nwd_matrix[max_idx, idxs]
#
#         # ==========================================
#         # 核心：尺度解耦抑制逻辑 (Scale-Decoupled)
#         # ==========================================
#
#         # 1. 全尺度保底：任何只要 IoU > 0.3 的，全部按基线逻辑抑制
#         suppress_mask = ious > iou_thr
#
#         # 2. 小目标增强：如果当前老大哥是小目标 (边长 <= 32)
#         # 即使它们 IoU 不高，但只要 NWD 极高 (中心点重合度极高)，也无情抑制
#         is_small_target = sqrt_areas[max_idx] <= C_scale
#         if is_small_target:
#             suppress_mask = suppress_mask | (nwds > nwd_thr)
#
#         # 剔除需要抑制的框，只保留 False 的索引
#         idxs = idxs[~suppress_mask]
#
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)


#wr20260509-1



# def compute_nwd_matrix(boxes, C_nwd=32.0):
#     cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
#     cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#
#     center_dist_sq = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
#     scale_dist_sq = ((w[:, None] - w[None, :]) / 2.0) ** 2 + ((h[:, None] - h[None, :]) / 2.0) ** 2
#
#     dist = torch.sqrt(torch.clamp(center_dist_sq + scale_dist_sq, min=0.0))
#     return torch.exp(-dist / C_nwd)
#
#
# def safit_nms(boxes, scores, gate_thr=0.3, sigma=0.1, C=32.0, min_score=0.001):
#     """
#     Gated Safit-Soft-NMS (带门槛的尺度感知平滑 NMS)
#     综合了所有的优化经验：
#     1. 严格对齐 Safit 评估的动态权重。
#     2. 引入 Gate 阈值，阻断 NWD 导致的全局分数磨损 (修复 V5)。
#     3. 采用高斯衰减代替硬删除，极大地保护密集场景下的 AR (修复 V6)。
#     4. 降低 min_score 截断，对齐 COCO 评测逻辑。
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#
#     # 1. 计算对齐 GT 的尺度权重
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#     sqrt_areas = torch.sqrt(w * h)
#     weights = torch.sigmoid((sqrt_areas - C) / C)
#
#     # 2. 预计算相似度矩阵
#     iou_matrix = ops.box_iou(boxes, boxes)
#     nwd_matrix = compute_nwd_matrix(boxes, C_nwd=C)
#
#     scores_cloned = scores.clone()
#     keep = []
#     idxs = torch.arange(scores.numel(), device=scores.device)
#
#     while idxs.numel() > 0:
#         max_idx_local = torch.argmax(scores_cloned[idxs])
#         max_idx = idxs[max_idx_local]
#
#         # 调低截断阈值以保护 PR 曲线的后半段召回
#         if scores_cloned[max_idx] < min_score:
#             break
#
#         keep.append(max_idx.item())
#         if idxs.numel() == 1:
#             break
#
#         idxs = torch.cat([idxs[:max_idx_local], idxs[max_idx_local + 1:]])
#
#         ious = iou_matrix[max_idx, idxs]
#         nwds = nwd_matrix[max_idx, idxs]
#
#         w_max = weights[max_idx]
#
#         # 精确对齐你的评估代码 Safit 相似度
#         sims = w_max * ious + (1.0 - w_max) * nwds
#
#         # ==========================================
#         # 核心：Gated Soft-NMS (门槛式衰减)
#         # ==========================================
#         # 只有相似度大于阈值的框，才会被惩罚！彻底解决全局雪崩问题。
#         penalty_mask = sims > gate_thr
#
#         if penalty_mask.any():
#             # 仅对超过阈值的嫌疑框进行高斯衰减
#             penalized_idxs = idxs[penalty_mask]
#
#             # 衰减公式：相似度越高，衰减越厉害
#             decay = torch.exp(-(sims[penalty_mask] ** 2) / sigma)
#
#             # 更新嫌疑框的分数
#             scores_cloned[penalized_idxs] *= decay
#
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# #wr20260509-2
# def compute_nwd_matrix(boxes, C_nwd=32.0):
#     cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
#     cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#
#     center_dist_sq = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
#     scale_dist_sq = ((w[:, None] - w[None, :]) / 2.0) ** 2 + ((h[:, None] - h[None, :]) / 2.0) ** 2
#
#     dist = torch.sqrt(torch.clamp(center_dist_sq + scale_dist_sq, min=0.0))
#     return torch.exp(-dist / C_nwd)
#
#
# def safit_nms(boxes, scores, hard_thr=0.75, soft_thr=0.3, sigma=0.1, C=32.0, min_score=0.01):
#     """
#     V8: Hybrid Safit NMS (Hard + Soft 混合制)
#     1. soft_thr (0.3): 对齐基线。高于 0.3 开始高斯惩罚。
#     2. hard_thr (0.75): 绝对冗余线。高于 0.75 的直接置 0 斩首，防止垃圾框挤占 Top-K。
#     3. 调低 sigma (0.1): 让惩罚更加剧烈，加速垃圾框沉底。
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#     sqrt_areas = torch.sqrt(w * h)
#     weights = torch.sigmoid((sqrt_areas - C) / C)
#
#     iou_matrix = ops.box_iou(boxes, boxes)
#     nwd_matrix = compute_nwd_matrix(boxes, C_nwd=C)
#
#     scores_cloned = scores.clone()
#     keep = []
#     idxs = torch.arange(scores.numel(), device=scores.device)
#
#     while idxs.numel() > 0:
#         max_idx_local = torch.argmax(scores_cloned[idxs])
#         max_idx = idxs[max_idx_local]
#
#         # 适当提高一点 min_score，防止 NMS 算太久且放过太多噪声
#         if scores_cloned[max_idx] < min_score:
#             break
#
#         keep.append(max_idx.item())
#         if idxs.numel() == 1:
#             break
#
#         idxs = torch.cat([idxs[:max_idx_local], idxs[max_idx_local + 1:]])
#
#         ious = iou_matrix[max_idx, idxs]
#         nwds = nwd_matrix[max_idx, idxs]
#         w_max = weights[max_idx]
#
#         sims = w_max * ious + (1.0 - w_max) * nwds
#
#         # 1. Hard Drop 阶段：极度相似的框直接删掉 (剔除索引)
#         hard_mask = sims <= hard_thr
#         idxs = idxs[hard_mask]
#         sims = sims[hard_mask]  # 同步缩小 sims
#
#         if idxs.numel() == 0:
#             continue
#
#         # 2. Soft Penalty 阶段：中度相似的框进行重度扣分
#         soft_mask = sims > soft_thr
#         if soft_mask.any():
#             penalized_idxs = idxs[soft_mask]
#             # sigma=0.1 会让惩罚变得非常严厉
#             decay = torch.exp(-(sims[soft_mask] ** 2) / sigma)
#             scores_cloned[penalized_idxs] *= decay
#
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)

#wr20260509-3



# def compute_nwd_matrix(boxes, C_nwd=32.0):
#     cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
#     cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#
#     center_dist_sq = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
#     scale_dist_sq = ((w[:, None] - w[None, :]) / 2.0) ** 2 + ((h[:, None] - h[None, :]) / 2.0) ** 2
#
#     dist = torch.sqrt(torch.clamp(center_dist_sq + scale_dist_sq, min=0.0))
#     return torch.exp(-dist / C_nwd)
#
#
# def safit_nms(boxes, scores, iou_thr=0.3, nwd_thr=0.85, sigma=0.2, C=32.0, min_score=0.001):
#     """
#     V9: Dynamic-Gated Safit Soft-NMS (终极跨越版)
#     核心逻辑：既然相似度 (sims) 是动态加权的，那么判定是否惩罚的门槛 (dynamic_gate) 也必须动态加权！
#     彻底解决“大目标放水、小目标错杀”的阈值错配悖论。
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#     sqrt_areas = torch.sqrt(w * h)
#
#     # 尺度权重
#     weights = torch.sigmoid((sqrt_areas - C) / C)
#
#     iou_matrix = ops.box_iou(boxes, boxes)
#     nwd_matrix = compute_nwd_matrix(boxes, C_nwd=C)
#
#     scores_cloned = scores.clone()
#     keep = []
#     idxs = torch.arange(scores.numel(), device=scores.device)
#
#     while idxs.numel() > 0:
#         max_idx_local = torch.argmax(scores_cloned[idxs])
#         max_idx = idxs[max_idx_local]
#
#         if scores_cloned[max_idx] < min_score:
#             break
#
#         keep.append(max_idx.item())
#         if idxs.numel() == 1:
#             break
#
#         idxs = torch.cat([idxs[:max_idx_local], idxs[max_idx_local + 1:]])
#
#         ious = iou_matrix[max_idx, idxs]
#         nwds = nwd_matrix[max_idx, idxs]
#
#         w_max = weights[max_idx]
#
#         # 1. 动态融合的相似度
#         sims = w_max * ious + (1.0 - w_max) * nwds
#
#         # 2. 核心魔法：动态融合的惩罚门槛！
#         # 大目标逼近 0.3，小目标逼近 0.85
#         dynamic_gate = w_max * iou_thr + (1.0 - w_max) * nwd_thr
#
#         # 3. 只有相似度越过了它自己“专属的门槛”，才会被认定为冗余并惩罚
#         penalty_mask = sims > dynamic_gate
#
#         if penalty_mask.any():
#             penalized_idxs = idxs[penalty_mask]
#
#             # 使用高斯平滑衰减，sigma 适中 (0.2)，保证足够的惩罚力度又不会误杀密集目标
#             decay = torch.exp(-(sims[penalty_mask] ** 2) / sigma)
#             scores_cloned[penalized_idxs] *= decay
#
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)


#wr20260509-4



# def compute_nwd_matrix(boxes, C_nwd=32.0):
#     cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
#     cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#
#     center_dist_sq = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
#     scale_dist_sq = ((w[:, None] - w[None, :]) / 2.0) ** 2 + ((h[:, None] - h[None, :]) / 2.0) ** 2
#
#     dist = torch.sqrt(torch.clamp(center_dist_sq + scale_dist_sq, min=0.0))
#     return torch.exp(-dist / C_nwd)
#
#
# def safit_nms(boxes, scores, iou_thr=0.3, nwd_thr=0.88, C=32.0, min_score=0.001):
#     """
#     V10: Dual-Criteria Safit NMS (终极大成版)
#     彻底抛弃加权融合带来的数学漏洞，采用“基线拦截 + 极限小目标补漏”的双重逻辑。
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#
#     # 预计算相似度矩阵
#     iou_matrix = ops.box_iou(boxes, boxes)
#     nwd_matrix = compute_nwd_matrix(boxes, C_nwd=C)
#
#     scores_cloned = scores.clone()
#     keep = []
#     idxs = torch.arange(scores.numel(), device=scores.device)
#
#     while idxs.numel() > 0:
#         max_idx_local = torch.argmax(scores_cloned[idxs])
#         max_idx = idxs[max_idx_local]
#
#         # 过滤低分噪音，对齐评测标准
#         if scores_cloned[max_idx] < min_score:
#             break
#
#         keep.append(max_idx.item())
#         if idxs.numel() == 1:
#             break
#
#         idxs = torch.cat([idxs[:max_idx_local], idxs[max_idx_local + 1:]])
#
#         ious = iou_matrix[max_idx, idxs]
#         nwds = nwd_matrix[max_idx, idxs]
#
#         # ==========================================
#         # 核心：Dual-Criteria 独立截断逻辑 (逻辑或)
#         # ==========================================
#
#         # 防线 1：基线拦截。只要 IoU 大于 0.3，必然是冗余，斩！(死保中大目标 AP)
#         iou_suppress = ious > iou_thr
#
#         # 防线 2：小目标狙击。即使 IoU 没超标，但只要极其靠近 (NWD > 0.88)
#         # 并且【有物理交集】(ious > 0.0)，也认定为冗余，斩！
#         # 这完美清理了小目标的位置抖动，且绝不误杀密集不重叠的真实目标。
#         nwd_suppress = (nwds > nwd_thr) & (ious > 0.0)
#
#         # 综合抑制：触发任意一条防线即被剔除
#         suppress_mask = iou_suppress | nwd_suppress
#
#         # 只保留未被抑制的框
#         idxs = idxs[~suppress_mask]
#
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)
#

#wr20260511-1


# def compute_nwd_matrix(boxes, C_nwd=32.0):
#     """
#     计算全体 boxes 之间的 NWD 相似度矩阵
#     """
#     cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
#     cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#
#     # 防止宽度或高度为 0 导致 nan
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#
#     center_dist_sq = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
#     scale_dist_sq = ((w[:, None] - w[None, :]) / 2.0) ** 2 + ((h[:, None] - h[None, :]) / 2.0) ** 2
#
#     dist = torch.sqrt(torch.clamp(center_dist_sq + scale_dist_sq, min=0.0))
#     return torch.exp(-dist / C_nwd)
#
#
# def safit_nms(boxes, scores, labels, nwd_thr=0.85, C_nwd=32.0, min_score=0.001):
#     """
#     Safit-WBF: 基于 NWD 距离的加权框融合 (Weighted Boxes Fusion)
#
#     Args:
#         boxes: Tensor, [N, 4], 格式为 [x1, y1, x2, y2]
#         scores: Tensor, [N], 置信度得分
#         labels: Tensor, [N], 类别标签
#         nwd_thr: float, NWD 相似度阈值，大于此值认为属于同一个簇
#         C_nwd: float, NWD 计算的平滑常数
#         min_score: float, 忽略极低分数的框，加速计算
#
#     Returns:
#         fused_boxes: Tensor, [M, 4], 融合后的新框
#         fused_scores: Tensor, [M], 融合后的新得分
#         fused_labels: Tensor, [M], 融合后的新类别标签
#     """
#     # 0. 极端情况防护
#     if boxes.numel() == 0:
#         return torch.empty((0, 4), device=boxes.device), \
#             torch.empty((0,), device=scores.device), \
#             torch.empty((0,), dtype=torch.long, device=labels.device)
#
#     # 1. 过滤极低分框
#     valid_mask = scores >= min_score
#     boxes = boxes[valid_mask]
#     scores = scores[valid_mask]
#     labels = labels[valid_mask]
#
#     if boxes.numel() == 0:
#         return torch.empty((0, 4), device=boxes.device), \
#             torch.empty((0,), device=scores.device), \
#             torch.empty((0,), dtype=torch.long, device=labels.device)
#
#     # 2. 预计算 NWD 矩阵
#     nwd_matrix = compute_nwd_matrix(boxes, C_nwd=C_nwd)
#
#     # 强行把不同类别的 NWD 设为 0 (不允许跨类别融合)
#     label_match = labels[:, None] == labels[None, :]
#     nwd_matrix[~label_match] = 0.0
#
#     # 3. 按得分降序排列
#     order = scores.argsort(descending=True)
#
#     fused_boxes = []
#     fused_scores = []
#     fused_labels = []
#
#     processed = torch.zeros_like(scores, dtype=torch.bool)
#
#     # 4. 核心聚类与融合循环
#     for i in range(order.numel()):
#         idx = order[i]
#
#         if processed[idx]:
#             continue
#
#         # 找到属于同一个簇的所有框
#         cluster_mask = (nwd_matrix[idx] > nwd_thr) & (~processed)
#
#         cluster_boxes = boxes[cluster_mask]
#         cluster_scores = scores[cluster_mask]
#
#         # 加权融合坐标
#         weight_sum = cluster_scores.sum()
#         fused_box = (cluster_boxes * cluster_scores[:, None]).sum(dim=0) / weight_sum
#
#         # 融合得分 (保留簇内最高分)
#         fused_score = cluster_scores.max()
#
#         # 获取类别
#         fused_label = labels[idx]
#
#         fused_boxes.append(fused_box)
#         fused_scores.append(fused_score)
#         fused_labels.append(fused_label)
#
#         processed[cluster_mask] = True
#
#     return torch.stack(fused_boxes), torch.stack(fused_scores), torch.stack(fused_labels)

#wr20260511-2

# def compute_nwd_matrix(boxes, C_nwd=32.0):
#     # (此函数与之前完全一致，保持不变)
#     cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
#     cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#
#     center_dist_sq = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
#     scale_dist_sq = ((w[:, None] - w[None, :]) / 2.0) ** 2 + ((h[:, None] - h[None, :]) / 2.0) ** 2
#
#     dist = torch.sqrt(torch.clamp(center_dist_sq + scale_dist_sq, min=0.0))
#     return torch.exp(-dist / C_nwd)
#
#
# def safit_nms(boxes, scores, labels, nwd_thr=0.85, C_nwd=32.0, min_score=0.001, filter_iou_thr=0.05,
#                  boost_alpha=0.05):
#     """
#     Safit-WBF v2: 带有方差门槛 (Variance Gating) 与分数增强 (Score Boosting) 的终极融合算法
#
#     新增参数:
#         filter_iou_thr: 方差门槛。簇内候选框与最高分框(Boss)的 IoU 低于此值，将被踢出坐标融合计算。
#         boost_alpha: 分数增强系数。融合后给 Boss 增加的额外分数奖励权重。
#     """
#     if boxes.numel() == 0:
#         return torch.empty((0, 4), device=boxes.device), \
#             torch.empty((0,), device=scores.device), \
#             torch.empty((0,), dtype=torch.long, device=labels.device)
#
#     valid_mask = scores >= min_score
#     boxes = boxes[valid_mask]
#     scores = scores[valid_mask]
#     labels = labels[valid_mask]
#
#     if boxes.numel() == 0:
#         return torch.empty((0, 4), device=boxes.device), \
#             torch.empty((0,), device=scores.device), \
#             torch.empty((0,), dtype=torch.long, device=labels.device)
#
#     nwd_matrix = compute_nwd_matrix(boxes, C_nwd=C_nwd)
#
#     label_match = labels[:, None] == labels[None, :]
#     nwd_matrix[~label_match] = 0.0
#
#     order = scores.argsort(descending=True)
#
#     fused_boxes = []
#     fused_scores = []
#     fused_labels = []
#
#     processed = torch.zeros_like(scores, dtype=torch.bool)
#
#     for i in range(order.numel()):
#         idx = order[i]
#
#         if processed[idx]:
#             continue
#
#         # 1. 第一阶段：利用 NWD 极其宽容的特性，把附近所有的框“一网打尽”构成一个大簇
#         cluster_mask = (nwd_matrix[idx] > nwd_thr) & (~processed)
#
#         candidate_boxes = boxes[cluster_mask]
#         candidate_scores = scores[cluster_mask]
#
#         # 2. 第二阶段：方差门槛过滤 (Variance Gating)
#         # 以当前得分最高的框 (Boss) 为核心
#         boss_box = boxes[idx:idx + 1]
#
#         # 计算簇内所有候选小弟与 Boss 的物理重叠度 (IoU)
#         ious_with_boss = ops.box_iou(boss_box, candidate_boxes)[0]
#
#         # 核心魔法：如果小弟偏得连一丁点交集都没有 (或者交集极小)，它大概率是个“脏框”
#         # Boss 自己的 IoU 是 1.0，永远满足条件
#         gate_mask = ious_with_boss >= filter_iou_thr
#
#         # 只保留通过安检的“干净框”来参与坐标融合
#         valid_boxes = candidate_boxes[gate_mask]
#         valid_scores = candidate_scores[gate_mask]
#
#         # 3. 坐标加权融合 (提纯后的坐标)
#         weight_sum = valid_scores.sum()
#         fused_box = (valid_boxes * valid_scores[:, None]).sum(dim=0) / weight_sum
#
#         # # 4. 分数增强 (Score Boosting)
#         # boss_score = valid_scores.max()  # 其实就是 scores[idx]
#         # # 把簇内其他干净框的残余分数，按比例奖励给 Boss
#         # # 这样簇越密集，最终生成的框分数越高，AP 排序越靠前
#         # fused_score = boss_score + (weight_sum - boss_score) * boost_alpha
#         # # 确保得分不会爆表超过 1.0
#         # fused_score = torch.clamp(fused_score, max=1.0)
#
#         #融合得分（保留簇内最高分）
#         fused_score = valid_scores.max()
#
#         fused_label = labels[idx]
#
#         fused_boxes.append(fused_box)
#         fused_scores.append(fused_score)
#         fused_labels.append(fused_label)
#
#         # 【极其关键的一步】：
#         # 不管候选框有没有通过 gate_mask，只要它最初被划进了 NWD 簇，
#         # 就必须把它标记为已处理 (processed)！
#         # 相当于：对于“脏框”，我们拒绝它的坐标，但我们通过 processed 变相把它“删掉”了 (类似 NMS)。
#         # 这有效解决了密集目标的假阳性 FP。
#         processed[cluster_mask] = True
#
#     return torch.stack(fused_boxes), torch.stack(fused_scores), torch.stack(fused_labels)


#wr20260513
# def compute_nwd_matrix(boxes, C_nwd=32.0):
#     cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
#     cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#
#     center_dist_sq = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
#     scale_dist_sq = ((w[:, None] - w[None, :]) / 2.0) ** 2 + ((h[:, None] - h[None, :]) / 2.0) ** 2
#
#     dist = torch.sqrt(torch.clamp(center_dist_sq + scale_dist_sq, min=0.0))
#     return torch.exp(-dist / C_nwd)
#
#
# def safit_nms(boxes, scores, labels=None, nwd_thr=0.85, C=32.0, min_score=0.001):
#     """
#     工程落地版 Safit-Cluster-NMS (强制去冗余版)
#     核心逻辑：无视物理交集(IoU)，只要中心点靠得足够近(NWD>阈值)，且属于同一类别，
#     就认定为冗余预测并强制剔除。保证画面极度干净。
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#
#     valid_mask = scores >= min_score
#     boxes = boxes[valid_mask]
#     scores = scores[valid_mask]
#     if labels is not None:
#         labels = labels[valid_mask]
#
#     # 记录原始索引，以便最后返回
#     original_idxs = torch.nonzero(valid_mask, as_tuple=False).squeeze()
#     if original_idxs.dim() == 0:
#         original_idxs = original_idxs.unsqueeze(0)
#
#     # 计算尺度权重 (大目标倾向 IoU，小目标倾向 NWD)
#     w_box = boxes[:, 2] - boxes[:, 0]
#     h_box = boxes[:, 3] - boxes[:, 1]
#     sqrt_areas = torch.sqrt(torch.clamp(w_box * h_box, min=1e-3))
#     weights = torch.sigmoid((sqrt_areas - C) / C)
#
#     iou_matrix = ops.box_iou(boxes, boxes)
#     nwd_matrix = compute_nwd_matrix(boxes, C_nwd=C)
#
#     # 动态融合相似度矩阵
#     # 加上权重的目的是：防止小目标框误杀了刚好路过的大目标框
#     sim_matrix = weights[:, None] * iou_matrix + (1.0 - weights[:, None]) * nwd_matrix  #计算第i个框和第j个框的相似度，只使用第i个框的权重，
#
#     # 如果传入了 labels，阻断跨类别抑制
#     if labels is not None:
#         label_match = labels[:, None] == labels[None, :]
#         sim_matrix[~label_match] = 0.0
#
#     keep = []
#     idxs = torch.argsort(scores, descending=True)
#
#     while idxs.numel() > 0:
#         # 获取当前最高分框的索引
#         current_idx = idxs[0]
#         keep.append(original_idxs[current_idx].item())
#
#         if idxs.numel() == 1:
#             break
#
#         # 提取当前最高分框与其余所有候选框的融合相似度
#         current_sims = sim_matrix[current_idx, idxs[1:]]
#
#         # 核心：不管 IoU 是多少，只要综合相似度（小目标主要看距离）越过门槛，全部杀掉！
#         # 这个门槛比纯 IoU 严苛得多，是清理“框云”的利器
#         retain_mask = current_sims <= nwd_thr
#
#         # 只保留没被抑制的框，进入下一轮循环
#         idxs = idxs[1:][retain_mask]
#
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# wr20260514
# def compute_nwd_matrix(boxes, C_nwd=32.0):
#     cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
#     cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
#     w = boxes[:, 2] - boxes[:, 0]
#     h = boxes[:, 3] - boxes[:, 1]
#
#     w = torch.clamp(w, min=1e-3)
#     h = torch.clamp(h, min=1e-3)
#
#     center_dist_sq = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
#     scale_dist_sq = ((w[:, None] - w[None, :]) / 2.0) ** 2 + ((h[:, None] - h[None, :]) / 2.0) ** 2
#
#     dist = torch.sqrt(torch.clamp(center_dist_sq + scale_dist_sq, min=0.0))
#     return torch.exp(-dist / C_nwd)
#
#
# def safit_nms(boxes, scores, labels=None, thr_loose = 0.65, thr_strict = 0.75, C=32.0, min_score=0.001):
#     """
#     工程落地版 Safit-Cluster-NMS (强制去冗余版)
#     核心逻辑：无视物理交集(IoU)，只要中心点靠得足够近(NWD>阈值)，且属于同一类别，
#     就认定为冗余预测并强制剔除。保证画面极度干净。
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#
#     valid_mask = scores >= min_score
#     boxes = boxes[valid_mask]
#     scores = scores[valid_mask]
#     if labels is not None:
#         labels = labels[valid_mask]
#
#     # 记录原始索引，以便最后返回
#     original_idxs = torch.nonzero(valid_mask, as_tuple=False).squeeze()
#     if original_idxs.dim() == 0:
#         original_idxs = original_idxs.unsqueeze(0)
#
#     # 计算尺度权重 (大目标倾向 IoU，小目标倾向 NWD)
#     w_box = boxes[:, 2] - boxes[:, 0]
#     h_box = boxes[:, 3] - boxes[:, 1]
#     sqrt_areas = torch.sqrt(torch.clamp(w_box * h_box, min=1e-3))
#     weights = torch.sigmoid((sqrt_areas - C) / C)
#
#     iou_matrix = ops.box_iou(boxes, boxes)
#     nwd_matrix = compute_nwd_matrix(boxes, C_nwd=C)
#
#     # 动态融合相似度矩阵
#     # 加上权重的目的是：防止小目标框误杀了刚好路过的大目标框
#     sim_matrix = weights[:, None] * iou_matrix + (1.0 - weights[:, None]) * nwd_matrix  #计算第i个框和第j个框的相似度，只使用第i个框的权重，
#
#     # 如果传入了 labels，阻断跨类别抑制
#     if labels is not None:
#         label_match = labels[:, None] == labels[None, :]
#         sim_matrix[~label_match] = 0.0
#
#     keep = []
#     idxs = torch.argsort(scores, descending=True)
#
#     # thr_loose = 0.70  # 对待低分框的严苛过滤线 (清理稀疏场景冗余)
#     # thr_strict = 0.85  # 对待高分框的宽容保护线 (保护密集场景独立目标)
#     thr_range = thr_strict - thr_loose
#
#     while idxs.numel() > 0:
#         current_idx = idxs[0]
#         keep.append(original_idxs[current_idx].item())
#
#         if idxs.numel() == 1:
#             break
#
#         current_sims = sim_matrix[current_idx, idxs[1:]]
#
#         # 1. 获取老大和小弟们的分数
#         boss_score = scores[current_idx]
#         neighbor_scores = scores[idxs[1:]]
#
#         # 2. 计算分数比值 (0 ~ 1 之间)
#         score_ratio = neighbor_scores / boss_score
#
#         # # 2. 计算分数比值的指数惩罚(例如3次方)
#         # score_ratio = ( neighbor_scores / boss_score ) ** 3
#
#         # 3. 计算动态阈值：小弟分数越接近老大，阈值越高（越难被抑制）
#         dynamic_thr = thr_loose + thr_range * score_ratio
#
#         # 4. 判断：只有相似度小于这个动态阈值的，才能活下来
#         retain_mask = current_sims <= dynamic_thr
#
#         idxs = idxs[1:][retain_mask]
#
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)


#wr20260421

# def safit_nms(boxes, scores, iou_threshold=0.3, C=32, **kwargs):
#     """
#     基于 max(IoU, NWD) 的非极大值抑制，使用尺度自适应的动态阈值。
#
#     Args:
#         boxes: (N, 4) 边界框坐标 (x1, y1, x2, y2)
#         scores: (N,) 置信度得分
#         iou_threshold: IoU 部分的固定阈值（默认0.3）
#         C: NWD 计算中的尺度参数
#     Returns:
#         keep: 保留的边界框索引
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#
#     # 按得分降序排序
#     order = scores.sort(0, descending=True)[1]
#     keep = []
#
#     # 预计算常数：与 IoU=0.3 等效的中心偏移比例 d/s = 1 - sqrt(0.3)
#     # 使用精确值 1 - sqrt(0.3) ≈ 0.452277
#     k = 1 - math.sqrt(0.3)   # 约 0.452277
#
#     while order.numel() > 0:
#         i = order[0]
#         keep.append(i)
#         if order.numel() == 1:
#             break
#
#
#         current_box = boxes[i].unsqueeze(0)
#         rest_boxes = boxes[order[1:]]
#
#
#         # 计算 IoU 和 NWD
#         ious = bbox_overlaps(current_box, rest_boxes,is_aligned=False).clamp(min=1e-7).squeeze(0)  # (M,)
#         nwd = compute_nwd(current_box, rest_boxes, C).squeeze(0)                  # (M,)
#
#         # 取最大值作为相似度度量
#         metric = torch.max(ious, nwd)  # (M,)
#
#         # 计算被抑制框（rest_boxes）的面积及 sqrt(area)
#         areas = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
#         sqrt_area = torch.sqrt(areas + 1e-7)
#
#         # 动态 NWD 阈值：与 IoU=0.3 具有相同中心偏移比例
#         nwd_thresh = torch.exp(-k * sqrt_area / C)
#
#         # 决定每个 rest 框使用哪个阈值：若 ious >= nwd 则用 iou_threshold，否则用 nwd_thresh
#         use_iou = ious >= nwd
#         thresh = torch.where(use_iou,
#                              torch.tensor(iou_threshold, device=ious.device),
#                              nwd_thresh)
#
#         # 抑制条件：相似度大于阈值则抑制
#         sup_mask = metric > thresh
#
#         order = order[1:][~sup_mask] #~sup_mask：取反，True 表示未被抑制（Safit ≤ 阈值）的框，从其余框索引中筛选出未被抑制的框，作为下一轮循环的 order
#
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)


#wr20260421-1

#
# def safit_nms(boxes, scores, iou_threshold=0.3, C=32.0, **kwargs):
#     """
#     基于高分主导框尺度的 max(IoU, NWD) 动态阈值 NMS
#     """
#     if boxes.numel() == 0:
#         return torch.empty(0, dtype=torch.long, device=boxes.device)
#
#     # 按得分降序排序
#     order = scores.sort(0, descending=True)[1]
#     keep = []
#
#     # 预计算常数：与 IoU=0.3 等效的中心偏移比例
#     k = 1.0 - math.sqrt(iou_threshold)  # 约 0.452277 (若 iou_thr=0.3)
#
#     while order.numel() > 0:
#         i = order[0]
#         keep.append(i)
#         if order.numel() == 1:
#             break
#
#         current_box = boxes[i].unsqueeze(0)  # Boss 框: (1, 4)
#         rest_boxes = boxes[order[1:]]  # 小弟框: (M, 4)
#
#         # 1. 计算 IoU 和 NWD
#         ious = bbox_overlaps(current_box, rest_boxes, is_aligned=False).clamp(min=1e-7).squeeze(0)  # (M,)
#         nwd = compute_nwd(current_box, rest_boxes, C).squeeze(0)  # (M,)
#
#         # 2. 取最大值作为最终的相似度度量
#         metric = torch.max(ious, nwd)  # (M,)
#
#         # ==========================================
#         # 👑 核心修改：尺度阈值由当前的 Boss（高分框）决定
#         # ==========================================
#         boss_w = current_box[0, 2] - current_box[0, 0]
#         boss_h = current_box[0, 3] - current_box[0, 1]
#         boss_area = torch.clamp(boss_w * boss_h, min=1e-7)
#         sqrt_boss_area = torch.sqrt(boss_area)
#
#         # 算出一个统一的动态 NWD 阈值 (这是一个标量 Scalar)
#         # 这个阈值代表了 Boss 的“绝对容忍度”
#         nwd_thresh = torch.exp(-k * sqrt_boss_area / C)
#
#         # ==========================================
#
#         # 3. 决定每个 rest 框的裁判标准
#         use_iou = ious >= nwd
#
#         # 此时 nwd_thresh 会利用 PyTorch 的广播机制，自动复制成 (M,) 与 ious 对齐
#         thresh = torch.where(use_iou,
#                              torch.tensor(iou_threshold, device=ious.device),
#                              nwd_thresh)
#
#         # 4. 执行抑制
#         sup_mask = metric > thresh
#         order = order[1:][~sup_mask]
#
#     return torch.tensor(keep, dtype=torch.long, device=boxes.device)

#wr20260517-1
import torch
from torchvision.ops import box_iou


# def safit_nms(pred_boxes, pred_scores,
#                                   probe_nwd_thr=0.85,
#                                   density_thresh=2,
#                                   sparse_kill_nwd_thr=0.30,
#                                   dense_kill_iou_thr=0.30,
#                                   absolute_kill_nwd_thr=0.95,
#                                   C_nwd=32.0):
#     """
#     纯 PyTorch 实现的双半径解耦混合 NMS (已内置 IoU 和 NWD 矩阵计算)
#     输入接口完美对齐传统 NMS (boxes, scores)
#
#     参数:
#         pred_boxes: Tensor (N, 4) 格式 [x1, y1, x2, y2]
#         pred_scores: Tensor (N,) 置信度得分
#         其余参数同前述算法设计。
#     返回:
#         keep: Tensor 包含保留下来的预测框的索引
#     """
#     # 0. 判空保护
#     if pred_scores.numel() == 0:
#         return torch.empty((0,), dtype=torch.long, device=pred_scores.device)
#
#     # ==========================================
#     # 🚀 第一部分：在 GPU 上极速预计算相似度矩阵
#     # ==========================================
#     # 1. 计算 IoU 矩阵 (N, N)
#     iou_matrix = box_iou(pred_boxes, pred_boxes)
#
#     # 2. 计算 NWD 矩阵 (N, N)
#     w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1e-3)
#     h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1e-3)
#     cx = pred_boxes[:, 0] + w / 2
#     cy = pred_boxes[:, 1] + h / 2
#
#     # 组成向量 [cx, cy, w/2, h/2] -> (N, 4)
#     vec = torch.stack([cx, cy, w / 2, h / 2], dim=1)
#
#     # 利用 PyTorch 广播机制计算两两欧氏距离
#     diff = vec.unsqueeze(1) - vec.unsqueeze(0)  # (N, N, 4)
#     dist = torch.sqrt(torch.sum(diff ** 2, dim=2))
#     nwd_matrix = torch.exp(-dist / C_nwd)
#
#     # ==========================================
#     # 🚀 第二部分：执行双半径混合 NMS 逻辑
#     # ==========================================
#     # 在 GPU 上按得分降序
#     order = torch.argsort(pred_scores, descending=True)
#     keep = []
#
#     while order.numel() > 0:
#         boss_idx = order[0].item()
#         keep.append(boss_idx)
#
#         # 如果只剩最后一个框，直接结束
#         if order.numel() == 1:
#             break
#
#         # 获取剩下的所有小弟框的索引
#         rest_indices = order[1:]
#
#         # 提取 Boss 框与小弟框的相似度数组
#         current_iou_sims = iou_matrix[boss_idx, rest_indices]
#         current_nwd_sims = nwd_matrix[boss_idx, rest_indices]
#
#         # 👑 1. 探测密度 (用 .item() 转为 python 标量)
#         local_density = torch.sum(current_nwd_sims > probe_nwd_thr).item()
#
#         # 👑 2. 场景解耦分流
#         if local_density < density_thresh:
#             # 稀疏发散区：使用宽容的 NWD 半径，强力斩杀远处冗余
#             kill_mask = current_nwd_sims > sparse_kill_nwd_thr
#         else:
#             # 密集群体区：使用保守的 IoU，保护真实独立目标
#             kill_mask = current_iou_sims > dense_kill_iou_thr
#
#         # 👑 3. 绝对排斥区安全阀：秒杀贴脸重影
#         absolute_kill_mask = current_nwd_sims > absolute_kill_nwd_thr
#
#         # 最终被杀掉的框
#         final_kill_mask = kill_mask | absolute_kill_mask
#
#         # 剥夺死亡框的资格，只保留存活的框进入下一轮
#         order = rest_indices[~final_kill_mask]
#
#     # 将 keep 列表转换回与原设备一致的 Tensor
#     return torch.tensor(keep, dtype=torch.long, device=pred_scores.device)

#wr20260517-2
# import torch
# from torchvision.ops import box_iou
#
#
# def safit_nms(pred_boxes, pred_scores,
#                                  probe_nwd_thr=0.80,  # 稍微放宽探测尺，防漏探
#                                  density_thresh=2,
#                                  sparse_kill_nwd_thr=0.50,  # 收缩屠龙刀半径，0.5 约等于 22 像素
#                                  score_ratio_thr=0.80,  # 👑 新增：分数悬念保护比例
#                                  dense_kill_iou_thr=0.30,
#                                  absolute_kill_nwd_thr=0.95,
#                                  C_nwd=32.0):
#     """
#     置信度感知的双半径混合 NMS：在强力清场的同时，保护高分“备胎”框。
#     """
#     if pred_scores.numel() == 0:
#         return torch.empty((0,), dtype=torch.long, device=pred_scores.device)
#
#     # 1. 预计算矩阵
#     iou_matrix = box_iou(pred_boxes, pred_boxes)
#     w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1e-3)
#     h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1e-3)
#     cx = pred_boxes[:, 0] + w / 2
#     cy = pred_boxes[:, 1] + h / 2
#     vec = torch.stack([cx, cy, w / 2, h / 2], dim=1)
#     diff = vec.unsqueeze(1) - vec.unsqueeze(0)
#     dist = torch.sqrt(torch.sum(diff ** 2, dim=2))
#     nwd_matrix = torch.exp(-dist / C_nwd)
#
#     # 2. NMS 循环
#     order = torch.argsort(pred_scores, descending=True)
#     keep = []
#
#     while order.numel() > 0:
#         boss_idx = order[0].item()
#         boss_score = pred_scores[boss_idx]  # 获取 Boss 分数
#         keep.append(boss_idx)
#
#         if order.numel() == 1:
#             break
#
#         rest_indices = order[1:]
#         current_iou_sims = iou_matrix[boss_idx, rest_indices]
#         current_nwd_sims = nwd_matrix[boss_idx, rest_indices]
#         rest_scores = pred_scores[rest_indices]
#
#         local_density = torch.sum(current_nwd_sims > probe_nwd_thr).item()
#
#         if local_density < density_thresh:
#             # 🔪 【稀疏场景】：启动长刀清场
#             distance_kill = current_nwd_sims > sparse_kill_nwd_thr
#
#             # 👑 【核心改进】：分数悬念保护
#             # 如果小弟的分数 / Boss的分数 < 0.8 (例如 Boss=0.9, 小弟<0.72)，才允许被杀
#             # 高分备胎将被免死金牌保护！
#             score_ratio = rest_scores / boss_score
#             score_allow_kill = score_ratio < score_ratio_thr
#
#             # 只有距离够近，且分数确实差得多的冗余，才被真正杀掉
#             kill_mask = distance_kill & score_allow_kill
#         else:
#             # 🛡️ 【密集场景】：IoU 保护
#             kill_mask = current_iou_sims > dense_kill_iou_thr
#
#         # 绝对物理排斥 (太贴脸了，不管分数多高直接杀)
#         absolute_kill_mask = current_nwd_sims > absolute_kill_nwd_thr
#         final_kill_mask = kill_mask | absolute_kill_mask
#
#         order = rest_indices[~final_kill_mask]
#
#     return torch.tensor(keep, dtype=torch.long, device=pred_scores.device)

#wr20260518-1
import torch
from torchvision.ops import box_iou

#
# def safit_nms(pred_boxes, pred_scores,
#                              probe_nwd_thr=0.80,
#                              local_density_thresh=2,
#                              global_bg_thresh=10,  # 👑 新增：全局背景密度阈值
#                              sparse_kill_nwd_thr=0.50,
#                              score_ratio_thr=0.80,
#                              dense_kill_iou_thr=0.30,
#                              absolute_kill_nwd_thr=0.95,
#                              C_nwd=32.0):
#     """
#     终极版 NMS：宏观全局感知 + 局部双半径解耦 + 置信度悬念保护
#     """
#     if pred_scores.numel() == 0:
#         return torch.empty((0,), dtype=torch.long, device=pred_scores.device)
#
#     # 1. 预计算相似度矩阵 (GPU极速计算)
#     iou_matrix = box_iou(pred_boxes, pred_boxes)
#     w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1e-3)
#     h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1e-3)
#     cx = pred_boxes[:, 0] + w / 2
#     cy = pred_boxes[:, 1] + h / 2
#     vec = torch.stack([cx, cy, w / 2, h / 2], dim=1)
#     diff = vec.unsqueeze(1) - vec.unsqueeze(0)
#     dist = torch.sqrt(torch.sum(diff ** 2, dim=2))
#     nwd_matrix = torch.exp(-dist / C_nwd)
#
#     order = torch.argsort(pred_scores, descending=True)
#     keep = []
#
#     while order.numel() > 0:
#         boss_idx = order[0].item()
#         boss_score = pred_scores[boss_idx]
#         keep.append(boss_idx)
#
#         if order.numel() == 1:
#             break
#
#         rest_indices = order[1:]
#         current_iou_sims = iou_matrix[boss_idx, rest_indices]
#         current_nwd_sims = nwd_matrix[boss_idx, rest_indices]
#         rest_scores = pred_scores[rest_indices]
#
#         # ==========================================
#         # 👑 核心逻辑 1：计算微观局部密度
#         # ==========================================
#         local_density = torch.sum(current_nwd_sims > probe_nwd_thr).item()
#
#         # ==========================================
#         # 👑 核心逻辑 2：计算宏观背景密度 (上帝视角)
#         # 当前全图剩下的所有框的数量 - 聚在 Boss 身边的数量
#         # ==========================================
#         current_total_boxes = order.numel()
#         background_density = current_total_boxes - local_density
#
#         # ==========================================
#         # 👑 核心逻辑 3：联合场景判定
#         # ==========================================
#         # 只要背景极其干净 (判定为全局稀疏)，不管局部有几个框，统统视为发散冗余！
#         is_sparse_scene = (background_density < global_bg_thresh) or (local_density < local_density_thresh)
#
#         if is_sparse_scene:
#             # 🔪 【稀疏场景】：启动长刀清场 + 高分备胎保护
#             distance_kill = current_nwd_sims > sparse_kill_nwd_thr
#             score_ratio = rest_scores / boss_score
#             score_allow_kill = score_ratio < score_ratio_thr
#
#             kill_mask = distance_kill & score_allow_kill
#         else:
#             # 🛡️ 【密集场景】：启动 IoU 保护
#             kill_mask = current_iou_sims > dense_kill_iou_thr
#
#         # 绝对物理排斥：防贴脸重影
#         # absolute_kill_mask = current_nwd_sims > absolute_kill_nwd_thr
#         # final_kill_mask = kill_mask | absolute_kill_mask
#         final_kill_mask = kill_mask
#         # 更新进入下一轮的框
#         order = rest_indices[~final_kill_mask]
#
#     return torch.tensor(keep, dtype=torch.long, device=pred_scores.device)

#20260518-3
# import torch
# from torchvision.ops import box_iou
#
#
# def safit_nms(pred_boxes, pred_scores,
#                                    probe_nwd_thr=0.80,
#                                    local_density_thresh=2,
#                                    global_bg_thresh=10,
#                                    sparse_kill_nwd_thr=0.50,  # 22像素大刀
#                                    score_ratio_thr=0.80,
#                                    dense_kill_iou_thr=0.30,
#                                    absolute_kill_nwd_thr=0.95,
#                                    bg_score_thr=0.15,
#                                    ex_tiny_area_thr=144,  # 👑 新增：极小目标豁免面积 (例如 12x12)
#                                    C_nwd=32.0):
#     """
#     击败 Baseline 的终极形态：尺度感知的宏微观混合 NMS
#     """
#     if pred_scores.numel() == 0:
#         return torch.empty((0,), dtype=torch.long, device=pred_scores.device)
#
#     # 1. 预计算矩阵
#     iou_matrix = box_iou(pred_boxes, pred_boxes)
#     w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1e-3)
#     h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1e-3)
#     cx = pred_boxes[:, 0] + w / 2
#     cy = pred_boxes[:, 1] + h / 2
#     vec = torch.stack([cx, cy, w / 2, h / 2], dim=1)
#     diff = vec.unsqueeze(1) - vec.unsqueeze(0)
#     dist = torch.sqrt(torch.sum(diff ** 2, dim=2))
#     nwd_matrix = torch.exp(-dist / C_nwd)
#
#     # 提前计算所有框的面积，用于尺度感知
#     areas = w * h
#
#     order = torch.argsort(pred_scores, descending=True)
#     keep = []
#
#     while order.numel() > 0:
#         boss_idx = order[0].item()
#         boss_score = pred_scores[boss_idx]
#         boss_area = areas[boss_idx].item()  # 👑 获取 Boss 框的物理面积
#         keep.append(boss_idx)
#
#         if order.numel() == 1:
#             break
#
#         rest_indices = order[1:]
#         current_iou_sims = iou_matrix[boss_idx, rest_indices]
#         current_nwd_sims = nwd_matrix[boss_idx, rest_indices]
#         rest_scores = pred_scores[rest_indices]
#
#         # 宏微观密度判定
#         local_density = torch.sum(current_nwd_sims > probe_nwd_thr).item()
#         # current_total_boxes = order.numel()
#         # background_density = current_total_boxes - local_density
#         # is_sparse_scene = (background_density < global_bg_thresh) or (local_density < local_density_thresh)
#
#         # 2. 宏观背景密度 (上帝视角 + 火眼金睛)
#         # 获取当前全图剩下的所有框的得分
#         current_all_scores = pred_scores[order]
#
#         # 过滤掉幽灵底噪！只统计得分大于 bg_score_thr 的“有效高分实体”
#         valid_total_boxes = torch.sum(current_all_scores > bg_score_thr).item()
#
#         # 背景有效密度 = 全图有效框数 - 聚集在 Boss 身边的框数
#         # (使用 max(0, ...) 防止局部极其密集的低分框导致出现负数)
#         background_density = max(0, valid_total_boxes - local_density)
#
#         # 3. 联合判定
#         is_sparse_scene = (background_density < global_bg_thresh) or (local_density < local_density_thresh)
#
#         # ==========================================
#         # 👑 核心改进：尺度感知决策分流
#         # ==========================================
#         if boss_area < ex_tiny_area_thr:
#             # 【尺度豁免区】：Boss 是 ex_tiny 或 tiny 目标！
#             # 它们太脆弱，网络定位误差极大，绝不能用大范围的 NWD 去乱杀。
#             # 强制回退到最保守的 IoU 保护模式，与 IoU-NMS 同台竞技，保住所有的低分真实备胎！
#             kill_mask = current_iou_sims > dense_kill_iou_thr
#
#         elif is_sparse_scene:
#             # 【稀疏大中目标区】：目标够大，且处于稀疏场景。
#             # 这是我们反超 IoU-NMS 的得分点！抽出 NWD 大刀，彻底清理发散的 FPs。
#             distance_kill = current_nwd_sims > sparse_kill_nwd_thr
#             score_ratio = rest_scores / boss_score
#             score_allow_kill = score_ratio < score_ratio_thr
#             kill_mask = distance_kill & score_allow_kill
#
#         else:
#             # 【密集大中目标区】：目标够大，但是处于密集重叠场景。
#             # 启动正常 IoU 保护。
#             kill_mask = current_iou_sims > dense_kill_iou_thr
#
#         # 绝对物理排斥：防完美贴脸重影 (NWD > 0.95 即距离 < 1.6像素，通杀所有尺度)
#         absolute_kill_mask = current_nwd_sims > absolute_kill_nwd_thr
#         final_kill_mask = kill_mask | absolute_kill_mask
#
#         # 更新进入下一轮的框
#         order = rest_indices[~final_kill_mask]
#
#     return torch.tensor(keep, dtype=torch.long, device=pred_scores.device)


#20260519-1
import torch
import torchvision


def safit_nms(pred_boxes, pred_scores,
                                        dense_kill_iou_thr=0.30,
                                        global_bg_thresh=10,  # 区分全图疏密的实体数量阈值
                                        sparse_kill_nwd_thr=0.75,  # 稀疏场景下 NWD 清场阈值 (0.50约22像素)
                                        absolute_kill_nwd_thr=0.95,  # 绝对重影秒杀线
                                        C_nwd=32.0):
    """
    两阶段级联自适应 NMS 引擎
    阶段 1: 全局高效 IoU-NMS 基线纯化
    场景感知: 基于阶段 1 剩余实体数进行全局疏密判定
    阶段 2: 仅针对稀疏场景下的 [ex_tiny, tiny, small] 目标进行 NWD 靶向抑制
    """
    if pred_scores.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=pred_scores.device)

    # 提前计算全图所有框的面积，用于尺度隔离
    # ex_tiny(<8^2=64), tiny(<16^2=256), small(<32^2=1024)
    # 统一归纳为面积小于 1024 像素的框
    w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1e-3)
    h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1e-3)
    areas = w * h
    small_scale_area_thr = 1024.0

    # =======================================================
    # 🚀 【第一阶段】: 全局高效 IoU-NMS 基线过滤
    # =======================================================
    # 调用 Torchvision 原生底层 CUDA 算子，极速纯化空间重叠框
    keep_stage1 = torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=dense_kill_iou_thr)

    # =======================================================
    # 🚀 【核心创新点】: 真实全局疏密感知 (上帝视角)
    # =======================================================
    # 此时数出来的框，是排除了冗余后的“真实物理实体预测数”，彻底免疫低分底噪欺骗！
    total_remaining_entities = keep_stage1.numel()

    if total_remaining_entities >= global_bg_thresh:
        # 🛡️ 【全局密集场景】: 目标交织复杂，直接信任并返回第一阶段结果，触发零损伤保护
        return keep_stage1

    # =======================================================
    # 🚀 【第二阶段】: 全局稀疏场景 —— NWD 微观靶向清场
    # =======================================================
    # 提取第一阶段幸存者的全部特征
    boxes_s1 = pred_boxes[keep_stage1]
    scores_s1 = pred_scores[keep_stage1]
    areas_s1 = areas[keep_stage1]

    # 一次性在 GPU 上预计算第一阶段幸存者之间的 NWD 相似度矩阵 (矩阵极小，运算耗时微秒级)
    cx_s1 = boxes_s1[:, 0] + (boxes_s1[:, 2] - boxes_s1[:, 0]) / 2
    cy_s1 = boxes_s1[:, 1] + (boxes_s1[:, 3] - boxes_s1[:, 1]) / 2
    w_s1 = (boxes_s1[:, 2] - boxes_s1[:, 0]).clamp(min=1e-3)
    h_s1 = (boxes_s1[:, 3] - boxes_s1[:, 1]).clamp(min=1e-3)
    vec_s1 = torch.stack([cx_s1, cy_s1, w_s1 / 2, h_s1 / 2], dim=1)

    diff_s1 = vec_s1.unsqueeze(1) - vec_s1.unsqueeze(0)  # (M, M, 4)
    dist_s1 = torch.sqrt(torch.sum(diff_s1 ** 2, dim=2))
    nwd_matrix_s1 = torch.exp(-dist_s1 / C_nwd)  # (M, M)

    # 按照置信度得分降序排列进入二阶段循环
    order = torch.argsort(scores_s1, descending=True)
    final_keep_global_indices = []

    while order.numel() > 0:
        boss_idx_in_s1 = order[0].item()
        # 转换回最初传给整个函数的全局索引，用于最终交付
        boss_idx_global = keep_stage1[boss_idx_in_s1].item()
        final_keep_global_indices.append(boss_idx_global)

        if order.numel() == 1:
            break

        rest_indices_in_s1 = order[1:]
        current_nwd_sims = nwd_matrix_s1[boss_idx_in_s1, rest_indices_in_s1]

        # 👑 核心改进：尺度隔离型 NWD 斩杀权
        # 只有当当前的 Boss 框属于 [ex_tiny, tiny, small] (面积 < 1024) 时，才激活 NWD 抑制逻辑
        if areas_s1[boss_idx_in_s1].item() < small_scale_area_thr:

            # 🔪 斩杀线：距离满足 NWD > sparse_kill_nwd_thr，且被杀的小弟也必须是 [ex_tiny, tiny, small] 尺度
            distance_kill = current_nwd_sims > sparse_kill_nwd_thr
            target_scale_match = areas_s1[rest_indices_in_s1] < small_scale_area_thr

            kill_mask = distance_kill & target_scale_match
        else:
            # 如果 Boss 是大中型目标（medium, large），它已经在第一阶段靠 IoU 扫干净了
            # 它的二阶段斩杀判定直接关闭，绝不在微观层面误杀任何人
            kill_mask = torch.zeros_like(rest_indices_in_s1, dtype=torch.bool)

        # 🛡️ 绝对排斥安全阀：无视尺度，只要两框近乎完美贴脸(距离<1.6像素)，作为严重分数溢出重影直接秒杀
        absolute_kill_mask = current_nwd_sims > absolute_kill_nwd_thr
        final_kill_mask = kill_mask | absolute_kill_mask

        # 剥夺死亡框的资格
        order = rest_indices_in_s1[~final_kill_mask]

    return torch.tensor(final_keep_global_indices, dtype=torch.long, device=pred_scores.device)