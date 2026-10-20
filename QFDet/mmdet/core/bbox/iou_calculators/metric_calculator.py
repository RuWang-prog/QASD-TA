import math
import torch

from .builder import IOU_CALCULATORS


@IOU_CALCULATORS.register_module()
class BboxDistanceMetric(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='wasserstein', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def bbox_overlaps(bboxes1, bboxes2, mode='siwd', is_aligned=False, eps=1e-6, constant=12.8 , weight=2, img_shape=(512, 640)):
    assert mode in ['iou', 'iof', 'giou', 'normalized_giou', 'ciou', 'diou', 'nwd', 'siwd', 'safit', 'QAsafit',
                    'dotd','box1_box2','focaliou2', 'focaliou3',
                    'swdv2', 'center_distance', 'center_distance2', 'kl', 'gwd', 'bcd', 'gjsd'], f'Unsupported mode {mode}'   #wr20260520
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)

    if rows * cols == 0:
        return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if mode in ['box1_box2']:
        box1_box2 = area1[...,None] / area2[None,...]
        return box1_box2

    lt = torch.max(bboxes1[..., :, None, :2],
                    bboxes2[..., None, :, :2])  # [B, rows, cols, 2] 逐元素取最大值得到所有框对的左上角坐标（交集区域的左上角是两框左上角的最大值）
    rb = torch.min(bboxes1[..., :, None, 2:],
                    bboxes2[..., None, :, 2:])  # [B, rows, cols, 2] 同理取最小值得到右下角坐标。 结果 lt 和 rb 的形状均为 [B, rows, cols, 2]，存储每一对框的交集区域坐标（交集区域的右下角是两框右下角的最小值）

    wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2] #计算交集宽高，并钳位到非负
    overlap = wh[..., 0] * wh[..., 1] #宽高相乘得到交集面积 overlap

    union = area1[..., None] + area2[..., None, :] - overlap + eps  #并集面积

    if mode in ['giou', 'normalized_giou', 'ciou', 'diou']:
        enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                bboxes2[..., None, :, :2])
        enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                bboxes2[..., None, :, 2:])
        

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union #计算 IoU，结果形状 [B, rows, cols]
    
    if mode in ['iou', 'iof']:
        return ious

    if mode in ['focaliou2']:
        area1=area1.float()
        focalious = ious.pow(torch.sqrt(torch.sqrt(area1[..., None]) / 12.8))
        return focalious

    if mode in ['focaliou3']:
        vt_ind= (area1 < 64)
        t_ind = (area1 >= 64) & (area1 < 16*16)
        s_ind = (area1 >= 16*16) & (area1 < 32*32)
        m_ind = (area1 >= 32*32)
        focalious = ious *(vt_ind[...,None]*1.3 + t_ind[...,None]*1.3 + s_ind[...,None]*0.9+ m_ind[...,None]*0.8)
        return focalious
    # calculate gious
    if mode in ['giou', 'normalized_giou', 'ciou', 'diou']:
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps)
        gious = ious - (enclose_area - union) / enclose_area

    if mode == 'giou':
        return gious

    if mode == 'normalized_giou':
        gious = (1 + gious) / 2
        
        return gious

    if mode == 'diou':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #distances of center points between gt and pre

        enclosed_diagonal_distances = enclose_wh[..., 0] * enclose_wh[..., 0] + enclose_wh[..., 1] * enclose_wh[..., 1] # distances of diagonal of enclosed bbox
        
        dious = ious - center_distance / torch.max(enclosed_diagonal_distances, eps)
        
        dious = torch.clamp(dious,min=-1.0,max = 1.0)
        
        return dious

    if mode == 'ciou':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps # distances of center points between gt and pre

        enclosed_diagonal_distances = enclose_wh[..., 0] * enclose_wh[..., 0] + enclose_wh[..., 1] * enclose_wh[..., 1] # distances of diagonal of enclosed bbox

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]  + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]  + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]  + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]  + eps

        factor = 4 / math.pi ** 2
        v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        cious = ious - (center_distance / torch.max(enclosed_diagonal_distances, eps) + v ** 2 / torch.max(1 - ious + v, eps))

        cious = torch.clamp(cious, min=-1.0, max=1.0)
        
        return cious
    
    if mode == 'siwd':
        H = img_shape[0]
        W = img_shape[1]

        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2 #计算中心点坐标及宽高。（x1,y1）+(x2,y2) 得到中心点的坐标
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]   #（cx1,cy1）-(cx2,cy2)  两个中心点在x和y方向上的差值
        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]

        center_distance = 4*whs[..., 0] * whs[..., 0] / (w1 + w2)**2 + 4*whs[..., 1] * whs[..., 1] / (h1 + h2)**2
        wh_distance = (w1 - w2) ** 2 / (w1 + w2)**2 + (h1 - h2) ** 2 / (h1 + h2)**2
        wassersteins = torch.sqrt(center_distance + wh_distance) #中心距离按宽高和进行缩放，宽高距离也归一化，两者相加开方得到 Wasserstein 距离
        
        normalized_wasserstein = torch.exp(-wassersteins*2) #通过指数函数将其转换为 [0,1] 范围内的相似度。 SIWD
        return normalized_wasserstein

#wr 20260609
    if mode == 'QAsafit':
        # 推断当前输入格式为 xyxy (左上角, 右下角)
        # 1. 计算宽高与中心点
        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]
        cx1 = bboxes1[..., :, None, 0] + w1 / 2.0
        cy1 = bboxes1[..., :, None, 1] + h1 / 2.0

        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]
        cx2 = bboxes2[..., None, :, 0] + w2 / 2.0
        cy2 = bboxes2[..., None, :, 1] + h2 / 2.0

        # 2. 计算 GT 的真实尺度 (基于真实框 bboxes2)
        area2 = w2 * h2
        sqrtA = torch.sqrt(torch.clamp(area2, min=1e-6))

        # 3. 计算微观尺度补偿 QA-NWD
        dx = cx1 - cx2
        dy = cy1 - cy2
        dw = w1 - w2
        dh = h1 - h2

        w2_sq = dx ** 2 + dy ** 2 + (dw ** 2 + dh ** 2) / 4.0
        w2_dist = torch.sqrt(torch.clamp(w2_sq, min=1e-12))

        # 核心：使用 C_min=16.0 进行极小目标的“保底挖掘”
        # C_dyn = torch.clamp(sqrtA, min=16.0)

       #wr20260713 下调Cmin到8,激活星探的显微镜效用，拉开正负样本对比度
        # C_dyn = torch.clamp(sqrtA, min=8.0)

        # wr20260715 下调Cmin到4,试探边界
        # C_dyn = torch.clamp(sqrtA, min=4.0)

        # wr20260715 下调Cmin到12,消融实验
        C_dyn = torch.clamp(sqrtA, min=12.0)

        qa_nwd = torch.exp(-w2_dist / C_dyn)

        # 4. 计算宏观尺度约束 Native IoU
        # 纯张量化计算交并比，支持多维广播
        inter_x1 = torch.max(bboxes1[..., :, None, 0], bboxes2[..., None, :, 0])
        inter_y1 = torch.max(bboxes1[..., :, None, 1], bboxes2[..., None, :, 1])
        inter_x2 = torch.min(bboxes1[..., :, None, 2], bboxes2[..., None, :, 2])
        inter_y2 = torch.min(bboxes1[..., :, None, 3], bboxes2[..., None, :, 3])

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        area1 = w1 * h1
        union_area = area1 + area2 - inter_area
        iou = inter_area / torch.clamp(union_area, min=1e-6)

        # 5. 尺度路由与动态融合 (以 IoU 为中心，微小目标让权)
        # 当 sqrtA < 16, w_iou -> 0 (NWD主导)；当 sqrtA > 48, w_iou -> 1 (IoU主导)
        w_iou = 1.0 / (1.0 + torch.exp(-(sqrtA - 32.0) / 4.0))

        qasafit_matrix = w_iou * iou + (1.0 - w_iou) * qa_nwd

        return qasafit_matrix
#wr20260521
    # if mode == 'safit':
    #     C_nwd = 32.0  # safit 的核心常数
    #
    #     # =====================================================
    #     # 1. 极速并行计算 IoU 矩阵 [..., N, M]
    #     # =====================================================
    #     # 计算交集区域的左上角和右下角
    #     lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
    #     rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    #     wh = (rb - lt).clamp(min=0)  # 限制最小值为0，没有交集的为0
    #     inter = wh[..., 0] * wh[..., 1]  # 交集面积
    #
    #     # 计算各自的面积
    #     w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]
    #     h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]
    #     area1 = w1 * h1
    #
    #     w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]
    #     h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]
    #     area2 = w2 * h2
    #
    #     # 计算并集和 IoU
    #     union = area1 + area2 - inter
    #     iou_matrix = inter / union.clamp(min=1e-6)
    #
    #     # =====================================================
    #     # 2. 🔥 修正版：完整计算 2nd Wasserstein 距离矩阵 [..., N, M]
    #     # =====================================================
    #     # 计算中心点坐标
    #     center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
    #     center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
    #
    #     # 中心点偏差平方项 (Δcx^2, Δcy^2)
    #     delta_cx = center1[..., 0] - center2[..., 0]
    #     delta_cy = center1[..., 1] - center2[..., 1]
    #
    #     # 👑 核心修正：加入半宽高偏差平方项 (Δ(w/2)^2, Δ(h/2)^2)
    #     delta_w_half = (w1 - w2) / 2
    #     delta_h_half = (h1 - h2) / 2
    #
    #     # 完整的 2nd Wasserstein 距离计算 (包含中心位置距离与形状尺度距离)
    #     wasserstein_distance_sq = delta_cx ** 2 + delta_cy ** 2 + delta_w_half ** 2 + delta_h_half ** 2
    #     wasserstein_distance = torch.sqrt(wasserstein_distance_sq.clamp(min=1e-6))
    #
    #     # 转换为 NWD 相似度矩阵
    #     nwd_matrix = torch.exp(-wasserstein_distance / C_nwd)
    #
    #     # =====================================================
    #     # 3. 极速计算动态权重 w (绝对依赖于 GT真实框 的尺度)
    #     # =====================================================
    #     # area2 的 shape 是 [..., 1, M]，这代表了 1 个预测框对应 M 个真实框
    #     # 这个维度刚好可以和 [..., N, M] 的矩阵进行完美广播！
    #     sqrt_area2 = torch.sqrt(area2.clamp(min=1e-3))
    #
    #     # x = (sqrt_area - C) / C
    #     x = (sqrt_area2 - C_nwd) / C_nwd
    #     w_weight = torch.sigmoid(x)  # shape: [..., 1, M]
    #
    #     # =====================================================
    #     # 4. 融合计算最终的 safit 矩阵
    #     # =====================================================
    #     # 同一个 GT 列下的所有预测框，共享相同的权重评判标准
    #     safit_matrix = w_weight * iou_matrix + (1 - w_weight) * nwd_matrix
    #
    #     return safit_matrix

    # wr20260522
    # if mode == 'safit':
    #     # === 👑 safit v2 超参数设定 (严苛模式) ===
    #     min_C = 8.0  # 极小目标的最后避风港 (保底容忍度)
    #     scale_factor = 0.5  # 严苛度系数 (边长缩放比，越小越严苛)
    #     pivot_size = 16.0  # Sigmoid 权重交接的转折点 (原为 32.0)
    #     temp_factor = 2.0  # 温度系数，让 Sigmoid 的非线性切换更陡峭
    #
    #     # =====================================================
    #     # 1. 极速并行计算 IoU 矩阵 [..., N, M]
    #     # =====================================================
    #     lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
    #     rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    #     wh = (rb - lt).clamp(min=0)
    #     inter = wh[..., 0] * wh[..., 1]
    #
    #     w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]
    #     h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]
    #     area1 = w1 * h1
    #
    #     w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]
    #     h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]
    #     area2 = w2 * h2
    #
    #     union = area1 + area2 - inter
    #     iou_matrix = inter / union.clamp(min=1e-6)
    #
    #     # =====================================================
    #     # 2. 计算 2nd Wasserstein 距离 [..., N, M]
    #     # =====================================================
    #     center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
    #     center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
    #
    #     delta_cx = center1[..., 0] - center2[..., 0]
    #     delta_cy = center1[..., 1] - center2[..., 1]
    #     delta_w_half = (w1 - w2) / 2
    #     delta_h_half = (h1 - h2) / 2
    #
    #     wasserstein_distance_sq = delta_cx ** 2 + delta_cy ** 2 + delta_w_half ** 2 + delta_h_half ** 2
    #     wasserstein_distance = torch.sqrt(wasserstein_distance_sq.clamp(min=1e-6))
    #
    #     # =====================================================
    #     # 3. 👑 核心改造一：动态计算极其严苛的 C_nwd
    #     # =====================================================
    #     sqrt_area2 = torch.sqrt(area2.clamp(min=1e-3))  # shape: [..., 1, M]
    #
    #     # 动态容忍度：目标边长的 0.5 倍，且绝不低于 8.0 像素
    #     dynamic_C = torch.clamp(sqrt_area2 * scale_factor, min=min_C)
    #
    #     # 计算 NWD 矩阵
    #     nwd_matrix = torch.exp(-wasserstein_distance / dynamic_C)
    #
    #     # =====================================================
    #     # 4. 👑 核心改造二：权重重心左移与陡峭化
    #     # =====================================================
    #     # 将转折点从 32 压低到 16，并乘以 temp_factor(2.0) 加速 Sigmoid 收敛
    #     x = ((sqrt_area2 - pivot_size) / pivot_size) * temp_factor
    #     w_weight = torch.sigmoid(x)  # shape: [..., 1, M]
    #
    #     # =====================================================
    #     # 5. 融合生成 safit_v2 矩阵
    #     # =====================================================
    #     safit_matrix = w_weight * iou_matrix + (1 - w_weight) * nwd_matrix
    #
    #     return safit_matrix

    #wr20260525
    # if mode == 'safit':
    #     # === 👑 safit v3 超参数设定 (终极切割模式) ===
    #     # 1. 放弃动态 C，使用固定绝对容忍度。对于16x16以下目标，16.0是一个黄金兜底值。
    #     C_nwd = 16.0
    #
    #     # 2. 权重切换的中心转折点：16像素 (区分 tiny 和 small 的物理分界线)
    #     pivot_size = 16.0
    #
    #     # 3. 极其陡峭的温度系数：2.0。确保 >20像素的目标，IoU权重高达90%以上！
    #     temp_factor = 2.0
    #
    #     # =====================================================
    #     # 1. 计算 IoU 矩阵 [..., N, M]
    #     # =====================================================
    #     lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
    #     rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    #     wh = (rb - lt).clamp(min=0)
    #     inter = wh[..., 0] * wh[..., 1]
    #
    #     w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]
    #     h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]
    #     area1 = w1 * h1
    #
    #     w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]
    #     h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]
    #     area2 = w2 * h2
    #
    #     union = area1 + area2 - inter
    #     iou_matrix = inter / union.clamp(min=1e-6)
    #
    #     # =====================================================
    #     # 2. 计算 2nd Wasserstein 距离 [..., N, M]
    #     # =====================================================
    #     center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
    #     center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
    #
    #     delta_cx = center1[..., 0] - center2[..., 0]
    #     delta_cy = center1[..., 1] - center2[..., 1]
    #     delta_w_half = (w1 - w2) / 2
    #     delta_h_half = (h1 - h2) / 2
    #
    #     wasserstein_distance_sq = delta_cx ** 2 + delta_cy ** 2 + delta_w_half ** 2 + delta_h_half ** 2
    #     wasserstein_distance = torch.sqrt(wasserstein_distance_sq.clamp(min=1e-6))
    #
    #     # =====================================================
    #     # 3. 计算 NWD (使用固定的宽容光环 C_nwd)
    #     # =====================================================
    #     nwd_matrix = torch.exp(-wasserstein_distance / C_nwd)
    #
    #     # =====================================================
    #     # 4. 👑 核心魔法：陡峭的自适应权重分配
    #     # =====================================================
    #     sqrt_area2 = torch.sqrt(area2.clamp(min=1e-3))  # shape: [..., 1, M]
    #
    #     # 数学推演：
    #     # 如果 GT是 8x8 (ex_tiny), x = -8/16 * 2.0 = -1.0 -> w_weight = 0.26 (NWD占74%)
    #     # 如果 GT是 16x16 (tiny),  x = 0 -> w_weight = 0.50 (平分秋色)
    #     # 如果 GT是 24x24 (small), x = 8/16 * 2.0 = 1.0 -> w_weight = 0.73 (IoU占主导)
    #     # 如果 GT是 32x32 (medium),x = 16/16 * 2.0 = 2.0 -> w_weight = 0.88 (IoU绝对碾压)
    #     x = ((sqrt_area2 - pivot_size) / pivot_size) * temp_factor
    #     w_weight = torch.sigmoid(x)
    #
    #     # =====================================================
    #     # 5. 融合生成最终分配矩阵
    #     # =====================================================
    #     safit_matrix = w_weight * iou_matrix + (1 - w_weight) * nwd_matrix
    #
    #     return safit_matrix

    #wr20260526
    # if mode == 'safit':
    #     # === 👑 safit v4: 终极双标底座 (NWD + SIWD 融合) ===
    #     C_nwd = 16.0  # 极小目标的绝对物理宽容度兜底
    #     pivot_size = 16.0  # 宏微观尺度分界线
    #     temp_factor = 2.0  # 陡峭切换温度系数
    #
    #     # --- 1. 准备基础宽高和面积 ---
    #     w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]
    #     h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]
    #     w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]
    #     h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]
    #     area2 = w2 * h2
    #
    #     center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
    #     center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
    #     whs = center1[..., :2] - center2[..., :2]
    #
    #     # =====================================================
    #     # 2. 宏观世界主宰：计算 SIWD 矩阵 (替换原有的 IoU)
    #     # =====================================================
    #     # 严格复刻 siwd 的归一化距离逻辑
    #     center_distance_siwd = 4 * whs[..., 0] ** 2 / (w1 + w2) ** 2 + 4 * whs[..., 1] ** 2 / (h1 + h2) ** 2
    #     wh_distance_siwd = (w1 - w2) ** 2 / (w1 + w2) ** 2 + (h1 - h2) ** 2 / (h1 + h2) ** 2
    #     wassersteins_siwd = torch.sqrt(center_distance_siwd + wh_distance_siwd)
    #
    #     siwd_matrix = torch.exp(-wassersteins_siwd * 2.0)
    #
    #     # =====================================================
    #     # 3. 微观世界主宰：计算 NWD 矩阵
    #     # =====================================================
    #     delta_w_half = (w1 - w2) / 2
    #     delta_h_half = (h1 - h2) / 2
    #
    #     wasserstein_distance_sq_nwd = whs[..., 0] ** 2 + whs[..., 1] ** 2 + delta_w_half ** 2 + delta_h_half ** 2
    #     wasserstein_distance_nwd = torch.sqrt(wasserstein_distance_sq_nwd.clamp(min=1e-6))
    #
    #     nwd_matrix = torch.exp(-wasserstein_distance_nwd / C_nwd)
    #
    #     # =====================================================
    #     # 4. 动态权重融合 (Sigmoid 门控机制)
    #     # =====================================================
    #     sqrt_area2 = torch.sqrt(area2.clamp(min=1e-3))
    #
    #     # x 越小，w_weight 越接近 0 (NWD主导)；x 越大，w_weight 越接近 1 (SIWD主导)
    #     x = ((sqrt_area2 - pivot_size) / pivot_size) * temp_factor
    #     w_weight = torch.sigmoid(x)  # shape: [..., 1, M]
    #
    #     # =====================================================
    #     # 5. 生成最终矩阵：大目标用 SIWD，小目标用 NWD
    #     # =====================================================
    #     safit_matrix = w_weight * siwd_matrix + (1 - w_weight) * nwd_matrix
    #
    #     return safit_matrix

    #wr20260527
    # if mode == 'safit':
    #     # === 👑 safit v5: 绝对领域解耦 (Hard Routing 路由模式) ===
    #     C_nwd = 16.0  # 极小目标的绝对物理宽容度兜底
    #     pivot_size = 16.0  # 宏微观尺度绝对分界线
    #
    #     # --- 1. 准备基础宽高和面积 ---
    #     w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]
    #     h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]
    #     w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]
    #     h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]
    #     area2 = w2 * h2
    #
    #     center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
    #     center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
    #     whs = center1[..., :2] - center2[..., :2]
    #
    #     # =====================================================
    #     # 2. 宏观世界主宰：纯粹的 SIWD 矩阵
    #     # =====================================================
    #     center_distance_siwd = 4 * whs[..., 0] ** 2 / (w1 + w2) ** 2 + 4 * whs[..., 1] ** 2 / (h1 + h2) ** 2
    #     wh_distance_siwd = (w1 - w2) ** 2 / (w1 + w2) ** 2 + (h1 - h2) ** 2 / (h1 + h2) ** 2
    #     wassersteins_siwd = torch.sqrt(center_distance_siwd + wh_distance_siwd)
    #
    #     siwd_matrix = torch.exp(-wassersteins_siwd * 2.0)
    #
    #     # =====================================================
    #     # 3. 微观世界主宰：纯粹的 NWD 矩阵
    #     # =====================================================
    #     delta_w_half = (w1 - w2) / 2
    #     delta_h_half = (h1 - h2) / 2
    #
    #     wasserstein_distance_sq_nwd = whs[..., 0] ** 2 + whs[..., 1] ** 2 + delta_w_half ** 2 + delta_h_half ** 2
    #     wasserstein_distance_nwd = torch.sqrt(wasserstein_distance_sq_nwd.clamp(min=1e-6))
    #
    #     nwd_matrix = torch.exp(-wasserstein_distance_nwd / C_nwd)
    #
    #     # =====================================================
    #     # 4. 👑 核心魔法：零模糊的硬截断路由 (Hard Mask)
    #     # =====================================================
    #     sqrt_area2 = torch.sqrt(area2.clamp(min=1e-3))
    #
    #     # 生成一个 0 和 1 的布尔掩码（Float类型以便计算）
    #     # 只要 GT 边长小于 16，is_micro 为 1；否则为 0。
    #     is_micro = (sqrt_area2 < pivot_size).float()  # shape: [..., 1, M]
    #
    #     # =====================================================
    #     # 5. 生成最终矩阵：纯净交接，互不干扰
    #     # =====================================================
    #     # 对于任何一列 GT，它的所有候选框都会 100% 采用同一种度量方式
    #     safit_matrix = is_micro * nwd_matrix + (1 - is_micro) * siwd_matrix
    #
    #     return safit_matrix

    #wr20260528
    if mode == 'safit':
        # === 👑 safit v6: 动态截断路由 (Dynamic Hard Routing) ===
        pivot_size = 16.0  # 宏微观尺度绝对分界线

        # --- 1. 准备基础宽高和面积 ---
        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]
        area2 = w2 * h2

        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        # =====================================================
        # 2. 宏观世界主宰：纯粹的 SIWD 矩阵
        # =====================================================
        center_distance_siwd = 4 * whs[..., 0] ** 2 / (w1 + w2) ** 2 + 4 * whs[..., 1] ** 2 / (h1 + h2) ** 2
        wh_distance_siwd = (w1 - w2) ** 2 / (w1 + w2) ** 2 + (h1 - h2) ** 2 / (h1 + h2) ** 2
        wassersteins_siwd = torch.sqrt(center_distance_siwd + wh_distance_siwd)

        siwd_matrix = torch.exp(-wassersteins_siwd * 2.0)

        # =====================================================
        # 3. 微观世界主宰：带动态保底的 NWD 矩阵
        # =====================================================
        delta_w_half = (w1 - w2) / 2
        delta_h_half = (h1 - h2) / 2

        wasserstein_distance_sq_nwd = whs[..., 0] ** 2 + whs[..., 1] ** 2 + delta_w_half ** 2 + delta_h_half ** 2
        wasserstein_distance_nwd = torch.sqrt(wasserstein_distance_sq_nwd.clamp(min=1e-6))

        # 👑 核心优化：微观领域的内部自适应
        sqrt_area2 = torch.sqrt(area2.clamp(min=1e-3))
        # 让 C 随极小目标自身尺寸变化，最低保底 8.0，最高不超过 pivot_size (16.0)
        dynamic_C_nwd = torch.clamp(sqrt_area2, min=8.0, max=pivot_size)

        nwd_matrix = torch.exp(-wasserstein_distance_nwd / dynamic_C_nwd)

        # =====================================================
        # 4. 硬截断路由 (Hard Mask)
        # =====================================================
        is_micro = (sqrt_area2 < pivot_size).float()  # shape: [..., 1, M]

        # =====================================================
        # 5. 生成最终矩阵
        # =====================================================
        safit_matrix = is_micro * nwd_matrix + (1 - is_micro) * siwd_matrix

        return safit_matrix


    if mode == 'nwd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]  + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]  + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]  + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]  + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / (weight**2)

        wassersteins = torch.sqrt(center_distance + wh_distance)
        # constant = 11.7
        normalized_wasserstein = torch.exp(-wassersteins/constant)

        return normalized_wasserstein



    if mode == 'dotd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #

        distance = torch.sqrt(center_distance)

        dotd = torch.exp(-distance/constant)

        return dotd

    if mode == 'center_distance':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance2 = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #

        distance = torch.sqrt(center_distance2)
    
        return distance

    if mode == 'center_distance2':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance2 = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #

        return center_distance2


    '''
    if mode == 'swd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  #

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

        wassersteins = torch.sqrt(center_distance + wh_distance)

        factor = torch.sqrt(area1[...,None]*area2[...,None,:])**(1/3)

        swd = torch.exp(-wassersteins /factor)

        return swd
    '''
    if mode == 'swdv2':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  #

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

        wassersteins = torch.sqrt(center_distance + wh_distance)

        factor = ((w1+w2)+(h1+h2))/4
        #swd = np.exp(-wassersteins / factor)
        swd = torch.exp(- wassersteins**0.5/ factor**0.5)

        return swd


    if mode == 'kl':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        kl=(w2**2/w1**2+h2**2/h1**2+4*whs[..., 0]**2/w1**2+4*whs[..., 1]**2/h1**2+torch.log(w1**2/w2**2)+torch.log(h1**2/h2**2)-2)/2

        kld = 1/(1+kl)

        return kld

    if mode == 'gwd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  #

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
        wasserstein = center_distance + wh_distance

        gwd = 1/(1+wasserstein)

        return gwd

    if mode == 'bcd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        x1 = center1[..., 0]
        x2 = center2[..., 0]
        y1 = center1[..., 1]
        y2 = center2[..., 1]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        a1 = w1**2/4
        a2 = w2**2/4
        b1 = h1**2/4
        b2 = h2**2/4
 
        bcd = ((x1-x2)**2/(a1+a2)+(y1-y2)**2/(a1+a2))/4 + torch.log((a1+a2)*(b1+b2))/2 - torch.log(a1*a2*b1*b2)/4

        bc = torch.exp(-bcd)

        hd = torch.sqrt(1-bc)

        bcd1 = 1-hd

        return bcd1

    if mode == 'gjsd':
        alpha = 1/2
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        x1 = center1[..., 0]
        x2 = center2[..., 0]
        y1 = center1[..., 1]
        y2 = center2[..., 1]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        gjsd = 0.5 * (4*(1-alpha)*x1**2/(w1**2)+4*(1-alpha)*y1**2/(h1**2)+4*alpha*(x2**2)/(w2**2)+4*alpha*y2**2/(h2**2)-4*(((1-alpha)*x1/(w1**2)+alpha*x2/(w2**2))**2/((1-alpha)/(w1**2)+alpha/(w2**2)) + ((1-alpha)*y1/(h1**2)+alpha*y2/(h2**2))**2/((1-alpha)/(h1**2)+alpha/(h2**2))) + torch.log(16*((w1*h1)**2/16)**(1-alpha)*((w2*h2)**2/16)**(alpha)*((1-alpha)/w1**2+alpha/w2**2)*((1-alpha)/h1**2+alpha/h2**2)))

        gjsd = 1/(1+gjsd)

        return gjsd


