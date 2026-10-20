# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class QLSAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    If ``alpha`` is not None, it means that the dynamic cost
    ATSSAssigner is adopted, which is currently only used in the DDOD.

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 topk,
                 alpha=None,
                 quality='x',
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 iou_mode='iou',
                 overlap_mode='hybrid',
                 ignore_iof_thr=-1):
        self.topk = topk
        self.alpha = alpha
        self.quality = quality
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.iou_mode = iou_mode
        self.overlap_mode = overlap_mode
        self.ignore_iof_thr = ignore_iof_thr

    """Assign a corresponding gt bbox or background to each bbox.

    Args:
        topk (int): number of bbox selected in each level.
        alpha (float): param of cost rate for each proposal only in DDOD.
            Default None.
        iou_calculator (dict): builder of IoU calculator.
            Default dict(type='BboxOverlaps2D').
        ignore_iof_thr (int): whether ignore max overlaps or not.
            Default -1 (1 or -1).
    """

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py
    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               cls_scores=None,
               bbox_preds=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt

        If ``alpha`` is not None, and ``cls_scores`` and `bbox_preds`
        are not None, the overlaps calculation in the first step
        will also include dynamic cost, which is currently only used in
        the DDOD.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO. Default None.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
            cls_scores (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes. Default None.
            bbox_preds (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4. Default None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0) #真实框数量 num_gt 和锚框数量 num_bboxes

        message = 'Invalid alpha parameter because cls_scores or ' \
                  'bbox_preds are None. If you want to use the ' \
                  'cost-based ATSSAssigner,  please set cls_scores, ' \
                  'bbox_preds and self.alpha at the same time. '

        if self.alpha is None or cls_scores is None or bbox_preds is None: #首先判断是否使用动态成本（self.alpha 存在且 cls_scores 和 bbox_preds 都提供）。如果是，则进行动态成本计算：
            # ATSSAssigner
            # Here bboxes are actually anchors.
            overlaps = self.iou_calculator(bboxes, gt_bboxes, mode=self.iou_mode)
            #否则，执行标准ATSS：直接用IoU计算overlaps
            if cls_scores is not None or bbox_preds is not None:
                warnings.warn(message)
        else:
            # Dynamic cost ATSSAssigner in DDOD
            assert cls_scores is not None and bbox_preds is not None, message

            # compute cls cost for bbox and GT
            cls_cost = torch.sigmoid(cls_scores[:, gt_labels]) #计算分类成本：取 cls_scores 中对应真实类别 gt_labels 的值，经过 sigmoid 得到概率，作为分类成本；  计算每个预测对于每个真实目标（GT）的分类概率，为后续的标签分配（如匈牙利匹配）构建分类代价矩阵
            
            # compute iou between all bbox and gt
            if self.overlap_mode == 'preds':
                overlaps = self.iou_calculator(bbox_preds, gt_bboxes, mode=self.iou_mode)
            elif self.overlap_mode == 'hybrid': #根据 overlap_mode 计算 IoU 成本
                overlaps1 = self.iou_calculator(bboxes, gt_bboxes, mode=self.iou_mode) #锚框与真实框的 IoU
                overlaps2 = self.iou_calculator(bbox_preds, gt_bboxes, mode=self.iou_mode) #预测框（bbox_preds）与真实框的 IoU
                assert overlaps1.shape == overlaps2.shape
                overlaps = torch.stack([overlaps1, overlaps2])
                overlaps = torch.max(overlaps, dim=0)[0] #根据 overlap_mode 计算 IoU 成本：可以使用预测框（bbox_preds）与真实框的 IoU，或者使用锚框与真实框的 IoU，或者取两者的最大值（hybrid）。 HLQ(i,j)
            else:
                raise NotImplementedError(self.overlap_mode)
            
            # make sure that we are in element-wise multiplication
            assert cls_cost.shape == overlaps.shape
            # overlaps is actually a cost matrix
            # ME:quality
            if self.quality == 'x': #将分类成本与 IoU 成本融合 乘性融合
                overlaps = cls_cost**(1 - self.alpha) * overlaps**self.alpha #这里 overlaps 实际上成了融合后的质量分数，但仍被称为 overlaps=混合定位分数*分类概率
            elif self.quality == '+':
                overlaps = cls_cost*(1 - self.alpha) + overlaps*self.alpha
            # import pdb; pdb.set_trace()

        # assign 0 by default  处理无真实框或锚框的情况
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt 空间先验
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1) #真实框的中心点 gt_points

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1) #锚框的中心点 bboxes_points

        if bbox_preds is not None:
            bboxes_cxp = (bbox_preds[:, 0] + bbox_preds[:, 2]) / 2.0
            bboxes_cyp = (bbox_preds[:, 1] + bbox_preds[:, 3]) / 2.0
            bboxes_pointsp = torch.stack((bboxes_cxp, bboxes_cyp), dim=1) #预测框 bbox_preds中心点，后续用于限制中心点在GT内
#欧氏距离
        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt() #计算每个锚框到每个真实框的欧氏距离；通过广播机制，两个张量被扩展为形状 (N, M, D)，然后逐元素相减，得到每个点对在每个维度上的差值
        #处理需忽略的真实框（crowd）
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance 为每个真实框选择候选正样本（根据中心距离）
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]  #从一个全局的距离矩阵中提取出属于当前特征层的部分
            selectable_k = min(self.topk, bboxes_per_level)

            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)   #目标检测的标签分配（如 ATSS）中，为每个真实目标从当前特征层中选取距离最近的 selectable_k 个锚框
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold 计算IoU阈值并确定候选框中满足条件的正样本
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)] #每个候选框与其对应真实框的QAF。  PyTorch 的整数数组索引（advanced indexing），从 QAF矩阵 overlaps 中提取出每个真实目标（GT）对应的候选锚框的 QAF 值。
        overlaps_mean_per_gt = candidate_overlaps.mean(0)  #对每个真实框，计算其所有候选框 QAF的均值和标准差，
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt #阈值 = 均值 + 标准差。

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :] #标记候选框中 QAF大于等于阈值的那些。筛选正样本 IoU>=阈值

        # limit the positive sample's center in gt  限制正样本中心点必须在真实框内部
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes  #为每个真实目标（ground truth）的候选锚框索引添加一个偏移量，使得不同目标的候选索引映射到互不重叠的数值区间，从而便于后续处理（例如将锚框-目标对作为唯一标识符）。
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1) #（这个扁平化张量包含了每个真实框对应的锚框中心点，排列为 [gt0_anchor0, gt0_anchor1, ..., gt1_anchor0, ...]）
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)  #所有元素按行优先顺序被展平成一个长向量，每个元素代表锚框中心到真实框中心的欧氏距离

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side 锚框的中心到真实框四边的距离 普通计算方式判断锚框中心是否在真实框内
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01 #要求最小距离大于 0.01（考虑浮点误差），从而判断中心是否在框内。 在张量的第1维（dim=1）上计算最小值，并只返回最小值数值（不返回对应的索引）
        
        if bbox_preds is not None: #如果提供了预测框，也检查预测框的中心是否在真实框内，并与锚框中心取或
            ep_bboxes_cxp = bboxes_cxp.view(1, -1).expand(
                num_gt, num_bboxes).contiguous().view(-1)
            ep_bboxes_cyp = bboxes_cyp.view(1, -1).expand(
                num_gt, num_bboxes).contiguous().view(-1)
            l_p = ep_bboxes_cxp[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0] #预测框的中心到真实框四边的距离
            t_p = ep_bboxes_cyp[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
            r_p = gt_bboxes[:, 2] - ep_bboxes_cxp[candidate_idxs].view(-1, num_gt)
            b_p = gt_bboxes[:, 3] - ep_bboxes_cyp[candidate_idxs].view(-1, num_gt)
            is_in_gts_p = torch.stack([l_p, t_p, r_p, b_p], dim=1).min(dim=1)[0] > 0.01

            is_in_gts = torch.logical_or(is_in_gts, is_in_gts_p) # 锚框中心在真实框内（或其预测框中心在真实框内）。


        is_pos = is_pos & is_in_gts  #最终正样本候选需要同时满足：IoU >= 阈值 且 中心在真实框内（或其预测框中心在真实框内）。

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected. 处理一个锚框可能匹配多个真实框的情况，选择 IoU 最高的
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]  #从所有候选锚框-目标对中筛选出正样本对应的索引 长度等于正样本的数量
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]  #从 IoU 矩阵中提取出正样本锚框-目标对的 IoU 值，并将这些值赋给另一个张量 overlaps_inf 中对应的位置
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()  #.t()转置 将经过稀疏赋值（如仅填充正样本 IoU）的 overlaps_inf 还原为与输入 overlaps 一致的形状，以便后续计算（例如 IoU 损失或质量损失）

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1) #对每个锚框，取其在所有真实框上的最大 IoU（即按列取 max）从 IoU 矩阵 overlaps_inf 中找出每个锚框的最大 IoU 值max_overlaps以及对应的真实目标索引argmax_overlaps
        assigned_gt_inds[  #max_overlaps != -INF 生成一个布尔张量，标记哪些锚框至少与一个 GT 有非无效的 IoU
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1  #将每个有效锚框分配给它对应的真实目标（通过最大 IoU 确定），并将目标索引（从 0 开始）转换为从 1 开始的格式，因为 assigned_gt_inds 中 0 通常表示背景，正样本从 1 开始编号，存储在 assigned_gt_inds 中
        #分配标签
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)  #初始化标签数组
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()  #从 assigned_gt_inds 中提取所有正样本锚框的索引
            if pos_inds.numel() > 0: #返回张量中元素总数，检查张量 pos_inds 中是否存在正样本
                assigned_labels[pos_inds] = gt_labels[  #assigned_labels 中正样本位置存储了它们应预测的类别 ID，而其他位置（背景或忽略）保持原值（如 0 或 -1），用于后续分类损失的计算
                    assigned_gt_inds[pos_inds] - 1] #为正样本锚框分配类别标签 assigned_gt_inds[pos_inds]：取出每个正样本锚框所分配到的真实目标索引（1-based，例如 1 表示第 1 个真实目标） 将目标索引转换为 0-based，以便从 gt_labels 中正确索引（因为 Python 索引从 0 开始）。
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
