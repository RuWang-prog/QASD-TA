# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.ops import batched_nms
from mmcv.runner import BaseModule, force_fp32

from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmdet.core.post_processing.safit_nms import safit_nms #wr20260403

class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)

    def init_weights(self):
        super(BaseDenseHead, self).init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None: # score_factors指的是centerness
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id) #从多尺度特征图（multi-level）的预测结果中，提取出单个图像对应的数据
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs) #目标检测框架（如 MMDetection）中单张图像后处理的核心调用，用于将模型的多层预测结果转换为最终的检测框。
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs] #过滤——只保留与最终检测框相关的分数因子

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs) #后处理多层级预测结果的核心方法调用，它接收来自多个特征层的分类分数、标签、边界框预测以及相关元信息，经过一系列转换（如缩放、解码、NMS）后，返回最终的检测结果。

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                return det_bboxes, mlvl_labels
# #wr20260427 尺度自适应融合IoU+真正NWD的NMS
            if cfg.nms is not None and cfg.nms.get('type') == 'safit_nms':
                # ---- Safit NMS (修正版) ----
                # 读取分尺度阈值（提供默认值以防未配置）
                # thr_small = cfg.nms.get('thr_small', 0.7)
                # thr_medium = cfg.nms.get('thr_medium', 0.5)

                # 偏移方法：为每个类别生成不同的偏移量，使不同类别的框在坐标空间上完全分离
                max_coordinate = mlvl_bboxes.max()
                offsets = mlvl_labels.to(mlvl_bboxes) * (max_coordinate + 1)
                boxes_for_nms = mlvl_bboxes + offsets[:, None]
#
                # keep = safit_nms(boxes_for_nms, mlvl_scores,
                #                  iou_threshold=iou_thr,
                #                  C=C, C_nwd=C_nwd,
                #                  soft_mode=cfg.nms.get('soft_mode', False))

                keep = safit_nms(boxes_for_nms, mlvl_scores)



                det_bboxes = torch.cat([mlvl_bboxes[keep], mlvl_scores[keep].reshape(-1, 1)], dim=1)
                keep_idxs = keep

                det_bboxes = det_bboxes[:cfg.max_per_img]  # 目标检测后处理中限制每张图像最终输出检测框数量的常见操作。
                det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]  # 获取最终保留的检测框对应的类别标签，并限制数量
                return det_bboxes, det_labels


            # wr20260511 safit-WBF
            # if cfg.nms is not None and cfg.nms.get('type') == 'safit_nms':  # 或者改成 safit_wbf
            #     # ---- Safit WBF (加权融合版) ----
            #
            #     # 1. 提取超参 (建议从 cfg 中获取，以便在 config 文件中调参)
            #     # nwd_thr = cfg.nms.get('nwd_thr', 0.85)
            #     # C_nwd = cfg.nms.get('C_nwd', 32.0)
            #     # min_score = cfg.nms.get('min_score', 0.001)
            #
            #     # 2. 直接调用 WBF，内部已用 labels 处理好多类别分离，无需 offsets！
            #     # 注意我们这里接收了 3 个返回值 (新框，新分数，新类别)
            #     fused_bboxes, fused_scores, det_labels = safit_nms(
            #         boxes=mlvl_bboxes,
            #         scores=mlvl_scores,
            #         labels=mlvl_labels,
            #     )
            #
            #     # 3. 拼接为 MMDet 期望的 [N, 5] 格式 (坐标 x1, y1, x2, y2 + score)
            #     if fused_bboxes.numel() == 0:
            #         det_bboxes = torch.cat([fused_bboxes, fused_scores.reshape(-1, 1)], dim=1)
            #     else:
            #         det_bboxes = torch.cat([fused_bboxes, fused_scores.reshape(-1, 1)], dim=1)
            #
            #     # 1. 限制检测框数量
            #     det_bboxes = det_bboxes[:cfg.max_per_img]
            #
            #     # 2. 【核心修改】直接截断 WBF 传出来的 det_labels，绝对不能再去 mlvl_labels 里索引！
            #     det_labels = det_labels[:cfg.max_per_img]
            #
            #     return det_bboxes, det_labels


                # [避坑警告] 往下看你的 base_dense_head.py 代码，
                # 如果在这段代码下面有一行类似： det_labels = mlvl_labels[keep_idxs]
                # 你必须把它注释掉或者包在 else 里！因为我们的 det_labels 已经直接生成好了！

            #wr20260403 按偏移抑制
            # if cfg.nms is not None and cfg.nms.get('type') == 'safit_nms':
            #     # 自定义 Safit NMS
            #     iou_thr = cfg.nms.get('iou_threshold', 0.3)
            #     C = cfg.nms.get('C', 32)
            #     adaptive = cfg.nms.get('adaptive', False)
            #     base_thresh = cfg.nms.get('base_thresh', 0.3)
            #     scale = cfg.nms.get('scale', 0.3)
            #     #偏移方法：为每个类别生成不同的偏移量，使不同类别的框在坐标空间上完全分离
            #     max_coordinate = mlvl_bboxes.max()
            #     offsets = mlvl_labels.to(mlvl_bboxes) * (max_coordinate + 1)
            #     boxes_for_nms = mlvl_bboxes + offsets[:, None]
            #     keep = safit_nms(boxes_for_nms, mlvl_scores, iou_threshold=iou_thr, C=C)
            #     det_bboxes = torch.cat([mlvl_bboxes[keep], mlvl_scores[keep].reshape(-1, 1)], dim=1)
            #     keep_idxs = keep
            else:
                det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                    mlvl_labels, cfg.nms) #返回的 det_bboxes 形状为 (K,5)，已按分数降序排列；keep_idxs 是保留框在原始输入中的索引（顺序与 det_bboxes 一致）。
                det_bboxes = det_bboxes[:cfg.max_per_img]  # 目标检测后处理中限制每张图像最终输出检测框数量的常见操作。
                det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img] #获取最终保留的检测框对应的类别标签，并限制数量
                return det_bboxes, det_labels
            # #wr20260418  按类别抑制
            # if cfg.nms is not None and cfg.nms.get('type') == 'safit_nms':
            #     iou_thr = cfg.nms.get('iou_threshold', 0.3)
            #     C = cfg.nms.get('C', 32)  # 从配置中读取 C，如果没有则默认 32
            #     adaptive = cfg.nms.get('adaptive', False)  # 是否启用自适应阈值
            #     base_thresh = cfg.nms.get('base_thresh', 0.3)  # 基础阈值（大目标使用）
            #     scale = cfg.nms.get('scale', 0.3)  # 阈值调节幅度
            #     # 按类别单独执行 Safit NMS
            #     unique_labels = mlvl_labels.unique()
            #     all_keep = []
            #     for label in unique_labels:
            #         mask = mlvl_labels == label
            #         cls_boxes = mlvl_bboxes[mask]
            #         cls_scores = mlvl_scores[mask]
            #         if cls_boxes.numel() == 0:
            #             continue
            #         # 调用单类别 Safit NMS（不需要偏移量）
            #         cls_keep = safit_nms(cls_boxes, cls_scores, iou_thr, adaptive, base_thresh, scale, C)
            #         # 将局部索引转换为全局索引
            #         global_keep = torch.where(mask)[0][cls_keep]
            #         all_keep.append(global_keep)
            #     if len(all_keep) > 0:
            #         keep = torch.cat(all_keep, dim=0)
            #     else:
            #         keep = torch.empty(0, dtype=torch.long, device=mlvl_bboxes.device)
            #     det_bboxes = torch.cat([mlvl_bboxes[keep], mlvl_scores[keep].reshape(-1, 1)], dim=1)
            #     keep_idxs = keep
            # else:
            #     det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
            #                                         mlvl_labels, cfg.nms)

            # det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
            #                                     mlvl_labels, cfg.nms) #多类别非极大值抑制的函数

        #     det_bboxes = det_bboxes[:cfg.max_per_img] #目标检测后处理中限制每张图像最终输出检测框数量的常见操作。
        #     det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img] #获取最终保留的检测框对应的类别标签，并限制数量
        #     return det_bboxes, det_labels
        # else:
        #     return mlvl_bboxes, mlvl_scores, mlvl_labels


        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def onnx_export(self,
                    cls_scores,
                    bbox_preds,
                    score_factors=None,
                    img_metas=None,
                    with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            score_factors (list[Tensor]): score_factors for each s
                cale level with shape (N, num_points * 1, H, W).
                Default: None.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc. Default: None.
            with_nms (bool): Whether apply nms to the bboxes. Default: True.

        Returns:
            tuple[Tensor, Tensor] | list[tuple]: When `with_nms` is True,
            it is tuple[Tensor, Tensor], first tensor bboxes with shape
            [N, num_det, 5], 5 arrange as (x1, y1, x2, y2, score)
            and second element is class labels of shape [N, num_det].
            When `with_nms` is False, first tensor is bboxes with
            shape [N, num_det, 4], second tensor is raw score has
            shape  [N, num_det, num_classes].
        """
        assert len(cls_scores) == len(bbox_preds)

        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shape = img_metas[0]['img_shape_for_onnx']

        cfg = self.test_cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_priors)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)

        # e.g. Retina, FreeAnchor, etc.
        if score_factors is None:
            with_score_factors = False
            mlvl_score_factor = [None for _ in range(num_levels)]
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True
            mlvl_score_factor = [
                score_factors[i].detach() for i in range(num_levels)
            ]
            mlvl_score_factors = []

        mlvl_batch_bboxes = []
        mlvl_scores = []

        for cls_score, bbox_pred, score_factors, priors in zip(
                mlvl_cls_scores, mlvl_bbox_preds, mlvl_score_factor,
                mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(0, 2, 3,
                                       1).reshape(batch_size, -1,
                                                  self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = scores.sigmoid()
                nms_pre_score = scores
            else:
                scores = scores.softmax(-1)
                nms_pre_score = scores

            if with_score_factors:
                score_factors = score_factors.permute(0, 2, 3, 1).reshape(
                    batch_size, -1).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            priors = priors.expand(batch_size, -1, priors.size(-1))
            # Get top-k predictions
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:

                if with_score_factors:
                    nms_pre_score = (nms_pre_score * score_factors[..., None])
                else:
                    nms_pre_score = nms_pre_score

                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = nms_pre_score.max(-1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = nms_pre_score[..., :-1].max(-1)
                _, topk_inds = max_scores.topk(nms_pre)

                batch_inds = torch.arange(
                    batch_size, device=bbox_pred.device).view(
                        -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                transformed_inds = bbox_pred.shape[1] * batch_inds + topk_inds
                priors = priors.reshape(
                    -1, priors.size(-1))[transformed_inds, :].reshape(
                        batch_size, -1, priors.size(-1))
                bbox_pred = bbox_pred.reshape(-1,
                                              4)[transformed_inds, :].reshape(
                                                  batch_size, -1, 4)
                scores = scores.reshape(
                    -1, self.cls_out_channels)[transformed_inds, :].reshape(
                        batch_size, -1, self.cls_out_channels)
                if with_score_factors:
                    score_factors = score_factors.reshape(
                        -1, 1)[transformed_inds].reshape(batch_size, -1)

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_batch_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            if with_score_factors:
                mlvl_score_factors.append(score_factors)

        batch_bboxes = torch.cat(mlvl_batch_bboxes, dim=1)
        batch_scores = torch.cat(mlvl_scores, dim=1)
        if with_score_factors:
            batch_score_factors = torch.cat(mlvl_score_factors, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment

        from mmdet.core.export import add_dummy_nms_for_onnx

        if not self.use_sigmoid_cls:
            batch_scores = batch_scores[..., :self.num_classes]

        if with_score_factors:
            batch_scores = batch_scores * (batch_score_factors.unsqueeze(2))

        if with_nms:
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_bboxes, batch_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        else:
            return batch_bboxes, batch_scores
