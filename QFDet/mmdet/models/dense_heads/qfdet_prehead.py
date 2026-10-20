# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead


@HEADS.register_module()
class QFDetPreHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 pred_kernel_size=3,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 reg_decoded_bbox=True,
                 centerness=None,
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_quality=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='atss_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(QFDetPreHead, self).__init__(
            num_classes,
            in_channels,
            reg_decoded_bbox=reg_decoded_bbox,
            init_cfg=init_cfg,
            **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_quality = build_loss(loss_quality)
        self.centerness = centerness

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        pred_pad_size = self.pred_kernel_size // 2
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.atss_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.atss_centerness = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 1,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.atss_quality = nn.Conv2d(
            self.feat_channels * 2,
            self.num_base_priors * 1,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        x_v, x_t = feats
        return multi_apply(self.forward_single, x_v, x_t, self.scales)

    def forward_single(self, x_v, x_t, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat_v = x_v
        cls_feat_t = x_t
        reg_feat_v = x_v
        reg_feat_t = x_t

        for cls_conv in self.cls_convs:
            cls_feat_t = cls_conv(cls_feat_t)
            cls_feat_v = cls_conv(cls_feat_v)
        for reg_conv in self.reg_convs:
            reg_feat_t = reg_conv(reg_feat_t)
            reg_feat_v = reg_conv(reg_feat_v)
        cls_score_t = self.atss_cls(cls_feat_t)
        cls_score_v = self.atss_cls(cls_feat_v)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred_t = scale(self.atss_reg(reg_feat_t)).float()
        centerness_t = self.atss_centerness(reg_feat_t)
        quality_pred_t = torch.sigmoid(self.atss_quality(torch.cat([reg_feat_t, cls_feat_t], dim=1)))
        bbox_pred_v = scale(self.atss_reg(reg_feat_v)).float()
        centerness_v = self.atss_centerness(reg_feat_v)
        quality_pred_v = torch.sigmoid(self.atss_quality(torch.cat([reg_feat_v, cls_feat_v], dim=1)))

        return cls_score_t, bbox_pred_t, centerness_t, quality_pred_t, \
                cls_score_v, bbox_pred_v, centerness_v, quality_pred_v
    
    def forward_test(self, feats):
        x_vs, x_ts = feats
        return multi_apply(self.forward_test_single, x_vs, x_ts, self.scales)

    def forward_test_single(self, x_v, x_t, scale):
        cls_feat_v = x_v
        cls_feat_t = x_t
        reg_feat_v = x_v
        reg_feat_t = x_t

        for cls_conv in self.cls_convs:
            cls_feat_t = cls_conv(cls_feat_t)
            cls_feat_v = cls_conv(cls_feat_v)

        for reg_conv in self.reg_convs:
            reg_feat_t = reg_conv(reg_feat_t)
            reg_feat_v = reg_conv(reg_feat_v)

        quality_pred_t = torch.sigmoid(self.atss_quality(torch.cat([reg_feat_t, cls_feat_t], dim=1)))
        quality_pred_v = torch.sigmoid(self.atss_quality(torch.cat([reg_feat_v, cls_feat_v], dim=1)))
        
        return quality_pred_t, quality_pred_v

    def loss_single(self, anchors, cls_score, bbox_pred, centerness, quality_pred, labels,
                    label_weights, bbox_targets, quality_targets, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        quality_pred = quality_pred.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        quality_targets = quality_targets.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        quality_targets=quality_targets.detach().requires_grad_()  #将张量 quality_targets 从当前计算图中分离，并重新启用梯度追踪，使其成为一个需要梯度的独立叶子张量
        loss_quality = self.loss_quality(quality_pred, quality_targets)
        
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1) #从标签张量中提取所有正样本锚框的索引 0 到 num_classes-1：正样本的类别索引 num_classes：背景类别（即 bg_class_ind
        
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds] #正样本对应的回归目标
            pos_bbox_pred = bbox_pred[pos_inds] #模型预测的正样本回归值
            pos_anchors = anchors[pos_inds] #正样本对应的锚框
            pos_centerness = centerness[pos_inds] #正样本的中心度预测

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode( #将预测的回归偏移量解码为实际的边界框坐标
                pos_anchors, pos_bbox_pred)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,  #预测边界框 xyxy
                pos_bbox_targets,  #真实边界框 xyxy
                weight=centerness_targets, #为每个正样本边界框分配一个独立的损失权重
                avg_factor=1.0)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets,
                avg_factor=num_total_samples)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_centerness, loss_quality, centerness_targets.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'quality_preds'))
    def loss(self,
             cls_scores_t,
             bbox_preds_t,
             centernesses_t,
             quality_preds_t,
             cls_scores_v,
             bbox_preds_v,
             centernesses_v,
             quality_preds_v,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores_t]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        num_level = len(cls_scores_t)
        batch_size = cls_scores_t[0].shape[0]

        device = cls_scores_t[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets_t = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            cls_scores=cls_scores_t,
            bbox_preds=bbox_preds_t,
            label_channels=label_channels)
        if cls_reg_targets_t is None:
            return None
        
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device) #目标检测框架（如 MMDetection）中获取锚框（anchor）及其有效性标志的常见操作

        cls_reg_targets_v = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            cls_scores=cls_scores_v,
            bbox_preds=bbox_preds_v,
            label_channels=label_channels)
        if cls_reg_targets_v is None:
            return None

        (anchor_list_t, labels_list_t, label_weights_list_t, bbox_targets_list_t,
         bbox_weights_list_t, quality_targets_list_t, num_total_pos_t, num_total_neg_t, _) = cls_reg_targets_t

        (anchor_list_v, labels_list_v, label_weights_list_v, bbox_targets_list_v,
         bbox_weights_list_v, quality_targets_list_v, num_total_pos_v, num_total_neg_v, pos_matched_gt_inds) = cls_reg_targets_v
        
        num_total_samples_t = reduce_mean(
            torch.tensor(num_total_pos_t, dtype=torch.float,
                         device=device)).item()  #在分布式训练中计算所有 GPU 上正样本数量的平均值，检查当前是否处于有效的分布式训练环境中，如果不是则直接返回输入张量，不执行任何分布式操作
        num_total_samples_t = max(num_total_samples_t, 1.0)
        
        num_total_samples_v = reduce_mean(
            torch.tensor(num_total_pos_v, dtype=torch.float,
                         device=device)).item()
        num_total_samples_v = max(num_total_samples_v, 1.0)
        
        losses_cls_t, losses_bbox_t, loss_centerness_t, loss_quality_t,\
            bbox_avg_factor_t = multi_apply(
                self.loss_single,
                anchor_list_t,
                cls_scores_t,
                bbox_preds_t,
                centernesses_t,
                quality_preds_t,
                labels_list_t,
                label_weights_list_t,
                bbox_targets_list_t,
                quality_targets_list_t,
                num_total_samples=num_total_samples_t) #对热红外模态（用 _t 表示）的多个特征层同时计算损失 损失函数 self.loss_single 并行应用到每个特征层的输入数据上，并收集各层返回的损失分量
        
        losses_cls_v, losses_bbox_v, loss_centerness_v, loss_quality_v,\
            bbox_avg_factor_v = multi_apply(
                self.loss_single,
                anchor_list_v,
                cls_scores_v,
                bbox_preds_v,
                centernesses_v,
                quality_preds_v,
                labels_list_v,
                label_weights_list_v,
                bbox_targets_list_v,
                quality_targets_list_v,
                num_total_samples=num_total_samples_v)

        bbox_avg_factor_t = sum(bbox_avg_factor_t)
        bbox_avg_factor_t = reduce_mean(bbox_avg_factor_t).clamp_(min=1).item()
        losses_bbox_t = list(map(lambda x: x / bbox_avg_factor_t, losses_bbox_t)) #列表 losses_bbox_t 中的每个元素（通常是各特征层计算出的未归一化的边界框损失）除以全局归一化因子 bbox_avg_factor_t，从而得到归一化后的各层损失，并更新原列表

        bbox_avg_factor_v = sum(bbox_avg_factor_v)
        bbox_avg_factor_v = reduce_mean(bbox_avg_factor_v).clamp_(min=1).item()
        losses_bbox_v = list(map(lambda x: x / bbox_avg_factor_v, losses_bbox_v))

        return dict(
            loss_cls_pre=losses_cls_t+losses_cls_v,
            loss_bbox_pre=losses_bbox_t+losses_bbox_v,
            loss_centerness_pre=loss_centerness_t+loss_centerness_v,
            loss_quality_pre=loss_quality_t+loss_quality_v,
            )

    def centerness_target(self, anchors, gts):
        # only calculate pos centerness targets, otherwise there may be nan
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]  #计算锚框中心到真实框四条边的距离
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        if self.centerness == 0: #（FCOS 原始公式）
            centerness = torch.sqrt(
                (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
                (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        elif self.centerness == 1:  #Manhattaness
            r = (top_bottom.sum(dim=-1) + left_right.sum(dim=-1)) / 2
            centerness = (left_right.min(dim=-1)[0] + top_bottom.min(dim=-1)[0]) / r
        centerness = torch.clamp(centerness, min=0, max=1)
        assert not torch.isnan(centerness).any()
        return centerness

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    cls_scores=None,
                    bbox_preds=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i]) #为了方便后续逐图像处理，将每张图像的所有级别锚点拼接成一个一维张量（形状为 [total_anchors_i, 4]）
            valid_flag_list[i] = torch.cat(valid_flag_list[i]) #同时将有效标志也拼接成一个一维布尔张量。这样每张图像只有一个锚点集合。

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        if cls_scores is None:
            cls_scores = [None for _ in range(num_imgs)]
        if bbox_preds is None:
            bbox_preds = [None for _ in range(num_imgs)]
        
        num_levels = len(cls_scores)
        cls_score_list = [] #预测分数
        bbox_pred_list = [] #边界框预测

        mlvl_cls_score_list = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.num_base_priors * self.cls_out_channels)
            for cls_score in cls_scores
        ]
        mlvl_bbox_pred_list = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_base_priors * 4)
            for bbox_pred in bbox_preds
        ]

        for i in range(num_imgs):
            mlvl_cls_tensor_list = [
                mlvl_cls_score_list[j][i] for j in range(num_levels)
            ]
            mlvl_bbox_tensor_list = [
                mlvl_bbox_pred_list[j][i] for j in range(num_levels)
            ]
            cat_mlvl_cls_score = torch.cat(mlvl_cls_tensor_list, dim=0)
            cat_mlvl_bbox_pred = torch.cat(mlvl_bbox_tensor_list, dim=0)
            cls_score_list.append(cat_mlvl_cls_score)
            bbox_pred_list.append(cat_mlvl_bbox_pred)


        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_quality_targets, pos_inds_list, neg_inds_list, pos_matched_gt_inds) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             cls_score_list,
             bbox_pred_list,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors ATSS 检测头中 get_targets 方法的最后一部分，负责处理多图像批量生成的目标，并进行后处理，最终返回便于损失函数计算的数据结构。
        if any([labels is None for labels in all_labels]): # 检查是否有无效图像
            return None
        # sampled anchors of all images 统计正负样本总数
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        
        # split targets to a list w.r.t. multiple levels 按特征图层级拆分目标； 这种拆分是为了后续损失函数能够按层级遍历，并与网络输出的特征图大小对齐（网络输出也是按层级组织的）。
        anchors_list = images_to_levels(all_anchors, num_level_anchors) #images_to_levels 是 MMDetection 提供的工具函数，作用是将每张图像独立的目标重新组织成按特征图层级聚合的形式。
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        quality_targets_list = images_to_levels(all_quality_targets,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, quality_targets_list, num_total_pos,
                num_total_neg, pos_matched_gt_inds)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           cls_scores,
                           bbox_preds,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            cls_scores (Tensor): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W)
            bbox_preds (Tensor): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W)
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, #用于生成一个布尔标志向量 inside_flags，指示哪些锚框位于图像内部
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :] #只保留在图像内部的锚点

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags) #统计每个特征层上位于图像内部的锚框数量
        
        dec_bbox_preds = self.bbox_coder.decode(anchors, bbox_preds) #预测的边界框偏移量解码后得到预测框的绝对坐标
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels,
                                             cls_scores=cls_scores,
                                             bbox_preds=dec_bbox_preds) # ATSS assigner 使用 ATSS 分配器进行分配  至此

        sampling_result = self.sampler.sample(assign_result, anchors,  #.sample() 方法根据输入的分配结果、锚框坐标和真实边界框，决定哪些锚框实际参与损失计算（即最终的训练样本）
                                              gt_bboxes) #根据标签分配结果（assign_result）进一步对锚框进行采样，以平衡正负样本或筛选困难样本
        
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors) #回归目标 存回归目标，预留给正样本填坐标差值。
        bbox_weights = torch.zeros_like(anchors) #回归权重  回归损失的开关（0代表不计筭回归损失，1代表计算）
        labels = anchors.new_full((num_valid_anchors, ), #类别标签 存分类标签
                                  self.num_classes, #即背景类索引，通常等于类别数，因为类别从 0 开始编号，背景用 num_classes 表示
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float) #标签权重   分类损失开关
        quality_targets = torch.clamp(assign_result.max_overlaps, min=0, max=1) #质量目标 将 assign_result.max_overlaps 中的数值裁剪到 [0, 1] 区间内，生成用于训练的质量目标 quality_targets
        
        #处理正样本
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox: # reg_decoded_bbox 为 True，直接使用真实框坐标（已解码）
                pos_bbox_targets = sampling_result.pos_gt_bboxes #SamplingResult 对象，其中 pos_gt_bboxes 属性存储了每个正样本所匹配的真实边界框坐标（形状为 (num_pos, 4)）
            else:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets #将回归目标填入 bbox_targets
            bbox_weights[pos_inds, :] = 1.0  #并将 bbox_weights 对应位置设为 1.0（表示参与回归损失），正样本的回归权重设为1,只有正样本才计算回归框的损失
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds] #从 gt_labels 中取出对应真实框的类别
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0  #正样本参与分类损失计算
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0: #处理负样本
            label_weights[neg_inds] = 1.0 #负样本的标签权重设为 1.0，表示它们参与分类损失（作为背景类），网络预测错了背景，要被扣分类的分，计算分类损失

        # map up to original set of anchors 映射回原始锚点集合（可选）
        if unmap_outputs: #越界黑户的还原
            num_total_anchors = flat_anchors.size(0) #原始所有锚框的总数
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights, quality_targets,
                pos_inds, neg_inds, sampling_result.pos_assigned_gt_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

