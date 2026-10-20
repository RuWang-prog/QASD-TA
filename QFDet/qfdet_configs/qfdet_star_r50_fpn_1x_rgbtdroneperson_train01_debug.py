dataset_type = 'RGBTyxy'
# data_root = '/media/zhx/DATA/yxy/rgbt_tiny/rgbt_mmrgbt/'
data_root = '../annotations_coco/'  #wr
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True, #表示将输入图像从 BGR 转为 RGB（因 OpenCV 默认 BGR）。
    mean_list=([115.37, 121.82, 122.63], [93.1, 93.1, 93.1]),
    std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]))
train_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 640), keep_ratio=True),  #keep_ratio=True 保持长宽比（不足部分填充）
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='MultiNormalize',
        mean_list=([115.37, 121.82, 122.63], [93.1, 93.1, 93.1]),
        std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]),
        to_rgb=True),
    dict(type='Pad', size_divisor=32), #将图像填充至能被 32 整除的尺寸（适配 FPN 等网络的下采样倍数）
    dict(type='DefaultFormatBundle'), #将数据转换为 MMDetection 内部使用的统一格式（如将 numpy 数组转为 Tensor）。
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']) #保留后续模型所需的键值，包括图像 img（此时为多模态拼接或堆叠后的张量）、真实边界框和标签。
]
test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(
        type='MultiScaleFlipAug', #测试时数据增强封装器
        img_scale=(512, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='MultiNormalize',
                mean_list=([115.37, 121.82, 122.63], [93.1, 93.1, 93.1]),
                std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]),
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']) #内部变换与训练基本一致，但无需加载标注，Collect 只保留图像
        ])
]
data = dict(
    samples_per_gpu=2, # 每个 GPU 的 batch size
    workers_per_gpu=4,  # 每个 GPU 的数据加载线程数
    train=dict(
        type='RGBTyxy',
        # ann_file='/media/zhx/DATA/yxy/rgbt_tiny/rgbt_mmrgbt/train01.json',
        ann_file='../annotations_coco/instances_01_train2017_debug.json',  #wr
        # img_prefix='/media/zhx/DATA/yxy/rgbt_tiny/rgbt_mmrgbt/train/',
        img_prefix='../annotations_coco/train_debug01/',  #wr
        pipeline=[  #pipeline=train_pipeline
            dict(
                type='LoadImagePairFromFile',
                spectrals=('visible', 'thermal')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(512, 640), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='MultiNormalize',
                mean_list=([115.37, 121.82, 122.63], [93.1, 93.1, 93.1]),
                std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]),
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='RGBTyxy',
        # ann_file='/media/zhx/DATA/yxy/rgbt_tiny/rgbt_mmrgbt/val01.json',
        ann_file='../annotations_coco/instances_01_test2017_debug.json',  #wr
        # img_prefix='/media/zhx/DATA/yxy/rgbt_tiny/rgbt_mmrgbt/val/',
        img_prefix='../annotations_coco/val_debug01/',   #wr
        pipeline=[
            dict(
                type='LoadImagePairFromFile',
                spectrals=('visible', 'thermal')),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='MultiNormalize',
                        mean_list=([115.37, 121.82,
                                    122.63], [93.1, 93.1, 93.1]),
                        std_list=([85.13, 89.01, 88.27], [50.24, 50.24,
                                                          50.24]),
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='RGBTyxy',
        # ann_file='/media/zhx/DATA/yxy/rgbt_tiny/rgbt_mmrgbt/val01.json',
        ann_file='../annotations_coco/instances_01_test2017_debug.json',  #wr
        # img_prefix='/media/zhx/DATA/yxy/rgbt_tiny/rgbt_mmrgbt/val/',
        img_prefix='../annotations_coco/val_debug01/',  #wr
        pipeline=[
            dict(
                type='LoadImagePairFromFile',
                spectrals=('visible', 'thermal')),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='MultiNormalize',
                        mean_list=([115.37, 121.82,
                                    122.63], [93.1, 93.1, 93.1]),
                        std_list=([85.13, 89.01, 88.27], [50.24, 50.24,
                                                          50.24]),
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')  #评估与优化策略
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  # 梯度裁剪
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])  # 在第 8 和 11 epoch 时将学习率乘以 0.1（默认衰减因子）
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)  #日志、检查点与钩子
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]   # 检查类别数是否与模型一致
dist_params = dict(backend='nccl') # 分布式与运行环境
log_level = 'INFO'
load_from = None  #不加载预训练权重
resume_from = None  # 不恢复训练
workflow = [('train', 1)]   # 仅训练，无验证循环
opencv_num_threads = 0  # OpenCV 线程数，0 表示自动
mp_start_method = 'fork'    # 多进程启动方式
auto_scale_lr = dict(enable=False, base_batch_size=16) # 关闭自动学习率缩放
model = dict(     #模型定义（核心）
    type='QFDet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,  # 冻结第一个 stage 的参数
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5),  # 输出 5 层特征图（P3-P7）
    bbox_head=dict(
        type='ATSSQHead',  # ATSS 变体，可能引入质量预测
        num_classes=7,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        centerness=1,
        anchor_generator=dict(  # 生成锚框，步长对应 FPN 各层
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(  # 边界框编解码方式
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        # loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_bbox=dict(type='SafitLoss', eps=1e-7, C=32, reduction='mean', loss_weight=2.0),   #wr20260403
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    bbox_prehead=dict(  # 辅助预头，可能用于质量估计
        type='QFDetPreHead',
        num_classes=7,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        centerness=1,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5),
        # loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_bbox=dict(type='SafitLoss', eps=1e-7, C=32, reduction='mean', loss_weight=1.0),  #wr20260403
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5),
        loss_quality=dict(type='MSELoss', loss_weight=0.5)),  # 新增质量损失
    base_fusion='cat',
    quality_attention=True, # 启用质量注意力机制
    poolupsample=1,
    train_cfg=dict(
        assigner=dict(
            type='QLSAssigner',  #创新点 创新标签分配器，基于质量感知
            topk=9,
            alpha=0.8,
            quality='x',  # 质量度量方式
            iou_calculator=dict(type='BboxDistanceMetric'),
            iou_mode='siwd',   # 一种新的 IoU 计算方式（如尺度不变）
            overlap_mode='hybrid'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.3),
        # nms=dict(type='safit_nms', iou_threshold=0.3, C=32),   #wr20260424
        max_per_img=100))
# work_dir = '/media/zhx/DATA/yxy/0827/mmdet-rgbtdroneperson-main/workdir/cx'
work_dir = '../workdir/cx'  #wr
auto_resume = False
gpu_ids = [0]
