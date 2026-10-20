_base_ = [
    '../configs/_base_/datasets/coco_detection.py',   #已经改过
    '../configs/_base_/schedules/schedule_1x.py', '../configs/_base_/default_runtime.py'
]
model = dict(
    type='QFDet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0, # from p2
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSQHead',
        num_classes=7,               #cx   默认是3
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
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    bbox_prehead=dict(
        type='QFDetPreHead',
        num_classes=7,   #cx   默认是3
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
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5),
        loss_quality=dict(type='MSELoss', loss_weight=0.5)),
    base_fusion='cat',
    quality_attention=True,
    poolupsample=1,
    # reweight=True,
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='QLSAssigner', 
                      topk=9,
                      alpha=0.8,
                      quality='x',
                      iou_calculator=dict(type='BboxDistanceMetric'),
                      iou_mode='siwd',
                      overlap_mode='hybrid',
                      ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.3),
        max_per_img=100))

# dataset settings
# dataset_type = 'DronePerson'                 #在datasets/中修改类别
dataset_type = 'RGBTyxy'                                     
# data_root = '/media/zhx/DATA/yxy/RGBTDronePerson/'     #cx 改
# data_root='/media/zhx/DATA/yxy/rgbt_tiny/rgbt_mmrgbt/'
data_root='/home/wang/RGBT-Tiny_data/'   #wr 改
# data_root='/media/zhx/DATA/yxy/rgbt_tiny/rgbt_tiny_origin/'
img_norm_cfg = dict(
    mean_list=([115.37, 121.82, 122.63], [93.10, 93.10, 93.10]),
    std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]), to_rgb=True)
train_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 640), keep_ratio=True),     #cx  默认（640，512）
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='MultiNormalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 640),   #cx  默认（640，512）
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='MultiNormalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,   #cx batchsize!
    workers_per_gpu=4,
    # samples_per_gpu=5,   #wr
    # workers_per_gpu=8,   #wr
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'train00.json',   #cx,尚且不知道文件夹中的annotation是否有用，因为这里已经用了json，怀疑json是anno中的xml转的
        ann_file=data_root + 'annotations_coco/instances_00_train2017.json',    #wr
        # img_prefix=data_root + 'train/',   #cx  改！！！！！！！！！！！train
        img_prefix=data_root + 'train/',       #wr
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'val00.json',   #cx
        ann_file=data_root + 'annotations_coco/instances_00_test2017.json',
        img_prefix=data_root + 'val/',      #cx
        pipeline=test_pipeline), 
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations_cx_new/instances_01_test2017.json',   #cx
        # img_prefix=data_root + 'test2017_cx/01/',      #cx
        # ann_file=data_root + 'val00.json',   #cx
        ann_file=data_root + 'annotations_coco/instances_00_test2017.json',   #wr
        img_prefix=data_root + 'val/',      #cx
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(_delete_=True, max_norm=35, norm_type=2))

# work_dir = '/media/zhx/DATA/yxy/0827/mmdet-rgbtdroneperson-main/workdir/cx00'   #cx
work_dir = '/home/wang/QFDet/code/QFDet/mmdet-rgbtdroneperson-main/workdir/cx00'     #wr