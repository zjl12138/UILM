_base_ = [
    './_base_/datasets/coco_with_sub_image_detection.py',
    './_base_/schedules/schedule_1x.py', './_base_/default_runtime.py'
]
sub_images=[
    'default.png',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='MultiLoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MultiResize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='MultiRandomFlip', flip_ratio=0.5),
    dict(type='MultiNormalize', **img_norm_cfg),
    dict(type='MultiPad', size_divisor=32),
    dict(type='MultiDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='MultiLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='MultiResize', keep_ratio=True),
            dict(type='MultiRandomFlip'),
            dict(type='MultiNormalize', **img_norm_cfg),
            dict(type='MultiPad', size_divisor=32),
            dict(type='MultiImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# model settings
model = dict(
    type='FusionFasterRCNN',
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
        num_outs=5),
    rpn_head=dict(
        type='GARPNHead',
        in_channels=256,
        feat_channels=256,
        approx_anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=8,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        square_anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[8],
            strides=[4, 8, 16, 32, 64]),
        anchor_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.07, 0.07, 0.14, 0.14]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.07, 0.07, 0.11, 0.11]),
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.05, 0.05, 0.1, 0.1]),            
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),

    train_cfg=dict(
        rpn=dict(
            ga_assigner=dict(
                type='ApproxMaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            ga_sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),    
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            pos_weight=-1,
            debug=False,
            allowed_border=-1,
            center_ratio=0.2,
            ignore_ratio=0.5),
        rpn_proposal=dict(
            nms_post=1000,
            nms_pre=2000,
            max_per_img=300,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms_post=300,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=1e-3,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)),
    sub_images=sub_images)

# optimizer setting
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

# data setting
classes = ('merge',)
dataset_folder = 'my_dataset_new'
data = dict(
    samples_per_gpu=4,  # Batch size of a single GPU
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix=f'{dataset_folder}/images',
        classes=classes,
        ann_file=f'{dataset_folder}/train.json',
        sub_images=sub_images,
        pipeline=train_pipeline
    ),
    val=dict(
        img_prefix=f'{dataset_folder}/images',
        classes=classes,
        ann_file=f'{dataset_folder}/test.json',
        sub_images=sub_images,
        pipeline=test_pipeline
    ),
    test=dict(
        img_prefix=f'{dataset_folder}/images',
        classes=classes,
        ann_file=f'{dataset_folder}/test.json',
        sub_images=sub_images,
        pipeline=test_pipeline
    )
)
