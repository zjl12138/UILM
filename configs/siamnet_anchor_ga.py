_base_ = [
    './_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/datasets/coco_with_sub_image_detection.py',
    './_base_/schedules/schedule_1x.py', './_base_/default_runtime.py'
]

sub_images=[
    'filled.png',
    'default.png'
]

model = dict(
    # type='SiameseRPNV2',
    type='SiameseRPNV3',
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
        in_channels=[512, 1024, 2048, 4096],
        out_channels=512,
        num_outs=5),
    rpn_head=dict(
        _delete_=True,
        type='GARPNHead',
        in_channels=512,
        feat_channels=512,
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
            target_means=[.0, .0, .0, .0],
            target_stds=[0.07, 0.07, 0.14, 0.14]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
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
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=512,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            in_channels=512,
            num_classes=1,
            bbox_coder=dict(target_stds=[0.05, 0.05, 0.1, 0.1]))),
    # model training and testing settings
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
            allowed_border=-1,
            center_ratio=0.2,
            ignore_ratio=0.5),
        rpn_proposal=dict(nms_post=1000, max_per_img=200),
        rcnn=dict(
            assigner=dict(pos_iou_thr=0.6, neg_iou_thr=0.6, min_pos_iou=0.6),
            sampler=dict(type='RandomSampler', num=256))),
    test_cfg=dict(
        rpn=dict(nms_post=1000, max_per_img=200), rcnn=dict(score_thr=1e-3)),
    sub_images = sub_images
)
# optimizer
optimizer = dict(lr=0.005)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

classes = ('merge',)
data = dict(
    samples_per_gpu=3,  # Batch size of a single GPU
    workers_per_gpu=3,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix='my-dataset/train',
        classes=classes,
        ann_file='my-dataset/train/train.json',
        sub_images=sub_images
    ),
    val=dict(
        img_prefix='my-dataset/test',
        classes=classes,
        ann_file='my-dataset/test/test.json',
        sub_images=sub_images
    ),
    test=dict(
        img_prefix='my-dataset/test',
        classes=classes,
        ann_file='my-dataset/test/test.json',
        sub_images=sub_images
    )
)