_base_ = [
    './_base_/models/faster_rcnn_r50_caffe_c4.py',
    './_base_/datasets/coco_with_sub_image_detection.py',
    './_base_/schedules/schedule_1x.py', './_base_/default_runtime.py'
]

sub_images = [
    'default.png'
]
norm_cfg = dict(type='BN', requires_grad=False)
model = dict(
    type='SiameseRPN',
    backbone=dict(
        type='TridentResNet',
        trident_dilations=(1, 2, 3),
        num_branch=3,
        test_branch_idx=-1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck = dict(
        type='FPN',
        in_channels=[1024,1024,1024],
        out_channels=1024,
        num_outs=3,
    ),
    rpn_head=dict(
        _delete_=True,
        type='GARPNHead',
        in_channels=1024,
        feat_channels=1024,
        approx_anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=8,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[16,32,64]),
        square_anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[8],
            strides=[16,32,64]),
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
        type='StandardRoIHead',
        shared_head=dict(
            type='ResLayer',
            depth=50,
            stage=3,
            stride=2,
            dilation=1,
            style='caffe',
            norm_cfg=norm_cfg,
            norm_eval=True),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16,32,64]),
        bbox_head=dict(
            type='BBoxHead',
            with_avg_pool=True,
            roi_feat_size=7,
            in_channels=2048,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
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
        rpn_proposal=dict(nms_post=1000, max_per_img=300),
        rcnn=dict(
            assigner=dict(pos_iou_thr=0.6, neg_iou_thr=0.6, min_pos_iou=0.6),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.25, add_gt_as_proposals=True))),
    test_cfg=dict(
        rpn=dict(nms_post=1000, max_per_img=300), rcnn=dict(score_thr=1e-3)),
    sub_images=sub_images
)
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# optimizer
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
classes = ('merge',)
data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix='my-dataset/images',
        classes=classes,
        ann_file='my-dataset/train.json',
        sub_images=sub_images
    ),
    val=dict(
        img_prefix='my-dataset/images',
        classes=classes,
        ann_file='my-dataset/test.json',
        sub_images=sub_images
    ),
    test=dict(
        img_prefix='my-dataset/images',
        classes=classes,
        ann_file='my-dataset/test.json',
        sub_images=sub_images
    )
)
