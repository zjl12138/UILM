# _base_ = ['../tridentnet/tridentnet_r50_caffe_1x_coco.py']
# _base_ = ['../tridentnet/tridentnet_r50_caffe_mstrain_3x_coco.py']
# _base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco.py'
_base_ = [
    './_base_/models/faster_rcnn_r50_caffe_c4.py',
    './_base_/datasets/coco_detection.py',
    './_base_/schedules/schedule_1x.py', './_base_/default_runtime.py'
]
classes = ('merge',)
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
rpn_weight = 0.7
model = dict(
    type='TridentFasterRCNN',
    backbone=dict(
        type='TridentResNet',
        trident_dilations=(1, 2, 3),
        num_branch=3,
        test_branch_idx=1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    rpn_head=dict(
        _delete_=True,
        type='CascadeRPNHead',
        num_stages=2,
        # self.num_levels = len(self.strides)
        # self.num_levels == len(featmap_sizes)
        stages=[
            dict(
                type='StageCascadeRPNHead',
                in_channels=1024,
                feat_channels=1024,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[1.0],
                    strides=[16]),
                adapt_cfg=dict(type='dilation', dilation=3),
                bridged_feature=True,
                sampling=False,
                with_cls=False,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.5, 0.5)),
                # loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight)),
            dict(
                type='StageCascadeRPNHead',
                in_channels=1024,
                feat_channels=1024,
                adapt_cfg=dict(type='offset'),
                bridged_feature=False,
                sampling=True,
                with_cls=True,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0 * rpn_weight),
                # loss_bbox=dict(type='L1Loss', loss_weight=1.0))
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight))
        ]
    ),
    roi_head=dict(
        type='TridentRoIHead',
        num_branch=3,
        test_branch_idx=1,
        bbox_head=dict(
            bbox_coder=dict(target_stds=[0.04, 0.04, 0.08, 0.08]),
            num_classes=1,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.5),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=[
            dict(
                assigner=dict(
                    type='RegionAssigner', center_ratio=0.2, ignore_ratio=0.5),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        rpn_proposal=dict(max_per_img=300, nms=dict(iou_threshold=0.8)),
        rcnn=dict(
            assigner=dict(
                pos_iou_thr=0.65, neg_iou_thr=0.65, min_pos_iou=0.65),
            sampler=dict(type='RandomSampler', num=256))),
    test_cfg=dict(
        rpn=dict(max_per_img=300, nms=dict(iou_threshold=0.8)),
        rcnn=dict(score_thr=1e-3))
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix='my-dataset/train',
        classes=classes,
        ann_file='my-dataset/train/train.json',
        pipeline=train_pipeline
    ),
    val=dict(
        img_prefix='my-dataset/test/images',
        classes=classes,
        ann_file='my-dataset/test/images/train.json',
        pipeline=test_pipeline
    ),
    test=dict(
        img_prefix='my-dataset/test/images',
        classes=classes,
        ann_file='my-dataset/test/images/train.json',
        pipeline=test_pipeline
    )
)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
# optimizer
# adjust learning rate to avoid grad boom
# todo better way to avoid grad boom
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)
# load_from = 'checkpoints/tridentnet_r50_caffe_mstrain_3x_coco_20201130_100539-46d227ba.pth'
