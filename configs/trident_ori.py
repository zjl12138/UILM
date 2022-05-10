_base_ = '../tridentnet/tridentnet_r50_caffe_1x_coco.py'
classes = ('merge',)
model = dict(
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
    )
)
data = dict(
    samples_per_gpu=1,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix='my-dataset/train',
        classes=classes,
        ann_file='my-dataset/train/train.json',
    ),
    val=dict(
        img_prefix='my-dataset/test/images',
        classes=classes,
        ann_file='my-dataset/test/images/train.json',
    ),
    test=dict(
        img_prefix='my-dataset/test/images',
        classes=classes,
        ann_file='my-dataset/test/images/train.json',
    )
)
log_config = dict(
    interval=1,
)