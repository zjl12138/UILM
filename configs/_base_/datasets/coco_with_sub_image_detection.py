# dataset settings
dataset_type = 'CocoDatasetWithSubImage'
data_root = 'data/coco/'
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
sub_images=[]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        sub_images=sub_images,
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        sub_images=sub_images,
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        sub_images=sub_images,
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
