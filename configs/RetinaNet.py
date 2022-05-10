_base_ = [
    './_base_/models/retinanet_r50_fpn.py',
    './_base_/datasets/coco_with_sub_image_detection.py',
    './_base_/schedules/schedule_1x.py', './_base_/default_runtime.py'
]
sub_images=[
    'default.png',
    #'filled.png'
   #'default-opacity.png'
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
    type='FusionRetinaNet',
    bbox_head=dict(num_classes=1),
    sub_images=sub_images)


# optimizer setting
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# data setting
classes = ('merge',)
dataset_folder = 'my_dataset_new'
data = dict(
    samples_per_gpu=8,  # Batch size of a single GPU
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
