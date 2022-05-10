_base_ = [
    './_base_/models/rpn_r50_fpn.py', './_base_/datasets/coco_with_sub_image_detection.py',
    './_base_/schedules/schedule_1x.py', './_base_/default_runtime.py'
]
data = dict(
    samples_per_gpu=1,  # Batch size of a single GPU
    workers_per_gpu=1,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix='my-dataset/train',
        ann_file='my-dataset/train/train_defaut.json',
    ),
    val=dict(
        img_prefix='my-dataset/test',
        ann_file='my-dataset/test/test_default.json',
    ),
    test=dict(
        img_prefix='my-dataset/test',
        ann_file='my-dataset/test/test_default.json',
    )
)
runner = dict(type=('EpochBasedRunner'), max_epochs=12)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
evaluation = dict(interval=1, metric='proposal_fast')