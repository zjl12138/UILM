_base_ = [
    './_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/datasets/coco_with_sub_image_detection.py',
    './_base_/schedules/schedule_1x.py', './_base_/default_runtime.py'
]
sub_images=[
    'default.png'
]
model = dict(
    type='SiameseRPNV2',
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,)),
    sub_images=sub_images)
classes = ('merge',)
data = dict(
    samples_per_gpu=4,  # Batch size of a single GPU
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU
    train=dict(
        # img_prefix='my-dataset/train',
        # classes=classes,
        # ann_file='my-dataset/train/train.json',
        img_prefix='my-dataset/train',
        classes=classes,
        ann_file='my-dataset/train/train.json',
        sub_images=sub_images,
    ),
    val=dict(
        img_prefix='my-dataset/test',
        classes=classes,
        ann_file='my-dataset/test/test.json',
        sub_images=sub_images,
    ),
    test=dict(
        img_prefix='my-dataset/test',
        classes=classes,
        ann_file='my-dataset/test/test.json',
        sub_images=sub_images,
    )
)