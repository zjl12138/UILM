_base_ = ['../cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py']
classes = ('merge',)
rpn_weight = 0.7
sub_images = [
    'default.png'
]
model = dict(
    type='SiameseRPNV2',
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    rpn_head=dict(
        _delete_=True,
        type='CascadeRPNHead',
        num_stages=2,
        stages=[
            dict(
                type='StageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[1.0],
                    strides=[4, 8, 16, 32, 64]),
                adapt_cfg=dict(type='dilation', dilation=3),
                bridged_feature=True,
                sampling=False,
                with_cls=False,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.5, 0.5)),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight)),
            dict(
                type='StageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
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
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight))
        ]),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
            )),
    # model training and testing settings
    sub_images=sub_images
)
data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
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