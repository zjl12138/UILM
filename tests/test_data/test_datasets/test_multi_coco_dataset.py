# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import pytest

from mmdet_ext.datasets import MultiCocoDataset


def _create_ids_error_multi_coco_json(json_name):
    sub_image = [{
        'id': 0,
        'file_name': 'fake0.png'
    }, {
        'id': 1,
        'file_name': 'fake1.png'
    }]
    image = {
        'id': 0,
        'width': 640,
        'height': 640,
        'file_name': 'fake_name',
    }

    annotation_1 = {
        'id': 1,
        'image_id': 0,
        'category_id': 0,
        'area': 400,
        'bbox': [50, 60, 20, 20],
        'iscrowd': 0,
    }

    annotation_2 = {
        'id': 1,
        'image_id': 0,
        'category_id': 0,
        'area': 900,
        'bbox': [100, 120, 30, 30],
        'iscrowd': 0,
    }

    categories = [{
        'id': 0,
        'name': 'car',
        'supercategory': 'car',
    }]

    fake_json = {
        'sub_images': sub_image,
        'images': [image],
        'annotations': [annotation_1, annotation_2],
        'categories': categories
    }
    mmcv.dump(fake_json, json_name)


def test_coco_annotation_ids_unique():
    tmp_dir = tempfile.TemporaryDirectory()
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    _create_ids_error_multi_coco_json(fake_json_file)

    # test annotation ids not unique error
    with pytest.raises(AssertionError):
        MultiCocoDataset(ann_file=fake_json_file, classes=('car',), pipeline=[])
