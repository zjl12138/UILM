import copy
import os.path as osp

import mmcv
import numpy as np
import pytest
from mmcv.utils import build_from_cfg

from mmdet.datasets.builder import PIPELINES
from mmdet_ext.datasets.pipelines import (MultiLoadImageFromFile)


class TestMulti:
    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../../data')

    def test_multi_load_img(self):
        results = dict(
            img_prefix=self.data_prefix, img_info=dict(filename=['color.jpg', 'color.jpg']))
        transform = MultiLoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['filename'] == [osp.join(self.data_prefix, x) for x in ['color.jpg', 'color.jpg']]
        assert results['ori_filename'] == ['color.jpg', 'color.jpg']
        assert results['img'][0].shape == (288, 512, 3)
        assert results['img'][1].shape == (288, 512, 3)
        assert results['img'][0].dtype == np.uint8
        assert results['img'][1].dtype == np.uint8
        assert results['img_shape'] == (288, 512, 3)
        assert results['ori_shape'] == (288, 512, 3)
        assert repr(transform) == transform.__class__.__name__ + \
               "(to_float32=False, color_type='color', " + \
               "file_client_args={'backend': 'disk'})"

        # no img_prefix
        results = dict(
            img_prefix=None, img_info=dict(filename=['tests/data/color.jpg', 'tests/data/color.jpg']))
        transform = MultiLoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['filename'] == ['tests/data/color.jpg', 'tests/data/color.jpg']
        assert results['ori_filename'] == ['tests/data/color.jpg', 'tests/data/color.jpg']
        assert results['img'][0].shape == (288, 512, 3)
        assert results['img'][1].shape == (288, 512, 3)

        # to_float32
        transform = MultiLoadImageFromFile(to_float32=True)
        results = transform(copy.deepcopy(results))
        assert results['img'][0].dtype == np.float32
        assert results['img'][1].dtype == np.float32

        # gray image
        results = dict(
            img_prefix=self.data_prefix, img_info=dict(filename=['gray.jpg', 'gray.jpg']))
        transform = MultiLoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['img'][0].shape == (288, 512, 3)
        assert results['img'][1].shape == (288, 512, 3)
        assert results['img'][0].dtype == np.uint8
        assert results['img'][1].dtype == np.uint8

        transform = MultiLoadImageFromFile(color_type='unchanged')
        results = transform(copy.deepcopy(results))
        assert results['img'][0].shape == (288, 512)
        assert results['img'][1].shape == (288, 512)
        assert results['img'][0].dtype == np.uint8
        assert results['img'][1].dtype == np.uint8

    def test_multi_resize(self):
        # test assertion if img_scale is a list
        with pytest.raises(AssertionError):
            transform = dict(type='MultiResize', img_scale=[1333, 800], keep_ratio=True)
            build_from_cfg(transform, PIPELINES)

        # test assertion if len(img_scale) while ratio_range is not None
        with pytest.raises(AssertionError):
            transform = dict(
                type='MultiResize',
                img_scale=[(1333, 800), (1333, 600)],
                ratio_range=(0.9, 1.1),
                keep_ratio=True)
            build_from_cfg(transform, PIPELINES)

        # test assertion for invalid multiscale_mode
        with pytest.raises(AssertionError):
            transform = dict(
                type='MultiResize',
                img_scale=[(1333, 800), (1333, 600)],
                keep_ratio=True,
                multiscale_mode='2333')
            build_from_cfg(transform, PIPELINES)

        # test assertion if both scale and scale_factor are setted
        with pytest.raises(AssertionError):
            results = dict(
                img_prefix=osp.join(osp.dirname(__file__), '../../data'),
                img_info=dict(filename=['color.jpg', 'color.jpg']))
            load = dict(type='MultiLoadImageFromFile')
            load = build_from_cfg(load, PIPELINES)
            transform = dict(type='MultiResize', img_scale=(1333, 800), keep_ratio=True)
            transform = build_from_cfg(transform, PIPELINES)
            results = load(results)
            results['scale'] = (1333, 800)
            results['scale_factor'] = 1.0
            results = transform(results)

        transform = dict(type='MultiResize', img_scale=(1333, 800), keep_ratio=True)
        resize_module = build_from_cfg(transform, PIPELINES)

        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        results['img'] = [img, copy.deepcopy(img)]
        results['img2'] = [copy.deepcopy(img), copy.deepcopy(img)]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['img_fields'] = ['img', 'img2']

        results = resize_module(results)
        assert np.equal(results['img'][0], results['img2'][0]).all()
        assert np.equal(results['img'][1], results['img2'][1]).all()

        results.pop('scale')
        results.pop('scale_factor')
        transform = dict(
            type='MultiResize',
            img_scale=(1280, 800),
            multiscale_mode='value',
            keep_ratio=False)
        resize_module = build_from_cfg(transform, PIPELINES)
        results = resize_module(results)
        assert np.equal(results['img'][0], results['img2'][0]).all()
        assert np.equal(results['img'][1], results['img2'][1]).all()
        assert results['img_shape'] == (800, 1280, 3)
        assert results['img'][0].dtype == results['img2'][0].dtype == np.uint8
        assert results['img'][1].dtype == results['img2'][1].dtype == np.uint8

        results_seg = {
            'img': [img, copy.deepcopy(img)],
            'img_shape': img.shape,
            'ori_shape': img.shape,
            'gt_semantic_seg': copy.deepcopy(img),
            'gt_seg': copy.deepcopy(img),
            'seg_fields': ['gt_semantic_seg', 'gt_seg']
        }
        transform = dict(
            type='MultiResize',
            img_scale=(640, 400),
            multiscale_mode='value',
            keep_ratio=False)
        resize_module = build_from_cfg(transform, PIPELINES)
        results_seg = resize_module(results_seg)
        assert results_seg['gt_semantic_seg'].shape == results_seg['gt_seg'].shape
        assert results_seg['img_shape'] == (400, 640, 3)
        assert results_seg['img_shape'] != results_seg['ori_shape']
        assert results_seg['gt_semantic_seg'].shape == results_seg['img_shape']
        assert np.equal(results_seg['gt_semantic_seg'],
                        results_seg['gt_seg']).all()

    def test_multi_random_flip(self):
        # test assertion for invalid flip_ratio
        with pytest.raises(AssertionError):
            transform = dict(type='MultiRandomFlip', flip_ratio=1.5)
            build_from_cfg(transform, PIPELINES)
        # test assertion for 0 <= sum(flip_ratio) <= 1
        with pytest.raises(AssertionError):
            transform = dict(
                type='MultiRandomFlip',
                flip_ratio=[0.7, 0.8],
                direction=['horizontal', 'vertical'])
            build_from_cfg(transform, PIPELINES)

        # test assertion for mismatch between number of flip_ratio and direction
        with pytest.raises(AssertionError):
            transform = dict(type='MultiRandomFlip', flip_ratio=[0.4, 0.5])
            build_from_cfg(transform, PIPELINES)

        # test assertion for invalid direction
        with pytest.raises(AssertionError):
            transform = dict(
                type='MultiRandomFlip', flip_ratio=1., direction='horizonta')
            build_from_cfg(transform, PIPELINES)

        transform = dict(type='MultiRandomFlip', flip_ratio=1.)
        flip_module = build_from_cfg(transform, PIPELINES)

        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        original_img = copy.deepcopy(img)
        results['img'] = [img, copy.deepcopy(img)]
        results['img2'] = [copy.deepcopy(img), copy.deepcopy(img)]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img', 'img2']

        results = flip_module(results)
        assert np.equal(results['img'][0], results['img2'][0]).all()
        assert np.equal(results['img'][1], results['img2'][1]).all()

        flip_module = build_from_cfg(transform, PIPELINES)
        results = flip_module(results)
        assert np.equal(results['img'][0], results['img2'][0]).all()
        assert np.equal(results['img'][1], results['img2'][1]).all()
        assert np.equal(original_img, results['img'][0]).all()
        assert np.equal(original_img, results['img'][1]).all()

        # test flip_ratio is float, direction is list
        transform = dict(
            type='MultiRandomFlip',
            flip_ratio=0.9,
            direction=['horizontal', 'vertical', 'diagonal'])
        flip_module = build_from_cfg(transform, PIPELINES)

        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        original_img = copy.deepcopy(img)
        results['img'] = [img, copy.deepcopy(img)]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img']
        results = flip_module(results)
        if results['flip']:
            assert np.array_equal(
                mmcv.imflip(original_img, results['flip_direction']),
                results['img'][0])
            assert np.array_equal(
                mmcv.imflip(original_img, results['flip_direction']),
                results['img'][1])
        else:
            assert np.array_equal(original_img, results['img'][0])
            assert np.array_equal(original_img, results['img'][1])

        # test flip_ratio is list, direction is list
        transform = dict(
            type='MultiRandomFlip',
            flip_ratio=[0.3, 0.3, 0.2],
            direction=['horizontal', 'vertical', 'diagonal'])
        flip_module = build_from_cfg(transform, PIPELINES)

        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        original_img = copy.deepcopy(img)
        results['img'] = [img, copy.deepcopy(img)]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img']
        results = flip_module(results)
        if results['flip']:
            assert np.array_equal(
                mmcv.imflip(original_img, results['flip_direction']),
                results['img'][0])
            assert np.array_equal(
                mmcv.imflip(original_img, results['flip_direction']),
                results['img'][1])
        else:
            assert np.array_equal(original_img, results['img'][0])
            assert np.array_equal(original_img, results['img'][1])

    def test_multi_normalize(self):
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)
        transform = dict(type='MultiNormalize', **img_norm_cfg)
        transform = build_from_cfg(transform, PIPELINES)
        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        original_img = copy.deepcopy(img)
        results['img'] = [img, copy.deepcopy(img)]
        results['img2'] = [copy.deepcopy(img), copy.deepcopy(img)]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img', 'img2']

        results = transform(results)
        assert np.equal(results['img'][0], results['img2'][0]).all()
        assert np.equal(results['img'][1], results['img2'][1]).all()

        mean = np.array(img_norm_cfg['mean'])
        std = np.array(img_norm_cfg['std'])
        converted_img = (original_img[..., ::-1] - mean) / std
        assert np.allclose(results['img'][0], converted_img)
        assert np.allclose(results['img'][1], converted_img)

    def test_multi_pad(self):
        # test assertion if both size_divisor and size is None
        with pytest.raises(AssertionError):
            transform = dict(type='MultiPad')
            build_from_cfg(transform, PIPELINES)

        transform = dict(type='MultiPad', size_divisor=32)
        transform = build_from_cfg(transform, PIPELINES)
        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        original_img = copy.deepcopy(img)
        results['img'] = [img, copy.deepcopy(img)]
        results['img2'] = [copy.deepcopy(img), copy.deepcopy(img)]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img', 'img2']

        results = transform(results)
        assert np.equal(results['img'][0], results['img2'][0]).all()
        assert np.equal(results['img'][1], results['img2'][1]).all()
        # original img already divisible by 32
        assert np.equal(results['img'][0], original_img).all()
        assert np.equal(results['img'][1], original_img).all()
        img_shape = results['img'][0].shape
        assert img_shape[0] % 32 == 0
        assert img_shape[1] % 32 == 0

        resize_transform = dict(
            type='MultiResize', img_scale=(1333, 800), keep_ratio=True)
        resize_module = build_from_cfg(resize_transform, PIPELINES)
        results = resize_module(results)
        results = transform(results)
        img_shape = results['img'][0].shape
        assert np.equal(results['img'][0], results['img2'][0]).all()
        assert np.equal(results['img'][1], results['img2'][1]).all()
        assert img_shape[0] % 32 == 0
        assert img_shape[1] % 32 == 0

        # test the size and size_divisor must be None when pad2square is True
        with pytest.raises(AssertionError):
            transform = dict(type='MultiPad', size_divisor=32, pad_to_square=True)
            build_from_cfg(transform, PIPELINES)

        transform = dict(type='MultiPad', pad_to_square=True)
        transform = build_from_cfg(transform, PIPELINES)
        results['img'] = [img, copy.deepcopy(img)]
        results = transform(results)
        assert results['img'][0].shape[0] == results['img'][0].shape[1]
        assert results['img'][1].shape[0] == results['img'][1].shape[1]

        # test the pad_val is converted to a dict
        transform = dict(type='MultiPad', size_divisor=32, pad_val=0)
        with pytest.deprecated_call():
            transform = build_from_cfg(transform, PIPELINES)

        assert isinstance(transform.pad_val, dict)
        results = transform(results)
        img_shape = results['img'][0].shape
        assert img_shape[0] % 32 == 0
        assert img_shape[1] % 32 == 0
        img_shape = results['img'][1].shape
        assert img_shape[0] % 32 == 0
        assert img_shape[1] % 32 == 0

    def test_default_format_bundle(self):
        results = dict(
            img_prefix=osp.join(osp.dirname(__file__), '../../data'),
            img_info=dict(filename=['color.jpg', 'color.jpg']))
        load = dict(type='MultiLoadImageFromFile')
        load = build_from_cfg(load, PIPELINES)
        bundle = dict(type='MultiDefaultFormatBundle')
        bundle = build_from_cfg(bundle, PIPELINES)
        results = load(results)
        assert 'pad_shape' not in results
        assert 'scale_factor' not in results
        assert 'img_norm_cfg' not in results
        results = bundle(results)
        assert 'pad_shape' in results
        assert 'scale_factor' in results
        assert 'img_norm_cfg' in results
