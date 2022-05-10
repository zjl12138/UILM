import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.pipelines.formating import DefaultFormatBundle, to_tensor, ImageToTensor
from mmdet.datasets.pipelines.loading import LoadImageFromFile
from mmdet.datasets.pipelines.transforms import Resize, RandomFlip, Normalize, Pad
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class MultiLoadImageFromFile(LoadImageFromFile):
    def __call__(self, results):
        # assert mmcv.is_list_of(results['img_info']['filename'], str)
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = [osp.join(results['img_prefix'], x)
                        for x in results['img_info']['filename']]
        else:
            filename = results['img_info']['filename']

        imgs = []  # single object to list
        for sub_filename in filename:
            img_bytes = self.file_client.get(sub_filename)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)  # single object to list

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = imgs  # single object to list
        results['img_shape'] = imgs[0].shape  # use first index
        results['ori_shape'] = imgs[0].shape  # use first index
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class MultiResize(Resize):
    def __call__(self, results):
        # assert mmcv.is_list_of(results['img'], ndarray)
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'][0].shape[:2]  # use first index
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def _resize_img(self, results):
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                imgs = []
                for sub_img in results[key]:
                    img, scale_factor = mmcv.imrescale(
                        sub_img,
                        results['scale'],
                        return_scale=True,
                        backend=self.backend)
                    imgs.append(img)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = imgs[0].shape[:2]
                h, w = results[key][0].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                imgs = []
                w_scale = 1
                h_scale = 1
                for sub_img in results[key]:
                    img, w_scale, h_scale = mmcv.imresize(
                        sub_img,
                        results['scale'],
                        return_scale=True,
                        backend=self.backend)
                    imgs.append(img)
            results[key] = imgs

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = imgs[0].shape
            # in case that there is no padding
            results['pad_shape'] = imgs[0].shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio


@PIPELINES.register_module()
class MultiRandomFlip(RandomFlip):
    def __call__(self, results):
        # assert mmcv.is_list_of(results['img'], ndarray)
        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = [mmcv.imflip(
                    sub_img, direction=results['flip_direction']) for sub_img in results[key]]
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
        return results


@PIPELINES.register_module()
class MultiNormalize(Normalize):
    def __call__(self, results):
        # assert mmcv.is_list_of(results['img'], ndarray)
        for key in results.get('img_fields', ['img']):
            results[key] = [mmcv.imnormalize(sub_img, self.mean, self.std,
                                             self.to_rgb) for sub_img in results[key]]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@PIPELINES.register_module()
class MultiPad(Pad):
    def _pad_img(self, results):
        pad_val = self.pad_val.get('img', 0)
        for key in results.get('img_fields', ['img']):
            padded_imgs = []
            for sub_img in results[key]:
                if self.pad_to_square:
                    max_size = max(sub_img.shape[:2])
                    self.size = (max_size, max_size)
                if self.size is not None:
                    padded_img = mmcv.impad(
                        sub_img, shape=self.size, pad_val=pad_val)
                elif self.size_divisor is not None:
                    padded_img = mmcv.impad_to_multiple(
                        sub_img, self.size_divisor, pad_val=pad_val)
                padded_imgs.append(padded_img)
            results[key] = padded_imgs
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        pad_val = self.pad_val.get('masks', 0)
        for key in results.get('mask_fields', []):
            results[key] = [sub_img.pad(pad_shape, pad_val=pad_val)
                            for sub_img in results[key]]

    def _pad_seg(self, results):
        pad_val = self.pad_val.get('seg', 255)
        for key in results.get('seg_fields', []):
            results[key] = [mmcv.impad(
                sub_img, shape=results['pad_shape'][:2], pad_val=pad_val) for sub_img in results[key]]

    def __call__(self, results):
        # assert mmcv.is_list_of(results['img'], ndarray)
        return super(MultiPad, self).__call__(results)


@PIPELINES.register_module()
class MultiDefaultFormatBundle(DefaultFormatBundle):
    def __call__(self, results):
        # assert mmcv.is_list_of(results['img'], ndarray)
        if 'img' in results:
            # add default meta keys
            results = self._add_default_meta_keys(results)
            imgs = []
            for img in results['img']:
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                imgs.append(DC(to_tensor(img), stack=True))
            results['img'] = imgs
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        return results

    def _add_default_meta_keys(self, results):
        img = results['img']
        results.setdefault('pad_shape', img[0].shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results


@PIPELINES.register_module()
class MultiImageToTensor(ImageToTensor):
    def __call__(self, results):
        for key in self.keys:
            imgs = []
            # assert mmcv.is_list_of(results[key], ndarray)
            for sub_img in results[key]:
                img = sub_img
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                imgs.append((to_tensor(img.transpose(2, 0, 1))).contiguous())
            results[key] = imgs
        return results
