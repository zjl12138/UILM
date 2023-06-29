from functools import lru_cache
import math
import os
import tempfile
from typing import List, Tuple, Union
from PIL import Image
import cv2
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.utils.config import Config
from mmdet.apis.inference import init_detector
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import replace_ImageToTensor
from mmdet.models.detectors.base import BaseDetector
import numpy as np


BboxLabel = Tuple[float, float, float, float, float]

@lru_cache(maxsize=4)
def get_detector(config: Union[str, Config], checkpoint: str, device: str) -> BaseDetector:
    return init_detector(config, checkpoint, device=device)

def get_img_bbox(img: Union[str, Image.Image], config: Union[str, Config], checkpoint: str, device: str = 'cuda:0') -> List[BboxLabel]:
    image = Image.open(img) if isinstance(img, str) else img
    tmpdir = tempfile.mkdtemp()
    temp_name = next(tempfile._get_candidate_names())+'.png'
    length = min(image.height, image.width)
    matrix = [[0, x] for x in np.arange(0, math.ceil(image.height/length), 0.5)] if length == image.width else [
        [x, 0] for x in np.arange(0, math.ceil(image.width/length), 0.5)]
    all_label = []
    detector = get_detector(config, checkpoint, device)
    for x, y in matrix:
        crop_img_path = os.path.join(
            tmpdir, f"{x}-{y}-{temp_name}")
        crop_img = Image.new(image.mode, (length, length), "white")
        original_crop = image.crop((int(x*length),
                                    int(y*length),
                                    min((x+1)*length, image.width),
                                    min((y+1)*length, image.height)))
        crop_img.paste(original_crop)
        crop_img.save(crop_img_path)
        labels = filter_img_bbox(
            inference_detector(detector, crop_img_path)
        )
        all_label = join_img_bboxes(all_label, labels, (x*length, y*length))
    return all_label

def get_labeled_img(imgs: Union[str, np.ndarray, List[str], List[np.ndarray]], config: Union[str, Config], checkpoint: str, device: str = 'cuda:0'):
    detector = get_detector(config, checkpoint, device)
    result = inference_detector(detector, imgs)
    labeled_img = detector.show_result(imgs, result)
    rgb_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)
    return rgb_img

def filter_img_bbox(inference_result: List[np.ndarray], score_thr: float = 0.3) -> List[BboxLabel]:
    bboxes: np.ndarray = inference_result[0]
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    return list(map(
        lambda x: tuple(x),
        bboxes))

def label_area(label: BboxLabel) -> float:
    return (label[2]-label[0])*(label[3]-label[1])

def intersection(label_a: BboxLabel, label_b: BboxLabel) -> float:
    dx = min(label_a[2], label_b[2]) - max(label_a[0], label_b[0])
    dy = min(label_a[3], label_b[3]) - max(label_a[1], label_b[1])
    if dx > 0 and dy > 0:
        return dx*dy
    else:
        return 0

def join_img_bboxes(full_list: List[BboxLabel], add_list: List[BboxLabel], offset: Tuple[float, float], thr=0.9) -> List[BboxLabel]:
    res = full_list[:]
    for new_label in add_list:
        mapped_new_label = (
            new_label[0] + offset[0],
            new_label[1] + offset[1],
            new_label[2] + offset[0],
            new_label[3] + offset[1],
            new_label[4]
        )
        is_existed = False
        new_area = label_area(mapped_new_label)
        for old_label in full_list:
            old_area = label_area(old_label)
            intersection_area = intersection(mapped_new_label, old_label)
            overlay_score = intersection_area / \
                (new_area+old_area-intersection_area)
            if overlay_score > thr:
                is_existed = True
                break
        if not is_existed:
            res.append(mapped_new_label)
    return res

def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    # imgs = [imgs]
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    if isinstance(data['img'][0], list):
        data['img'] = data['img'][0]
        data['img'] = [[img.data for img in data['img']]]
    else:
        data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results
