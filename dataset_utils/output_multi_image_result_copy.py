# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from PIL import Image
from matplotlib import patches
from mmcv.ops.nms import nms
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import os
import json
import numpy as np

from mmdet.core.post_processing.bbox_nms import multiclass_nms
from mmdet.datasets.pipelines.compose import Compose
from tools.analysis_tools.analyze_results import bbox_map_eval


def NMS(dets, thresh):
    # x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的，得到的是排序的本来的索引，不是排完序的原数组
    order = scores.argsort()[::-1]
    # ::-1表示逆序

    temp = []
    while order.size > 0:
        i = order[0]
        temp.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
        # 由于numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return dets[temp]


def replace_MultiImageToTensor(pipelines):
    """Replace the MultiImageToTensor transform in a data pipeline to
    MultiDefaultFormatBundle, which is normally useful in batch inference.

    Args:
        pipelines (list[dict]): Data pipeline configs.

    Returns:
        list: The new pipeline list with all MultiImageToTensor replaced by
            MultiDefaultFormatBundle.
    """
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_MultiImageToTensor(
                pipeline['transforms'])
        elif pipeline['type'] == 'MultiImageToTensor':
            warnings.warn(
                '"MultiImageToTensor" pipeline is replaced by '
                '"MultiDefaultFormatBundle" for batch inference. It is '
                'recommended to manually replace it in the test '
                'data pipeline in your config file.', UserWarning)
            pipelines[i] = {'type': 'MultiDefaultFormatBundle'}
    return pipelines


def predict_img(model, img_path, img_name, sub_images=()):
    """predict image with trained model

    Args:
        model (nn.Module): trained model with pipeline is multi input format
        img_path (str): the folder where sub_images store
        sub_images (tuple, optional): the sub_images to predict. Defaults to ().

    Returns:
        predict_bbox: the result of model predicted bboxes
    """
    imgs = [os.path.join(img_path, sub_image) for sub_image in sub_images]
    cfg = model.cfg
    device = next(model.parameters()).device
    pipeline = replace_MultiImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(pipeline)
    data = dict(img_info=dict(filename=imgs), img_prefix=None)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [[img.data[0] for img in imgs] for imgs in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    # print(img)
    img_concat = model.show_result(imgs[0], result[0], bbox_color=(0, 0, 255), show=False, score_thr=0.3)
    # cv2.imwrite(os.path.join("/media/sda1/cyn-workspace/mmdetection/demo/result_new2", f'{img_name}.png'), img_concat)
    return result, img_concat


def merge_proposal(img_root, img_root_path, result_d, result_f):
    result = np.concatenate((*result_d, *result_f))
    det_result = NMS(result, thresh=0.3)  # merge two proposal
    # img = os.path.join(img_root_path, img_root, 'default.png')
    bbox_list = []
    label_list = []
    with open(img_root_path + 'test.json') as f:
        data = json.loads(f.read())

    for j in range(len(data["images"])):
        if data["images"][j]["file_name"] == img_root:
            id = j
            break
    for i in range(len(data["annotations"])):
        if data["images"][id]["id"] == data["annotations"][i]["image_id"]:
            bbox = data["annotations"][i]["bbox"]
            bbox_list.append(bbox)
            label_list.append(np.array(0))

    annotation = dict(
        bboxes=bbox_list,
        labels=label_list
    )
    map = bbox_map_eval([[det_result]], annotation)
    print(f"map:{map}")


def plot_pre(img, result_bbox):
    # print(result_bbox)
    img = cv2.imread(img)
    #
    for i in range(0, result_bbox[0].size):
        ans = []
        for id in result_bbox:
            ans.append(id[i])
        x1, y1, x2, y2 = ans
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def cv_plot_gt_label(img, img_root, label_file_path, out_file=None, win_name='', thickness=2):
    """plot img with bbox

    Args:
        img ([type]): img file path
        img_root_name ([type]): img root name
        label_file_path ([type]): the json file
        out_file ([type], optional): [description]. Defaults to None.
        win_name (str, optional): [description]. Defaults to ''.
        thickness (int, optional): [description]. Defaults to 1.
    """
    img = cv2.imread(img)

    bboxes = []
    coords = []
    with open(label_file_path) as f:
        data = json.loads(f.read())
    for j in range(len(data["images"])):
        if data["images"][j]["file_name"] == img_root:
            id = j
            break

    for i in range(len(data["annotations"])):
        if data["images"][id]["id"] == data["annotations"][i]["image_id"]:
            bbox = data["annotations"][i]["bbox"]
            bboxes.append(bbox)
    bboxes = np.array(bboxes)

    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        coords.append([bbox_int[0], bbox_int[1], bbox_int[2], bbox_int[3]])

    for _, coord in enumerate(coords):
        x, y, w, h = coord
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.imwrite(out_file,img)
    # return plt.imread(out_file)
    return img


def concat_img(img_list: list, img_name, orientation='horizontal', save_dir=None):
    w1, h1, _ = img_list[0].shape
    img_reshaped = []
    for img in img_list:
        img = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_CUBIC)
        img_reshaped.append(img)
    if orientation == 'horizontal':
        img_concat = np.concatenate((img_reshaped), axis=1)
    cv2.imwrite(os.path.join(save_dir, f'{img_name}.png'), img_concat)


def mkdir_all(save_dir_path_dlable, save_dir_path_flable, save_dir_path_fuison, save_dir_path_default):
    os.mkdir(save_dir_path_dlable)
    os.mkdir(save_dir_path_flable)
    os.mkdir(save_dir_path_fuison)
    os.mkdir(save_dir_path_default)


def get_bbox(dets, score_thr):
    scores = dets[:, -1]
    inds = scores > score_thr
    dets = dets[inds, :]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    bbox = []
    bbox.append(x1)
    bbox.append(y1)
    bbox.append(x2)
    bbox.append(y2)
    return bbox


if __name__ == '__main__':

    img_root_path = '/media/sda1/cyn-workspace/mmdetection/3.9/'
    save_dir_path_dlable = "/media/sda1/cyn-workspace/mmdetection/demo/3.9_small/dlable"
    save_dir_path_flable = "/media/sda1/cyn-workspace/mmdetection/demo/3.9_small/flable"
    save_dir_path_fuison = "/media/sda1/cyn-workspace/mmdetection/demo/3.9_small/fusion"
    save_dir_path_default = "/media/sda1/cyn-workspace/mmdetection/demo/3.9_small/default"

    #mkdir_all(save_dir_path_dlable, save_dir_path_flable, save_dir_path_fuison, save_dir_path_default)

    fusion_config = '/media/sda1/cyn-workspace/mmdetection/work_dirs/final_experiment/UIML/UIML+fusion+augdata/resnet_fpn_cascadeRPN_cascadeROI_default.py'
    fusion_checkpoint = '/media/sda1/cyn-workspace/mmdetection/work_dirs/final_experiment/UIML/UIML+fusion+augdata/latest.pth'
    default_config = '/media/sda1/cyn-workspace/mmdetection/work_dirs/final_experiment/UIML/UIML+fusion+augdata/resnet_fpn_cascadeRPN_cascadeROI_default.py'
    default_checkpoint = '/media/sda1/cyn-workspace/mmdetection/work_dirs/final_experiment/UIML/UIML+fusion+augdata/latest.pth'

    with open(img_root_path + 'test.json') as f:
        coco_data = json.load(f)

    model_fusion = init_detector(fusion_config, fusion_checkpoint, device='cuda')
    model_default = init_detector(default_config, default_checkpoint, device='cuda')

    for img in tqdm(coco_data["images"]):
        img_root = os.path.join(img_root_path + "test/", img["file_name"])
        result_d, img_pred_default = predict_img(model_default, img_root, img["file_name"], ('default.png',))
        result_f, img_pred_fusion = predict_img(model_fusion, img_root, img["file_name"], ('default-opacity.png',))

        #        merge_proposal(img_root, img_root_path, result_d, result_f)
        img_images_default = os.path.join(img_root_path, "images", img["file_name"], 'default.png')
        img_images_fusion = os.path.join(img_root_path, "images", img["file_name"], 'default-opacity.png')

        bbox_d = get_bbox(result_d[0][0], 0.3)
        bbox_f = get_bbox(result_f[0][0], 0.3)

        img_pred_default = plot_pre(img_images_default, bbox_d)
        img_pred_fusion = plot_pre(img_images_fusion, bbox_f)

        default_lable = cv_plot_gt_label(img_images_default, img["file_name"],
                                         label_file_path=img_root_path + 'test.json', out_file='demo/result.png')
        fusion_lable = cv_plot_gt_label(img_images_fusion, img["file_name"],
                                        label_file_path=img_root_path + 'test.json', out_file='demo/result.png')

        concat_img([img_pred_fusion], img["file_name"], save_dir=save_dir_path_fuison)  # fusion pre
        concat_img([fusion_lable], img["file_name"], save_dir=save_dir_path_flable)  # fusion lable
        concat_img([img_pred_default], img["file_name"], save_dir=save_dir_path_default)  # default pre
        concat_img([default_lable], img["file_name"], save_dir=save_dir_path_dlable)  # default lable
