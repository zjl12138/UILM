# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from re import X
from PIL import Image
from matplotlib import patches
from mmcv.ops.nms import nms
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


def predict_img(model, img_root, img_root_path, img_suffix):
    """use trained model to predict image

        return 
    """
    img = os.path.join(img_root_path, img_root, img_suffix)
    # test a single image
    result = inference_detector(model, img)
    return result, model.show_result(img, result, bbox_color=(0, 0, 255), show=False, score_thr=0.3)


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

# def merge_proposal(img_root, img_root_path, img_suffix=(), model=()):
#     assert img_suffix[0] == 'filled.png'
#     # model_filled, model_default = model
#     # filled, default = img_suffix
#     # img_filled = os.path.join(img_root_path, img_root, filled)
#     # img_default = os.path.join(img_root_path, img_root, default)
#     result_list = []
#     for model_i, img_suffix_i in zip(model, img_suffix):
#         img = os.path.join(img_root_path, img_root, img_suffix_i)
#         result = inference_detector(model_i, img)
#         result_list.append(result)
#     # the result shape [[bbox], scores] [x, 5]
#     dets = np.concatenate((*result_list[0], *result_list[1]))
#     # bboxes = np.concatenate((*result_list[0], *result_list[1]))[:, :4]
#     # scores = np.concatenate((*result_list[0], *result_list[1]))[:, -1:]
#     # bboxes= torch.from_numpy(bboxes).contiguous()
#     # scores = torch.from_numpy(scores).contiguous()
#     output = NMS(dets, 0.7)
#     return output


def plot_gt_label(img_root, img_root_path, label_file_path, out_file=None, win_name='', thickness=2):
    """plot img with bbox

    Args:
        img ([type]): img file path
        img_root_name ([type]): img root name
        label_file_path ([type]): the json file
        out_file ([type], optional): [description]. Defaults to None.
        win_name (str, optional): [description]. Defaults to ''.
        thickness (int, optional): [description]. Defaults to 1.
    """
    img = os.path.join(img_root_path, img_root, 'default.png')
    img = plt.imread(img)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    currentAxis = fig.gca()
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
    total_obj = len(bboxes)
    bboxes = np.array(bboxes)

    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        coords.append([bbox_int[0], bbox_int[1], bbox_int[2], bbox_int[3]])

    for _, coord in enumerate(coords):
        rect = patches.Rectangle((coord[0], coord[1]), coord[2], coord[3],
                                 linewidth=thickness, edgecolor='r', facecolor='none')
        currentAxis.add_patch(rect)
    plt.title(f'total_obj:{total_obj}')
    plt.axis('off')
    if out_file is not None:
        plt.savefig(out_file, pad_inches=0.0, bbox_inches='tight')
    return plt.imread(out_file)


def concat_img(img_list: list, img_root, orientation='horizontal', save_dir='demo/result'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    w1, h1, _ = img_list[0].shape
    img_reshaped = []
    for img in img_list:
        img = cv2.resize(img, (w1, w1), interpolation=cv2.INTER_CUBIC)
        img_reshaped.append(img)
    if orientation == 'horizontal':
        img_concat = np.concatenate((img_reshaped), axis=1)
        cv2.imwrite(os.path.join(save_dir, f'{img_root}.png'), img_concat)


def init_model(img_suffix):
    if img_suffix == 'filled.png':
        config = './configs/my-dataset/siamnet_ga_filled.py'
        checkpoint = './work_dirs/siamnet_anchor_ga/img_filled/latest.pth'
    elif img_suffix == 'default.png':
        config = './configs/my-dataset/siamnet_ga_default.py'
        checkpoint = './work_dirs/siamnet_anchor_ga/img_default/latest.pth'
    else:
        raise "invalid image suffix, please use filled or default"
    # build the model from a config file and a checkpoint file
    return init_detector(config, checkpoint, device='cuda')


if __name__ == '__main__':

    img_root_path = './my-dataset/test/'
    with open(img_root_path + 'test.json') as f:
        data = json.loads(f.read())
    model_filled = init_model('filled.png')
    model_default = init_model('default.png')
    for i in tqdm(range(len(data["images"]))):
        img_root = data["images"][i]["file_name"]
        # merge_proposal(img_root, img_root_path, ('filled.png',
        #                'default.png'), (model_filled, model_default))
        result_f, img_pred_default = predict_img(
            model_default, img_root, img_root_path, img_suffix='default.png')
        result_d, img_pred_filled = predict_img(
            model_filled, img_root, img_root_path, img_suffix='filled.png')
        merge_proposal(img_root, img_root_path, result_d, result_f)
        img_oringin = plot_gt_label(img_root, img_root_path, label_file_path=img_root_path +
                                    'test.json', out_file='demo/result.png')
        img_oringin = cv2.cvtColor(img_oringin, cv2.COLOR_RGBA2BGR)
        img_oringin = img_oringin * 255.0
        concat_img([img_pred_default, img_pred_filled, img_oringin], img_root)
