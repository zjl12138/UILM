import warnings

import mmcv
import torch
from mmcv.image import tensor2imgs
from torch import nn

from mmdet.core import bbox_mapping
from mmdet.models.detectors.faster_rcnn import FasterRCNN
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.base import BaseDetector

@DETECTORS.register_module()
class SiameseRPN(FasterRCNN):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 sub_images=()):

        super(SiameseRPN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.num_branch = self.roi_head.num_branch
        self.test_branch_idx = self.roi_head.test_branch_idx
        self.backbones = nn.ModuleList([build_backbone(backbone) for i in sub_images])
    
    def extract_feat(self, imgs):
        assert isinstance(imgs, tuple) or isinstance(imgs, list)
        feature_list = [self.backbones[i](img) for i,img in enumerate(imgs)]
        x = tuple(torch.cat(x, dim=0) for x in zip(*feature_list)) # x should be [batch*2, c, h, w]
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, **kwargs):
        trident_gt_bboxes = tuple(gt_bboxes * self.num_branch)
        trident_gt_labels = tuple(gt_labels * self.num_branch)
        trident_img_metas = tuple(img_metas * self.num_branch)
        return super(SiameseRPN,
                     self).forward_train(img, trident_img_metas,
                                         trident_gt_bboxes, trident_gt_labels)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            num_branch = (self.num_branch if self.test_branch_idx == -1 else 1)
            trident_img_metas = img_metas * num_branch
            proposal_list = self.rpn_head.simple_test_rpn(x, trident_img_metas)
        else:
            proposal_list = proposals
        return self.roi_head.simple_test(
            x, proposal_list, trident_img_metas, rescale=rescale)


    def aug_test(self, imgs, img_metas, rescale=False):
        x = self.extract_feats(imgs)
        num_branch = (self.num_branch if self.test_branch_idx == -1 else 1)
        trident_img_metas = [img_metas * num_branch for img_metas in img_metas]
        proposal_list = self.rpn_head.aug_test_rpn(x, trident_img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)