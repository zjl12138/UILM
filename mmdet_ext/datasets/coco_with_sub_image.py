import os.path
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class CocoDatasetWithSubImage(CocoDataset):
    def __init__(self, sub_images=(), **kwargs):
        self.sub_images = sub_images
        super(CocoDatasetWithSubImage, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        data_infos = super().load_annotations(ann_file)
        for index,info in enumerate(data_infos):
            data_infos[index]['filename'] = [os.path.join(info['filename'], sub_image_name) for sub_image_name in self.sub_images]
        return data_infos


    def _parse_ann_info(self, img_info, ann_info):
        filename_list = img_info['filename']
        img_info['filename'] = filename_list[0]
        ann = super()._parse_ann_info(img_info, ann_info)
        img_info['filename'] = filename_list
        return ann
