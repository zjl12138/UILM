import json
import os

from PIL import Image
from tqdm import tqdm


def format_yolo2coco(img_folder_path, label_folder_path, out_file, sub_images=()):
    annotations = []
    images = []
    label_file_list = os.listdir(label_folder_path)
    image_id = 0
    annotation_id = 0

    for label_file_name in tqdm(label_file_list, desc='format'):
        file_prefix = os.path.splitext(os.path.basename(label_file_name))[0]
        if len(sub_images) > 0:
            image_path = os.path.join(img_folder_path, file_prefix, sub_images[0])
        else:
            image_path = os.path.join(img_folder_path, file_prefix + '.png')
        # print('image:', image_path)
        label_path = os.path.join(label_folder_path, label_file_name)
        # print('label:', label_path)
        img_file = Image.open(image_path)
        img_width, img_height = img_file.size

        images.append(dict(
            id=image_id,
            file_name=file_prefix + '' if len(sub_images) > 0 else '.png',
            height=int(img_height),
            width=int(img_width)))

        # yolo format - (class_id, x_center, y_center, width, height)
        # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)
        with open(label_path, 'r') as label_fp:
            # read label list from file
            label_list = [
                (float(x.split(" ")[1]), float(x.split(" ")[2]), float(x.split(" ")[3]), float(x.split(" ")[4]))
                for x in label_fp.readlines()]
            for label in label_list:
                x_center, y_center, width, height = label
                x_center, y_center, width, height = int(x_center * img_width), \
                                                    int(y_center * img_height), int(width * img_width), int(
                    height * img_height)
                x_min, y_min = x_center - width / 2, y_center - height / 2
                data_anno = dict(
                    image_id=image_id,
                    id=annotation_id,
                    category_id=0,
                    bbox=[x_min, y_min, width, height],
                    area=width * height,
                    segmentation=[],
                    iscrowd=0)
                annotations.append(data_anno)
                annotation_id += 1

        image_id += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 0, 'name': 'merge'}]
    )
    if len(sub_images) > 0:
        coco_format_json['sub_images'] = [{'id': index, 'file_name': sub_file_name} for index, sub_file_name in
                                          enumerate(sub_images)]
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(coco_format_json, f, ensure_ascii=False, indent=4)
    print('** json created! **')


if __name__ == '__main__':
    file_name = os.listdir('train_label')
    # for name in file_name:
    #     suffix = os.path.splitext(os.path.basename(name))[1]
    #
    #     if suffix != '.txt':
    #         os.remove('../labels/' + name)
    print(len(file_name))
    format_yolo2coco(img_folder_path='./train/', label_folder_path='./train_label/', out_file='train/train.json',
                     sub_images=('filled.png', 'default.png'))
    format_yolo2coco(img_folder_path='./test/', label_folder_path='./test_label/', out_file='test/test.json',
                     sub_images=('filled.png', 'default.png'))
__all__ = [
    'format_yolo2coco'
]
