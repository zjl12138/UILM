# 对预测结果进行更细的分类
# 1. 推断图片，计算bbox面积
# 2. 判断面积大小，对应类别加1
# 找出预测得不好的例子
from fileinput import filename
import shutil
from dataset_utils.mmdetapi import *
import json
from tqdm import tqdm


class Analysis:

    class_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    save_flag_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    class_list_gt = [0, 0, 0, 0, 0, 0, 0, 0, 0]


    def count_class(self, img_path, detector):

        label_list = filter_img_bbox(inference_detector(detector, img_path))
        # print(label_list)
        for label in label_list:
            area = label_area(label)
            img_root_path = img_path[:-len('default-opacity.png')] # default-opacity.png
            self.add(area, img_root_path)
            # print(area)
        print(f'pred: {self.class_list}')

    def add(self, area, img_root_path):
        if area > 0 and area < 50 * 50:
            self.class_list[0] += 1
            if self.save_flag_list[0] == 0:
                img_path = img_root_path + 'default-labeled.png'
                shutil.copy(src=img_path, dst='/media/sda1/cyn-workspace/mmdetection/dataset_utils/class_example/0.png')
                self.save_flag_list[0] = 1
        elif area > 50 * 50 and area < 100 * 100:
            self.class_list[1] += 1
            if self.save_flag_list[1] == 0:
                img_path = img_root_path + 'default-labeled.png'
                shutil.copy(src=img_path, dst='/media/sda1/cyn-workspace/mmdetection/dataset_utils/class_example/1.png')
                self.save_flag_list[1] = 1
        elif area > 150 * 150 and area < 200 * 200:
            self.class_list[2] += 1
            if self.save_flag_list[2] == 0:
                img_path = img_root_path + 'default-labeled.png'
                shutil.copy(src=img_path, dst='/media/sda1/cyn-workspace/mmdetection/dataset_utils/class_example/2.png')
                self.save_flag_list[2] = 1
        elif area > 250 * 250 and area < 300 * 300:
            self.class_list[3] += 1
            if self.save_flag_list[3] == 0:
                img_path = img_root_path + 'default-labeled.png'
                shutil.copy(src=img_path, dst='/media/sda1/cyn-workspace/mmdetection/dataset_utils/class_example/3.png')
                self.save_flag_list[3] = 1
        elif area > 350 * 350 and area < 400 * 400:
            self.class_list[4] += 1
            if self.save_flag_list[4] == 0:
                img_path = img_root_path + 'default-labeled.png'
                shutil.copy(src=img_path, dst='/media/sda1/cyn-workspace/mmdetection/dataset_utils/class_example/4.png')
                self.save_flag_list[4] = 1
        elif area > 450 * 450 and area < 500 * 500:
            self.class_list[5] += 1
            if self.save_flag_list[5] == 0:
                img_path = img_root_path + 'default-labeled.png'
                shutil.copy(src=img_path, dst='/media/sda1/cyn-workspace/mmdetection/dataset_utils/class_example/5.png')
                self.save_flag_list[5] = 1
        elif area > 500 * 500 and area < 600 * 600:
            self.class_list[6] += 1
            if self.save_flag_list[6] == 0:
                img_path = img_root_path + 'default-labeled.png'
                if img_path == '/media/sda1/cyn-workspace/mmdetection/dataset_assests/my_dataset_new/images/02743EAF-4B86-47C9-BF5B-A9FAC8005566-1-0-0.0/default-labeled.png':
                    return
                with open("dataset_utils/class_example/image_path.txt", "w") as txt_file:
                    txt_file.write(img_path)
                shutil.copy(src=img_path, dst='/media/sda1/cyn-workspace/mmdetection/dataset_utils/class_example/6.png')
                self.save_flag_list[6] = 1
        elif area > 600 * 600 and area < 650 * 650:
            self.class_list[7] += 1
            if self.save_flag_list[7] == 0:
                img_path = img_root_path + 'default-labeled.png'
                shutil.copy(src=img_path, dst='/media/sda1/cyn-workspace/mmdetection/dataset_utils/class_example/7.png')
                self.save_flag_list[7] = 1
        else:
            self.class_list[8] += 1
            if self.save_flag_list[8] == 0:
                img_path = img_root_path + 'default-labeled.png'
                shutil.copy(src=img_path, dst='/media/sda1/cyn-workspace/mmdetection/dataset_utils/class_example/8.png')
                self.save_flag_list[8] = 1
    
    def add_gt(self, area):
        if area > 0 and area < 50 * 50:
            self.class_list_gt[0] += 1
        elif area > 50 * 50 and area < 100 * 100:
            self.class_list_gt[1] += 1
        elif area > 150 * 150 and area < 200 * 200:
            self.class_list_gt[2] += 1
        elif area > 250 * 250 and area < 300 * 300:
            self.class_list_gt[3] += 1
        elif area > 350 * 350 and area < 400 * 400:
            self.class_list_gt[4] += 1
        elif area > 450 * 450 and area < 500 * 500:
            self.class_list_gt[5] += 1
        elif area > 500 * 500 and area < 600 * 600:
            self.class_list_gt[6] += 1
        elif area > 600 * 600 and area < 650 * 650:
            self.class_list_gt[7] += 1
        else:
            self.class_list_gt[8] += 1

    def main(self):
        config = '/media/sda1/cyn-workspace/mmdetection/work_dirs/final_experiment/UIML/UIML+fusion+augdata/resnet_fpn_cascadeRPN_cascadeROI_default.py'
        checkpoint = '/media/sda1/cyn-workspace/mmdetection/work_dirs/final_experiment/UIML/UIML+fusion+augdata/latest.pth'
        img_root_path = '/media/sda1/cyn-workspace/mmdetection/dataset_assests/my_dataset_new'
        # img_path = '/media/sda1/cyn-workspace/mmdetection/dataset_assests/my_dataset_new/images/6AD29565-12AE-4ABE-9F56-E1E9D0EF273A-1-0-0.0/default-opacity.png'
        label_json_path = '/media/sda1/cyn-workspace/mmdetection/dataset_assests/my_dataset_new/test.json'
        with open(label_json_path, 'r') as label_file:
            data = json.load(label_file)
        detector = get_detector(config, checkpoint, 'cuda:0')
        for i in tqdm(range(len(data["images"]))):
            img_path = data["images"][i]["file_name"]
            img_path = img_root_path + f'/images/{img_path}' + '/default-opacity.png'
            self.count_class(img_path, detector)
            
        # for i in tqdm(range(len(data["annotations"]))):
        #     gt_area = data["annotations"][i]["area"]
        #     self.add_gt(gt_area)
        #     print(f'gt: {self.class_list_gt}')
if __name__ == '__main__':
    # todo 
    # 找出预测与gt数目差大于2的图片
    ana = Analysis()
    ana.main()