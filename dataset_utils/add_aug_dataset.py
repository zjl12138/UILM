import os
from re import template
import shutil
from count_num import count_num, check
from tqdm.std import tqdm
from format_convert import format_yolo2coco
from split_data import split_file, split_label, split_file_with_union, split_file_with_template
import random

def add_aug(aug_data_path,aug_label_path,aug_data,
            src_data_path,src_label_path,
            dst_data_path,dst_label_path,ratio):

        for label in os.listdir(src_label_path):
            label_prefix = os.path.splitext(label)[0]
            aug_new_data=aug_data+"-new"
            if aug_data == label_prefix and random.random()<ratio:
                # copy aug data to train folder
                shutil.copytree(src=os.path.join(aug_data_path, aug_data),
                                dst=os.path.join(dst_data_path, aug_new_data))
                shutil.copy(src=os.path.join(aug_label_path, label),
                            dst=os.path.join(dst_label_path, label_prefix+'-new.txt')) 

def find_augmented_data(origin_folder,aug_folder,dst_folder,ratio):

    # path
    aug_data_path=aug_folder+'/images'
    aug_label_path=aug_folder+'/labels'

    train_data_path=origin_folder+'/train'
    train_label_path=origin_folder+'/train_labels'
    test_data_path=origin_folder+'/test'
    test_label_path=origin_folder+'/test_labels'

    dst_data=dst_folder+'/images'
    dst_label=dst_folder+'/labels'

    dst_train_data=dst_folder+'/train'
    dst_train_lable=dst_folder+'/train_labels'
    dst_test_data=dst_folder+'/test'
    dst_test_lable=dst_folder+'/test_labels'    

    os.makedirs(dst_data)
    os.makedirs(dst_label)
    os.makedirs(dst_train_data)
    os.makedirs(dst_train_lable)
    os.makedirs(dst_test_data)
    os.makedirs(dst_test_lable)

    # add aug data
    aug_data_list = os.listdir(aug_data_path)
    for aug_data in tqdm(aug_data_list, desc='add augment data'):
        add_aug(aug_data_path,aug_label_path,aug_data,train_data_path,train_label_path,dst_train_data,dst_train_lable,ratio)
        #add_aug(aug_data_path,aug_label_path,aug_data,test_data_path,test_label_path,dst_test_data,dst_test_lable,ratio)
                
    # add origin data
    for data in tqdm(os.listdir(train_data_path),desc='add train data'):
        shutil.copytree(src=os.path.join(train_data_path, data),
            dst=os.path.join(dst_train_data, data)) 

    for lable in tqdm(os.listdir(train_label_path),desc='add train lable'):
        shutil.copy(src=os.path.join(train_label_path, lable),
                            dst=os.path.join(dst_train_lable, lable)) 

    for data in tqdm(os.listdir(test_data_path),desc='add test data'):
        shutil.copytree(src=os.path.join(test_data_path, data),
            dst=os.path.join(dst_test_data, data))   

    for lable in tqdm(os.listdir(test_label_path),desc='add test label'):
        shutil.copy(src=os.path.join(test_label_path, lable),
                            dst=os.path.join(dst_test_lable, lable))   
                  
    format_yolo2coco(img_folder_path=f'{dst_folder}/train/', label_folder_path=f'{dst_folder}/train_labels/', out_file=f'{dst_folder}/train.json',
                     sub_images=('default-opacity.png','filled.png', 'default.png', 'default-labeled.png', 'filled-labeled.png'))
    format_yolo2coco(img_folder_path=f'{dst_folder}/test/', label_folder_path=f'{dst_folder}/test_labels/', out_file=f'{dst_folder}/test.json',
                     sub_images=('default-opacity.png','filled.png', 'default.png', 'default-labeled.png', 'filled-labeled.png'))

    add_image_and_label(dst_data,dst_label,dst_train_data,dst_train_lable,dst_test_data,dst_test_lable)

def add_image_and_label(dst_data,dst_label,dst_train_data,dst_train_lable,dst_test_data,dst_test_lable):
    for data in tqdm(os.listdir(dst_train_data),desc='add image_train'):
        shutil.copytree(src=os.path.join(dst_train_data, data),
            dst=os.path.join(dst_data, data))   

    for lable in tqdm(os.listdir(dst_train_lable),desc='add label_train'):
        shutil.copy(src=os.path.join(dst_train_lable, lable),
                            dst=os.path.join(dst_label, lable)) 

    for data in tqdm(os.listdir(dst_test_data),desc='add image_test'):
        shutil.copytree(src=os.path.join(dst_test_data, data),
            dst=os.path.join(dst_data, data))   

    for lable in tqdm(os.listdir(dst_test_lable),desc='add label_test'):
        shutil.copy(src=os.path.join(dst_test_lable, lable),
                            dst=os.path.join(dst_label, lable))     

if __name__ == '__main__':
    keep_ratio=1
    origin_folder = '/media/sda1/cyn-workspace/mmdetection/my_dataset_new'
    aug_folder='/media/sda1/cyn-workspace/mmdetection/my_dataset_new_aug'
    dst_folder='/media/sda1/cyn-workspace/mmdetection/my_dataset_union'
    
    #find_augmented_data(origin_folder,aug_folder,dst_folder,keep_ratio)
    count_num(origin_folder)
    check(origin_folder)
    
