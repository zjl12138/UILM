import os
from re import template
import shutil

from tqdm.std import tqdm
from format_convert import format_yolo2coco
from split_data import split_file, split_label, split_file_with_union, split_file_with_template


def find_augmented_data(aug_data_path, train_data_path, train_label_path):
    aug_data_list = os.listdir(aug_data_path)
    for aug_data in tqdm(aug_data_list, desc='augment data'):
        aug_data_withoutnew = aug_data[4:]
        for label in os.listdir(train_label_path):
            label_prefix = os.path.splitext(label)[0]
            if aug_data_withoutnew == label_prefix:
                # copy aug data to train folder
                shutil.copytree(src=os.path.join(aug_data_path, aug_data),
                                dst=os.path.join(train_data_path, aug_data))
                shutil.copy(src=os.path.join(train_label_path, label),
                            dst=os.path.join(train_label_path, 'new_'+label))


def generate_new_dataset(src_dataset, aug_dataset):
    dst = 'my-dataset/new_images'
    if not os.path.exists('my-dataset/new_images'):
        os.makedirs('my-dataset/new_images')
    for data in tqdm(os.listdir(src_dataset)):
        shutil.copytree(src=os.path.join(src_dataset, data),
                        dst=os.path.join(dst, data))
    for data in tqdm(os.listdir(aug_dataset)):
        shutil.copytree(src=os.path.join(aug_dataset, data),
                        dst=os.path.join(dst, data))


def main(folder, template_folder=None):
    # generate_new_dataset('my-dataset/images',
    #                      'my-dataset/merge_background_images')
    if template_folder is not None:
        # split origin dataset with template to train and test set
        split_file_with_template(src_file_path=f'{folder}/images', template_train_path=f'{template_folder}/train',
                                 template_test_path=f'{template_folder}/test',
                                 train_path=f'{folder}/train', test_path=f'{folder}/test')
    # split origin dataset to train and test set
    split_file_with_union(src_file_path=f'{folder}/images',
                          train_path=f'{folder}/train/', test_path=f'{folder}/test/')

    # split origin label to train and test label set
    split_label(src_file_path=f'{folder}/train/', label_path=f'{folder}/labels/',
                train_path=f'{folder}/train_labels/')
    split_label(src_file_path=f'{folder}/test/', label_path=f'{folder}/labels/',
                train_path=f'{folder}/test_labels/')
    # find_augmented_data(aug_data_path='my-dataset/merge_background_images',
    #                     train_data_path='my-dataset/train', train_label_path='my-dataset/train_labels')
    # convert label to coco format, and generate train and test json file
    format_yolo2coco(img_folder_path=f'{folder}/train/', label_folder_path=f'{folder}/train_labels/', out_file=f'{folder}/train.json',
                     sub_images=('filled.png', 'default.png', 'default-labeled.png', 'filled-labeled.png'))
    format_yolo2coco(img_folder_path=f'{folder}/test/', label_folder_path=f'{folder}/test_labels/', out_file=f'{folder}/test.json',
                     sub_images=('filled.png', 'default.png', 'default-labeled.png', 'filled-labeled.png'))


if __name__ == '__main__':
    main('my_dataset_fill_layer')
