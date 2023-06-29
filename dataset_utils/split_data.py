import os
import shutil
from tqdm import tqdm
import random


def split_file_with_template(src_file_path, template_train_path, template_test_path, train_path, test_path):
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    template_train_lsit = os.listdir(template_train_path)
    template_test_lsit = os.listdir(template_test_path)
    # using template to generate dataset
    for template in tqdm(template_train_lsit, desc='template'):
        for file in os.listdir(src_file_path):
            if file == template:
                if not os.path.exists(os.path.join(train_path, file)):
                    shutil.copytree(os.path.join(
                        src_file_path, file), os.path.join(train_path, file))
    for template in tqdm(template_test_lsit, desc='template'):
        for file in os.listdir(src_file_path):
            if file == template:
                if not os.path.exists(os.path.join(test_path, file)):
                    shutil.copytree(os.path.join(
                        src_file_path, file), os.path.join(test_path, file))


def split_file_with_union(src_file_path, train_path, test_path):
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    src_file_list = os.listdir(src_file_path)
    uuid_list = []
    union_file_list = []
    # find the root name for all images
    for file_name in src_file_list:
        file_name = file_name.split('-')
        file_name = file_name[0] + file_name[1] + file_name[2] + file_name[3]
        if file_name not in uuid_list:
            uuid_list.append(file_name)
    # union same root files
    for uuid in tqdm(uuid_list, desc='find uuid'):
        file_name_list = []
        for file_name in src_file_list:
            name = file_name.split('-')
            name = name[0] + name[1] + name[2] + name[3]
            if uuid == name:
                file_name_list.append(file_name)
        union_file_list.append(file_name_list)
    src_file_list = random.sample(union_file_list, len(union_file_list))
    for data_list in tqdm(src_file_list[:int(len(src_file_list)*0.8)], desc='train'):
        for data in data_list:
            if not os.path.exists(os.path.join(train_path, data)):
                shutil.copytree(os.path.join(src_file_path, data),
                                os.path.join(train_path, data))
    for data_list in tqdm(src_file_list[int(len(src_file_list)*0.8):], desc='test'):
        for data in data_list:
            if not os.path.exists(os.path.join(test_path, data)):
                shutil.copytree(os.path.join(src_file_path, data),
                                os.path.join(test_path, data))


def split_file(src_file_path, train_path, test_path):
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    dataset_len = len(os.listdir(src_file_path))
    src_file_list = os.listdir(src_file_path)
    # print(src_file_list[0:5])
    src_file_list = random.sample(src_file_list, len(src_file_list))
    # print(src_file_list[0:5])

    for i, data in tqdm(enumerate(src_file_list[0:int(len(src_file_list) * 0.8)])):
        if not os.path.exists(os.path.join(train_path, data)):
            shutil.copytree(os.path.join(src_file_path, data),
                            os.path.join(train_path, data))
    for i, data in tqdm(enumerate(src_file_list[int(len(src_file_list) * 0.8):])):
        if not os.path.exists(os.path.join(test_path, data)):
            shutil.copytree(os.path.join(src_file_path, data),
                            os.path.join(test_path, data))


def split_label(src_file_path, label_path, train_path):
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    dataset_len = len(os.listdir(src_file_path))
    print(dataset_len)
    for i, data in tqdm(enumerate(os.listdir(src_file_path)), desc='split'):
        file_name = data + '.txt'
        if not os.path.exists(os.path.join(train_path, data)):
            shutil.copy(os.path.join(label_path, file_name),
                        os.path.join(train_path, file_name))


if __name__ == '__main__':
    # split_file('images/', 'train/', 'test/')
    # split_label('train/', 'labels/', 'train_label/')
    # split_label('test/', 'labels/', 'test_label/')
    split_file_with_union(src_file_path='new_my_dataset/images',
                          train_path='new_my_dataset/train/', test_path='new_my_dataset/test/')
__all__ = [
    'split_label',
    'split_file',
    'split_file_with_union',
    'split_file_with_template'
]
