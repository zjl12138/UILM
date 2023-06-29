import os
import json

from PIL.Image import OPEN


def count_num(folder_name):
    
    print('total images:', len(os.listdir(f'{folder_name}/images')))
    print('total labels:', len(os.listdir(f'{folder_name}/labels')))
    
    print('train iamges:', len(os.listdir(f'{folder_name}/train')))
    print('test images:', len(os.listdir(f'{folder_name}/test')))
    
    test_json = json.load(open(f'{folder_name}/test.json', 'r'))
    train_json = json.load(open(f'{folder_name}/train.json', 'r'))
    
    print('train json:', len(train_json['images']))
    print('test json:', len(test_json['images']))
    print('<------------------------------------>')

def check(folder_name):
    train_dataset = json.load(open(f'{folder_name}/train.json', 'r'))['images']
    test_dataset = json.load(open(f'{folder_name}/test.json', 'r'))['images']
    for image in train_dataset:
        image_name = image['file_name']
        for test_image in test_dataset:
            test_image_name = test_image['file_name']
            if image_name == test_image_name:
                print(test_image_name)
                raise "the dataset generate false!"


if __name__ == '__main__':
    count_num('my_dataset')
    check('my_dataset')    
