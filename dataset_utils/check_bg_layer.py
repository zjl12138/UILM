import os
from PIL import Image, ImageDraw
from tqdm import tqdm

def label_bg_layer(img_path, label_path, img_type):
    bg_data_list = os.listdir(img_path)
    label_list = os.listdir(label_path)
    label_prefix_list = []
    for label in label_list:
        label = os.path.splitext(label)[0]
        label_prefix_list.append(label)
    # find backgound label
    for bg_data in tqdm(bg_data_list):
        bg_data_withoutnew = bg_data[4:]
        if bg_data_withoutnew in label_prefix_list:
            single_label_path = os.path.join(label_path, bg_data_withoutnew + '.txt')
            single_img_path = os.path.join(img_path, bg_data, img_type)
            with open(single_label_path, 'r') as label_fp:
                label_list = [
                    (float(x.split(" ")[1]), float(x.split(" ")[2]),
                    float(x.split(" ")[3]), float(x.split(" ")[4]),)
                    for x in label_fp.readlines()
                ]
                image = Image.open(single_img_path)
                h, w = image.size
                image_draw = ImageDraw.Draw(image)
                for label in label_list:
                    # draw label
                    image_draw.rectangle(
                        [(label[0] - label[2] / 2) * w, (label[1] - label[3] / 2) * h, (label[0] + label[2] / 2) * w,
                        (label[1] + label[3] / 2) * h],
                        fill=None, outline='red', width=3
                    )
                # save labeled image
                image.save(os.path.splitext(single_img_path)[
                        0] + '-labeled' + os.path.splitext(single_img_path)[1])


if __name__ == '__main__':

    label_bg_layer('my-dataset/merge_background_images',
                   'my-dataset/labels',
                   img_type='filled.png')
