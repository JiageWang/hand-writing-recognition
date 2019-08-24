import os
import struct
import pickle
import threading

import numpy as np
from PIL import Image


# 处理单个gnt文件获取图像与标签
def read_from_gnt_dir(gnt_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size:
                break
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            label = header[5] + (header[4] << 8)
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            if header_size + width * height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
            yield image, label

    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, label in one_file(f):
                    yield image, label


def gnt_to_img(gnt_dir, img_dir):
    counter = 0
    for image, label in read_from_gnt_dir(gnt_dir=gnt_dir):
        label = struct.pack('>H', label).decode('gb2312')
        img = Image.fromarray(image)
        dir_name = os.path.join(img_dir, '%0.5d' % char_dict[label])
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        img.convert('RGB').save(dir_name + '/' + str(counter) + '.png')
        print("train_counter=", counter)
        counter += 1


# 路径
data_dir = './data'
train_gnt_dir = os.path.join(data_dir, 'HWDB1.1trn_gnt')
test_gnt_dir = os.path.join(data_dir, 'HWDB1.1tst_gnt')
train_img_dir = os.path.join(data_dir, 'train')
test_img_dir = os.path.join(data_dir, 'test')
if not os.path.exists(train_img_dir):
    os.mkdir(train_img_dir)
if not os.path.exists(test_img_dir):
    os.mkdir(test_img_dir)

# 获取字符集合
char_set = set()
for _, tagcode in read_from_gnt_dir(gnt_dir=test_gnt_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    char_set.add(tagcode_unicode)
char_list = list(char_set)
char_dict = dict(zip(sorted(char_list), range(len(char_list))))
print(len(char_dict))
print("char_dict=", char_dict)

with open('char_dict', 'wb') as f:
    pickle.dump(char_dict, f)

train_thread = threading.Thread(target=gnt_to_img, args=(train_gnt_dir, train_img_dir)).start()
test_thread = threading.Thread(target=gnt_to_img, args=(test_gnt_dir, test_img_dir)).start()
train_thread.join()
test_thread.join()
