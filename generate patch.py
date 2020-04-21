import numpy as np
import cv2 as cv
import os
import random

img_pth = r"D:\BaiduNetdiskDownload\image"
mask_pth = r"D:\BaiduNetdiskDownload\mask"

img_save_pth = r"D:\BaiduNetdiskDownload\gen_img"
mask_save_pth = r"D:\BaiduNetdiskDownload\gen_mask"


def generate_patch(img_path, mask_path, size=256, img_num=128):
    img_list = os.listdir(img_path)
    mask_list = os.listdir(mask_path)

    length = len(img_list)

    image_gen_path = []
    mask_gen_path = []

    count = 0

    image_each = img_num // length

    for i in range(length):
        icount = 0

        # 读入图片和ground truth
        img = cv.imread((img_pth + '/' + img_list[i]))
        mask = cv.imread((mask_pth + '/' + mask_list[i]), cv.COLOR_BGR2GRAY)

        height, width, _ = img.shape

        while (icount < image_each):
            # 生成随机点的横纵坐标。
            random_height = random.randint(0, height - size - 1)
            random_width = random.randint(0, width - size - 1)

            # 生成256*256的patch
            img_gen = img[random_height: random_height + size, random_width: random_width + size, :]
            mask_gen = mask[random_height: random_height + size, random_width: random_width + size]

            # 保存到list中
            image_gen_path.append(img_save_pth + '/' + '%05d.png' % count)
            mask_gen_path.append(mask_save_pth + '/' + '%05d.png' % count)

            cv.imwrite(img_save_pth + '/' + '%05d.png' % count, img_gen)
            cv.imwrite(mask_save_pth + '/' + '%05d.png' % count, mask_gen)

            count += 1
            icount += 1

    with open(img_save_pth + '/img_list.txt', 'w') as f:
        f.writelines(image_gen_path)

    with open(mask_save_pth + '/mask_list.txt', 'w') as f:
        f.writelines(mask_gen_path)

generate_patch(img_pth, mask_pth)

# class Dataset(object):
#     def __init__(self, img_save_pth, mask_save_pth):
#         self.image_path = img_save_pth
#         self.mask_path = mask_save_pth
#         self.batch_count = 0
#
#     def next_batch(self, batch_size):
#         sample_num = os.listdir(self.image_path)
#         length = len(sample_num)
#
#         index = np.random.permutation(length)
#
#         start = (self.batch_count * batch_size) % length
#         end = min(start + batch_size, length)
#
#         img = []
#         mask = []
#         for i in range(start, end):
#             img_ = cv.imread((self.image_path + '/%05d.png' % index[i]))
#             mask_ = cv.imread((self.mask_path + '/%05d.png' % index[i]), 0)
#
#             img.append(cv.imread((self.image_path + '/%05d.png' % index[i])))
#             mask.append(cv.imread((self.mask_path + '/%05d.png' % index[i]), 0))
#
#         return np.array(img), np.array(mask)


# generate_patch(img_pth, mask_pth)

# dataset = Dataset(img_save_pth, mask_save_pth)
# image, label = dataset.next_batch(5)
#
# print(label[0][190])