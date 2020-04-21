from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from scipy.ndimage.interpolation import shift


class RelaxedBoundaryLossToTensor(object):
    """
    Boundary Relaxation
    """

    def __init__(self, ignore_id, num_classes):
        self.ignore_id = ignore_id
        self.num_classes = num_classes

    def new_one_hot_converter(self, a):
        ncols = self.num_classes + 1
        out = np.zeros((a.size, ncols), dtype=np.uint8)
        out[np.arange(a.size), a.ravel()] = 1
        out.shape = a.shape + (ncols,)
        return out

    def __call__(self, img):

        img_arr = np.array(img)
        img_arr[img_arr == self.ignore_id] = self.num_classes

        # if cfg.STRICTBORDERCLASS != None:
        #     one_hot_orig = self.new_one_hot_converter(img_arr)
        #     mask = np.zeros((img_arr.shape[0], img_arr.shape[1]))
        #     for cls in cfg.STRICTBORDERCLASS:
        #         mask = np.logical_or(mask, (img_arr == cls))
        one_hot = 0

        border = 1
        # if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
        #     border = border // 2
        #     border_prediction = find_boundaries(img_arr, mode='thick').astype(np.uint8)

        for i in range(-border, border + 1):
            for j in range(-border, border + 1):
                shifted = shift(img_arr, (i, j), cval=self.num_classes)
                one_hot += self.new_one_hot_converter(shifted)

        one_hot[one_hot > 1] = 1

        # if cfg.STRICTBORDERCLASS != None:
        #     one_hot = np.where(np.expand_dims(mask, 2), one_hot_orig, one_hot)

        one_hot = np.moveaxis(one_hot, -1, 0)

        # if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
        #     one_hot = np.where(border_prediction, 2 * one_hot, 1 * one_hot)
        #     # print(one_hot.shape)
        return torch.from_numpy(one_hot).byte()


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        # print(idx, "\n")

        mask_file = glob(self.masks_dir + "\\" + idx + '*')
        # print(mask_file)
        img_file = glob(self.imgs_dir + "\\" + idx + '*')
        # print(img_file)

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        # img = self.preprocess(img, self.scale)
        # mask = self.preprocess(mask, self.scale)
        img = np.array(img)
        mask = np.array(mask)

        img = np.transpose(img, (2, 0, 1))
        # mask = np.expand_dims(mask, axis=0)
        # print(img.shape)
        # print(mask.shape)
        target_train_transform = RelaxedBoundaryLossToTensor(0, 5)
        mask = target_train_transform(mask)

        return {'image': torch.from_numpy(img), 'mask': mask}
        # return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
        # return {'image': np.array(img), 'mask': np.array(mask)}