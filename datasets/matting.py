import torch
import torch.utils.data as data
import numpy as np
import os

from PIL import Image

from .utils import *
import cv2


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class Matting(data.Dataset):
    cmap = voc_cmap()

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = []
        self.masks = []
        for (dirpath, dirnames, filenames) in tqdm(os.walk(img_dir)):
            L = [os.path.join(dirpath, x) for x in filenames if x.endswith(('.jpg', 'png'))]
            self.images += L
            maskpath = dirpath.replace('clip_img', 'matting').replace('clip', 'matting')

            for x in filenames:
                if os.path.exists(os.path.join(maskpath, x.replace('.jpg', '.png'))):
                    self.masks += [os.path.join(maskpath, x.replace('.jpg', '.png'))]
                else:
                    self.images.remove(os.path.join(dirpath, x))

        print(len(self.images))
        #     for (dirpath, dirnames, filenames) in os.walk(mask_dir):
        #         self.masks += [os.path.join(dirpath,x) for x in filenames if x.endswith(('.jpg','png'))]
        # Check if all matting exists
        for f in self.masks:
            if os.path.isfile(f):
                continue
            else:
                self.images.remove(f.replace())
                raise Exception("ALL MASKS NOT FOUND {}".format(f))
        print(len(self.masks))
        assert len(self.masks) == len(self.images), " Number of images and masks not same"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = self.images[i]
        mask_path = self.masks[i]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGBA"), dtype=np.float32)[:, :, 3]
        mask[mask < 255] = 0

        mask[mask == 255.0] = 1
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        return image, mask

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]
