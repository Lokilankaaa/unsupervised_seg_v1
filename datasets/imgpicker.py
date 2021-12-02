import random

import torch
import torch.utils.data as data
import numpy as np
import os
from .utils import *
import cv2
from PIL import Image


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


class ImgPicker(data.Dataset):
    cmap = voc_cmap()

    def __init__(self, root, transform=None):
        self.imgs = list_files(root, 'jpg', True)
        self.imgs.sort()


        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        r = Image.open(random.choice(self.imgs)).convert("RGB")
        return self.transform(img, r), self.transform(r, img)

    def __len__(self):
        return len(self.imgs)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]
