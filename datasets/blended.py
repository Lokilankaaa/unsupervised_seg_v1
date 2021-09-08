import torch
import torch.utils.data as data
import numpy as np
import os
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


class Blended(data.Dataset):
    cmap = voc_cmap()

    def __init__(self, root, transform=None):
        self.imgs = list_files(os.path.join(root, 'imgs'), 'jpg', True)
        self.labels = list_files(os.path.join(root, 'labels'), 'jpg', True)
        self.reference_img = list_files(os.path.join(root, 'reference_img'), 'jpg', True)
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406],
        #                          [0.229, 0.224, 0.225], True),
        # ]) if transform is None else transform

        self.transform = transform

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx], cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.labels[idx], cv2.CV_8UC1)
        r = cv2.imread(self.reference_img[idx], cv2.COLOR_BGR2RGB)
        return self.transform(img, label), self.transform(r, label)

    def __len__(self):
        return len(self.imgs)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]
