import random

import torch
from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean / std
    _std = 1 / std
    return normalize(tensor, _mean, _std)


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean / std
        self._std = 1 / std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)
        return normalize(tensor, self._mean, self._std)


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def exchange_in_ext(img, res, num, radius):
    img = img.to('cpu', dtype=torch.float32)
    res = res.to('cpu', dtype=torch.long)
    for i, r in enumerate(res):
        front = np.argwhere(r == 1).T
        back = np.argwhere(r == 0).T
        front_indices = random.sample(range(len(front)), num)
        back_indices = random.sample(range(len(back)), num)

        def check_boundary(x, y, h, w):
            l, r, t, b = x - radius, x + radius, y - radius, y + radius

            if l < 0:
                l, r = 0, 2 * radius
            if r >= h:
                l, r = h - 2 * radius, h
            if t < 0:
                t, b = 0, 2 * radius
            if b >= w:
                t, b = w - 2 * radius, w

            return l, r, t, b

        def swap_pixels(front_index, back_index, rr):
            _, h, w = rr.shape
            x1, y1 = front_index[1], front_index[2]
            x2, y2 = back_index[1], back_index[2]
            l1, r1, t1, b1 = check_boundary(x1, y1, h, w)
            l2, r2, t2, b2 = check_boundary(x2, y2, h, w)

            t = img[i, :, l1: r1, t1: b1].clone()
            img[i, :, l1: r1, t1: b1] = img[i, :, l2: r2, t2:b2]
            img[i, :, l2: r2, t2:b2] = t

            t = rr[:, l1: r1, t1: b1].clone()
            rr[:, l1: r1, t1: b1] = rr[:, l2: r2, t2:b2]
            rr[:, l2: r2, t2: b2] = t

        for f, b in zip(front_indices, back_indices):
            swap_pixels(front[f], back[b], r)
    img = img.to('cuda')
    res = res.to('cuda')
    return img, res
