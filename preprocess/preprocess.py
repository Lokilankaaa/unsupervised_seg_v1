import cv2
import numpy as np
import argparse
import os
import random
import multiprocessing as mp


def build_args():
    paser = argparse.ArgumentParser()
    paser.add_argument('-n', type=int, default=100)
    return paser


def read_img(path, resize=(256, 256)):
    img = cv2.imread(path)
    if resize is not None:
        img = cv2.resize(img, resize)
    return img


def create_patch_index(split):
    index = np.array(range(split))
    return np.dstack(np.meshgrid(index, index)).reshape(-1, 2)


def sampler(num, patch_size, positive_pt, indexes, f_directory, b_directory, resize, imglist, dst):
    for i in range(num):
        pairs = [random.choice(imglist['f']), random.choice(imglist['b'])]
        img1 = read_img(os.path.join(f_directory, pairs[0]), resize)
        img2 = read_img(os.path.join(b_directory, pairs[1]), resize)
        res = img2.copy()
        labels = np.zeros(resize)

        p = int(len(indexes) * positive_pt)
        pp = np.array(random.sample(indexes.tolist(), p))
        for c in pp:
            res[c[0] * patch_size[0]:(c[0] + 1) * patch_size[0],
            c[1] * patch_size[1]:(c[1] + 1) * patch_size[0], :] = img1[
                                                                  c[0] * patch_size[0]:(c[0] + 1) * patch_size[
                                                                      0],
                                                                  c[1] * patch_size[1]:(c[1] + 1) * patch_size[
                                                                      0], :]
            labels[c[0] * patch_size[0]:(c[0] + 1) * patch_size[0],
            c[1] * patch_size[1]:(c[1] + 1) * patch_size[0]] = 1

        cv2.imwrite(os.path.join(dst, 'imgs', pairs[0].split('.')[0] + '_' + pairs[1].split('.')[0] + '.jpg'), res)
        cv2.imwrite(os.path.join(dst, 'labels', pairs[0].split('.')[0] + '_' + pairs[1].split('.')[0] + '_labels.jpg'),
                    labels)
        cv2.imwrite(
            os.path.join(dst, 'reference_img', pairs[0].split('.')[0] + '_' + pairs[1].split('.')[0] + '_r.jpg'),
            img1)


def sample_from_directory(f_directory, b_directory, num, positive_pt, split, dst, workers=10, resize=(256, 256)):
    """
    :param directory: source directory to sample training data from
    :param num: number of training examples
    :param positive_pt: the percentage of positive patches in img1
    :param split: number of how many patch to be split in img1
    :param dst: destination directory to save result img
    :param workers: number of workers to run sampling
    :param resize: the size of training img
    :return: None
    """
    if not os.path.exists(f_directory) or not os.path.exists(b_directory):
        return None
    else:
        os.mkdir(dst) if not os.path.exists(dst) else None
        os.mkdir(os.path.join(dst, 'imgs')) if not os.path.exists(os.path.join(dst, 'imgs')) else None
        os.mkdir(os.path.join(dst, 'labels')) if not os.path.exists(os.path.join(dst, 'labels')) else None
        os.mkdir(os.path.join(dst, 'reference_img')) if not os.path.exists(os.path.join(dst, 'reference_img')) else None
        imglist = dict()
        imglist['f'] = os.listdir(f_directory)
        imglist['b'] = os.listdir(b_directory)
        indexes = create_patch_index(split)
        patch_size = int(resize[0] / split), int(resize[1] / split)

        if len(imglist) < 2:
            raise Exception('No enough imgs')

        pool = [mp.Process(target=sampler,
                           args=(
                           num // workers, patch_size, positive_pt, indexes, f_directory, b_directory, resize, imglist,
                           dst))
                for _ in range(workers)]
        for p in pool:
            p.start()

        for p in pool:
            p.join()
        return True


if __name__ == '__main__':
    sample_from_directory('../datasets/data/portrait', '../datasets/data/scene', 1000, 0.5, 8, '../test_sample_out', 12)
