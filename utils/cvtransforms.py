# coding: utf-8
import random
import cv2
import numpy as np
import torch


def TensorRandomFlip(tensor):
    # (b, c, t, h, w)
    if (random.random() > 0.5):
        return torch.flip(tensor, dims=[4])
    return tensor


def TensorRandomCrop(tensor, size):
    h, w = tensor.size(-2), tensor.size(-1)
    tw, th = size
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return tensor[:, :, :, x1:x1 + th, y1:y1 + w]


def CenterCrop(batch_img, size):
    w, h = batch_img.shape[2], batch_img.shape[1]
    th, tw = size
    th = th // 2
    img = np.zeros((batch_img.shape[0], th * 2, tw))
    x1 = int(round((w - tw)) / 2.)
    y1 = int(round((h - th)) / 2.)
    tmpimg = batch_img[:, y1:y1 + th, x1:x1 + tw]
    for i_iter, input in enumerate(tmpimg):
        tmp = cv2.resize(input, (th * 2, tw), fx=1, fy=2)
        img[i_iter, :, :] = tmp
    return img


def RandomCrop(batch_img, size):
    w, h = batch_img.shape[2], batch_img.shape[1]
    th, tw = size
    img = np.zeros((batch_img.shape[0], th, tw))
    x1 = random.randint(0, 8)
    y1 = random.randint(0, 8)
    img = batch_img[:, y1:y1 + th, x1:x1 + tw]
    return img


def Crop(batch_img, size):
    w, h = batch_img.shape[2], batch_img.shape[1]
    th, tw = size
    img = np.zeros((batch_img.shape[0], th, tw))
    x1 = 15
    y1 = 15
    img = batch_img[:, y1:y1 + th, x1:x1 + tw]
    return img


def HorizontalFlip(batch_img):
    if random.random() > 0.5:
        batch_img = np.ascontiguousarray(batch_img[:, :, ::-1])
    return batch_img


def gaussian_noise(image, mean, sigma):
    for i in image:
        img_noise = i.copy()
        # 将图片灰度标准化
        img_noise = img_noise / 255
        # 产生高斯 noise
        noise = np.random.normal(mean, sigma, img_noise.shape)
        # 将噪声和图片叠加
        i = img_noise + noise
        # 将超过 1 的置 1，低于 0 的置 0
        i = np.clip(i, 0, 1)
        # 将图片灰度范围的恢复为 0-255
        i = np.uint8(i * 255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return image  # 这里也会返回噪声，注意返回值
