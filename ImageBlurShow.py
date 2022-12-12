# coding: utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt


def addsalt_pepper(img, SNR):
    img_ = img.copy()
    c, h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, c, axis=0)  # 按channel 复制到 与img具有相同的shape
    img_[mask == 1] = 255  # 盐噪声
    img_[mask == 2] = 0  # 椒噪声

    return img_


img = cv2.imread('images/SpatialDomain/Raw.bmp')

SNR_list = [0.9, 0.7, 0.5, 0.3]

img_s = addsalt_pepper(img.transpose(2,1,0),SNR_list[0])
img_s = img_s.transpose(2,1,0)
cv2.imwrite('images/SpatialDomain/SaltNoise.png',img_s)