import cv2
import numpy as np
from matplotlib import pyplot as plt

name = 'P1'  # SpatialDomain 或 CameraMan


def ReadImages(name):
    """
    读入图片
    :param name: 读入目录
    :return:返回读入图像
    """
    img = cv2.imread(f'images/HistogramEqualization/{name}/Raw.jpg', 0)
    return img


# 打印直方图
def HistogramShow(imgs, name):
    """

    :param imgs: 输入字典
    :param name: 输出目录
    :return:
    """
    plt.figure()
    i = 1
    for key, value in imgs.items():
        hist, bins = np.histogram(value.flatten(), 256, [0, 256])
        # 计算累积分布图
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        plt.subplot(2, len(imgs), i)
        plt.imshow(value,cmap="gray")
        plt.title(key)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2,len(imgs),i+len(imgs))
        plt.plot(cdf_normalized, color='b')
        plt.hist(value.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.title(key)
        plt.legend(('cdf', 'histogram'), loc='upper left')
        i += 1
    plt.tight_layout()
    plt.savefig(f'images/HistogramEqualization/{name}.png')


def HistogramEqualization(img, name):
    """
    直方图均衡化函数
    :param imgs: 输入图片
    :param name: 目录
    :return:均衡化处理后的图片
    """
    equ = cv2.equalizeHist(img)
    cv2.imwrite(f'images/HistogramEqualization/{name}/HistogramEqualization.png', equ)
    return equ


def Clahe(img, name):
    """
    有限对比适应性直方图均衡化
    :param imgs:输入图片
    :param name:目录
    :return:均衡化处理后的图片
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cls = clahe.apply(img)
    cv2.imwrite(f'images/HistogramEqualization/{name}/Clahe.png', cls)
    return cls


# 读入图片
img = ReadImages(name)
equ = HistogramEqualization(img, name)
cls = Clahe(img, name)
imgs = {'raw': img, 'HistogramEqualization': equ, 'Clahe': cls}
HistogramShow(imgs, name + '/Histogram')
