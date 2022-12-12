import cv2
import matplotlib.pyplot as plt
import numpy as np
import HistogramEqualization
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ImageName = ['PepperNoise', 'SaltNoise', 'SaltandPepperNoise']




def Saveplt(imgs, name):
    '''

    :param imgs: 输入图像字典
    :param name: 输出命名
    :return:
    '''
    plt.figure()
    i = 1
    for key, value in imgs.items():
        plt.subplot(2, 2, i)
        i += 1
        plt.imshow(value, cmap="gray")
        plt.title(key)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'images/SpatialDomain/remove{name}.png')


def SpatialDomain(img_name):
    '''

    :param img_name: 图片列表
    :return:
    '''
    for name in img_name:
        img = cv2.imread(f'images/SpatialDomain/{name}.png', 0)

        # 算数均值滤波
        img1 = cv2.blur(img, (3, 3))
        # 谐波均值滤波器
        img2 = 1 / cv2.blur(1 / (img + 1E-10), (3, 3))  # 加1E-10防止除0操作
        # 中值滤波
        img3 = cv2.medianBlur(img, 3)
        imgs = {name: img, '算数均值滤波': img1, '谐波均值滤波': img2, '中值滤波': img3}
        Saveplt(imgs, name)
        HistogramEqualization.HistogramShow(imgs, f'SpatialDomain/{name}')


def Laplacian(name):
    '''

    :param name: 原图片的命名
    :return:
    '''
    img = cv2.imread(f'images/SpatialDomain/{name}.bmp', 0)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    edge_img = cv2.filter2D(img, -1, kernel)
    output_img = cv2.add(img, edge_img)
    imgs = {'Raw':img,'edge':edge_img,'output':output_img}
    plt.figure()
    i = 1
    for key, value in imgs.items():
        plt.subplot(1, 3, i)
        i += 1
        plt.imshow(value, cmap="gray")
        plt.title(key)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig('images/SpatialDomain/Laplacian.png')
    HistogramEqualization.HistogramShow(imgs,f'SpatialDomain/Laplacian')


SpatialDomain(ImageName)
Laplacian('Raw')
