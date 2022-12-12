import cv2
import numpy as np
import matplotlib.pyplot as plt
import  HistogramEqualization
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img = cv2.imread('images/FrequencyDomain/Raw.bmp', 0)
rows, cols = img.shape[:2]
d0 = 80

'''
对图像进行傅里叶变换，并返回换位后的频率矩阵
步骤② ：填充输入图像img
步骤③ ：对输入图像进行傅里叶变换得到fft_mat
步骤④ ：对fft_mat进行换位，低频部分移到中间，高频部分移到四周
'''
# 计算最优尺寸
nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)
# 根据新尺寸，建立新变换图像
nimg = np.zeros((nrows, ncols))
nimg[:rows, :cols] = img
# 得到的fft_mat有2个通道，实部和虚部
fft_mat = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)
# 反换位，低频部分移到中间，高频部分移到四周
fft_mat = np.fft.fftshift(fft_mat)

'''
计算D(u,v)
'''


def fft_distances(m, n):
    u = np.array([i - m / 2 for i in range(m)], dtype=np.float32)
    v = np.array([i - n / 2 for i in range(n)], dtype=np.float32)
    ret = np.ones((m, n))
    for i in range(m):
        for j in range(n):
            ret[i][j] = np.sqrt(u[i] * u[i] + v[j] * v[j])
    u = np.array([i if i <= m / 2 else m - i for i in range(m)], dtype=np.float32)
    v = np.array([i if i <= m / 2 else m - i for i in range(m)], dtype=np.float32)
    return ret


'''
步骤① ：选择低通滤波器
'''


def change_filter(flag):
    # 理想低通滤波器
    if flag == 1:
        # 初始化滤波器，因为fft_mat有2个通道，filter_mat也需要2个通道
        filter_mat = np.zeros((nrows, ncols, 2), np.float32)
        # 将filter_mat中以(ncols/2, nrows/2)为圆心、d0为半径的圆内的值设置为1
        cv2.circle(filter_mat, (np.int_(ncols / 2), np.int_(nrows / 2)), d0, (1, 1, 1), -1)
    # 布特沃斯低通滤波
    elif flag == 2:
        n = 2  # 2阶
        filter_mat = None
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = 1 / (1 + np.power(duv / d0, 2 * n))
        filter_mat = cv2.merge((filter_mat, filter_mat))  # fliter_mat 需要2个通道
    # 高斯低通滤波（σ为d0）
    else:
        filter_mat = None
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = np.exp(-(duv * duv) / (2 * d0 * d0))
        filter_mat = cv2.merge((filter_mat, filter_mat))  # fliter_mat 需要2个通道
    return filter_mat


'''
对图像进行傅里叶反变换，返回反变换图像
步骤⑥ ：对fft_mat进行换位，低频部分移到四周，高频部分移到中间
步骤⑦ ：傅里叶反变换，并计算幅度，提取左上角的 M×N 区域
'''


def ifft(fft_mat):
    # 反换位，低频部分移到四周，高频部分移到中间
    f_ishift_mat = np.fft.ifftshift(fft_mat)
    # 傅里叶反变换
    img_back = cv2.idft(f_ishift_mat)
    # 将复数转换为幅度, sqrt(re^2 + im^2)
    img_back = cv2.magnitude(*cv2.split(img_back))
    # 标准化到0~255之间
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(img_back))[:rows, :cols]


'''
选择不同的滤波器对图像进行滤波
flag:1理想低通 2布特沃斯低通 3高斯低通
利用 HP+LP=1可计算高频滤波器
对滤波后的结果进行傅里叶反变换，得到输出图像
'''
img1 = ifft(change_filter(1) * fft_mat)  # 理想低通
img2 = ifft(change_filter(2) * fft_mat)  # 布特沃斯低通
img3 = ifft(change_filter(3) * fft_mat)  # 高斯低通
img4 = ifft((1 - change_filter(1)) * fft_mat)  # 理想高通
img5 = ifft((1 - change_filter(2)) * fft_mat)  # 布特沃斯高通
img6 = ifft((1 - change_filter(3)) * fft_mat)  # 高斯高通

imgs_low = {'原图': img, '理想低通': img1, '布特沃斯低通': img2, '高斯低通': img3}
imgs_high = {'原图': img, '理想高通': img4, '布特沃斯高通': img5, '高斯高通': img6}
HistogramEqualization.HistogramShow(imgs_low,'FrequencyDomain/low.png')
HistogramEqualization.HistogramShow(imgs_high,'FrequencyDomain/high.png')
# cv2.imwrite('images/FrequencyDomain/ILP.png', img1)
# cv2.imwrite('images/FrequencyDomain/BWLP.png', img2)
# cv2.imwrite('images/FrequencyDomain/GLP.png', img3)
# cv2.imwrite('images/FrequencyDomain/IHP.png', img4)
# cv2.imwrite('images/FrequencyDomain/BWHP.png', img5)
# cv2.imwrite('images/FrequencyDomain/GHP.png', img6)
