import datetime

import cv2.cv2 as cv2
import numpy as np
from numba import jit


# import cupy as np


# Gaussian filter
@jit
def gaussion(img_input, sigma):
    starttime = datetime.datetime.now()
    k_size = int(round(6 * sigma - 1) // 2 * 2 + 1)  # 预设卷积核大小

    # 图像数据读取、调整
    if len(img_input.shape) == 3:
        H, W, C = img_input.shape
    else:
        img = np.expand_dims(img_input, axis=-1)
        H, W, C = img.shape

    # 对图像进行 zreo padding
    pad_size = int(k_size // 2)
    img_output = np.zeros((H + pad_size * 2, W + pad_size * 2, C), dtype=np.float)
    img_output[pad_size: pad_size + H, pad_size: pad_size + W] = img_input.copy().astype(np.float)
    tmp = img_output.copy()

    #############################
    # # 构造卷积核
    # # 构造二维卷积核
    # k2 = np.zeros((k_size, k_size), dtype=np.float)
    # for x in range(0, k_size):
    #     for y in range(0, k_size):
    #         k2[y, x] = np.exp(-((x-k_size//2) ** 2 + (y-k_size//2) ** 2) / (2 * (sigma ** 2)))
    # k2 /= (2 * np.pi * sigma * sigma)
    # #归一化
    # k2 /= k2.sum()
    # #print(k2)
    #
    # # 二维卷积操作
    # for h in range(H):
    #     for w in range(W):
    #         for c in range(C):
    #             img_output[h + pad_size, w + pad_size, c] = np.sum(k2 * tmp[h:h + k_size, w:w + k_size, c])
    #

    ## 构造一维卷积核
    k1 = np.zeros(k_size, dtype=np.float)
    for x in range(0, k_size):
        k1[x] = np.exp((-(x - k_size // 2) ** 2) / (2 * (sigma ** 2)))

    k1 /= np.sqrt(2 * np.pi * sigma * sigma)

    k1 /= k1.sum()
    print(k1)
    ##############################################################

    # 一维卷积操作
    # 一维横向滤波
    for h in range(H):
        for w in range(W):
            for c in range(C):
                img_output[h + pad_size, w + pad_size, c] = np.sum(k1 * tmp[h:h + k_size, w, c])
                # print(tmp[h:h + k_size, w, c])
    # 一维纵向滤波
    for h in range(H):
        for w in range(W):
            for c in range(C):
                k = np.transpose(k1)
                img_output[h + pad_size, w + pad_size, c] = np.sum(k * tmp[h, w:w + k_size, c])

    # 输出图像预处理
    img_output = img_output[pad_size: pad_size + H, pad_size: pad_size + W].astype(np.uint8)
    np.clip(img_output, 0, 255)
    endtime = datetime.datetime.now()
    print("高斯滤波时间：", endtime - starttime)
    return img_output


if __name__ == "__main__":
    # 读取并展示图片
    img_path = r'./image/2-1/a.jpg'
    img = cv2.imread(img_path)
    cv2.imshow('image of salt and pepper noise', img)
    cv2.waitKey(0)
    ans_img = gaussion(img_input=img, sigma=1)
    cv2.imshow('image without salt and pepper noise', ans_img)
    cv2.waitKey(0)
    # 关闭窗口
    cv2.destroyAllWindows()
