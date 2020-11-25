import warnings
from time import time

import numpy as np
from cv2
from numba import jit

warnings.filterwarnings('ignore')


# 采样函数，org_img：图像矩阵，multiple:采样倍数
@jit
def sampled(org_img, multiple):
    # 读取原图像的shape
    if len(org_img.shape) == 3:
        org_shape = org_img.shape
    else:
        org_img = np.expand_dims(org_img, axis=-1)
        org_shape = img.shape
    org_h, org_w, org_c = org_shape

    # 设置目标图像的shape
    dst_shape = [int(org_h * multiple), int(org_w * multiple), org_c]
    # 目标图像初始化
    dst_img = np.zeros(shape=dst_shape, dtype=np.uint8)
    dst_h, dst_w, dst_c = dst_shape
    # print(dst_shape)

    # if dst_shape>org_shape:
    #     x = org_w
    #     y = org_h
    # else:
    #     x=dst_w
    #     y= dst_h
    # if multiple>1:
    #     multiple = 1/multiple
    # 向下采样
    for i in range(dst_h):
        for j in range(dst_w):

            # 原图像和目标图像的几何对齐
            src_x = j / multiple
            src_y = i / multiple
            src_x_int = int(j / multiple)
            src_y_int = int(i / multiple)
            a = src_x - src_x_int
            b = src_y - src_y_int
            if src_x_int + 1 == org_w or src_y_int + 1 == org_h:
                dst_img[i][j] = org_img[src_y_int][src_x_int]
                continue
            # print(src_x_int, src_y_int)
            dst_img[i][j] = a * b * org_img[src_y_int + 1][src_x_int + 1] + \
                            a * (1. - b) * org_img[src_y_int][src_x_int + 1] + \
                            (1. - a) * b * org_img[src_y_int + 1][src_x_int] + \
                            (1. - a) * (1. - b) * org_img[src_y_int][src_x_int]

    # 输出图像调整
    dst_img = dst_img.astype(np.uint8)
    np.clip(dst_img, 0, 255)
    return dst_img


# 联合双边滤波函数
@jit
def jbf(img_d, img_c, w, sigma_f, sigma_g):
    # 联合双边滤波
    # img_d为输入图像,三维矩阵
    # img_c为引导图像,三维矩阵
    # w为滤波窗口大小
    # sigma_f为spatial kernel标准差
    # sigma_g为range kernel 标准差
    # 返回处理后的图像的三维矩阵

    # 引导图像数据读取、调整
    if len(img_d.shape) == 3:
        H, W, C = img_c.shape
    else:
        img = np.expand_dims(img_c, axis=-1)
        H, W, C = img.shape

    # 输出图像初始化
    dst_img = img_d.copy()

    # 卷积核初始化
    distance = np.zeros([w, w], dtype=np.uint8)
    pad_size = w // 2
    # 计算欧式距离表
    for m in range(w):
        for n in range(w):
            distance[m, n] = (m - pad_size) ** 2 + (n - pad_size) ** 2
    # 由原图的欧式距离计算值域核
    f = np.exp(-0.5 * distance / (sigma_f ** 2))

    for i in range(pad_size, H - pad_size):
        for j in range(pad_size, W - pad_size):
            for d in range(C):
                # 计算当前窗口范围
                i1 = i - pad_size
                i2 = i + pad_size
                j1 = j - pad_size
                j2 = j + pad_size
                # 原图的当前窗口
                # window_s = img_d[i1:i2 + 1, j1: j2 + 1, d]
                # 引导图的当前窗口
                # window_g = img_c[i1:i2 + 1, j1: j2 + 1, d]
                # 由引导图像的灰度值差计算值域核
                g = np.exp(-0.5 * (img_c[i1:i2 + 1, j1: j2 + 1, d] - img_c[i, j, d]) ** 2 / (sigma_g ** 2))
                # f = np.exp(-0.5 * distance / (sigma_f ** 2))

                dst_img[i, j, d] = np.sum(g * f * img_d[i1:i2 + 1, j1: j2 + 1, d]) / np.sum(g * f)
    # 输出图像调整
    dst_img = dst_img.astype(np.uint8)
    np.clip(dst_img, 0, 255)

    return dst_img


if __name__ == "__main__":
    # 读取并显示图像
    img_path = r'./image/2-2/2_2.png'
    img = cv2.imread(img_path)
    cv2.imshow('org_img', img)

    # 向下采样并展示图像
    start_time1 = time()
    img_sub_sampled = sampled(img, 0.5)
    cv2.imshow('sub_sampled', img_sub_sampled)
    end_time1 = time()
    print("下采样用时：", end_time1 - start_time1)

    # 向上采样并展示图像
    start_time2 = time()
    img_up_sampled = sampled(img_sub_sampled, 2)
    cv2.imshow('up_sampling', img_up_sampled)
    end_time2 = time()
    print("上采样用时：", end_time2 - start_time2)

    # 对图像进行联合滤波处理
    start_time3 = time()
    img_jbf = jbf(img, img_up_sampled, 21, 10, 10)
    cv2.imshow('img_jbg', img_jbf)
    end_time3 = time()
    print("联合滤波用时：", end_time3 - start_time3)
    cv2.waitKey(0)
    # 关闭窗口
    cv2.destroyAllWindows()
