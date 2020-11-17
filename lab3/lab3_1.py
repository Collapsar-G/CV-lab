import warnings
from time import time

import cv2.cv2 as cv2
import numpy as np
from numba import jit

warnings.filterwarnings('ignore')


@jit
def Scale(image_input, sx, sy):
    ## 读取图像的长宽通道数
    if len(image_input.shape) == 3:
        H, W, C = image_input.shape
    else:
        img = np.expand_dims(image_input, axis=-1)
        H, W, C = img.shape

    # 设置目标图像的shape
    dst_shape = [int(H * sy), int(W * sx), C]
    # 目标图像初始化
    dst_img = np.zeros(shape=dst_shape, dtype=np.uint8)
    dst_h, dst_w, dst_c = dst_shape

    for i in range(dst_h):
        for j in range(dst_w):

            # 原图像和目标图像的几何对齐
            src_x = j / sx
            src_y = i / sy
            src_x_int = int(j / sx)
            src_y_int = int(i / sy)
            a = src_x - src_x_int
            b = src_y - src_y_int
            if src_x_int + 1 == W or src_y_int + 1 == H:
                dst_img[i][j] = image_input[src_y_int][src_x_int]
                continue
            # print(src_x_int, src_y_int)
            dst_img[i][j] = a * b * image_input[src_y_int + 1][src_x_int + 1] + \
                            a * (1. - b) * image_input[src_y_int][src_x_int + 1] + \
                            (1. - a) * b * image_input[src_y_int + 1][src_x_int] + \
                            (1. - a) * (1. - b) * image_input[src_y_int][src_x_int]

    # 输出图像调整
    dst_img = dst_img.astype(np.uint8)
    np.clip(dst_img, 0, 255)
    return dst_img


if __name__ == "__main__":
    shape_x = 1.8
    shape_y = 0.8

    # 读取并显示图像
    img_path = r'./image/lab2.png'
    img = cv2.imread(img_path)
    cv2.imshow('org_img', img)

    # 向下采样并展示图像
    start_time = time()
    img_sub_sampled = Scale(img, shape_x, shape_y)
    cv2.imshow('sub_sampled', img_sub_sampled)
    end_time = time()
    print("图像缩放用时：", end_time - start_time)

    cv2.waitKey(0)
    # 关闭窗口
    cv2.destroyAllWindows()
