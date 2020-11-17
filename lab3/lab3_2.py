import warnings
from time import time

import cv2.cv2 as cv2
import numpy as np
from numba import jit

warnings.filterwarnings('ignore')


@jit
def Deformation(image_input):
    ## 读取图像的长宽通道数
    if len(image_input.shape) == 3:
        H, W, C = image_input.shape
    else:
        img = np.expand_dims(image_input, axis=-1)
        H, W, C = img.shape

    # 设置目标图像的shape
    dst_shape = [H, W, C]
    # 目标图像初始化
    dst_img = np.zeros(shape=dst_shape, dtype=np.uint8)
    dst_h, dst_w, dst_c = dst_shape

    ## 遍历图像
    for i in range(H):
        for j in range(W):
            # 中心坐标归一化
            temp_x = (j - 0.5 * W) / (0.5 * W)
            temp_y = (i - 0.5 * H) / (0.5 * H)
            r = np.sqrt(temp_x ** 2 + temp_y ** 2)
            th = (1 - r) ** 2
            if r < 1:
                x = np.cos(th) * temp_x - np.sin(th) * temp_y
                y = np.sin(th) * temp_x + np.cos(th) * temp_y
            else:
                x = temp_x
                y = temp_y
            dst_img[i][j] = image_input[int((y + 1) * 0.5 * H)][int((x + 1) * 0.5 * W)]

    # 输出图像调整
    dst_img = dst_img.astype(np.uint8)
    np.clip(dst_img, 0, 255)
    return dst_img


if __name__ == "__main__":
    # 读取并显示图像
    img_path = r'./image/lab2.png'
    img = cv2.imread(img_path)
    cv2.imshow('org_img', img)

    # 向下采样并展示图像
    start_time = time()
    img_sub_sampled = Deformation(img)
    cv2.imshow('sub_sampled', img_sub_sampled)
    end_time = time()
    print("图像变形用时：", end_time - start_time)

    cv2.waitKey(0)
    # 关闭窗口
    cv2.destroyAllWindows()
