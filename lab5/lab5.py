import warnings
from time import time

# from skimage import measure, color
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from numba import jit

warnings.filterwarnings('ignore')


@jit
def compute_e(img_left, img_right, dmax, size):
    H, W, C = img_left.shape
    e = np.zeros([H, W, 40])
    e_avg = np.zeros([H, W, 40])
    disparity = np.zeros([H, W], dtype=np.int64)
    M, N = size

    for i in range(H):
        for j in range(W):
            for d in range(dmax):
                sum = 0
                for x in range(i, i + M):
                    if (i + M) >= H:
                        break
                    else:
                        for y in range(j, j + N):
                            if (j + d + N) >= W:
                                break
                            else:

                                for k in range(3):
                                    square_diff = (img_left[x, y + d, k] - img_right[x, y, k]) ** 2
                                    sum += square_diff
                e[i][j][d] = sum / (3 * M * N)
            # print(np.mean(e[i,j]))
    # 再计算e_avg（i,j，d）
    for i in range(H):
        for j in range(W):
            e_t = 999999
            for d in range(dmax):
                e_temp = 0
                if (j + d + N) >= W or (i + M) >= H:
                    break
                else:
                    for m in range(M):
                        for n in range(N):
                            e_temp += e[i + m, j + n, d]
                e_temp = e_temp / (M * N)
                e_avg[i, j, d] = e_temp
                if e_avg[i, j, d] < e_t:
                    e_t = e_avg[i, j, d]
                    disparity[i, j] = d + 1
                # print(e_temp)

    # for i in range(H):
    #     for j in range(W):
    #         e_temp = 999999999
    #         for d in range(dmax):
    #             if e_avg[i, j, d] < e_temp:
    #                 e_temp = e_avg[i, j, d]
    #                 disparity[i, j] = d
    # print(e.shape)
    return disparity


@jit
def reliable(img_left, img_right, d, Alpha, size):
    H, W, C = img_left.shape
    e_d = np.zeros([H, W])
    e_d_ = e_d.copy()
    d_ = d.copy()
    M, N = size
    for i in range(H):
        for j in range(W):
            sum = 0
            for x in range(i, i + M):
                if (i + M) >= H:
                    break
                else:
                    for y in range(j, j + N):
                        if (j + d[i, j] + N) >= W:
                            break
                        else:
                            for k in range(3):
                                sum += (img_left[x, y + d[i, j], k] - img_right[x, y, k]) ** 2
            e_d[i][j] = sum / (3 * M * N)
    ve = Alpha * np.mean(e_d)
    # print(ve)
    # print(ve)
    # print(np.sum(e_d))
    for i in range(H):
        for j in range(W):
            if e_d[i, j] > ve:
                d_[i, j] = 0
                # print('')
                e_d_[i, j] = 0

    r_d = 1 / (np.mean(e_d))
    # print(d_)
    return r_d, e_d.astype(np.uint8), e_d_, d_


def deep(d, f, T):
    # d_mid = cv2.medianBlur(d, 5)
    z = np.zeros(d.shape)
    H, W = d.shape
    for i in range(H):
        for j in range(W):
            if d[i, j] != 0:
                z[i, j] = f * T / d[i, j]
    return z


if __name__ == "__main__":
    dmax = 40
    windows_size = 3, 3
    Alpha = 1
    f = 30
    T = 20
    # 读取并显示图像
    img_path_left = r'./image/view1m.png'
    img_path_right = r'./image/view5m.png'

    img_left = cv2.imread(img_path_left)
    img_right = cv2.imread(img_path_right)
    cv2.imshow('img_left', img_left)
    cv2.imshow('img_right', img_right)
    start_time = time()

    disparity = compute_e(img_left.copy(), img_right.copy(), dmax, windows_size)
    end_time = time()
    print('视差图用时:', end_time - start_time)
    cv2.imshow('d', (disparity.copy()[0:disparity.shape[0], 0:180] * 255 / 40).astype(np.uint8))
    # print(np.mean(disparity))

    r_d, e_d, e_d_, d_ = reliable(img_left, img_right, disparity, Alpha, windows_size)
    # cv2.imshow('d_', (d_ * 255 / 40).astype(np.uint8))
    # cv2.imshow('ed', (e_d).astype(np.uint8))
    # cv2.imshow('ed_', (e_d_).astype(np.uint8))
    z = deep(disparity, f, T)
    cv2.imshow('Z', (z[0:z.shape[0], 0:180] * 255 / 120).astype(np.uint8))
    # print(np.max(z))
    cv2.imwrite(r'./image/ed.png', (disparity.copy()[0:disparity.shape[0], 0:180] * 255 / 40).astype(np.uint8))
    cv2.imwrite(r'./image/deep.png', (z[0:z.shape[0], 0:180] * 255 / 120).astype(np.uint8))

    # Load and format data
    # dem = cbook.get_sample_data(r'./image/deep.png', np_load=True)

    x = np.arange(z.shape[0], dtype=int)
    y = np.arange(180, dtype=int)
    x, y = np.meshgrid(x, y)

    # region = np.s_[5:50, 5:50]
    # x, y, z = x[region], y[region], z[region]

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    # rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, d_[0:z.shape[0], 0:180].T, cmap='bone')

    plt.show()

    cv2.waitKey(0)
    # 关闭窗口
    cv2.destroyAllWindows()
