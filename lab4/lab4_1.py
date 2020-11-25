import warnings
from time import time

import cv2
import numpy as np

warnings.filterwarnings('ignore')


def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out


# 模式“RGB”转换为“YCbCr”的公式如下：
#
# Y= 0.257R+0.504G+0.098B+16
# Cb = -0.148R-0.291G+0.439B+128
# Cr = 0.439R-0.368G-0.071*B+128


def bgr2ycbcr(gbr_img):
    mat = [[0.098, 0.504, 0.259], [0.439, -0.291, -0.148], [-0.071, -0.368, 0.439]]
    mat_inv = np.linalg.inv(mat)
    offset = np.array([16, 128, 128])

    ycbcr_img = np.zeros(gbr_img.shape)
    for x in range(gbr_img.shape[0]):
        for y in range(gbr_img.shape[1]):
            for z in range(3):
                ycbcr_img[x, y, z] = np.round(np.sum(mat[z] * gbr_img[x, y, :]) + offset[z])
    ycbcr_img = ycbcr_img.astype(np.uint8)
    return ycbcr_img


def skin(Y, Cb, Cr, R, G, B):
    # if 133 <= Cr <= 173 and 77 <= Cb <= 127 and R > 95 and G > 40 and B > 20 and max(R, G, B) - min(R, G,
    #                                                                                                 B) > 15 and abs(
    #     R - G) > 15 and R > G and R > B and Y >= 70:
    if 133 <= Cr <= 173 and 77 <= Cb <= 127 and Y >= 70:
        return True
    else:
        return False


def dfs(grid, r, c):
    grid[r][c] = 255
    nr, nc = len(grid), len(grid[0])
    for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1), (r + 1, c + 1), (r + 1, c - 1), (r - 1, c - 1),
                 (r - 1, c + 1)]:
        if 0 <= x < nr and 0 <= y < nc and grid[x][y] == 0:
            dfs(grid, x, y)


def get_skin(image: np.ndarray) -> np.ndarray:
    h, w, c = image.shape
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = (ycrcb[:, :, i] for i in range(3))
    res = np.zeros(shape=cr.shape, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if 133 < cr[i, j] < 173 and 77 < cb[i, j] < 127 and y[i, j] >= 70:
                res[i, j] = 255
            else:
                res[i, j] = 0
    return res


def find_eye(image):
    dst = 255 - image
    eyes = num(dst)
    count = 0
    for eye in eyes:
        top, right, bottom, left = eye
        height = bottom - top
        width = right - left
        if 0.05 < width / image.shape[1] < 1 / 3 and width >= height:
            count += 1
    if count >= 1:
        return True
    else:
        return False


def find_face(image):
    dst = image.copy()
    sum = np.sum(dst) / 255
    h, w = image.shape
    # print(sum/(h*w))
    if sum / (h * w) > 0.4:
        return True
    else:
        return False


def num(image: np.ndarray) -> list:
    h, w = image.shape
    des = image.copy()
    component = []
    for i in range(h):
        for j in range(w):
            if des[i, j] == 255:
                top, right, bottom, left = componment_bfs(des, j, i, h, 0, 0, w)
                component.append([top, right, bottom, left])
    return component


def componment_bfs(image, x, y, t, r, b, l):
    height, weight = image.shape
    queue = [[x, y]]
    top_max = min(y, t)
    right_max = max(x, r)
    bottom_max = max(y, b)
    left_max = min(x, l)
    while len(queue) != 0:
        item = queue.pop(0)
        x = item[0]
        y = item[1]
        image[y, x] = 1
        top = max(y - 1, 0)
        top_max = min(y, top_max)
        bottom = min(y + 1, height - 1)
        bottom_max = max(y, bottom_max)
        left = max(x - 1, 0)
        left_max = min(x, left_max)
        right = min(x + 1, weight - 1)
        right_max = max(x, right_max)
        for i in range(top, bottom + 1):
            for j in range(left, right + 1):
                if image[i, j] == 255:
                    image[i, j] = 2
                    queue.append([j, i])
    return top_max, right_max, bottom_max, left_max


def indentify(des_image):
    # gray = BGR2GRAY(des_image)
    H, W, C = des_image.shape
    gray = np.zeros((H, W))
    ycrcb = bgr2ycbcr(des_image)
    # ycbcr1 = cv2.cvtColor(des_image, cv2.COLOR_BGR2YCR_CB)
    # cv2.imshow('gray', gray)
    # cv2.imshow('ycbcr', ycrcb)
    # cv2.imshow('ycbcr1', ycbcr1)

    heighth = np.size(gray, 0)
    width = np.size(gray, 1)
    for i in range(heighth):
        for j in range(width):
            Y = ycrcb[i, j, 0]
            Cb = ycrcb[i, j, 1]
            Cr = ycrcb[i, j, 2]
            R = des_image[i, j, 2]
            G = des_image[i, j, 1]
            B = des_image[i, j, 0]
            if Y < 80:
                gray[i, j] = 0
            else:
                if skin(Y, Cb, Cr, R, G, B) == 1:
                    gray[i, j] = 255
                else:
                    gray[i, j] = 0
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # gray = cv2.erode(gray, se)
    # gray = cv2.dilate(gray, se)
    cv2.imshow('gray', gray)
    # cv2.waitKey(0)
    # print("start")
    num_component = num(gray.copy())
    # print(len(num_component))
    # print(num_component)
    for index in num_component:
        top, right, bottom, left = index
        h = bottom - top
        w = right - left
        pre_face = gray[top: bottom + 1, left:right + 1]
        # if w != 0:
        #     print(h / w)
        if w != 0 and h != 0 and 0.6 <= (h / w) <= 2 and w > heighth * 0.05 and h > width * 0.05:
            # print(h, w)
            if find_eye(pre_face) and find_face(pre_face):
                # print(h, w)

                des_image = cv2.rectangle(des_image, (left, bottom), (right, top), (0, 0, 255), 2)
    return des_image


if __name__ == "__main__":
    # 读取并显示图像
    # img_path = r'./image/Orical2.jpg'
    img_path = r'./image/Orical1.jpg'
    # img_path = r'./image/download.jpg'
    img = cv2.imread(img_path)
    cv2.imshow('org_img', img)
    start_time = time()
    des_image = img.copy()
    des_image = indentify(des_image)
    end_time = time()
    print('time:', end_time - start_time)
    cv2.imshow('des_image', des_image)
    cv2.waitKey(0)
    # 关闭窗口
    cv2.destroyAllWindows()
