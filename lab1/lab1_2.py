import cv2.cv2 as cv2

""""""
""""""
"lab 1-2 图像合成"
""""""
""""""
img_path_2 = r'./image/lab2/'

# 读入图像
a_png = cv2.imread(img_path_2 + "a.png", -1)
cv2.imshow("a.png", a_png)

# 将图像通道分离
b, g, r, a = cv2.split(a_png)

# 得到PNG图像的alpha通道，即alpha掩模
alpha = cv2.merge((a, a, a))
# 保存alpha图像
cv2.imwrite(img_path_2 + "alpha.jpg", alpha)
# 显示alpha通道
cv2.imshow("alpha", alpha)

# 将alpha通道数值的区间范围限制在0-1之间作为权
alpha = alpha.astype(float) / 255

# 读入背景图片
background = cv2.imread(img_path_2 + 'bg.png')
cv2.imshow("background", background)

# 将前景图片去掉alpha通道
foreground = cv2.merge((b, g, r))

# 将数值类型设置为浮点数防止溢出
foreground = foreground.astype(float)
background = background.astype(float)

# 对前景和背景利用alpha加权合成
out = cv2.multiply(foreground, alpha) + cv2.multiply(background, 1 - alpha)
# 保存合成的图像
cv2.imwrite(img_path_2 + "out.jpg", out)
# 显示合成的图形
cv2.imshow("out", out / 255)
cv2.waitKey(0)

# 关闭窗口
cv2.destroyAllWindows()
