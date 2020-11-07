import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import pyglet
from PIL import Image

""""""
""""""
"lab 1-1 图像显示"
""""""
""""""

img_path_1 = r'./image/lab1/'
img_name_1 = ['Img1.png', 'Img2.jpg', 'Img3.bmp', 'Img4.gif']
img = []
for i in range(0, 4):
    # image = cv2.imread(img_path_1 + img_name_1[i], 1)
    #
    # im_file = open(img_path_1 + img_name_1[i])
    # im_obj = imread(im_file)
    image = Image.open(img_path_1 + img_name_1[i]).convert('RGB')
    # cv2.imshow(img_name_1[i], img[i])
    img.insert(i, image)
    plt.subplot(2, 2, i + 1)
    plt.imshow(img[i])
    plt.title(img_name_1[i], fontsize=8)
    plt.xticks([])
    plt.yticks([])
plt.show()

# 由于图片所在的目录不在当前目录，因此我们需要告诉Pyglet去哪里找到它们：
pyglet.resource.path = ['./image/lab1/']
pyglet.resource.reindex()

# 选择一个gif文件路径
gif = pyglet.resource.animation('Img4.gif')
sprite = pyglet.sprite.Sprite(gif)
# for i in range(sprite.)

# 创建窗口并将其大小调整为图像大小
win = pyglet.window.Window(width=sprite.width, height=sprite.height, caption='Img4.gif')

# 设置窗口背景颜色 = r,g,b,alpha
# 每个值从0.0到1.0
value = 0, 1, 0, 1
pyglet.gl.glClearColor(*value)


@win.event
def on_draw():
    win.clear()
    sprite.draw()


pyglet.app.run()
cv2.waitKey(0)
# 关闭窗口
cv2.destroyAllWindows()
