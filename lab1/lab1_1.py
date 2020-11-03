import cv2
import pyglet

""""""
""""""
"lab 1-1 图像显示"
""""""
""""""

img_path_1 = r'./image/lab1/'
img_name_1 = ['Img1.png', 'Img2.jpg', 'Img3.bmp']
img = []
for i in range(0, 3):
    image = cv2.imread(img_path_1 + img_name_1[i], 1)
    img.insert(i, image)
    cv2.imshow(img_name_1[i], img[i])
    cv2.waitKey(0)

# 由于图片所在的目录不在当前目录，因此我们需要告诉Pyglet去哪里找到它们：
pyglet.resource.path = ['./image/lab1/']
pyglet.resource.reindex()

# 选择一个gif文件路径
gif = pyglet.resource.animation('Img4.gif')
sprite = pyglet.sprite.Sprite(gif)

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

# 关闭窗口
cv2.destroyAllWindows()
