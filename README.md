# CV-lab
> SDU软件计算机视觉实验报告

## lab1 图像加载、显示
### lab 1-1：图像显示
> **实验要求** 
>
>1.利用图像库的功能，实现从文件加载图像，并在窗口中进行显示的功能；
>
>2.利用常见的图像文件格式（.jpg； .png； .bmp； .gif）进行测试。
>
#### 实验结果：
![](https://cdn.jsdelivr.net/gh/Collapsar-G/image/img/20201029102431.png)



### lab 1-2：图像合成
> **实验要求：**
>
>1.现有一张4通道透明图像a.png:
>
>2.从其中提取出alpha通道并显示;
>
>3.用alpha混合，为a.png替换一张新的背景（bg.png）。

#### 实验结果
![](https://cdn.jsdelivr.net/gh/Collapsar-G/image/img/20201029102114.png)

## lab2 图像滤波处理
### lab2_1: 实现图像的高斯滤波处理

> **实验要求**
>
>通过调整高斯函数的标准差(sigma)来控制平滑程度；
>
>>给定函数：void Gaussian(const MyImage &input, MyImage &output, double sigma);
>
>2）滤波窗口大小取为[6*sigma-1]/2*2+1，[.]表示取整；
>
>3）利用二维高斯函数的行列可分离性进行加速；
>>* 先对每行进行一维高斯滤波，再对结果的每列进行同样的一维高斯滤波；
>>* 空间滤波=图像卷积；
>>* 高斯滤波=以高斯函数为卷积核的图像卷积。

#### 实验结果
![](https://cdn.jsdelivr.net/gh/Collapsar-G/image/img/20201030162332.png)

### lab2_2: 实现图像的联合双边滤波

#### 实验要求
![](https://cdn.jsdelivr.net/gh/Collapsar-G/image/img/20201030162655.png)

#### 实验结果
![](https://cdn.jsdelivr.net/gh/Collapsar-G/image/img/20201103204532.png)