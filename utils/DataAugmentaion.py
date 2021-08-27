import torch
import torch.nn as nn
import cv2
import numpy as np
import random

# 存放数据增强代码
class DataAugmentation():
    def __init__(self, mean=114):
        super().__init__()
        self.mean = mean

    # 色彩空间转换
    def convertColor(self, img, current='RGB'):
        if current == 'RGB':  # RGB分别代表三个基色（R-红色、G-绿色、B-蓝色）
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if current == 'HSV':  # HSV色彩空间（Hue-色调、Saturation-饱和度、Value-值）
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)  # 色彩空间转换必须单独进行，不能在读取的时候同时进行,但选择读灰度图还是彩色图时可以
        return img

    # 饱和度变换,转变HSV颜色空间下操作图片0.5-1.5
    def randomSaturation(self, img, lower=0.5, upper=1.5):
        if random.randint(0, 1):  # 随机生成整数：[a-b]区间的整数（包含两端），即0或者1，0的话则不进行数据增强，1的话进行数据增强
            img[:, :, 1] *= random.uniform(lower, upper)  # 随机生成下一个实数，它在[lower, upper]范围内,包含两端
        return img

    # 色调变化，转变HSV颜色空间下操作图片默认18
    def randomHue(self, img, delta=10.0):
        if random.randint(0, 1):
            img[:, :, 0] += random.uniform(-delta, delta)  # H的范围是[0，360），S和V的范围是[0，1]
            img[:, :, 0][img[:, :, 0] > 360] -= 360  # 超过360的，减去360
            img[:, :, 0][img[:, :, 0] < 0] += 360  # 小于0的，加上360
        return img

    # 对比度变化，在RGB空间下操作
    def randomContrast(self, img, lower=0.5, upper=1.5):
        if random.randint(0, 1):
            alpha = random.uniform(lower, upper)
            img *= alpha
        return img

    # 亮度变化，在RGB空间下操作默认32
    def randomBrightness(self, img, delta=15.0):
        if random.randint(0, 1):
            delta = random.uniform(-delta, delta)
            img += delta
        return img

    # 改变通道,swaps例子：（2，1，0）
    def _SwapChannels(self, img, swaps):
        img = img[:, :, swaps]
        return img

    # 通道变化
    def randomLightingNoise(self, img):
        perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
        if random.randint(0, 1):
            swap = perms[random.randint(0, len(perms) - 1)]  # 从元组中随机拿一个出来
            img = self._SwapChannels(img, swap)
        return img

    # 去均值
    def subtractMeans(self, img):
        img -= self.mean
        return img

    # 图像扩充
    def expand(self, img, labels):
        h, w, c = img.shape
        # 随机初始化扩充比例
        r = random.uniform(1, 1.5)
        # 随机初始化原图左边坐标，在0～扩张宽度之间
        left = random.uniform(0, w * r - w)
        # 随机初始化原图顶部坐标，在0～扩张高度之间
        top = random.uniform(0, h * r - h)
        nw = int(w * r)
        nh = int(h * r)
        expImg = np.zeros((nh, nw, c), dtype=img.dtype)
        # 用均值填充
        expImg += self.mean
        # 把图片放到扩展图片中
        expImg[int(top):int(top + h), int(left):int(left + w)] = img
        img = expImg
        img = cv2.resize(img, (w, h))  # 将扩充图像大小，重新调整为原图像大小

        # 标签坐标为xyxy格式,且未做归一化的数值
        box = labels[:, 1:]
        box += (int(left), int(top)) * 2  # 这里的乘以2,不是数值乘以2，而是元组格式扩展2倍，形成（left,top,left,top）的形式
        rate = np.array([w, h] * 2) / np.array([nw, nh] * 2)  # 获得缩放比例  结果为[rate_w,rate_h,rate_w,rate_h]
        box *= rate  # 对box的四角坐标进行缩放
        labels[:, 1:] = box

        return img, labels

    # 图像水平翻转，上下不变，左右互换
    def randomMirror(self, img, labels):
        if random.randint(0, 1):
            return img, labels
        w = img.shape[1]  # a[i:j:1]相当于a[i:j]，缺省默认最后步距为s=1，从前往后，当s<0时，从后往前

        # a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍，即倒序
        img = img[:, ::-1, :].copy()  # 步距为-1，将所有列，从后往前，倒序罗列一遍，实现水平翻转
        box = labels[:, 1:]

        # 切片，实际为i:j:s格式，只不过一般默认s=1,可忽略，此处s=2和-2，正即从前往后，负即从后往前
        box[:, 0::2] = w - box[:, 2::-2]  # 取出翻转前[x1,x2]，翻转前的[x2,x1]，w-x2即为翻转后的x1，画图就明白了
        labels[:, 1:] = box

        return img, labels

    def bbox_iou(self, box1, box2):
        # box1  nx4    ndarray
        # box2  4      ndarray
        lt = np.maximum(box1[:, :2], box2[:2])  # nx2
        rt = np.minimum(box1[:, 2:], box2[2:])  # nx2
        wh = np.maximum((rt - lt), 0)  # nx2

        inter = wh[:, 0] * wh[:, 1]  # nx1   wxh

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # nx1
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])  # 1

        union = area1 + area2 - inter

        iou = inter / (union + 1e-9)
        return iou

    # 图片随机裁剪, 未修改 不能使用
    def randomSampleCrop(self, img, labels):
        # 设置采样模式
        sampleOption = (
            # 不采样，维持原图
            None,
            # 采样一个图像块，设置最小重叠为0.1，0.3，0.5，0.7或0.9,即最小IOU,后面的是最大IOU
            (0.1, float('inf')),
            (0.3, float('inf')),
            (0.7, float('inf')),
            (0.9, float('inf')),
            # 真正随机采样一个图像块,最小和最大不作限定，为无穷
            (float('-inf'), float('inf')),
        )
        height, width, c = img.shape

        while True:
            mode = random.choice(sampleOption)  # 从采样模式中，随机选取一种
            if mode is None:
                return img, labels
            miniou, maxiou = mode

            for i in range(50):
                current_img = img.copy()
                # 随机初始化裁剪区域的宽度和高度，在0.3-1倍之间，即可定出裁剪框的大小
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)
                # 把裁剪区域长宽比控制在1：2的范围内，使裁剪区域不会过瘦或过胖
                if h / w < 0.5 or h / w > 2:
                    continue
                # 左边距离,在原图与采样图片的宽度差距中随机选取，定出裁剪框放在原图上哪个位置，要求不可超出原图边界
                left = random.uniform(0, width - w)
                # 顶部距离,在原图与采样图片的高度差距中随机选取
                top = random.uniform(0, height - h)
                # 采样块xmin,ymin,xmax,ymax
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])  # w,h为宽度和高度，left和top为坐标
                box = labels[:, 1:]  # 所有gtbox nx4
                overlap = self.bbox_iou(box, rect)  # nx1的矩阵，代表n个iou结果值
                # np.flatnonzero将overlap由数组降维到列表，找到其中非零数值，返回对应索引
                overlap = overlap.ravel()[np.flatnonzero(overlap)]  # .ravel()将overlap降维到列表，根据非零索引，找到对应元素值

                # 满足两个条件才能继续
                if len(overlap) == 0 or overlap.min() < miniou:
                    continue

                # 剪裁后的图片赋值到当前图片变量   存放到y2:y1,x2:x1的矩形区域内，原图被覆盖
                temp_img = current_img[rect[1]:rect[3], rect[0]:rect[2], :]
                temp_img = cv2.resize(temp_img, (width, height))
                # 中心坐标（x1+x2）/2  (y1+y2)/2    x,y
                centers = (box[:, :2] + box[:, 2:]) / 2.0
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])  # 判断x1<x,y1<y的关系,若都成立，结果为真
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])  # 判断x2>x,y2>y的关系,若都成立，结果为真
                mask = m1 * m2  # 若gt框中心在裁剪框内部，结果为真，否则为假,这里有多个gt框和一个裁剪框，是一个n维向量  F,T,F,T,T,F,F

                # any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True
                if not mask.any():  # 若结果为假，说明与所有gt框无中心点符合，不符合要求，加上not，即结果为假时执行跳过操作
                    continue
                # 选出那些与裁剪框有中心点交集的gt框   mx4
                mask = np.expand_dims(mask, axis=1)  # 做乘积的话，需要保持相同形状，不可广播
                currentbox = box * mask
                # 获取裁剪框和gt框的左上角交点
                currentbox[:, :2] = np.maximum(currentbox[:, :2], rect[:2])
                # 调整为切割后的坐标
                currentbox[:, :2] -= rect[:2]  # 啥意思

                # 获取裁剪框和gt框的右下角交点
                currentbox[:, 2:] = np.minimum(currentbox[:, 2:], rect[2:])
                # 调节坐标系，让boxes的左上角坐标变为切割后的坐标
                currentbox[:, 2:] -= rect[:2]  # 啥意思

                # 裁剪的图片缩放成原图
                currentbox[:, :2] *= np.array([width, height], dtype=np.float32) / np.array([w, h],
                                                                                            dtype=np.float32)  # 获取一个？比例
                currentbox[:, 2:] *= np.array([width, height], dtype=np.float32) / np.array([w, h],
                                                                                            dtype=np.float32)  # 获取一个？比例

                labels[:, 1:] = currentbox
                labels = labels * mask  # 留下修改过的gtbox的信息，过滤掉未修改的

                # argwhere函数返回矩阵中,非零元素的索引，即留下那些四个坐标均非零，有效的gtbox，
                # labels执行等于0的操作，结果为布尔矩阵，用all函数，得到结果为真的那些行，得到布尔矩阵，再用where得到那些结果为真的行索引
                idx = np.argwhere(np.all(labels == 0, axis=1))  # all(x)函数必须x中所有元素均为真，结果才为真，有一个假，结果就为假,与any相反
                labels = np.delete(labels, idx, axis=0)  # 排查所有行，将gtbox坐标信息全为0的行，进行删除，留下符合要求的行

                return temp_img, labels
