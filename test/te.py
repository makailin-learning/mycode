# 打印坐标图
import torch
import torch.nn.functional as func
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 定义数据x
# 将[-5,5]区间等份分割为200个小区间
x = torch.linspace(-5, 5, 200)
x = Variable(x)
# torch转numpy数据格式
np_x = x.data.numpy()

# 通过激活函数处理x
y_relu = func.relu(x).data.numpy()
y_sigmoid = func.sigmoid(x).data.numpy()
y_sigmoid_0 = func.sigmoid(x).data.numpy()*2-0.5
y_sigmoid_1 = func.sigmoid(x).data.numpy()*4-1.5
y_tanh = func.tanh(x).data.numpy()
y_softmax = func.softplus(x).data.numpy()

# 绘制激活函数图
plt.figure(1, figsize = (8, 6))   #创建图表编号:1 ，图标尺寸宽和高:8x6
plt.subplot(221)    # plt.subplot（ijn）形式，其中ij(2,2)是行列数，n(1)是第几个图，比如（221）则是一个有四个图，该图位于第一个
plt.plot(np_x, y_relu, c = 'red', label = 'relu')  # x轴数据，y轴数据，format_string控制曲线的格式字串,c颜色，label曲线名称
plt.ylim((-1, 5)) # 通过xlim，ylim方法设置坐标轴范围，上限5，下限-1
plt.legend(loc = 'best')  # 用于给图像加图例，loc:设置图列位置
#plt.scatter()函数用于生成一个scatter散点图

plt.figure(1, figsize = (8, 6))
plt.subplot(222)
plt.plot(np_x, y_sigmoid, c = 'red', label = 'sigmoid')
plt.ylim((0, 1))
plt.legend(loc = 'best')

plt.figure(1, figsize = (8, 6))
plt.subplot(223)
plt.plot(np_x, y_sigmoid_0, c = 'red', label = 'sigmoid_0')
plt.ylim((-1, 2))
plt.legend(loc = 'best')
"""
plt.figure(1, figsize = (8, 6))
plt.subplot(223)
plt.plot(np_x, y_tanh, c = 'red', label = 'tanh')
plt.ylim((-1, 1))
plt.legend(loc = 'best')

plt.figure(1, figsize = (8, 6))
plt.subplot(224)
plt.plot(np_x, y_softmax, c = 'red', label = 'softmax')
plt.ylim((-1, 5))
plt.legend(loc = 'best')
"""
plt.figure(1, figsize = (8, 6))
plt.subplot(224)
plt.plot(np_x, y_sigmoid_1, c = 'red', label = 'sigmoid_1')
plt.ylim((-2, 4))
plt.legend(loc = 'best')

plt.show()