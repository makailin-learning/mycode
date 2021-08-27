import torch
import torch.nn as nn
import numpy as np

# 将detect层输出的Bx75xHxW的张量进行处理，再返回
class YOLOLayer(nn.Module):
    def __init__(self, anchors, classes, img_size, cfg):
        super().__init__()  # 别忘了写super，因为要继承父类的一些默认函数
        self.anchors = anchors  # 一个二维数组：[[142, 110], [192, 243], [459, 401]]
        self.num_cls = int(classes)
        self.img_size = img_size
        self.num_anchor = len(anchors)
        self.cfg = cfg

    def forward(self, x):  # yolo上一层输出x进行输入，第1个为32x75x52x52，第2个为32x75x26x26，第3个为32x75x13x13

        # 消除网格敏感系数
        # 同样到得到0.9的结果，通过乘以大于1的因子后，原本的tx可以适当缩小，降低tx的预测难度
        scale_x_y = float(self.cfg['scale_x_y'])

        # 定义御用张量
        float_tensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        batch_size = x.size(0)
        grid_size = x.size(2)
        self.cfg['grid_size'] = grid_size  # 反向添加网格信息，传参进来的字典内容会被修改，原字典内容也会一并改变
        self.stride = self.img_size // grid_size  # 416//13=32
        """
        锚框缩放到特征图大小,从 [(142, 110), (192, 243), (459, 401)] :
        [[ 4.4375,  3.4375],
        [ 6.0000,  7.5938],
        [14.3438, 12.5312]]
        """
        self.scale_anchors = float_tensor([(a[0] / self.stride, a[1] / self.stride) for a in self.anchors])

        # 调整数据通道格式64x75x52x52  -> 64x3x25x52x52  -> 64x3x52x52x25
        pred = x.view(batch_size, self.num_anchor, self.num_cls + 5, grid_size, grid_size)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()

        # 推理阶段，进行处理后，按照shape(64, 507, 25)形式返回, 训练阶段，直接返回网络预测结果给后续loss函数处理
        if not self.training:
            pred = torch.sigmoid(pred)
            # 获取各个部分的信息 conf,x,y,w,h,cls
            pred_conf = pred[..., 0:1]  #64x3x52x52x1
            pred_x = pred[..., 1] * scale_x_y - (scale_x_y - 1) / 2  #64x3x52x52
            pred_y = pred[..., 2] * scale_x_y - (scale_x_y - 1) / 2
            pred_w = pred[..., 3]  # v4作者改进版本
            pred_h = pred[..., 4]
            pred_cls = pred[..., 5:]

            # torch.repeat进行张量复制，第一个参数表示的是行复制的次数，第二个参数表示列复制的次数,变为52x52的坐标张量
            grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(float_tensor)
            grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(float_tensor)
            """
            grid = np.arange(grid_size)
            c = np.meshgrid(grid, grid)  # 生成gridxgrid的网格坐标
            c = float_tensor(c)
            cx,cy = c
            cx = cx.view([1, 1, grid_size, grid_size])
            cy = cy.view([1, 1, grid_size, grid_size])
            """
            # 获取锚框的w,h, 一定要留意这样的写法，这样写会保持2维张量格式，若写成[:,1]，则会产生一个向量格式
            pw = self.scale_anchors[:, 0:1].view(1, self.num_anchor, 1, 1)  # 3x1 -> 1x3x1x1
            ph = self.scale_anchors[:, 1:2].view(1, self.num_anchor, 1, 1)

            # 创造一个用于后续计算loss的容器 64,3,52,52,4
            pred_loss = float_tensor(pred[..., :4].shape)

            # bx=cx+offset   bw=pw*e^tw
            pred_loss[..., 0] = pred_x + grid_x  #广播机制相加
            pred_loss[..., 1] = pred_y + grid_y
            pred_loss[...,2]=4*pred_w*pred_w*pw  #用4x平方，代替指数运算
            pred_loss[...,3]=4*pred_h*pred_h*ph

            # 13*13 shape(64, 507, 25)
            # 拼接成64x507x25的形状   conf,x,y,w,h,cls0,cls1...cls20
            output = torch.cat((pred_conf.view(batch_size, -1, 1),
                                pred_loss.view(batch_size, -1, 4) * self.stride,  # 还原到原图下的尺度
                                pred_cls.view(batch_size, -1, self.num_cls)), -1)

        return pred if self.training else output

