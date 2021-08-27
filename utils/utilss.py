import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.iou import xywh2xyxy,iou,diou
from torchvision.ops import nms


# 转成CPU计算
def to_cpu(tensor):
    return tensor.detach().cpu()

# tensorboard日志生成函数
class Logger(object):
    def __init__(self,logdir,log_hist=True):
        if log_hist:
            # 将当前的datetime转换为字符串str,将多个路径组合后返回save_path:  logdir\"%Y_%m_%d_%H_%M_%S"
            logdir=os.path.join(logdir,datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            # 创建接口(日志对象), logdir=事件输出文件夹地址，comment=不指定logdir时文件夹后缀
            self.writer=SummaryWriter(logdir,comment="YOLO")
            self.writer.close()

    def scalars_summary(self,tag,value,step):
        # 单个记录标量add_scalars() 数据名称，标量数值，训练轮次
        self.writer.add_scalar(tag,value,step)
        self.writer.close()

    def list_of_scalars_summary(self,tag_value_list,step):
        # 列表记录数据
        for tag,value in tag_value_list:
            self.writer.add_scalar(tag,value,step)
        self.writer.close()

    def creat_model(self,model,image_size):
        # 查看模型图 model必须是nn.module类型的模型 verbose:是否打印计算图结构信息
        inputs=torch.randn(1,3,image_size,image_size)
        self.writer.add_graph(model,input_to_model=inputs,verbose=False)
        self.writer.close()

# 权重初始化函数
def weights_init(m):
    """
    classname=m.__class__.__name__
    # Python find() 方法检测字符串中是否包含子字符串 str，函数找不到时返回为-1,找到返回字符所在位置索引
    if classname.find("Conv2d")!=-1:
        torch.nn.init.normal_(m.weight.data,0.0,0.02)
    """
    if isinstance(m, nn.Linear):  # 是否为线性层
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):  # 是否为卷积层
        # kaiming正态分布,张量中的值采样自U(-bound, bound)
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        # 正态分布初始化  torch.nn.init.normal_(tensor, mean=0, std=1)
        torch.nn.init.uniform_(m.weight, 0.02, 1)
        nn.init.constant_(m.bias, 0.0)  # 常数初始化

# NMS极大值抑制函数
def NMS(pred,conf_thres=0.25,nms_thres=0.45,max_det=300):
    if len(pred)==1:
        pred=pred[0]
    else:
        pred=torch.cat(pred,dim=1)  # 64xnx25

    #类别数 20
    nc=pred.shape[2]-5
    #置信度 64x10647  第一次筛选，提取过置信度阈值的候选框
    xc=pred[:,:,0]>conf_thres

    max_nms=100 #输入到torchvision.ops.nms()中去的候选框个数
    # conf,x1,y1,x2,y2,ccs,cls_id  [tensor[0,7],tensor[0,7]....] 创建一个list,存放结果 len(output)=64
    output = [torch.zeros((0, 7), device=pred.device)] * pred.shape[0]

    for xi,x in enumerate(pred):  #image_index,image_inference
        # xi=0-64 x:10647x25  xc[xi]第xi张图片10647个框的bool向量
        x=x[xc[xi]]   # 选取那些置信度超过阈值的候选框 [10647,25] -> [m,25]

        # 若该张图片没有合适的后续框，预测失败，进入下一张图片
        if not x.shape[0]:
            continue

        x[:, 5:]*=x[:, 0:1]  # conf = obj_conf * cls_conf  shape: [m,25] 计算conf_class_score后的
        box=xywh2xyxy(x[:,1:5]) # Box (center x, center y, width, height) to (x1, y1, x2, y2)  [m,4]

        #keepdim用于保持结果的形状同输入x一样为[m,1]的矩阵形式,否则为[m]维向量形式, conf_cls_score为value, cls_id为index(类别号)
        conf_cls_score,cls_id=x[:, 5:].max(dim=1, keepdim=True)
        # x: [m,7]  conf,x,y,x,y,conf_cls_socre,cls_id -> [n,7](n<m) 第二次筛选,提取置信度与分类的乘积得分过置信度阈值的候选框
        x=torch.cat((x[:,0:1],box, conf_cls_score, cls_id.float()), 1)[conf_cls_score.view(-1) > conf_thres]

        # 获取最终合格的box数量
        n=x.shape[0]
        if not n:
            continue
        # 筛选出的候选box数量超过规定数量，则进行排序裁减至指定数量
        elif n>max_nms:
            # 对ccs降序排列得到索引，再从原来x的对应提取行向量，重新定义x，并截取前max_nms行 [n,7] -> [max_nms,7]
            x=x[x[:,5].argsort(descending=True)][:max_nms]
        boxes,scores=x[:,1:5],x[:,5]
        # 返回的是最终合格box的索引向量[max_nms]->[i]
        i=nms(boxes,scores,nms_thres)

        # 限制最大检测结果box的个数
        if i.shape[0]>max_det:
            i=i[:max_det]

        output[xi]=x[i]   # [max_nms,7] -> [i,7] 存放到list中

    return output

"""
滑动平均(exponential moving average)，或指数移动平均，是一种  “给予近期数据更高权重”  的平均方法。
提升过去一段区间内的均值权重，降低当前取值的权重，能有效防止因当前值发生抖动，而影响模型的收敛趋势，提高鲁棒性
可以用来估计变量的局部均值，使得变量的更新与一段时间内的历史取值有关。滑动平均可以看作是变量的过去一段时间取值的均值，
“相比对变量直接赋值”  而言，滑动平均得到的值在图像上更加平缓光滑，抖动性更小，不会因为某次的异常取值而使得滑动平均值波动很大
decay 代表衰减率，该衰减率用于控制模型更新的速度

训练过程中，使用原有weight进行参数更新，同时产生一个shadow weight，在推理时替换原有weight来使用
在梯度下降的过程中，会一直维护着这个影子权重，但是这个影子权重并不会参与训练
模型权重在最后的n步内，会在实际的最优点处抖动，所以我们取最后n步的平均，能使得模型更加的鲁棒
因滑动平均从初始值0开始，开始部分误差较大，需引入bias来进行修正
"""
class EMA():
    def __init__(self,model,decay,bias_correction=False,num_updates=1):
        self.model=model
        self.decay=decay
        self.shadow={}
        self.backup={}
        self.num_updates=num_updates
        self.bias_correction=bias_correction

    def register(self):
        for name,param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name]=param.data.clone()  # 注册模型权重的克隆: 影子权重

    #影子变量的初始值与训练变量的初始值相同。当运行变量随机梯度更新时，每个影子变量都会按照公式策略去更新
    def update(self):
        debias_term=(1.0-self.decay**self.num_updates) # 对应公式: 1减去beta的t次方
        self.num_updates+=1
        for name,param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow  # 检查该变量是否在影子变量中，是否为需要维护的对象
                if self.bias_correction:
                    # 对应公式: vt / 1减去beta的t次方
                    new_average=((1.0-self.decay)*param.data+self.decay*self.shadow[name])/debias_term
                    self.shadow[name] = new_average.clone()  # 更新影子权重
                else:
                    new_average=(1.0-self.decay)*param.data+self.decay*self.shadow[name]
                    self.shadow[name]=new_average.clone()

    def apply_shadow(self):
        for name,param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name]=param.data
                param.data=self.shadow[name]   # 用影子权重覆盖原模型权重

    def restore(self):
        for name,param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data=self.backup[name]   # 恢复原模型权重
        self.backup={}

