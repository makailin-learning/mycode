# coding:utf-8
import onnx
import onnxruntime
import torch
import pickle
from torch.autograd import Variable
#from x1 import x_iou
import numpy as np
import numpy
import os
import time
import argparse

import random
from utils.iou2 import iou2
from utils.utils import *
from utils.dataset import ListDataset
from model.model import YOLOv4
import  time
import os

#推理统计onnx模型代码，方法参考count2.py

conf_thres=0.03   #置信度阈值
nms_thres=0.001
iou_thres=0.001   #iou阈值

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
T_detect=0      #预测对的个数
target_num=0     #标签瑕疵的个数
F_detect=0      #误检

max_t=0
min_t=float('inf')
max_wh=[]
min_wh=[]

#------------------------iou---------------------------
# def to_xxyy2(box1, box2):
#     #print( box1[0])
#     b1_x1, b1_x2 = box1[0], box1[ 2]
#     b1_y1, b1_y2 = box1[ 1],  box1[ 3]
#     b2_x1, b2_x2 = box2[ 0],  box2[ 2]
#     b2_y1, b2_y2 = box2[ 1] , box2[ 3]
#     return b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2
#
# # 相交面积
# def inter_box(b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y):
#     # Intersection area
#     inter = (torch.min(b1_max_x, b2_max_x) - torch.max(b1_min_x, b2_min_x)).clamp(0) * \
#     (torch.min(b1_max_y, b2_max_y) - torch.max(b1_min_y, b2_min_y)).clamp(0)
#     return inter
#
# # 相并面积
# def union_box(b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y, inter):
#     # Union Area
#     w1, h1 = b1_max_x - b1_min_x, b1_max_y - b1_min_y
#     w2, h2 = b2_max_x - b2_min_x, b1_max_y - b2_min_y
#     union = w1 * h1 + w2 * h2 - inter
#     return union
#
# # 最小包围框
# def c_box(box1, box2):
#     b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y = to_xxyy2(box1, box2)
#     c_top = torch.min(b1_min_y, b2_min_y)
#     c_bot = torch.max(b1_max_y, b2_max_y)
#     c_left = torch.min(b1_min_x, b2_min_x)
#     c_right = torch.max(b1_max_x, b2_max_x)
#
#     return c_top, c_bot, c_left, c_right
#
# def iou2(box1, box2):
#     b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y = to_xxyy2(box1, box2)
#     inter = inter_box(b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y)
#     union = union_box(b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y, inter)
#     iou = inter / (union + 0.000001)
#
#
# def giou2(box1, box2):
#     c_top, c_bot, c_left, c_right = c_box(box1, box2)
#     iou=0
#     w = c_right - c_left
#     h = c_bot - c_top
#     c = w * h
#
#     i_iou = iou2(box1, box2)
#
#     b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y = to_xxyy2(box1, box2)
#     inter = inter_box(b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y)
#     union = union_box(b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y, inter)
#
#     giou_term = (c - union) / c
#
#     if i_iou==None:
#         iou=0
#     elif i_iou is not None:
#         iou=i_iou - giou_term
#     return iou
# #------------------------------------------------------
#nms
def nms(prediction, conf_thres=0.9, nms_thres=0.8):
    # 因为输出是一个tuple，由损失和输出组成，非训练模式输出为0，所以输出预测应该prediction[1]
    # 输出形状为( batch_size, 10647, 25) 25为0-19分类预测，20置信度，21-24为盒子

    # print(prediction[0].shape,"prediction[0]")
    # print(len(prediction),"prediction")
    # print(prediction[1], "prediction[1]")
    # prediction=prediction[1]
    prediction = prediction[0]

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    # 把预测盒子由中心坐标转为四角坐标
    # prediction[..., 21:] = xywh2xyxy(prediction[..., 21:])
    prediction[..., 3:] = xywh2xyxy(prediction[..., 3:])
    # 按批次小大生成output
    output = [None for _ in range(len(prediction))]

    # 循环处理预测值image_i图片序号，image_pred单图片预测shape (10647, 25)
    for image_i, image_pred in enumerate(prediction):

        # 去掉了小于置信度的预测 shape (pred>=conf_thres,)
        ##image_pred = image_pred[image_pred[:,20]>=conf_thres]
        image_pred = image_pred[image_pred[:, -5] >= conf_thres]

        # 如果没有大于阈值的置信度，则跳过图片，说明图片没有预测到任何值
        if not image_pred.size(0):
            continue

        # 置信度×分类预测概率作为分数,max返回的不是tensor是value,index，所以要用[0]提取value
        ##score = image_pred[:, 20] * image_pred[:, :20].max(1)[0]
        score = image_pred[:, -5] * image_pred[:, :-5].max(1)[0]
        # .argsort()是对数组从小到大，(-score)相当于从大到小排序，并返回排序的索引，即把预测结果从大到小排序
        image_pred = image_pred[(-score).argsort()]
        ##ccc=image_pred[:, :20].max(1, keepdim=True)
        ccc = image_pred[:, :-5].max(1, keepdim=True)

        # class_confs是20个分类中最大的分类值，class_preds是20个分类中最大值的索引
        ##class_confs, class_preds=image_pred[:, :20].max(1, keepdim=True)
        class_confs, class_preds = image_pred[:, :2].max(1, keepdim=True)
        # 连接结果 shape (pred>=conf_thres, 7) 7为(object_conf, x1, y1, x2, y2, class_score, class_pred)
        ##detections = torch.cat((image_pred[:, 20:], class_confs.float(), class_preds.float()), 1)
        detections = torch.cat((image_pred[:, 2:], class_confs.float(), class_preds.float()), 1)
        keep_boxes = []
        while detections.size(0):
            # keep_boxes += [d]
            # 计算最大交并比，第一个是score最大的，unsqueeze增加一个维度,第一个与其余预测盒子交并比大于nms的阈值是无效的
            large_overlap = diou(detections[:1, 1:5], detections[:, 1:5]) > nms_thres

            # 第一个预测值分类与其他分类预测相等的记1否则为0
            label_match = detections[0, -1] == detections[:, -1]

            # 把交并比大于nms阈值的维度与分类相同的维度相与，得出无效的维度索引
            invalid = large_overlap & label_match
            # detection中要无效的维度索引到的置信度
            weights = detections[invalid, :1]
            # 无效的置信度乘以无效的边框，在0维相加，再除以无效的置信度之和,得到所有无效的预测的平均值
            detections[0, 1:5] = (weights * detections[invalid, 1:5]).sum(0) / weights.sum()
            # 把改预测放到保持的盒子里
            keep_boxes += [detections[0]]
            # ～是取反符号，无效索引取反就是有效索引，相当于去掉了无效值，重新赋值detection，然后再循环去重
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output

def detect(opt):
    # 设置 debug
    debug = False

    # 读取配置文件
    cfgs = parseCfgFile(opt.cfg)

    # 读取net部分配置
    net_cfg = cfgs[0]

    # 图片大小
    image_size = net_cfg['width']
    # 图片通道数
    channels = net_cfg['channels']

    # gpu设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    # 测试结构用cpu
    if debug:
        device = torch.device("cpu")

    # 初始化网络
    # net = YOLOv4(cfg=cfgs[1:], channels=channels).to(device)
    # # 加载网络参数
    # model = torch.load(opt.checkpoint)
    # net.load_state_dict(model)
    # net.eval()
    # net.fuse(is_rep=True)
    # net.half()

    onnx.checker.check_model(onnx.load(opt.onnx_save_path))
    sess = onnxruntime.InferenceSession(opt.onnx_save_path)
    #sess=sess.set_providers(['CUDAExecutionProvider'], [ {'device_id': 0}])
    out_name = ["output"]

    #print("1111111")
    # 预测图片地址
    detect_path = opt.detect

    # 标签名称和色彩处理
    label_name = opt.label_name.split(",")

    dataset = ListDataset(detect_path, opt.img, opt.label, opt.is_grey, label_name, img_size=image_size)
    # 数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )


    def result(dataloader,conf_thres,nms_thres,iou_thres):
        T_detect = 0
        F_detect = 0
        target_num = 0
        for batch_i, (img_paths, input_imgs,bb_targets) in enumerate(dataloader):

            # global T_detect
            # global F_detect
            # global target_num

            # 图片输入变量

            # input_imgs = Variable(input_imgs.to(device),requires_grad=False)
            # bb_targets = Variable(bb_targets.to(device), requires_grad=False)
            #input_imgs = Variable(input_imgs, requires_grad=False)
            #bb_targets = Variable(bb_targets, requires_grad=False)

            #input_imgs = input_imgs.half()
            with torch.no_grad():

                # 图片进网络

                #detections = net(input_imgs)
                detections = sess.run(out_name, {"input": np.array((input_imgs).cpu())})
                detections = torch.tensor(detections).half()

                #detections = non_max_suppression_v5(detections,conf_thres=conf_thres,nms_thres=nms_thres)
                detections=non_max_suppression_v5(detections, conf_thres=conf_thres, nms_thres=nms_thres)
                #print("nmsend")
                bb_targets=bb_targets
                detections=detections[0]

                targets=bb_targets[...,2:6]*1024

                #统计标签瑕疵总个数
                target_num=target_num+len(targets[0])
                T_num = 0

                if detections==None or len(detections)==0:
                    pass
                elif len(detections) >0:
                    for i in range(len(targets)):
                        targets=targets.squeeze(0)
                        x= targets[i][0]
                        y = targets[i][1]
                        w=targets[i][2]
                        h=targets[i][3]
                        targets[i][0] = x-w
                        targets[i][1] = y-h
                        targets[i][2]=x+w
                        targets[i][3] = y+h

                        num=len(detections)
                        iou_p = []         #每个标签的所有预测匹配的iou集合
                        # print("********************************")
                        # print("图片标签个数",len(targets))
                        # print("预测框个数：",num)
                        for j in range(num):
                            pre =detections[j][1:5]
                            iou = iou2(targets[i], pre)
                            iou_p.append(iou)
                        iou_max=max(iou_p)
                        #print(iou_max,"iou_max")

                        if iou_max>iou_thres:
                                T_detect=T_detect+1
                                T_num=T_num+1

                    F_detect = F_detect +(num-T_num)
        #             print("正检个数：",T_num)
        #             print("误检个数：", num - T_num)
        #
        #
        # print(" ")
        # print("预测对总个数：",T_detect)
        # print("误检总个数", F_detect)
        # print("标签总个数：",target_num)
        # print("准确率：","{}%".format((T_detect/target_num)*100))
        # print("误检率：","{}%".format((F_detect/target_num)*100))
        return T_detect,F_detect,target_num


    def result2(dataloader,conf_thres,nms_thres,iou_thres):

        for batch_i, (img_paths, input_imgs,bb_targets) in enumerate(dataloader):

            global T_detect
            global F_detect
            global target_num

            # 图片输入变量

            input_imgs = Variable(input_imgs,requires_grad=False)
            bb_targets = Variable(bb_targets, requires_grad=False)

            #input_imgs = input_imgs.half()
            with torch.no_grad():

                # 图片进网络
                t1=time.time()
                #detections = net(input_imgs)
                detections = sess.run(out_name, {"input": np.array((input_imgs).cpu())})
                detections=torch.tensor(detections)
                #print(detections, "detections11111")

                #detections = nms(detections,conf_thres=conf_thres,nms_thres=nms_thres)
                detections=non_max_suppression_v5(detections, conf_thres=conf_thres, nms_thres=nms_thres)
                t2 = time.time()
                print("时间：",t2-t1)
                #print("nmsend")
                bb_targets=bb_targets
                detections=detections[0]

                targets=bb_targets[...,2:6]*1024
                # print(targets,"targets")
                # print(detections,"detections")
                #统计标签瑕疵总个数
                target_num=target_num+len(targets[0])
                T_num = 0

                if detections==None or len(detections)==0:
                    pass
                elif len(detections) >0:
                    for i in range(len(targets)):
                        targets=targets.squeeze(0)
                        x= targets[i][0]
                        y = targets[i][1]
                        w=targets[i][2]
                        h=targets[i][3]
                        targets[i][0] = x-w
                        targets[i][1] = y-h
                        targets[i][2]=x+w
                        targets[i][3] = y+h

                        num=len(detections)
                        iou_p = []         #每个标签的所有预测匹配的iou集合
                        print("********************************")
                        print("图片标签个数",len(targets))
                        print("预测框个数：",num)
                        for j in range(num):
                            pre =detections[j][1:5]
                            iou = iou2(targets[i], pre)
                            iou_p.append(iou)
                        iou_max=max(iou_p)
                        print(iou_max,"iou_max")
                        #
                        if iou_max>iou_thres:
                                T_detect=T_detect+1
                        #         T_num=T_num+1
                        for p_iou in iou_p:
                            if p_iou>iou_thres:
                                T_num = T_num + 1


                    F_detect = F_detect +(num-T_num)
                    print("正检个数：",T_num)
                    print("误检个数：", num - T_num)


        print(" ")
        print("预测对总个数：",T_detect)
        print("误检总个数", F_detect)
        print("标签总个数：",target_num)
        print("准确率：","{}%".format((T_detect/target_num)*100))
        print("误检率：","{}%".format((F_detect/target_num)*100))
        return T_detect,F_detect,target_num

    cni=[]
    nn=1

    # for conf_thres in np.arange(0.01, 0.02, 0.01):
    #     for nms_thres in np.arange(0.0,0.1,0.01):   #误检增加
    #         for iou_thres in np.arange(0, 0.001, 0.001):#正检减少
    #             #print(f"{nn}/{9*8*15}")
    #             print(conf_thres,nms_thres,iou_thres)
    #             T_detect,F_detect,target_num = result(dataloader, conf_thres, nms_thres, iou_thres)
    #             print("T_detect",T_detect)
    #             print("F_detect", F_detect)
    #             if T_detect>40 and F_detect <10:
    #                 cni.append([conf_thres,nms_thres,iou_thres,T_detect,F_detect])
    #                 print("保存")
    #             else:
    #                 print("未满足条件")
    #             nn+=1
    # text_txt = 'count.txt'
    # with open(text_txt, 'wb') as text:
    #     pickle.dump(cni, text,0)


    T_detect, F_detect, target_num = result2(dataloader, conf_thres, nms_thres,iou_thres)


if __name__ == '__main__':
    project_path = 'F:/'  # environ['HOME'] + '/work/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default= r'F:/YOLOv4_project_ylc20210526/cfg/yolov5-s-s-up-v4-pro-eightrep-csp1_07171.cfg',
                        help='配置文件')
    parser.add_argument('--detect', type=str,
                        default=project_path + "YOLOv4_project_ylc20210526/custom_data/ylc_20210526/VOC2007_026/ImageSets/Main/val.txt",
                        help='预测数据集')
    parser.add_argument('--img', type=str,
                        default=project_path + 'YOLOv4_project_ylc20210526/custom_data/ylc_20210526/VOC2007_026/JPEGImages/',
                        help='图片地址')
    parser.add_argument('--label', type=str,
                        default=project_path + 'YOLOv4_project_ylc20210526/custom_data/ylc_20210526/VOC2007_026/Annotations/',
                        help='标签地址')
    parser.add_argument('--onnx_save_path', type=str, default='C:/Users/ylcServer/Desktop/0717_v813/yolov4_0717_v0813_2.onnx',
                        help='标签地址')
    # parser.add_argument('--checkpoint', type=str,
    #                     default=r'C:/Users/ylcServer/Desktop/s/YOLOv4_project_ylc20210526_ckpt_800.pth',
    #                     help='存档地址')
    parser.add_argument('--is_grey', action='store_true', default=False, help='是否是灰度图')
    parser.add_argument('--label_name', type=str, default='白点,黑点', help='分类标签')
    opt = parser.parse_args()
    detect(opt)
