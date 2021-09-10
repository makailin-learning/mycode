import sys
sys.path.append("D://github//Yolo_mkl//")  #终端运行时，需要将整个项目目录添加进来，才能导入候选的文件包
from model.models import *
from utils.dataset import *
from utils.utilss import *
import argparse

def infer(opt,model_id):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=parseCfgFile(opt.cfg)
    net_info,module_list=creat_module(cfg)
    image_size = int(net_info['width'])
    net=YOLOV4(module_list).to(device)

    # 加载原先为cuda_tensor的模型权重，在cpu上进行推理
    model=torch.load(opt.checkpoint+'yolov4_mkl_ckot_320.pth',map_location=lambda storage, loc: storage)
    net.load_state_dict(model)

    infer_dataset = Mydata(opt.image_path, opt.label_path, opt.txt_path, opt.classes,is_train=False,img_size=image_size)
    infer_loader = DataLoader(infer_dataset, batch_size=opt.batch, shuffle=False, collate_fn=infer_dataset.collate_fn)

    for i, data in enumerate(infer_loader):
        net.eval()
        imgi, label = data  # img是包含batch_size的维度，还需要继续剥离

        with torch.no_grad():
            imgi=imgi.to(device)
            label=label.to(device)
            outputs=net(imgi)
            # 输出 mx7 conf x,y,x,y,conf_class_score,cls_id
            outputs = NMS(outputs, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

        imgi = imgi.squeeze(0)
        imgi = imgi.permute(1, 2, 0).contiguous()  # 调整通道顺序,需要保证顺序存储,否则在GPU设备上跑会报错
        imgi = imgi.numpy()  # 转为numpy形式
        imgi = imgi * 255  # 去归一化
        imgi = imgi.astype(np.uint8)  # 格式转换

        outputs = np.array(outputs[0])
        outputs[:, 1:5] = outputs[:, 1:5]
        size = len(outputs)
        classes = opt.classes.split(',')
        for i in range(size):
            x1=int(outputs[i][1])
            y1=int(outputs[i][2])
            x2=int(outputs[i][3])
            y2=int(outputs[i][4])
            cls_name = classes[int(outputs[i][-1])]
            # 限制输出小数点后两位
            conf = str(round(outputs[i][0],2))

            imgi = cv2.rectangle(imgi, (x1,y1), (x2,y2),(0,255,0),1)
            # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            imgi = cv2.putText(imgi, cls_name, (x1, y1 - 2),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)
            imgi = cv2.putText(imgi, conf, (x1, y2 + 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)

        cv2.imshow('img', imgi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = 'E://ai_project//'
    data_path = 'F://VOC//VOC2012//'
    label_class = 'aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=model_path + 'cfg_yolov5_210901_mkl_rep1.cfg', help='配置文件')
    parser.add_argument('--txt_path', type=str, default=data_path + "ImageSets//Main//", help='数据集地址')
    parser.add_argument('--image_path', type=str, default=data_path + 'JPEGImages//', help='图片地址')
    parser.add_argument('--label_path', type=str, default=data_path + 'Annotations//', help='标签地址')
    parser.add_argument('--checkpoint', type=str, default=model_path + 'checkpoints20210831//', help='存档地址')
    parser.add_argument('--classes', type=str,default=label_class,help='分类标签')
    parser.add_argument('--iou_thres', type=float, default=.4, help='交叉验证iou阈值')
    parser.add_argument('--conf_thres', type=float, default=.1, help='交叉验证置信度阈值')
    parser.add_argument('--nms_thres', type=float, default=.2, help='交叉验证nms阈值')
    parser.add_argument('--is_ebr', action='store_true', default=False, help='是否ebr训练模型')
    parser.add_argument('--batch', type=int, default=1, help='批数量')
    parser.add_argument('--is_train', action='store_true', default=False, help='是否训练模式')
    opt = parser.parse_args()
    infer(opt,380)

