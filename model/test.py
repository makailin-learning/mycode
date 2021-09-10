from model.models import *
from utils.dataset import *
from model.yolo_loss import *
import argparse
from torch.autograd import Variable
from utils.utilss import Logger


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 模型测试代码
def model_test(opt):
    cfg=parseCfgFile(opt.cfg)
    # b = net_info, c = modules_list
    b,c=creat_module(cfg,3)
    x=torch.rand([1,3,opt.image_size,opt.image_size])
    model=YOLOV4(c)
    print(model(x)[0].shape)
    return model


# 日志测试代码
def log_test(opt):
    model=model_test(opt)
    log=Logger(opt.log)
    log.creat_model(model,opt.image_size)


# 损失测试
def loss_test(opt):
    model=model_test(opt)
    model=model.to(device)
    dataloader=data_test(opt,img_show=False)
    yolo_loss=YoloLoss(opt.class_scale,gr=0,is_ebr=opt.is_ebr,is_fl=opt.is_fl)

    for i,data in enumerate(dataloader):
        img,label=data
        # TODO Variable的使用场景？
        img=Variable(img.to(device))
        label=Variable(label.to(device),requires_grad=False)
        pred=model(img)
        loss=yolo_loss(pred,label,model.cfg)
        print('loss=',loss.item())
        #print(yolo_loss.metrics)


# 数据集测试代码
def data_test(opt,img_show=True):
    data=Mydata(opt.image_path, opt.label_path, opt.txt_path, opt.classes, is_train=opt.is_train,
                is_aug=opt.is_aug,is_img=opt.is_img, is_grey=opt.is_grey, is_mosaic=opt.is_mosaic,
                is_mixup=opt.is_mixup, img_size=opt.image_size)
    """
    数据加载由数据集和采样器组成
    DataLoader是PyTorch中数据读取的一个重要接口,将自定义的Dataset根据batch size大小、
    是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练
    RandomSampler等方法返回的就是DataSet中的索引位置(indices)
    sampler是一个迭代器，一次只返回一个样本索引，它的值由Randomsampler或者Sequentialsampler产生
    Batchsampler也是一个迭代器，但一次性返回batch_size个样本索引，它的值由Batchsampler产生
    索引产生后用__getitem__取得对应索引的img和label，再交给collate_fn函数进行打包处理，生成batch_size形状的张量
    """
    data_loader=DataLoader(data,batch_size=opt.batch,shuffle=True,collate_fn=data.collate_fn)
    data_loader.dataset.mosaic_close()
    if img_show:
        # for循环代码等价于: iters=iter(DataLoader)   data=next(iters)
        for i, data in enumerate(data_loader):
            imgi, label = data  # img是包含batch_size的维度，还需要继续剥离
            imgi = imgi.squeeze(0)
            imgi = imgi.permute(1, 2, 0)  # 调整通道顺序
            imgi = imgi.numpy()  # 转为numpy形式
            imgi = imgi * 255  # 去归一化
            imgi = imgi.astype(np.uint8)  # 格式转换

            # 在padding的时候归一化，并转换为xywh格式了
            # batch_id,cls_id,x,y,w,h
            h = opt.image_size
            lab=label[:,2:]*h
            label[:,2:]=lab
            size = len(label)
            classes=opt.classes.split(',')
            for i in range(size):
                point1 = (int(label[i][2]-label[i][4]/2), int(label[i][3]-label[i][5]/2))
                point2 = (int(label[i][2]+label[i][4]/2), int(label[i][3]+label[i][5]/2))
                imgi = cv2.rectangle(imgi, point1, point2, (0, 255, 255), 1)
                cls_name = classes[int(label[i][1])]

                # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                imgi = cv2.putText(imgi, cls_name, (point1[0], point1[1] - 2),
                                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,(0, 255, 0), 1)

            cv2.imshow('img', imgi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return data_loader

#optparse模块主要用来为脚本传递命令参数，采用预先定义好的选项来解析命令行参数
if __name__ == '__main__':
    model_path = 'E:/ai_project/'
    data_path = 'F:/VOC/VOC2012/'
    class_scale = '10.6787234,12.24146341,8.47804054,9.87992126,6.70093458,15.83280757,4.21410579,8.24137931,3.44474949,14.13802817,13.45576408,6.53515625,13.31299735,13.384,1.,9.01077199,9.86051081,12.57894737,15.34862385,12.18203883'
    label_class = 'aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=model_path + 'cfg_yolov5_210901_mkl_dbb.cfg',help='配置文件')
    parser.add_argument('--log', type=str, default=model_path + 'yolo_mkl/logs_test/', help='配置日志地址')
    parser.add_argument('--txt_path', type=str, default=data_path + "ImageSets/Main/", help='数据集地址')
    parser.add_argument('--image_size', type=int, default=512, help='图片尺寸')
    parser.add_argument('--image_path', type=str, default=data_path + 'JPEGImages/', help='图片地址')
    parser.add_argument('--label_path', type=str, default=data_path + 'Annotations/', help='标签地址')
    parser.add_argument('--checkpoint', type=str, default=model_path + 'checkpoints20210831/', help='存档地址')
    parser.add_argument('--checkpoint_name', type=str, default='yolov4_mkl', help='存档文件名称')
    parser.add_argument('--epoch', type=int, default=400, help='训练世代')
    parser.add_argument('--steps', type=str, default='40000,45000', help='改变学习率步数')
    parser.add_argument('--is_max_batches', action='store_true', default=False, help='是否启用配置中最大迭代次数，会覆盖epoch')
    parser.add_argument('--is_grey', action='store_true', default=False, help='是否是灰度图')
    parser.add_argument('--class_scale', type=str, default=class_scale, help='分类乘数')
    parser.add_argument('--classes', type=str, default=label_class, help='分类标签')
    parser.add_argument('--eva_fq', type=int, default=10, help='交叉验证频率')
    parser.add_argument('--block_size', type=int, default=7, help='丢弃块大小')
    parser.add_argument('--drop_prob', type=float, default=.1, help='丢弃块概率')
    parser.add_argument('--iou_thres', type=float, default=.5, help='交叉验证iou阈值')
    parser.add_argument('--conf_thres', type=float, default=.25, help='交叉验证置信度阈值')
    parser.add_argument('--nms_thres', type=float, default=.45, help='交叉验证nms阈值')
    parser.add_argument('--is_aug', action='store_true', default=True, help='是否数据增强')
    parser.add_argument('--is_img', type=str, default='saturation, hue, contrast, mirror',help='数据增强类型')
    parser.add_argument('--batch', type=int, default=1, help='批数量')
    parser.add_argument('--mini_batch', type=int, default=2, help='mini批数量')
    parser.add_argument('--is_train', action='store_true', default=True, help='是否训练模式')
    parser.add_argument('--is_mosaic', action='store_true', default=True, help='是否随机马赛克')
    parser.add_argument('--is_multi_scale', action='store_true', default=True, help='是否多尺度训练')
    parser.add_argument('--is_amp', action='store_true', default=False, help='是否混合精度训练')
    parser.add_argument('--is_ema', action='store_true', default=True, help='是否指数滑动平均训练')
    parser.add_argument('--is_ebr', action='store_true', default=True, help='是否ebr训练模型')
    parser.add_argument('--is_mixup', action='store_true', default=True, help='是否图像混合')
    parser.add_argument('--is_fl', action='store_true', default=True, help='是否focal_loss')
    parser.add_argument('--is_debug', action='store_true', default=False, help='是否调试模式')
    opt = parser.parse_args()

    #model_test(opt)
    data_test(opt)
    #loss_test(opt)
    #log_test(opt)
