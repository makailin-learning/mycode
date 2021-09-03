import sys
sys.path.append("D://github//Yolo_mkl//")  #终端运行时，需要将整个项目目录添加进来，才能导入候选的文件包
from model.models import *
from utils.dataset import *
from utils.utilss import *
from model.yolo_loss import YoloLoss
from torch import optim
import argparse
from math import cos,sin,pi,ceil
import time
import random
from terminaltables import AsciiTable
from utils.iou import *
import datetime

def train(opt):
    """
    是否混合精度训练,混合精度预示着有不止一种精度的Tensor:  torch.FloatTensor和torch.HalfTensor
    自动预示着Tensor的dtype类型会自动变化，也就是框架按需自动调整tensor的dtype
    针对不同的层，采用不同的数据精度进行计算，从而实现节省显存和加快速度的目的
    这个功能只能在cuda上使用
    autocast主要用作上下文管理器或者装饰器，来确定使用混合精度的范围
    GradScalar主要用来完成梯度缩放
    """
    if opt.is_amp:
        from torch.cuda.amp import autocast,GradScaler
        scaler=GradScaler()

    debug=opt.is_debug
    epochs=opt.epoch
    cfg_file=opt.cfg
    cfgs=parseCfgFile(cfg_file)
    image_size = int(cfgs[0]['width'])
    channels = int(cfgs[0]['channels'])
    net_info, modules_list = creat_module(cfgs, channels, opt.drop_prob, opt.block_size)

    #读取net部分配置信息
    momentum = float(net_info['momentum'])
    decay = float(net_info['decay'])

    batch=opt.batch
    mini_batch=opt.mini_batch
    batch_update=batch/mini_batch

    # 创建日志对象，存放日志
    log_dir=opt.log
    logger=Logger(log_dir)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if debug:
        device=torch.device("cpu")
        batch = 2
        mini_batch = 2

    # 自动回收我们不用的显存，类似于python的引用机制
    torch.cuda.empty_cache()

    # 网络搭建
    net=YOLOV4(modules_list=modules_list).to(device)
    net.apply(weights_init)

    # 训练数据集
    train_dataset = Mydata(opt.image_path, opt.label_path, opt.txt_path, opt.classes,
                           is_train=opt.is_train, is_aug=opt.is_aug,is_img=opt.is_img,
                           is_grey=opt.is_grey, is_mosaic=opt.is_mosaic,
                           is_mixup=opt.is_mixup,img_size=image_size)
    train_loader=DataLoader(train_dataset,batch_size=mini_batch,shuffle=True,collate_fn=train_dataset.collate_fn)

    val_dataset = Mydata(opt.image_path, opt.label_path, opt.txt_path, opt.classes,
                           is_train=False, is_mixup=opt.is_mixup, img_size=image_size)
    val_loader = DataLoader(val_dataset, batch_size=mini_batch, shuffle=True, collate_fn=val_dataset.collate_fn)

    # 数据长度
    size=len(train_loader)
    # 学习率及调整策略
    learning_rate = float(net_info['learning_rate'])
    learning_rate_min = 0.00001
    burn_in = int(net_info['burn_in'])
    # 改变学习率步数
    steps = opt.steps.split(',')
    # 最大迭代次数
    if opt.is_max_batches:
        max_batches = int(net_info['max_batches'])
        steps = net_info['steps'].split(',')
        epochs=max_batches//size + 1
    # 学习率计划
    def burnin_schedule(i):
        if i<burn_in:
            factor=learning_rate*pow(i/burn_in,4)
        else:
            et=i//size
            es=(i%size)/size
            et+=es
            factor=learning_rate_min+0.5*(learning_rate-learning_rate_min)*(1+cos((et/epochs)*pi))

        return factor
    # 滑动平均训练模式
    if opt.is_ema:
        ema_max = 0.999
        ema_min = 0.001
        ema = EMA(net, 0.001)
        # 注册影子权重
        ema.register()

        def ema_schedule(i):
            et = i // size
            es = (i % size) / size
            et += es
            factor = ema_min + (ema_max - ema_min) * pow(0.5 * (1 + sin((et / epochs) * pi - pi / 2)), 4)
            return factor
    tempEMADecay = 0.001

    # 优化器定义
    train_optimizer='adam'
    if train_optimizer=='adam':
        optimizer=optim.Adam(net.parameters(),lr=burnin_schedule(1),betas=(0.9,0.999),eps=1e-8)
    elif train_optimizer=='sgd':
        optimizer=optim.SGD(net.parameters(),lr=burnin_schedule(1),momentum=momentum,weight_decay=decay)
    templr=burnin_schedule(1)

    yolo_loss=YoloLoss(class_scale=opt.class_scale,gr=0,is_ebr=opt.is_ebr,is_fl=opt.is_fl,is_pa=opt.is_pa)

    # 评估参数
    metrics=["grid_size","loss","lbox","lobj","lcls","iou","conf"]
    log_str=''

    # 按世代训练 400
    for epoch in range(epochs):
        net.train()
        start_time=time.time()
        # 训练显示信息
        pbar='Epoch'+str(epoch+1)+'/'+str(epochs)+'['
        pbar_end=''
        mean_loss=[]

        # 按批次循环 每次取出1组mini_batch的数据,作为一个batch_i
        # 当取得batch_update组mini_batch数据时,进行梯度传播,参数更新
        for batch_i, (imgs,targets) in enumerate(train_loader):
            # 训练步数
            step=len(train_loader)*epoch+batch_i
            imgs=imgs.to(device)
            targets=targets.to(device)

            # 多尺度训练
            if opt.is_multi_scale:
                # 图像尺寸基数，需是32的倍数
                gs=32
                # 随机计算image_size 0.5-1.5倍之间的随机数，以32为间隔
                sz=random.randrange(image_size*0.5,image_size*1.5+gs,gs)
                # 与原图尺寸比例
                sf=sz/max(imgs.shape[2:])
                if sf!=1:
                    # 将原图上采样到指定尺寸
                    imgs=F.interpolate(imgs,size=[sz,sz],mode='bilinear',align_corners=False)

            # 多精度训练
            if opt.is_amp:
                with autocast():
                    outputs=net(imgs)
                    loss=yolo_loss(x=outputs,y=targets,c=net.cfg)
                scaler.scale(loss).backward()
            else:
                # outputs的尺寸一直跟随这多尺度训练在变化，而不是一直的52，26，13
                outputs=net(imgs)
                loss = yolo_loss(x=outputs, y=targets, c=net.cfg)
                # 每一步会计算损失，然后反向传播计算梯度，存放到对应变量里的梯度参数grad中，累计4步后再进行梯度参数更新，然后归零
                # 这一步是计算损失，然后反向传播计算出梯度，存储在参数的grad变量中，不会执行参数更新
                loss.backward()

            # 优化器及学习率计划
            # step: 即取得了step组mini_batch的数据了,当其等于更新次数4时或1个世代结束时，执行更新
            if (step+1)%batch_update==0 or (step+1)==size*epochs:
                # 更新学习率
                templr=burnin_schedule(step+1)
                for param_group in optimizer.param_groups:
                    param_group['lr']=templr

                if opt.is_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 根据每个参数中的grad梯度值，执行参数更新
                    optimizer.step()
                # 将参数内的梯度清零grad=0，进入下一个累积循环
                # 1个batch进行梯度清零，4步之内梯度叠加，实现利用低内存消耗，但具有相同的参数更新间隔
                optimizer.zero_grad()

                if opt.is_ema:
                    # 更新滑动平均的衰减率
                    tempEMADecay=ema_schedule(step+1)
                    ema.decay=tempEMADecay
                    ema.update()

            # 加入到一个epoch的损失里，即该epoch下的每个迭代步的loss值
            mean_loss.append(loss.item())

            log_str="\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch+1,epochs,batch_i+1,len(train_loader))
            # 字符串矩阵[['Metrics', 'YOLO Layer 0', 'YOLO Layer 1', 'YOLO Layer 2']]
            # *用于解开列表，提取列表中的元素出来
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(yolo_loss.metrics))]]]
            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):  # 遍历参数列表,生成一个n行4列的信息矩阵 ，['key','value1','value2','value3']
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                # # 遍历列表中的3个字典元素, dictionary.get(keyname, value)  字典 get() 函数返回指定键的值
                row_metrics = [formats[metric] % yolo.get(metric, 0) for yolo in yolo_loss.metrics]
                # %不是求余符号，是格式数据替换的符合，就print里的%作用
                # 将nx1和nx3的信息矩阵合并   *解开的序列，必须又存放到列表或字典中去，否则报错
                metric_table += [[metric, *row_metrics]]

            log_str += AsciiTable(metric_table).table  # AsciiTable是最简单的表。它使用+，|和-字符来构建边框
            log_str += f"\nTotal loss {loss}"  # 将loss信息加入到tabel末尾

            if step%200==0:
                tensorboard_log = []
                for j, yolo in enumerate(yolo_loss.metrics):
                    # items()方法把字典中每对 key和value 组成一个元组，并把这些元组放在列表中返回
                    for name, metric in yolo.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"train/{name}_{j + 1}", metric)]  # name为键，metric为值
                tensorboard_log += [("train/loss", loss)]
                logger.list_of_scalars_summary(tensorboard_log, step) # 按step存入损失信息
                # 记录每次更新下的学习率
                logger.list_of_scalars_summary([("learning_rate", templr)], step + 1)

            epoch_batches_left = len(train_loader) - (batch_i + 1)  # 当前迭代epoch中余下多少迭代step

            # datetime.timedelta对象代表两个时间之间的时间差，两个date或datetime对象相减就可以返回一个timedelta对象
            end_time = time.time()
            time_left = datetime.timedelta(seconds=epoch_batches_left * (end_time - start_time) / (batch_i + 1))  # 计算剩余时间
            log_str += f"\n---- ETA {time_left}"

            # 进度条，每个step都要打印loss值,以及之前所有step累计起来的loss的均值

            size_percent = 30
            info_str = f"Batch: {batch_i + 1}/{size} Loss:{loss.item():.6f} Mean loss:{np.mean(mean_loss):.6f} lr:{templr:.6f} ema:{tempEMADecay:.4f} Time:{str(time_left)[3:7]}                    "
            # IDE环境下会屏蔽'\r',需在终端中运行文件

            pbar_end = '>' + ' ' * (size_percent - batch_i // (size // size_percent)) + ']' + info_str + '\r'
            if batch_i % (size // size_percent) == 0:
                pbar += '='
            if batch_i == size - 1:
                pbar += '='
                pbar_end = '=]' + info_str + '\n'
            print(pbar, end=pbar_end)
            #time.sleep(0.1)
            """
            pbar_end=info_str+'\r'
            if batch_i == size - 1:
                pbar_end = info_str + '\n'

            try:
                with tqdm.tqdm(train_loader) as t:
                    for i in t:
                        print(pbar,end=pbar_end)
                        #time.sleep(1)
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            """

        if log_str != '':
            print(log_str)
            log_str = ""  # 每个epoch结束后，打印日志，然后清零，开始下一个epoch

        # 模型保存
        if (epoch + 1) % 20 == 0:
            if opt.is_ema:
                # 保存影子权重
                ema.apply_shadow()
            torch.save(net.state_dict(), f"%s%s_ckot_%d.pth"%(opt.checkpoint,opt.checkpoint_name,epoch+1))
            print("模型已保存：epoch=", epoch + 1)

        print('\n')  # 断隔开各个epoch之间的信息

        if (epoch+1) % opt.eva_fq == 0:
            print("交叉验证中...")
            if opt.is_ema:
                ema.apply_shadow()
            if opt.is_amp:
                net.half()
            net.eval()

            # 分类标签
            labels = []
            # 指标列表 (TP, confs, pred)
            sample_metrics = []
            val_size = len(val_loader)
            # Tqdm是一个快速，可扩展的Python进度条，可在Python长循环中添加一个进度提示信息，用户只需要封装任意的迭代器,
            # tqdm(iterator)
            for batch_j, (imgs,targets) in enumerate(tqdm.tqdm(val_loader, desc="Detecting objects")):
                if targets is None:
                    continue
                # tolist()，将ndarray或tensor的n维向量，转换为列表  b_id,cls_id,x,y,w,h
                # 将分类号提取出来，存放到列表中[2,4,5,1,0,3...]
                labels+=targets[:,1].tolist()
                # box坐标还原到原图尺寸 x,y,w,h
                targets[:, 2:] *= image_size
                # x,y,w,h -> x,y,x,y
                targets[:, 2:] = xywh2xyxy(targets[:, 2:])

                # pytorch除了训练都需要去梯度
                with torch.no_grad():
                    imgs=imgs.to(device)
                    targets=targets.to(device)
                    if opt.is_amp:
                        imgs=imgs.half()

                    # output: batch, 507, 25(conf, x, y, x, y, cls0 - 20)
                    outputs = net(imgs)
                    # 过NMS得到最后合格的那些预测box  output为list: [[n,7],[n,7],[n,7]...]   len(output)=64
                    outputs = NMS(outputs,conf_thres=opt.conf_thres,nms_thres=opt.nms_thres)
                # 计算指标
                sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=opt.iou_thres)

            if len(sample_metrics) == 0:
                return None
            if opt.is_amp:
                net.float()
            if opt.is_ema:
                # 交叉验证完后，把模型权重由影子权重恢复到正常
                ema.restore()

            # 拼接交叉验证参数
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
            print("\n")
            classes=opt.classes.split(',')
            print("Average Precisions:")
            for i, c in enumerate(ap_class):
                print(f"+ Class '{c}' ({classes[c]}) - AP: {AP[i]}")

            print(f"mAP: {AP.mean()}")
            logger.list_of_scalars_summary([("val/mAP", AP.mean())], epoch)


if __name__ == '__main__':
    model_path = 'E:/ai_project/'
    data_path = 'F:/VOC/VOC2012/'
    class_scale = '10.6787234,12.24146341,8.47804054,9.87992126,6.70093458,15.83280757,4.21410579,8.24137931,3.44474949,14.13802817,13.45576408,6.53515625,13.31299735,13.384,1.,9.01077199,9.86051081,12.57894737,15.34862385,12.18203883'
    label_class = 'aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=model_path + 'cfg_yolov5_210901_mkl_dbb.cfg', help='配置文件')
    parser.add_argument('--log', type=str, default=model_path + 'yolo_mkl/logs_test/', help='配置日志地址')
    parser.add_argument('--txt_path', type=str, default=data_path + "ImageSets/Main/", help='数据集地址')
    parser.add_argument('--image_path', type=str, default=data_path + 'JPEGImages/', help='图片地址')
    parser.add_argument('--label_path', type=str, default=data_path + 'Annotations/', help='标签地址')
    parser.add_argument('--checkpoint', type=str, default=model_path + 'checkpoints20210831/', help='存档地址')
    parser.add_argument('--checkpoint_name', type=str, default='yolov4_mkl', help='存档文件名称')
    parser.add_argument('--epoch', type=int, default=400, help='训练世代')
    parser.add_argument('--steps', type = str, default = '40000,45000', help = '改变学习率步数')
    parser.add_argument('--is_max_batches', action='store_true', default=False, help = '是否启用配置中最大迭代次数，会覆盖epoch')
    parser.add_argument('--is_grey', action='store_true', default=False, help='是否是灰度图')
    parser.add_argument('--class_scale', type=str,default=class_scale,help='分类乘数')
    parser.add_argument('--classes', type=str,default=label_class,help='分类标签')
    parser.add_argument('--eva_fq', type=int, default=10, help='交叉验证频率')
    parser.add_argument('--block_size', type=int, default=7, help='丢弃块大小')
    parser.add_argument('--drop_prob', type=float, default=.1, help='丢弃块概率')
    parser.add_argument('--iou_thres', type=float, default=.5, help='交叉验证iou阈值')
    parser.add_argument('--conf_thres', type=float, default=.25, help='交叉验证置信度阈值')
    parser.add_argument('--nms_thres', type=float, default=.45, help='交叉验证nms阈值')
    parser.add_argument('--is_aug', action='store_true', default=True, help='是否数据增强')
    parser.add_argument('--is_img', type=str,default='saturation, hue, contrast, mirror', help='数据增强类型')
    parser.add_argument('--batch', type=int, default=6, help='批数量')
    parser.add_argument('--mini_batch', type=int, default=2, help='mini批数量')
    parser.add_argument('--is_train', action='store_true', default=True, help='是否训练模式')
    parser.add_argument('--is_mosaic', action='store_true', default=True, help='是否随机马赛克')
    parser.add_argument('--is_mixup', action='store_true', default=True, help='是否随机mixup')
    parser.add_argument('--is_multi_scale', action='store_true', default=True, help='是否多尺度训练')
    parser.add_argument('--is_amp', action='store_true', default=False, help='是否混合精度训练')
    parser.add_argument('--is_ema', action='store_true', default=True, help='是否指数滑动平均训练')
    parser.add_argument('--is_ebr', action='store_true', default=True, help='是否ebr训练模型')
    parser.add_argument('--is_pa', action='store_true', default=False, help='是否正样本扩充')
    parser.add_argument('--is_fl', action='store_true', default=True, help='是否focal_loss')
    parser.add_argument('--is_debug', action='store_true', default=False, help='是否调试模式')
    opt = parser.parse_args()

    train(opt)