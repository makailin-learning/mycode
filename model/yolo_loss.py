import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utilss import to_cpu
import utils.iou as I

# 损失平滑，即设置target时，缩小target和pred之间的差距，也能推理出正确的结果
# 也即并非一定要pred接近1，才认为是合格的，但pred接近0.9时我就认为是预测OK
# 将原先的 n_target=0 p_target=1 切换为 n_target=0.1 p_target=0.9 ,减小预测的难度
def smooth_BCE(eps=0.1):
    return 1.0-0.5*eps,0.5*eps

class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=2, alpha=0.25):
        super(FocalLoss,self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn( pred, true)
        pred_prob = torch.sigmoid(pred)
        #论文作者定义: pt = p,y=1 or 1-p,otherwise
        p_t = true*pred_prob + (1-true)*(1-pred_prob)
        alpha_t = self.alpha*true + (1-self.alpha)*(1-true)
        gamma_t = (1.0-p_t)**self.gamma
        # 对应论文公式: FL(pt) = −αt*(1 − pt)γ*log(pt).
        loss = alpha_t*gamma_t*loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Focal Loss在连续label上的拓展形式之一
# 既保证Focal Loss此前的平衡正负、难易样本的特性，又需要让其支持连续数值的监督
class QFocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=2, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true, cp):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_t = (true==cp)*self.alpha + (true==1-cp)*(1-self.alpha)
        gamma_t = torch.abs(true-pred_prob)**self.gamma
        loss = alpha_t * gamma_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class YoloLoss(nn.Module):
    def __init__(self,class_scale,gr=0,is_ebr=False,is_fl=False):
        super(YoloLoss, self).__init__()
        self.cp, self.cn=smooth_BCE(eps=0.2)
        #类别乘数，缓解类别间的样本不平衡
        self.class_scale=class_scale.split(',')
        # map函数将按照指定的函数，对数据进行映射处理 map(function, iterable, iterable...)
        self.class_scale=list(map(float,self.class_scale))
        # 初始化gtbox的置信度时，取1.0和iou之间的协调比例 0,0.5,1
        self.gr=gr
        self.sort_obj_iou=False
        #self.image_size=image_size
        #二元交叉熵带softmax的分类损失函数
        self.BCEcls=nn.BCEWithLogitsLoss()
        self.BCEobj=nn.BCEWithLogitsLoss()
        self.BCEbox=nn.BCEWithLogitsLoss()

        #FocalLoss损失函数
        self.is_fl=is_fl
        if self.is_fl:
            self.BCEcls,self.BCEobj=QFocalLoss(self.BCEcls),FocalLoss(self.BCEobj)

        self.is_ebr=is_ebr
        self.ebr_outline=[3.,1.5,1.5]
        self.ebr_loss=10.

        self.loss_status="iou"

    def forward(self,x,y=None,c=None):
        # 存放网络性能参数
        self.metrics=[]

        # 多尺度下的img_size
        self.image_size=x[0].shape[2]*8

        # batch_size
        bs=x[0].shape[0]

        # 计算设备
        device=y.device

        #cls/box/obj损失初始化 0. 0. 0.
        lcls,lbox,lobj=torch.zeros(1,device=device),torch.zeros(1,device=device),torch.zeros(1,device=device)

        #读取配置文件
        cfg=c[0]

        # na为每个检测头的anchor数量3，nt为一个批次中所有目标的gt框数量12, nc为数据集中的目标类别数20
        na=len(cfg['mask'].split(','))
        nt=y.shape[0]
        nc=int(cfg['classes'])

        # 标准化网格空间增益[1.,1.,1.,1.,1.,1.,1.] b_id,cls_id,x,y,w,h,a_id
        gain=torch.ones(7,device=device)

        """
        torch.tensor([0,1,2]) 转换如下:
        torch.tensor([[0],
                      [1],
                      [2]])   转换如下:
        torch.tensor([[0,0,0,0....],
                      [1,1,1,1....],
                      [2,2,2,2....]])
        """
        ai=torch.arange(na,device=device).float().view(na,1).repeat(1,nt)
        # 对label进行改造,ai[:,:,None]会自动进行维度转换，升维
        """
        [[12,6]]      ->
        [[[12,6]],         [[[12,7]], 
         [[12,6]],          [[12,7]], 
         [[12,6]]]    ->    [[12,7]]]     batch_id,class_id,x,y,w,h,anchor_id(0,1,2)
        """
        # 由2维矩阵变为3维张量，厚度为na=3，即anchor个数，高度为nt=12，即bs张图片中的gt框个数，宽度为7  label_shape:[3,n,7]
        y=torch.cat((y.repeat(na,1,1),ai[:,:,None]),2)

        #偏置项(网格中心点)
        g=0.5
        #扩展正样本偏置项，正样本中心网格的3x3网格区域
        off=torch.tensor(
            [[0, 0],[ 1, 0],[0,1],
             [-1,0],[ 0,-1],[1,1],
             [-1,1],[-1,-1],[1,1]],device=device).float()*g

        for i,x_i in enumerate(x):
            x_type=x_i.dtype
            cfg=c[i]
            scale_x_y=float(cfg['scale_x_y'])*2
            grid_size=int(cfg['grid_size'])

            # 找出合适的gt类别号，gtbox, 选取预测box的索引，锚框
            tcls,tbox,indices,anchors=self.build_targets(y,grid_size,cfg,device,gain,nt,g,off,x_type)

            # tbox是gt的tx,ty,tw,th,网格内部的偏移量
            # 候选框索引: 图片索引，anchor索引，y网格索引，x网格索引,均为m维向量[m]
            b,a,gj,gi=indices  # gj和gi是gt框在检测图中的网格位置，检测图下的网格偏移量

            # 定义置信度，shape与预测tensor一样,(input,full_value: 0.1) 64x3x13x13
            pconf = x_i[..., 0].sigmoid()
            tobj=torch.full_like(pconf,self.cn,device=device)

            # 选取候选框数量 m，这么写比len(b)的鲁棒性更高
            n=b.shape[0]
            iou=torch.zeros(1,device=device)
            # 如果有目标存在，即本次检测头能够检测图片中存在物体，否则认为该批次的图片与该检测头不匹配
            # 计算lcls和lbox，否则全是背景，不计算，但lobj是针对整个检测图的，所以始终要计算
            # 如果n=0，则该检测头预测失败
            if n:
                # 获取正样本预测box的信息，shape[m,25] conf,x,y,w,h,cls...
                ps=x_i[b,a,gj,gi]
                # sigmoidx2-0.5,消除网格敏感
                pxy=ps[:,1:3].sigmoid()*scale_x_y-(scale_x_y-1.)/2.   # sigmoid*4-1.5
                # tw[m,2]*pw[m,2]
                pwh=(ps[:,3:5].sigmoid()*2)**2*anchors
                # 预测box的信息shape[m,4]
                pbox=torch.cat((pxy,pwh),1)
                # 预测分类信息  [m,20]
                pcls=ps[:,5:].sigmoid()

                #计算loss
                if self.is_ebr:
                    gtwh,ct=self.ebrv2(pwh,tbox[:,2:],outline=self.ebr_outline[i])
                    if ct>0:  #说明gtbox以及更改,需更新
                        tbox[:,2:]=gtwh
                        self.loss_status='ebr'
                    else:
                        self.loss_status='iou'
                iou=self.box_iou(pbox,tbox,ciou=True)  #[m] m维向量
                # box的iou平均损失
                lbox+=(1.0-iou).mean()
                #TODO 到底什么时候需要使用detach()
                score_iou=iou.detach().clamp(0).type(tobj.dtype)
                # 是否对iou进行排序
                if self.sort_obj_iou:
                    #排序后的iou索引
                    sort_id=torch.argsort(score_iou)
                    # 将原先的正样本boxid按照iou大小顺序重新定义
                    b,a,gj,gi,score_iou=b[sort_id],a[sort_id],gj[sort_id],gi[sort_id],score_iou[sort_id]

                # gt=0时，初始化为1，静态的; gt=1时，初始化为iou，动态的，随着iou越好，gtbox的conf也越好
                tobj[b,a,gj,gi]=(self.cp-self.gr)+self.gr*score_iou

                if nc>1:
                    t=torch.full_like(pcls,self.cn,device=device)  # [m,20]
                    # range(m个box),即逐行中的指定列进行赋值为0.9，其余19列值保持为0.1
                    t[range(n),tcls]=self.cp
                    #分类是针对box的，所以shape mx5
                    lcls+=self.BCEcls(pcls,t,self.cp)   #未乘以分类乘数

            # 置信度是针对整个检测图的所有网格的，所以shape 64x3x13x13
            lobj+=self.BCEobj(pconf,tobj)*float(cfg['obj_normalizer'])  #不同检测图的置信度损失权重

            #评估参数
            metric={
                #TODO 什么时候用to_cpu
                "grid_size":grid_size,
                "loss":to_cpu(lobj+lcls+lbox).item(),
                "lbox":to_cpu(lbox).item(),
                "lcls": to_cpu(lcls).item(),
                "lobj": to_cpu(lobj).item(),
                "iou":to_cpu(iou.mean()).item(),
                "conf":to_cpu(pconf.mean()).item()
            }
            self.metrics.append(metric)

            # 用于检查其参数是否是无穷大,有效数字返回true,无穷大或者nan则返回false
            if not torch.isfinite(lobj):
                print("lobj:grid_size",grid_size,grid_size*32,lobj,end='\n')
                print(tobj,float(cfg['obj_normalizer']),iou,tbox,pbox,anchors,b,a,gj,gi,x_i)
            if not torch.isfinite(lcls):
                print("lcls:grid_size",grid_size,grid_size*32,lcls,end='\n')
            if not torch.isfinite(lbox):
                print("lbox:grid_size",grid_size,grid_size*32,lbox,end='\n')

        lbox*=float(cfg["iou_normalizer"])  #box的损失权重
        lcls*=float(cfg["cls_normalizer"])  #cls的损失权重

        # 损失都是求的均值,所以要乘以batch_size，才得到batch_sum_loss
        return (lcls+lbox+lobj)*bs

    def box_iou(self,p_b,t_b,iou=False,giou=False,diou=False,ciou=False):
        iou_res=0.
        if iou:
            iou_res=I.iou(p_b,t_b)
        if giou:
            iou_res = I.giou(p_b, t_b)
        if diou:
            iou_res = I.diou(p_b, t_b)
        if ciou:
            iou_res = I.ciou(p_b, t_b)
        return iou_res

    def build_targets(self,y,grid_size,cfg,device,gain,nt,g,off,x_type):
        mask = cfg['mask'].split(',')
        mask = [int(a) for a in mask]
        anchors = cfg['anchors'].split(',')
        anchors = [int(a) for a in anchors]
        anchors = np.array([[anchors[i], anchors[i + 1]] for i in range(0, len(anchors), 2)])

        # 获得当前检测图尺寸下的anchors,ndarray和tensor才能这样取索引,list不行
        # TODO 这里需要除以原图尺寸吗？
        anchors=torch.tensor(anchors[mask],dtype=x_type,device=device)/self.image_size*grid_size
        gain[2:6]=torch.tensor([grid_size,grid_size,grid_size,grid_size],device=device) #[1.,13,13,13,13,1.]

        # 所有目标的gt框 x,y,w,h * 网格大小增益，将gtbox的尺寸由原图尺寸下的归一化，转换为检测图尺寸下的归一化
        t=y*gain  # batch_id,cls_id,x,y,w,h,anchor_id   [3,12,7]

        if nt:

            r=t[:,:,4:6]/anchors[:,None]   #gt框和锚框的wh尺寸比例，anchors的shape由 [3,2] -> [3,1,2]  [3,12,2]/[3,1,2]

            # TODO 没看懂这段代码，再确认
            # 判断r和1/r谁大，然后选出最后一个维度的最大值(w,h比例谁大)，gt与ac尺寸比是否超过4倍，max过后产生values和index两个张量
            j=torch.max(r,1./r).max(dim=2)[0]<4.  # j.shape: [3,12] bool
            # 选出那些gt和anchor的尺寸比小于四倍的那些gt框，shape[n,7],有可能n=0，即该batch张图片与该检测头匹配不成功
            # 例如图中全是大物体，且都只有一个物体，即nt=4，在52x52的检测头下，就会匹配失败，那么将会在26和13下匹配成功
            t=t[j]
            gxy=t[:,2:4]    #取得gt框的xy坐标 [n,2]
            gxi=gain[[2,3]]-gxy  #网格数-xy坐标，获取框在检测图上的方位偏移 [n,2]
            #gt框x,y坐标距离左边的距离小于中心点，且不是在第一个网格中那些xy坐标
            j,k=((gxy%1.<g)&(gxy>1.)).T
            j_s, k_s = ((gxy % 1. < 0.35) & (gxy > 1.)).T
            # gt框位置距离右边的距离小于中心点，且不是在最后一个网格中  shape均为 [n]
            l,m=((gxi%1.<g)&(gxi>1.)).T
            l_s, m_s = ((gxi % 1. < 0.35) & (gxi > 1.)).T
            o,p=j_s&k_s,k_s&l_s
            q,s=l_s&m_s,m_s&j_s
            j=torch.stack((torch.ones_like(j),j,k,l,m,o,p,q,s))  # [9,n]
            t=t.repeat((9,1,1))[j]  # [n,7]-[9,n,7][9,n]-[m,7]
            # [None]用于升维 [n,2]-[1,n,2] + [9,2]-[9,1,2] = [9,n,2][9,n] = [m,2]
            offsets=(torch.zeros_like(gxy)[None]+off[:,None])[j]
        else:
            t=y[0]
            offsets=0

        b,c=t[:,:2].long().T # batch_id,cls_id
        gxy=t[:,2:4]-offsets
        gwh=t[:,4:6]
        gij=gxy.long()  # grid_id  检测图中的网格位置索引
        gi,gj=gij.T     # .T=.t()
        a=t[:,6].long() # anchor_id
        # 返回的均是m维向量或者，mxn的矩阵，m为原先n个gt框经过筛选后的
        #    c_id([m]), [m,4 (tx,ty,tw,th)],(b_id([m]),a_id([m]),gy_id([m]),gx_id([m])),anchor([m,2])
        return c,torch.cat((gxy-gij,gwh),1),(b,a,gj.clamp(0,gain[3]-1),gi.clamp(0,gain[2]-1)),anchors[a]

    # 这是个动态的过程，不断训练的pbox会因gtbox的调整而不断优化
    def ebrv2(self,pdwh,gtwh,outline=1.0):
        # [m,2] and [m,2]
        diswh=pdwh-gtwh
        maskl=diswh<0
        # 未指定维度，就是整个[m,2]一起进行求和，取true的个数
        count=maskl.sum()
        if count>0:
            # 对gt框的宽高大于预测框的宽高的conunt个位置进行处理:加上差值，使gtbox更大
            # 注意gtwh=... 和gtwh[mask]=...的不同，前一个是一个新变量，覆盖了之前的gtwh;后一个是在之前变量的不同位置上新赋值
            gtwh[maskl]=gtwh[maskl]+torch.clamp(-diswh[maskl],0,outline) #torch.clamp(input,min=0,max=outline)

        return gtwh,count

