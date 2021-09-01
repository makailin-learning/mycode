import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset,DataLoader
from utils.DataAugmentaion import *
from torchvision.transforms import transforms

# 读取数据函数
class Mydata(Dataset):
    def __init__(self, image_path, label_path, txt_path, classes, is_train=True, is_aug=False, is_img=None,
                 is_grey=False, is_mosaic=False, img_size=416):
        super().__init__()
        self.image_path = image_path
        self.label_path = label_path
        self.classes = classes.split(',')

        self.is_train = is_train         # 读取训练数据集还是测试数据集
        self.is_grey = is_grey           # 是否灰度图
        self.is_aug = is_aug             # 是否图像增强
        self.is_mosaic = is_mosaic       # 是否马赛克增强
        #self.is_mosaic_weight=is_mosaic_weight  #是否使用权重马赛克
        self.img_size = img_size

        if is_img is not None:
            self.is_img = is_img.split(',')  # self.img_aug为字符串数组,记录需要进行图像增强的类型
        else:
            self.is_img = []

        if self.is_train:
            with open(txt_path+'train.txt') as f:
                self.filename = [a.strip() for a in f]
        else:
            with open(txt_path+'val.txt') as f:
                self.filename = [a.strip() for a in f]

        self.img_len = self.__len__()  # 获取数据个数
        self.indices = range(self.img_len)  # 获取数据集的序列，马赛克增强时选取另外3个图像时使用

    def __len__(self):
        return len(self.filename)

    # 解析xml标签文件函数
    def parsexml(self, index):
        with open(self.label_path + self.filename[index] + '.xml') as f:
            tree = ET.parse(f)  # 打开并解析xml文件,得到一个xml对象

        root = tree.getroot()  # 获得根元素
        size = root.find('size')  # 获得根元素下的size元素
        w = int(size.find('width').text)  # 获得size元素的width属性值,图像的宽度
        h = int(size.find('height').text)  # 获得size元素的height属性值，图像的高度
        obj_iter = root.iter('object')
        bbox = []
        for obj in obj_iter:
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes or difficult == 1:
                continue
            cls_id = self.classes.index(cls)  # 类别号
            box = obj.find('bndbox')
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)
            bbox.append(cls_id)
            bbox.append(xmin)
            bbox.append(ymin)
            bbox.append(xmax)
            bbox.append(ymax)

        if len(bbox) % 5 != 0:
            raise ValueError("File:" + self.label_path + self.filename[index] + ".xml" + "——bbox Extraction Error!")
        bbox = np.array(bbox, dtype=np.float32).reshape(-1, 5)  # 转换为2维数组，nx5格式

        return bbox, w, h

    # 读取图片和标签数据函数
    def load_img(self, index):

        # 加载图片,区分灰度图和彩色图，cv2.imread读取的图片格式为np.uint8，HxWxC
        if self.is_grey:  # 判断是否为灰度图
            try:
                img = cv2.imread(self.image_path + self.filename[index] + '.jpg', cv2.IMREAD_GRAYSCALE)
            except:
                img = cv2.imread(self.image_path + self.filename[index] + '.bmp', cv2.IMREAD_GRAYSCALE)
        else:
            try:
                img = cv2.imread(self.image_path + self.filename[index] + '.jpg', cv2.IMREAD_COLOR)
            except:
                img = cv2.imread(self.image_path + self.filename[index] + '.bmp', cv2.IMREAD_COLOR)

        # 加载标签
        bbox, w, h = self.parsexml(index)

        return img, bbox, w, h

    #按索引读取数据集函数
    def __getitem__(self, index):

        img ,bbox ,w, h = self.load_img(index)

        # 图像增强
        if self.is_aug:
            img, bbox = self.img_aug(img, bbox)

        # 马赛克增强
        if self.is_mosaic:
            if random.randint(0, 1):

                data = [(img, bbox, w, h)]  # 创造一个存放每个图像的相关数据
                indice = random.choices(self.indices, k=3)  # choices可取多次，choice只取1次

                for i in range(3):
                    # 随机选取3张图片

                    img_temp, bbox_temp, w_temp, h_temp = self.load_img(indice[i])
                    # 选取的图片进行数据增强
                    img_temp, bbox_temp = self.img_aug(img_temp, bbox_temp)
                    # 存放到列表中
                    data.append((img_temp, bbox_temp, w_temp, h_temp))

                # 马赛克合并，4合1
                img_temp, bbox_temp = self.img_mosaic(data)
                if bbox_temp.shape[0] != 0:  # 判断结果是否有效
                    img, bbox = img_temp, bbox_temp
                else:
                    raise ValueError('img_mosaic fail')

        img, bbox = self.img_padding(img, bbox)

        # torch.from_numpy会将ndarray转换为tensor，不会归一化，也不会改变通道顺序，依旧为 (H, W, C)
        # ToTensor()将shape为(H, W, C)的ndarray转为shape为(C, H, W)的tensor
        imgs = transforms.ToTensor()(img)
        # 用transforms.ToTensor()处理bbox，会生成[1,n,5]，用torch.from_numpy会生成[n,5]
        target = torch.zeros(len(bbox), 6)  # len(bbox)会只读取bbox的行数
        bbox = torch.from_numpy(bbox)
        # bbox=transforms.ToTensor()(bbox)  #这么转tensor会增加一个维度：1xnx5
        target[:, 1:] = bbox  # 这里又将bbox由 1xnx5转回为nx5的格式,所以前面用ToTensor还是from_numpy，不会产生影响

        return imgs, target  # imgs为张量且经归一化(ToTensor)，shape:CxHxW   target，张量经归一化（手动）,shape:HxW

    #数据集打包函数
    """
    default_collate是DataLoader的默认collate_fn,它将batch size个样本合成为一个batch（加了一个维度）
    __getiterm__ 返回的是 （img_tensor, label）
    所以传入collate_fn的参数就是一个list(batch): [(img_tensor0, label0),(img_tensor1, label1), ....]
    batch[0]=(img_tensor0, label0)
    然后利用torch.stack将数据堆叠起来，新增一个batch_size的维度
    所以重写collate_fn函数，必定会使用torch.stack函数
    """
    def collate_fn(self, batch):

        batch = [data for data in batch if data is not None]
        """
        type(batch):->list
        batch=[(img,label),(img,label)....]
        *batch进行解包，imgs=img,img,img...., labels=label,label,label....
        zip再进行压包，imgs=[img,img,img....], labels=[label,label,label....]
        解包和压包必须配合使用，不可单独使用
        所以imgs:->list/touple  labels:->list/touple
        len(imgs)=len(labels)=len(batch)=batch_size
        """
        imgs, labels = zip(*batch)
        imgs = torch.stack([img for img in imgs])  # stack会新增维度，3维-》4维
        for i, box in enumerate(labels):  # 每次从元组中取出一个label进行序号编码
            box[:, 0] = i  # 批次号赋值
        labels = torch.cat(labels, 0)  # cat必须指定维度，因为它不会新增维度，只会沿着指定维度续接
        return imgs, labels

    #图像普通增强函数
    def img_aug(self, img, box):  # 坐标要求为xyxy形式

        if len(self.is_img) == 0:  # 若传入空列表进来，原样返回，不做增强处理
            return img, box
        else:
            img = img.astype(np.float32)  # np.uint8->np.float32
            da = DataAugmentation()  # 生成数据增强类对象

            img = da.convertColor(img, 'RGB')  # 两种增强在HSV空间下进行

            if 'saturation' in self.is_img:  # 饱和度
                img = da.randomSaturation(img)

            if 'hue' in self.is_img:  # 色调
                img = da.randomHue(img)

            img = da.convertColor(img, 'HSV')  # 其余都在RGB空间下进行

            if 'contrast' in self.is_img:  # 对比度
                img = da.randomContrast(img)

            if 'brightness' in self.is_img:  # 亮度
                img = da.randomBrightness(img)

            if 'lighting_noise' in self.is_img:
                img = da.randomLightingNoise(img)  # 通道变换

            if 'crop' in self.is_img:
                img, box = da.randomSampleCrop(img, box)  # 随机裁剪

            if 'expand' in self.is_img:  # 图像扩充
                img, box = da.expand(img, box)

            if 'mirror' in self.is_img:
                img, box = da.randomMirror(img, box)  # 镜像翻转

            img = img.astype(np.uint8)   # np.float32->np.uint8
            return img, box

    #图像填充灰边函数
    def img_padding(self, img, bbox):  # 要求输入坐标为xywh形式，转换为xywh格式并归一化后返回

        h = img.shape[0]
        w = img.shape[1]

        padw = 0.
        padh = 0.
        temp = 0.
        if h > w:
            # 若高度大于宽度，则在图像两边进行填充，再缩放，同时这会使得原图像的原点变化，向左边移动了padw列
            padw = (h - w) // 2
            # 填充对象为img，有三个维度，填充数组为((0,0),(padw,padw),(0,0))，行方向和通道方向不填充，填充列方向，前后各填充 padw列，用114填充
            img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), 'constant', constant_values=114)

            # box的x方向坐标，向右移动padw个
            bbox[:, 1] = bbox[:, 1] + padw
            bbox[:, 3] = bbox[:, 3] + padw

            temp = h  # 记录填充后的图像边长
        elif w > h:
            # 若宽度大于高度，则在图像上下进行填充，再缩放，同时这回使得原图像的原点变化，向上边移动了padh行
            padh = (w - h) // 2
            img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), 'constant', constant_values=114)  # 填充方向：行，列，通道

            # box的y方向坐标，向下移动padw个
            bbox[:, 2] = bbox[:, 2] + padh
            bbox[:, 4] = bbox[:, 4] + padh

            temp = w
        else:
            temp = h

        # 缩放图片
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 格式转换(因为网络预测的形式是xywh,计算loss的时候会使用)    xywh
        box = np.zeros((len(bbox), 4))  # 注意格式：（r,l）有括号或者[]才行
        box[:, 0] = (bbox[:, 1] + bbox[:, 3]) / 2  # x1+x2  -> x
        box[:, 1] = (bbox[:, 2] + bbox[:, 4]) / 2
        box[:, 2] = bbox[:, 3] - bbox[:, 1]  # x2-x1  -> w
        box[:, 3] = bbox[:, 4] - bbox[:, 2]

        # 416尺寸下的归一化，获得gt框的所占图像比例
        box = box / temp
        bbox[:, 1:] = box

        return img, bbox

    #马赛克数据增强函数
    def img_mosaic(self, data):
        size = len(data)
        labels4 = []
        # 定义中心偏移
        mosaic_offset = [-self.img_size // 2, -self.img_size // 2]
        # 随机中心点坐标偏移量,即把一个图片随机分成四份。将原图放大两倍进行拼接，最后在缩回到原图像大小
        cut_y, cut_x = [int(random.uniform(-x, 2 * self.img_size + x)) for x in mosaic_offset]  # 限制切割点处于1/4到3/4之间

        for i in range(size):
            img, boxes, w, h = data[i]
            n_size = self.img_size * 2
            # 处理第一张,设置在左上角
            if i == 0:
                # cx/cy'=w/h   若cy<cy'，说明分割区域为 宽矮型，则以w为标准进行拼接
                o_size, flag = (w, 1) if cut_x / cut_y > w / h else (h, 0)  # 若cy>cy'，说明分割区域为窄高型，则以h为标准进行拼接
                if len(img.shape) < 3:  # 判断灰度图和彩色图
                    channel = 1
                else:
                    channel = img.shape[2]
                # 创建一个2倍的灰色填充图像，作为基础图，在此基础上进行分割与拼图
                img4 = np.full((n_size, n_size, channel), 114, dtype=np.uint8)  # np.full 构造一个数组，用指定值填充其元素

                if flag == 1:  # 以w为基准进行缩放
                    ratio = cut_x / o_size  # 获取拼接区域的缩放比例
                    temp_s = int(round(h * ratio))  # temp_s为等比例缩小的h'点，即对角线上，相似矩形的哪个h'
                    img = cv2.resize(img, (cut_x, temp_s))  # 原图等比例缩小，不会发生失真现象
                    x1b, y1b, x2b, y2b = 0, int(round(abs(temp_s - cut_y) / 2)), cut_x, temp_s - int(round(abs(temp_s - cut_y) / 2))
                else:  # 以h为基准进行缩放
                    ratio = cut_y / o_size  # h'/h
                    temp_s = int(round(w * ratio))
                    img = cv2.resize(img, (temp_s, cut_y))
                    x1b, y1b, x2b, y2b = int(round(abs(temp_s - cut_x) / 2)), 0, temp_s - int(round(abs(temp_s - cut_x) / 2)), cut_y

                x1a, y1a, x2a, y2a = 0, 0, cut_x, cut_y

            # 第二张图片设置在右上角
            elif i == 1:
                o_size, flag = (w, 1) if (n_size - cut_x) / cut_y > w / h else (h, 0)  # 若cy>cy'，说明分割区域为窄高型，则以h为标准进行拼接
                if flag == 1:
                    ratio = (n_size - cut_x) / o_size
                    temp_s = int(round(h * ratio))
                    img = cv2.resize(img, (n_size - cut_x, temp_s))
                    x1b, y1b, x2b, y2b = 0, int(round(abs(temp_s - cut_y) / 2)),\
                                         n_size - cut_x, temp_s - int(round(abs(temp_s - cut_y) / 2))
                else:
                    ratio = cut_y / o_size
                    temp_s = int(round(w * ratio))
                    img = cv2.resize(img, (temp_s, cut_y))
                    x1b, y1b, x2b, y2b = int(round(abs(temp_s - n_size + cut_x) / 2)), 0,\
                                         temp_s - int(round(abs(temp_s - n_size + cut_x) / 2)), cut_y

                x1a, y1a, x2a, y2a = cut_x, 0, n_size, cut_y

            # 第三张图片设置在左下角
            elif i == 2:
                o_size, flag = (w, 1) if cut_x / (n_size - cut_y) > w / h else (h, 0)
                if flag == 1:
                    ratio = cut_x / o_size
                    temp_s = int(round(h * ratio))
                    img = cv2.resize(img, (cut_x, temp_s))
                    x1b, y1b, x2b, y2b = 0, int(round(abs(temp_s - n_size + cut_y) / 2)), cut_x,\
                                         temp_s - int(round(abs(temp_s - n_size + cut_y) / 2))
                else:
                    ratio = (n_size - cut_y) / o_size
                    temp_s = int(round(w * ratio))
                    img = cv2.resize(img, (temp_s, n_size - cut_y))
                    x1b, y1b, x2b, y2b = int(round(abs(temp_s - cut_x) / 2)), 0,\
                                         temp_s - int(round(abs(temp_s - cut_x) / 2)), n_size - cut_y

                x1a, y1a, x2a, y2a = 0, cut_y, cut_x, n_size

            # 第四张图设置在右下角
            elif i == 3:
                o_size, flag = (w, 1) if (n_size - cut_x) / (n_size - cut_y) > w / h else (h, 0)
                if flag == 1:
                    ratio = (n_size - cut_x) / o_size
                    # 跟上面3次处理略有不同，原因？
                    img = cv2.resize(img, (n_size - cut_x, int(round(h * ratio))))
                    x1b, y1b, x2b, y2b = 0, int(round(abs(h * ratio - n_size + cut_y) / 2)), n_size - cut_x,\
                                         int(round(h * ratio - abs(h * ratio - n_size + cut_y) / 2))
                else:
                    ratio = (n_size - cut_y) / o_size

                    img = cv2.resize(img, (int(round(w * ratio)), n_size - cut_y))
                    x1b, y1b, x2b, y2b = int(round(abs(w * ratio - n_size + cut_x) / 2)), 0,\
                                         int(round(w * ratio - abs(w * ratio - n_size + cut_x) / 2)), n_size - cut_y

                x1a, y1a, x2a, y2a = cut_x, cut_y, n_size, n_size

            err1 = (y2b - y1b) - (y2a - y1a)  # 表示紫色框和红色框，高度的差值,理论上为0，实际有四舍五入的偏差，y2b-y1b略高于y2a-y1a一点点
            err2 = (x2b - x1b) - (x2a - x1a)  # 为紫色框和红色框，宽度的差值，理论上应当为0，两者错位，但宽度相同

            tempya=y2b
            tempxa=x2b
            if err1!=0:
                y2b=y2b-err1
            if err2!=0:
                x2b=x2b-err2

            if len(img.shape) < 3:  # 若为灰度图，则增加一个维度
                img = np.expand_dims(img, -1)  # 增加一个维度
            try:
                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # 相对于在缩放后的img上，裁去两端，保留中间部分，然后覆盖在img4上
            except Exception as e:
                print('err1', err1, y2b, y1b, y2a, y1a, tempya)
                print('err2', err2, x2b, x1b, x2a, x1a, tempxa)
                raise e

            if boxes.size:  # gt框数据有效
                if i == 0:  # 左上角
                    boxes[:, 1] *= ratio  # 先将gt框同图片一样，进行缩放
                    boxes[:, 1] -= x1b  # 将gt框向左移动x1b段距离，因原图两端裁去了x1b大小的像素
                    boxes[:, 1] = np.clip(boxes[:, 1], 0, cut_x)  # clip这个函数将将数组中的元素限制在0,cut_x之间，两端被裁掉的gt框不再显示

                    boxes[:, 2] *= ratio
                    boxes[:, 2] -= y1b  # 将gt框向上移动y1b段距离，因原图两端裁去了y1b大小的像素
                    boxes[:, 2] = np.clip(boxes[:, 2], 0, cut_y)

                    boxes[:, 3] *= ratio
                    boxes[:, 3] -= x1b
                    boxes[:, 3] = np.clip(boxes[:, 3], 0, cut_x)

                    boxes[:, 4] *= ratio
                    boxes[:, 4] -= y1b
                    boxes[:, 4] = np.clip(boxes[:, 4], 0, cut_y)

                if i == 1:  # 右上角
                    boxes[:, 1] *= ratio
                    boxes[:, 1] -= x1b
                    boxes[:, 1] = np.clip(boxes[:, 1], 0, n_size - cut_x)
                    boxes[:, 1] += cut_x

                    boxes[:, 2] *= ratio
                    boxes[:, 2] -= y1b
                    boxes[:, 2] = np.clip(boxes[:, 2], 0, cut_y)

                    boxes[:, 3] *= ratio
                    boxes[:, 3] -= x1b
                    boxes[:, 3] = np.clip(boxes[:, 3], 0, n_size - cut_x)
                    boxes[:, 3] += cut_x

                    boxes[:, 4] *= ratio
                    boxes[:, 4] -= y1b
                    boxes[:, 4] = np.clip(boxes[:, 4], 0, cut_y)

                if i == 2:  # 左下角
                    boxes[:, 1] *= ratio
                    boxes[:, 1] -= x1b
                    boxes[:, 1] = np.clip(boxes[:, 1], 0, cut_x)

                    boxes[:, 2] *= ratio
                    boxes[:, 2] -= y1b
                    boxes[:, 2] = np.clip(boxes[:, 2], 0, n_size - cut_y)
                    boxes[:, 2] += cut_y

                    boxes[:, 3] *= ratio
                    boxes[:, 3] -= x1b
                    boxes[:, 3] = np.clip(boxes[:, 3], 0, cut_x)

                    boxes[:, 4] *= ratio
                    boxes[:, 4] -= y1b
                    boxes[:, 4] = np.clip(boxes[:, 4], 0, n_size - cut_y)
                    boxes[:, 4] += cut_y

                if i == 3:  # 右下角
                    boxes[:, 1] *= ratio  # 想对分割区域的缩放
                    boxes[:, 1] -= x1b
                    boxes[:, 1] = np.clip(boxes[:, 1], 0, n_size - cut_x)
                    boxes[:, 1] += cut_x

                    boxes[:, 2] *= ratio
                    boxes[:, 2] -= y1b
                    boxes[:, 2] = np.clip(boxes[:, 2], 0, n_size - cut_y)
                    boxes[:, 2] += cut_y

                    boxes[:, 3] *= ratio
                    boxes[:, 3] -= x1b
                    boxes[:, 3] = np.clip(boxes[:, 3], 0, n_size - cut_x)
                    boxes[:, 3] += cut_x

                    boxes[:, 4] *= ratio
                    boxes[:, 4] -= y1b
                    boxes[:, 4] = np.clip(boxes[:, 4], 0, n_size - cut_y)
                    boxes[:, 4] += cut_y
                # 注意：boxes[:,3:4]结果为1x1的二维矩阵，boxes[:,3]结果为单元素向量
                idx_w = np.argwhere(np.all((boxes[:, 3:4] - boxes[:, 1:2]) == 0, axis=1))  # 获取w为非零，即有效的那些box的行索引
                boxes = np.delete(boxes, idx_w, axis=0)
                idx_h = np.argwhere(np.all((boxes[:, 4:] - boxes[:, 2:3]) == 0, axis=1))  # 获取h为非零，即有效的那些box的行索引
                boxes = np.delete(boxes, idx_h, axis=0)

            boxes[:, 1:] /= 2  # 前面的ratio是针对img4进行了缩放，即两倍大小的情况下，缩放了，故需要再缩小一倍，才符合原图的大小尺寸
            labels4.append(boxes)

        img4 = cv2.resize(img4, (self.img_size, self.img_size))  # 将2倍的背景图，再缩放会原图大小
        labels4 = np.concatenate(labels4, axis=0)  # 按轴axis连接array组成一个新的array,拼接行，将所有label的行拼接再一起
        return img4, labels4
