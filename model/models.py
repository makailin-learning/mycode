import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# sys.path.append('D:\\project\\yolov4_mkl')
from model.common import *
from model.yolo import YOLOLayer

# 配置文件解析函数，解析为一个字典列表
def parseCfgFile(cfgfile):
    """
    read()直接读取字节到字符串中，包括了换行符,读取整个文件，将文件内容放到一个字符串变量中   返回字符串
    '吴迪 177 70 13888888\n王思 170 50 13988888\n白雪 167 48 13324434\n黄蓉 166 46 13828382'

    readline()方法每次读取一行；返回的是一个字符串对象，保持当前行的内存,readline()  读取整行，包括行结束符   返回字符串
    '吴迪 177 70 13888888\n'

    一次性读取整个文件；自动将文件内容分析成一个行的列表，readlines()读取所有行然后把它们作为一个字符串列表返回   返回字符串列表
    ['吴迪 177 70 13888888\n', '王思 170 50 13988888\n', '白雪 167 48 13324434\n', '黄蓉 166 46 13828382']

    """
    try:
        with open(cfgfile, 'r', encoding='UTF-8') as f:
            lines = f.readlines()  # 按行读取，返回字符串列表，包含转义字符
            lines = [x for x in lines if x[0] != '\n']  # 过滤空白行,该行只包含一个'\n'转义字符，没有其他的
            lines = [x for x in lines if x[0] != '#']  # 过滤注释行
            lines = [x.strip('\n') for x in lines]  # 去除字符串中的回车字符,
    except:
        with open(cfgfile, 'r', encoding='gbk') as f:
            lines = f.readlines()  # 按行读取，返回字符串列表，包含转义字符
            lines = [x for x in lines if x[0] != '\n']  # 过滤空白行,该行只包含一个'\n'转义字符，没有其他的
            lines = [x for x in lines if x[0] != '#']  # 过滤注释行
            lines = [x.strip('\n') for x in lines]  # 去除字符串中的回车字符,

    block = {}  # 存放每一个层结构信息的模块
    blocks = []  # 存放所有层结构模块

    for line in lines:  # 逐行读取
        if line[0] == '[':  # 说明当前进入了一个结构层，以此为开始进行存储结构层信息，同时作为两个结构层的分界线
            if len(block) != 0:  # 当遇到下一个结构层名称时，将之前的block信息存入到blocks中去
                blocks.append(block)
                block = {}  # 然后清空block，开始存放下一个结构层
                block["name"] = line[1:-1].strip()  # 取出[]之间的字符，作为结构层名称，作为字典的键
            else:
                block["name"] = line[1:-1].strip()  # 当第一个结构层时，此时block还等于0，即存入‘net’作为键
        # 没遇到下一个‘[’之前，后续的内容都为该结构层的信息
        else:
            key, value = line.split('=')  # 以'='将字符串分隔成两部分 height=416  ->  height,416
            # 此处的值value均不做进一步处理，统统以字符串形式赋值，不同的字符串在对应的模块中去做详细的处理
            block[key.strip()] = value.strip().replace(" ","")  # 字典会自动在末尾创建键key，然后赋值value，并将v中空格替换掉

    blocks.append(block)  # 遍历完所有行时，最后一个block还没有存入列表中，因此跳出循环后，执行一次存入操作

    return blocks  # [{’name‘:'net',...},{'name':'conv',....}....]  字典列表,一个字典内包含该结构层的所有信息

# 模型结构解析函数，解析为模型列表
def creat_module(cfg_blocks, first_channel=3, img_size=416, drop_prob=0.1, block_size=7):  # 利用配置文件列表，创建模型
    net_info = cfg_blocks[0]  # 第一个字典元素为{net}模块，不算做模型结构，作为网络信息
    """
    #它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器

    nn.Sequential里面的模块按照顺序进行排列的，所以必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。
    而nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起，这些模块之间并没有什么先后顺序可言

    nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数。
    因此构建网络时，还需要重写forward函数

    加入到 nn.ModuleList 里面的 module 是会自动注册到整个网络上的，同时 module 的 parameters 也会自动添加到整个网络中
    使用Python中的list，则不会自动进行，导致网络模型创建错误，无法进行训练

    #可将不同网络层打包到nn.Sequential中，然后再存放到ModuleList中，作为列表元素，跟blocks的[{},{}]一个道理
    """
    module_list = nn.ModuleList()
    # 用来记录前一层网络的输出通道数，用于下一层网络的输入通道数，初始值为3
    output_dims = [first_channel]  # 记录整个网络前进过程中，每个网络层的通道数，便于路由层和捷径层按照层号索引进行读取
    routs={}  #存放网络中的分支filter数量，便于后面使用

    #遍历整个字典列表，根据模块名称创建网络层
    for index, c in enumerate(cfg_blocks[1:]):  # 跳过第一个字典元素net，index为字典元素序号，x为字典

        modules = nn.Sequential()  # 创建一个模型结构序列,用add_module添加网络结构层

        # 匹配卷积层
        if c['name'] == 'conv':
            params = c['params']
            modules = Conv(output_dims[-1], params)
            filters = modules.filter

        # 检测层
        elif c['name'] == 'detect':
            params = c['params']
            modules = Detect(output_dims[-1], params)
            filters = modules.filter

        # focus层
        elif c['name'] == 'focus':
            params = c['params']
            modules = Focus(output_dims[-1], params)
            filters = modules.filter

        # 残差层
        elif c['name'] == 'res':
            params = c['params']
            modules = Res(output_dims[-1], params, drop_prob, block_size)
            filters = modules.filter

        # rep层
        elif c['name'] == 'rep':
            params = c['params']
            modules = Rep(output_dims[-1], params, drop_prob, block_size)
            filters = modules.filter

        # resrep层
        elif c['name'] == 'resrep':
            params = c['params']
            modules = ResRep(output_dims[-1], params, drop_prob, block_size)
            filters = modules.filter

        # eightrep层
        elif c['name'] == 'eightrep':
            params = c['params']
            modules = EightRep(output_dims[-1], params, drop_prob, block_size)
            filters = modules.filter

        # transformer层
        elif c['name'] == 'transformer':
            params = c['params']
            modules = Transformer(output_dims[-1], params)
            filters = modules.filter

        # transformer conv层
        elif c['name'] == 'transformer_new':
            params = c['params']
            modules = TransformerNew(output_dims[-1], params)
            filters = modules.filter

        # se层
        elif c['name'] == 'se':
            params = c['params']
            modules = SEBlock(output_dims[-1], params)
            filters = modules.filter

        # sam层
        elif c['name'] == 'multisam':
            params = c['params']
            modules = MultiSam(output_dims[-1], params)
            filters = modules.filter

        # spp层
        elif c['name'] == 'spp':
            params = c['params']
            modules = Spp(output_dims[-1], params)
            filters = modules.filter

        # rfb层
        elif c['name'] == 'rfb':
            params = c['params']
            flag = c['flag']
            modules = RFB(output_dims[-1], params, flag)
            filters = modules.filter

        # csp层
        elif c['name'] == 'csp':
            convs = c['convs']
            block = ""
            if 'block' in c:
                block = c['block']
            seq = c['seq']
            super9 = 0
            if 'super9' in c:
                super9 = c['super9']
            super_add = 0
            if 'super_add' in c:
                super_add = c['super_add']
            is_catt = 0
            if 'is_catt' in c:
                is_catt = c['is_catt']
            modules = Csp(output_dims[-1], convs, block, seq, drop_prob, block_size, super9, super_add, is_catt)
            filters = modules.filter

        # fpn增强层
        elif c['name'] == 'AugFPN':
            params = c['params']
            ratio = c['ratio']
            is_Aug = c['is_Aug']
            modules = AugFPN(output_dims[-1], params, ratio, is_Aug)
            filters = modules.filter

        # 上采样层
        elif c['name'] == 'upsample':
            modules = nn.Upsample(scale_factor=c['stride'])

        # 输出层
        elif c['name'] == 'out':
            params = c['params']
            modules = Out(output_dims[-1], params)
            # filters = modules.filter
            routs[modules.id] = modules.filter

        # 拼接层
        elif c['name'] == 'cat':
            froms = c['froms']
            modules = Cat(output_dims[-1], routs, froms, drop_prob, block_size)
            filters = modules.filter

        # 连接层
        elif c['name'] == 'get':
            froms = c['froms']
            modules = Get(output_dims[-1], routs, froms)
            filters = modules.filter

        # BiFPN
        elif c['name'] == 'BiFpn':
            out_dim = c['out_dim']
            params = c['params']
            froms = c['froms']
            rout = int(c['rout'])
            bfblock = int(c['block'])
            if bfblock < 1:
                raise ValueError('bfblocks应大于等于1')
            else:
                modules = BiFPN(out_dim, params, froms, bfblock, rout)
                filters = modules.filter

        # SCConv
        elif c['name'] == 'scconv':
            params = c['params']
            modules = SC(output_dims[-1], params)
            filters = modules.filter

        # 匹配检测层
        elif c['name'] == 'yolo':
            mask = c['mask'].split(',')
            mask = [int(a) for a in mask]

            anchors = c['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            # 转换为2维矩阵形式，才能使用mask取
            anchors = np.array([[anchors[i], anchors[i + 1]] for i in range(0, len(anchors), 2)])
            classes = c['classes'] #这里的classes是类别个数，不是具体类型
            modules = YOLOLayer(anchors[mask],classes,img_size,c)  #将yolo网络层整个字典信息传入

        else:
            assert False, "出错：未加入结构" + x['type']  # assert根据计算的表达式结果，进行报错与不报错，这里强制设置为False,所以一定会报错

        module_list.append(modules)  # 将对应（用nn.Sequential）进行包装的网络层结构，加入到列表中
        # 将本次网络结构的输入通道存入列表中，作为下一层网络结构的输入通道
        output_dims.append(filters)  # 每一层的通道数装入到列表中，作为记录列表，用于按索引查找 [32, 64, 32, 64.....128, 256, 255, 255]

    return net_info, module_list

# 利用模型列表创建模型
class YOLOV4(nn.Module):
    def __init__(self, modules_list):
        super().__init__()
        self.modules_list = modules_list

    def forward(self,x):
        yolo_out, yolo_cfg, out = [], [], {}
        for i,module in enumerate(self.modules_list):
            # 获得模型结构层名称
            name = module.__class__.__name__

            # 如果是路由层或残差层特输处理
            if name == 'Cat':
                # 参数需要带上输出值
                x = module(x, out)
            elif name == 'Get':
                # 参数需要带上输出值
                x = module(x, out)
            elif name == 'Out':

                out[module.id] = module(x)   #{'-1':out, '0':out, '1',out}
            elif name == 'BiFPN':
                y = [out, x]
                x = module(y)

            # 如果是yolo层，把最后输出加到yolo_out列表中
            elif name == 'YOLOLayer':

                x = module(x)
                yolo_out.append(x)  #[64x..x25,64x...x25,64x....x25]
                yolo_cfg.append(module.cfg)  # 把3个yololayer对应的字典信息收集起来
            else:
                x = module(x)

        self.cfg = yolo_cfg   # [yolocfg0,yolocfg1,yolocfg2]

        return yolo_out       # [yoloout0,yoloout1,yoloout2]

