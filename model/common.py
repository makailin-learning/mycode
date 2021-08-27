import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import time

# Mish激活函数
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

# 丢弃块
class DropBlock2D(nn.Module):

    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size**2)

######### 新版网络结构
# 卷积层
# conv params = filters, kernel_size, stride, pad, batch_normalize, activation
class Conv(nn.Module):
    def __init__(self, in_channels, params, is_conv=True):
        super(Conv, self).__init__()
        params = params.split(",")
        self.filter = int(params[0])
        size = int(params[1])
        stride = int(params[2])
        pad = int(params[3])
        batch_normalize = int(params[4])
        activation = params[5]
        self.conv = nn.Identity()
        if is_conv:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=self.filter,
                                  kernel_size=size,
                                  stride=stride,
                                  padding=size // 2 if pad else 0,
                                  bias=not batch_normalize)

        self.bn = nn.Identity()
        if batch_normalize:
            self.bn = nn.BatchNorm2d(self.filter)

        self.ac = nn.Identity()
        # leaky relu激活函数
        if activation == 'leaky':
            self.ac = nn.LeakyReLU(0.1, inplace=True)

        # Mish激活函数
        elif activation == 'mish':
            self.ac = Mish()

        # 逻辑激活函数
        elif activation == 'logistic':
            self.ac = nn.Sigmoid()

        elif activation == 'linear' or activation == 'none':
            pass

        else:
            assert False, "未匹配的激活函数" + activation

    def forward(self, x):
        #self.compute_time()
        #self.compute_time(False)
        return self.ac(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.ac(self.conv(x))

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"Conv time:{time.time() - self.time}")


# detect 层
class Detect(nn.Module):
    def __init__(self, in_channels, params):
        super(Detect, self).__init__()
        self.res = Conv(in_channels, params)
        self.filter = self.res.filter

    def forward(self, x):
        return self.res(x)

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"detect time:{time.time() - self.time}")

# 残差层
# conv params = filters, kernel_size, stride, pad, batch_normalize, activation
class Res(nn.Module):
    def __init__(self, in_channels, params, drop_prob, block_size):
        super(Res, self).__init__()
        params = params.split("/")
        #self.filter = int(params[0])
        self.res = nn.Sequential()
        filter = in_channels
        for i, p in enumerate(params):
            self.res.add_module(f"Conv2d_{i}", Conv(filter, p))
            filter = self.res[-1].filter
        self.filter = filter

        self.drop_block = DropBlock2D(drop_prob, block_size)

    def forward(self, x):
        return self.drop_block(x + self.res(x))

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"res time:{time.time() - self.time}")

# spp层
class Spp(nn.Module):
    def __init__(self, in_channels, params):
        super(Spp, self).__init__()
        params = params.split("|")
        #self.filter = int(params[0])
        self.res = nn.ModuleList()
        for p in params:
            p = p.split(",")
            stride = int(p[0])
            size = int(p[1])
            self.res.append(nn.MaxPool2d(kernel_size=size, stride=stride, padding=size // 2))

        self.filter = in_channels * (len(params) + 1)

    def forward(self, x):
        return torch.cat([x] + [module(x) for module in self.res], 1)

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"spp time:{time.time() - self.time}")

# Csp层
# conv params = filters, kernel_size, stride, pad, batch_normalize, activation
class Csp(nn.Module):
    def __init__(self, in_channels, convs, block, seq, drop_prob, block_size, super9=0, super_add=0, is_catt=0):
        super(Csp, self).__init__()
        self.super9 = super9
        self.super9flag = False
        # 一定要注意传入的是str '1' 还是int '1',查看变量类型: type(变量)
        self.super_add = int(super_add)
        self.catt = int(is_catt)

        convs = convs.split("/")
        if block != "":
            block = block.split(":")
        seq = seq.split("/")

        self.res = nn.ModuleList()
        filter = in_channels
        out_dims = {}
        # is_number
        for s in seq:
            if "|" in s:
                s_1 = s.split("|")
                self.res.append(Conv(filter, convs[int(s_1[1])]))
                if self.catt:
                    self.branch=CAtt(filter, convs[int(s_1[0])], 32)
                else:
                    self.branch = Conv(filter, convs[int(s_1[0])])
                out_dims["0"] = self.branch.filter

            elif "block" in s:
                s_2 = s.split("*")
                if int(s_2[1]) == 9:
                    self.super9flag = True
                if block[0] == "res":
                    for i in range(int(s_2[1])):
                        self.res.append(Res(filter, block[1], drop_prob, block_size))
                        filter = self.res[-1].filter
                elif block[0] == "rep":
                    for i in range(int(s_2[1])):
                        self.res.append(Rep(filter, block[1], drop_prob, block_size))
                        filter = self.res[-1].filter
                elif block[0] == "resrep":
                    for i in range(int(s_2[1])):
                        self.res.append(ResRep(filter, block[1], drop_prob, block_size))
                        filter = self.res[-1].filter
                elif block[0] == "eightrep":
                    for i in range(int(s_2[1])):
                        self.res.append(EightRep(filter, block[1], drop_prob, block_size))
                        filter = self.res[-1].filter
                elif block[0] == "transformer":
                    for i in range(int(s_2[1])):
                        self.res.append(Transformer(filter, block[1]))
                        filter = self.res[-1].filter
                elif block[0] == "transformer_new":
                    for i in range(int(s_2[1])):
                        self.res.append(TransformerNew(filter, block[1]))
                        filter = self.res[-1].filter
                elif block[0] == "se":
                    for i in range(int(s_2[1])):
                        self.res.append(SEBlock(filter, block[1]))
                        filter = self.res[-1].filter
                elif block[0] == "multisam":
                    for i in range(int(s_2[1])):
                        self.res.append(MultiSam(filter, block[1]))
                        filter = self.res[-1].filter
                elif block[0] == "spp":
                    self.res.append(Spp(filter, block[1]))
                else:
                    assert False, "出错：未定义的结构" + block[0]

            elif "cat" in s:
                self.res.append(Cat(filter, out_dims, "0", drop_prob, block_size))
            else:
                s_3 = int(s)
                self.res.append(Conv(filter, convs[s_3]))

            filter = self.res[-1].filter

        self.filter = filter
        self.ac = Mish()

    def forward(self, x):

        if self.super_add:
            c0=x.clone()

        flag = False
        if self.super9 and self.super9flag:
            flag = True

        c1 = self.branch(x)

        for i, module in enumerate(self.res):
            name = module.__class__.__name__
            if name == 'Cat':
                x = module(x, {"0": c1})

                if self.super_add:

                    x+=c0
            else:
                x = module(x)

            if flag:
                if i == 3 and flag:
                    c1 = c1 + x
                if i == 6 and flag:
                    c1 = self.ac(c1 + x)

        return x

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"csp time:{time.time() - self.time}")

# 输出层
class Out(nn.Module):
    def __init__(self, in_channels, params):
        super(Out, self).__init__()

        self.res = nn.Sequential()
        filter = in_channels
        params = params.split(":")
        self.id = params[0]
        if len(params) > 1:
            pa = params[1].split("/")
            for i, p in enumerate(pa):
                self.res.add_module(f"Conv2d_{i}", Conv(filter, p))
                filter = self.res[-1].filter

        self.filter = filter

    def forward(self, x):
        if len(self.res) > 0:
            x = self.res(x)

        return x

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"out time:{time.time() - self.time}")

# 连接层
class Cat(nn.Module):
    def __init__(self, in_channels, out_dims, froms, drop_prob, block_size):
        super(Cat, self).__init__()

        self.froms = froms.split(",")

        if len(self.froms) == 1:
            self.filter = in_channels + out_dims[self.froms[0]]
        else:
            out_filter = 0
            for x in self.froms:
                out_filter += out_dims[x]
            self.filter = out_filter

        self.drop_block = DropBlock2D(drop_prob, block_size)

    def forward(self, x, out):

        if len(self.froms) == 1:
            x = torch.cat([out[self.froms[0]], x], 1)
        else:
            temp = [out[i] for i in self.froms]
            x = torch.cat(temp[::-1], 1)
        return self.drop_block(x)

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"cat time:{time.time() - self.time}")

# 直通层
class Get(nn.Module):
    def __init__(self, in_channels, out_dims, froms):
        super(Get, self).__init__()

        self.froms = froms
        self.filter = out_dims[self.froms[0]]

    def forward(self, x, out):
        return out[self.froms[0]]

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"get time:{time.time() - self.time}")

# Focus 层
class Focus(nn.Module):
    def __init__(self, in_channels, params):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()

        self.conv = Conv(in_channels * 4, params)

        self.filter = self.conv.filter

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)

        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"focus time:{time.time() - self.time}")

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

# repvgg层
# conv params = filters, kernel_size, stride, pad, batch_normalize, activation
class Rep(nn.Module):
    def __init__(self, in_channels, params, drop_prob, block_size):
        super(Rep, self).__init__()
        self.in_channels = in_channels
        params = params.split("|")
        #self.filter = int(params[0])
        self.res = nn.ModuleList()
        filter = in_channels
        for i, p in enumerate(params):
            self.res.append(Conv(filter, p))
            filter = self.res[-1].filter
        self.filter = filter
        self.res.append(nn.BatchNorm2d(num_features=self.filter))
        self.ac = Mish()
        self.drop_block = DropBlock2D(drop_prob, block_size)

        self.deploy = False
        self.res_rep = None

    def forward(self, x):
        if not self.deploy:
            return self.drop_block(self.ac(self.res[0](x) + self.res[1](x) + self.res[2](x)))
        else:
            return self.drop_block(self.ac(self.res_rep(x)))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.res[1])
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.res[0])
        kernelid, biasid = self._fuse_bn_tensor(self.res[2])
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.BatchNorm2d):
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        else:
            return branch.conv.weight.data, branch.conv.bias.data

    def switch_to_deploy(self):
        self.deploy = True
        kernel, bias = self.get_equivalent_kernel_bias()
        self.res_rep = nn.Conv2d(in_channels=self.res[1].conv.in_channels,
                                 out_channels=self.res[1].conv.out_channels,
                                 kernel_size=self.res[1].conv.kernel_size,
                                 stride=self.res[1].conv.stride,
                                 padding=self.res[1].conv.padding,
                                 dilation=self.res[1].conv.dilation,
                                 groups=self.res[1].conv.groups,
                                 bias=True)
        self.res_rep.weight.data = kernel
        self.res_rep.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('res')

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"res time:{time.time() - self.time}")

# transformer 层
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # GELU激活函数
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        # 内部维度 头维度×头数 64×16
        inner_dim = dim_head * heads

        #如果头数1且头维度等于dim，那么输出不做处理，见本方法倒数第1行
        project_out = not (heads == 1 and dim_head == dim)

        #头数
        self.heads = heads
        #缩放比例
        self.scale = dim_head**-0.5

        #softmax激活函数
        self.attend = nn.Softmax(dim=-1)

        #连结后的线性变换 输出维度为inner_dim * 3,因为后续要分为3块，保持维度不变需乘以3
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        #定义输出层
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        #获得batch数、patch数、头数
        b, n, _, h = *x.shape, self.heads

        #按最后一个维度 把x分成3块，qkv为内容为tensor的list 64,1025,256
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # 因为qkv是len等于3的list，每个内容是tensor并进行转换，再赋值给q k v,转换即是最后的维度拆分成头数*d,再进行位置变化
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # q、k张量点积并乘以缩放比例
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        #经过最后一个维度经过softmax
        attn = self.attend(dots)

        #再与v进行张量点积
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        #在进行变换，转换成输入的shape
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    # in_channels, params -> depth, heads, dim_head, mlp_dim, dropout = 0.
    def __init__(self, in_channels, params):
        super().__init__()
        params = params.split(",")
        self.depth = int(params[0])
        heads = int(params[1])
        dim_head = int(params[2])
        mlp_dim = int(params[3])
        dropout = float(params[4])
        #self.filter = int(params[0])
        #先层归一，再进入多头注意力msa
        self.msa = nn.Sequential(nn.LayerNorm(in_channels), Attention(in_channels, heads=heads, dim_head=dim_head, dropout=dropout))
        #先层归一，再进入mlp
        self.mlp = nn.Sequential(nn.LayerNorm(in_channels), FeedForward(in_channels, mlp_dim, dropout=dropout))
        self.filter = in_channels

    def forward(self, x):
        # x reshape
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        for _ in range(self.depth):
            #输入x与经过msa多头注意力相加，类似残差
            x = self.msa(x) + x
            #输入x与经过mlp相加，类似残差
            x = self.mlp(x) + x

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"res time:{time.time() - self.time}")

# transformer new 层
class FeedForwardConv(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., dropsize=7):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=hidden_dim, kernel_size=1, stride=1, bias=True), nn.BatchNorm2d(hidden_dim),
                                 Mish(), DropBlock2D(dropout, dropsize),
                                 nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1, stride=1, bias=True), nn.BatchNorm2d(dim), Mish(),
                                 DropBlock2D(dropout, dropsize))

    def forward(self, x):
        return self.net(x)

class AttentionConv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., dropsize=7):
        super().__init__()
        # 内部维度 头维度×头数 64×16
        inner_dim = dim_head * heads

        #如果头数1且头维度等于dim，那么输出不做处理，见本方法倒数第1行
        project_out = not (heads == 1 and dim_head == dim)

        #头数
        self.heads = heads
        #缩放比例
        self.scale = dim_head**-0.5

        #softmax激活函数
        self.attend = nn.Softmax(dim=-1)

        #连结后的线性变换 输出维度为inner_dim * 3,因为后续要分为3块，保持维度不变需乘以3
        #self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=inner_dim * 3, kernel_size=1, stride=1, bias=False),
                                    DropBlock2D(dropout, dropsize))

        #定义输出层
        self.to_out = nn.Sequential(nn.Conv2d(in_channels=inner_dim, out_channels=dim, kernel_size=1, stride=1, bias=False),
                                    DropBlock2D(dropout, dropsize)) if project_out else nn.Identity()

    def forward(self, x):
        #获得batch数、patch数、头数
        hs, ws = x.size()[2:]
        h = self.heads

        #按最后一个维度 把x分成3块，qkv为内容为tensor的list 64,512,64,64
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # 因为qkv是len等于3的list，每个内容是tensor并进行转换，再赋值给q k v,转换即是最后的维度拆分成头数*d,再进行位置变化
        #q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) hs ws -> b h (hs ws) d', h=h), qkv)

        # q、k张量点积并乘以缩放比例
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        #经过最后一个维度经过softmax
        attn = self.attend(dots)

        #再与v进行张量点积
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        #在进行变换，转换成输入的shape
        out = rearrange(out, 'b h (hs ws) d -> b (h d) hs ws', hs=hs, ws=ws)
        return self.to_out(out)

class TransformerConv(nn.Module):
    # in_channels, params -> depth, heads, dim_head, mlp_dim, dropout = 0.
    def __init__(self, in_channels, params):
        super().__init__()
        params = params.split(",")
        self.depth = int(params[0])
        heads = int(params[1])
        dim_head = int(params[2])
        mlp_dim = int(params[3])
        dropout = float(params[4])
        dropsize = int(params[5])
        #self.filter = int(params[0])
        #先层归一，再进入多头注意力msa
        self.ln = nn.LayerNorm(in_channels)
        self.flag = True
        self.msa = AttentionConv(in_channels, heads=heads, dim_head=dim_head, dropout=dropout, dropsize=dropsize)
        #先层归一，再进入mlp
        self.mlp = FeedForwardConv(in_channels, mlp_dim, dropout=dropout, dropsize=dropsize)
        self.filter = in_channels

    def forward(self, x):
        # x reshape
        b, c, h, w = x.size()
        for _ in range(self.depth):
            #输入x与经过msa多头注意力相加，类似残差
            x_ln = rearrange(x, 'b c h w -> b h w c')
            x_ln = self.ln(x_ln)
            x_ln = rearrange(x_ln, 'b h w c-> b c h w')
            x = self.msa(x_ln) + x
            #输入x与经过mlp相加，类似残差
            x_ln = rearrange(x, 'b c h w -> b h w c')
            x_ln = self.ln(x_ln)
            x_ln = rearrange(x_ln, 'b h w c-> b c h w')
            x = self.mlp(x_ln) + x

        #x = rearrange(x, 'b h w c -> b c h w')
        return x

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"res time:{time.time() - self.time}")

class TransformerNew(nn.Module):
    # in_channels, params -> depth, heads, dim_head, mlp_dim, dropout = 0.
    def __init__(self, in_channels, params):
        super().__init__()
        params = params.split(",")
        self.depth = int(params[0])
        self.heads = int(params[1])
        dim_scale = int(params[4])

        dim = in_channels // dim_scale
        self.patch = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(dim))

        dropout = float(params[2])
        dropsize = int(params[3])
        #self.filter = int(params[0])

        inner_dim = self.heads * in_channels // 2
        self.scale = (in_channels // 2)**-0.5
        #先层归一，再进入多头注意力msa
        self.to_qkv = nn.Conv2d(in_channels=dim, out_channels=inner_dim * 3, kernel_size=1, stride=1, bias=False)

        #定义输出层
        self.to_out = nn.Sequential(nn.Conv2d(in_channels=inner_dim, out_channels=in_channels, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(in_channels), Mish(), DropBlock2D(dropout, dropsize))

        self.attend = nn.Softmax(dim=1)
        self.filter = in_channels

    def forward(self, x):
        hs, ws = x.size()[2:]
        h = self.heads
        for _ in range(self.depth):
            #按最后一个维度 把x分成3块，qkv为内容为tensor的list 64,512,64,64
            qkv = self.to_qkv(self.patch(x)).chunk(3, dim=1)

            # 因为qkv是len等于3的list，每个内容是tensor并进行转换，再赋值给q k v,转换即是最后的维度拆分成头数*d,再进行位置变化
            q, k, v = map(lambda t: rearrange(t, 'b (h d) hs ws -> b h (hs ws) d', h=h), qkv)

            # q、k张量点积并乘以缩放比例
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

            #经过最后一个维度经过softmax
            attn = self.attend(dots)

            #再与v进行张量点积
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            #在进行变换，转换成输入的shape
            out = rearrange(out, 'b h (hs ws) d -> b (h d) hs ws', hs=hs, ws=ws)
            x = self.to_out(out) + x

        #x = rearrange(x, 'b h w c -> b c h w')
        return x

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"res time:{time.time() - self.time}")

# senet 注意力
class SEBlock(nn.Module):
    def __init__(self, in_channels, params):
        super(SEBlock, self).__init__()
        params = params.split(",")
        internal_neurons = in_channels // int(params[0])
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)

        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=in_channels, kernel_size=1, stride=1, bias=True)
        self.filter = in_channels

    def forward(self, inputs):
        kernel_size = inputs.size(3)
        if not isinstance(kernel_size, int):
            kernel_size = kernel_size.item()
        x = F.avg_pool2d(inputs, kernel_size=kernel_size)
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.filter, 1, 1)
        return inputs * x

# repvgg层
# conv params = filters, kernel_size, stride, pad, batch_normalize, activation
class ResRep(nn.Module):
    def __init__(self, in_channels, params, drop_prob, block_size):
        super(ResRep, self).__init__()
        params = params.split("/")
        #self.filter = int(params[0])
        self.res = nn.Sequential()
        filter = in_channels
        self.res.add_module(f"Conv", Conv(filter, params[0]))
        filter = self.res[-1].filter
        for i, p in enumerate(params[1:]):
            self.res.add_module(f"Rep_{i}", Rep(filter, p, drop_prob, block_size))
            filter = self.res[-1].filter
        self.filter = filter

        self.drop_block = DropBlock2D(drop_prob, block_size)

    def forward(self, x):
        return self.drop_block(x + self.res(x))

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"res time:{time.time() - self.time}")


class SingleRep(nn.Module):
    def __init__(self, in_channels, params, drop_prob, block_size):
        super(SingleRep, self).__init__()
        self.in_channels = in_channels
        #params = params.split("|")
        #self.filter = int(params[0])
        self.res = nn.ModuleList()
        self.res.append(Conv(in_channels, params))
        filter = self.res[-1].filter
        self.filter = filter
        self.res.append(nn.BatchNorm2d(num_features=self.filter))
        self.ac = Mish()
        self.drop_block = DropBlock2D(drop_prob, block_size)

        self.deploy = False
        self.res_rep = None

    def forward(self, x):
        if not self.deploy:
            return self.drop_block(self.ac(self.res[0](x) + self.res[1](x)))
        else:
            return self.drop_block(self.ac(self.res_rep(x)))

    def get_equivalent_kernel_bias(self):
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.res[0])
        kernelid, biasid = self._fuse_bn_tensor(self.res[1])
        return kernel1x1 + kernelid, bias1x1 + biasid

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.BatchNorm2d):
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels
                kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 0, 0] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        else:
            return branch.conv.weight.data, branch.conv.bias.data

    def switch_to_deploy(self):
        self.deploy = True
        kernel, bias = self.get_equivalent_kernel_bias()
        self.res_rep = nn.Conv2d(in_channels=self.res[0].conv.in_channels,
                                 out_channels=self.res[0].conv.out_channels,
                                 kernel_size=self.res[0].conv.kernel_size,
                                 stride=self.res[0].conv.stride,
                                 padding=self.res[0].conv.padding,
                                 dilation=self.res[0].conv.dilation,
                                 groups=self.res[0].conv.groups,
                                 bias=True)
        self.res_rep.weight.data = kernel
        self.res_rep.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('res')

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"res time:{time.time() - self.time}")


class EightRep(nn.Module):
    def __init__(self, in_channels, params, drop_prob, block_size):
        super(EightRep, self).__init__()
        params = params.split("/")
        #self.filter = int(params[0])
        self.res = nn.Sequential()
        filter = in_channels
        self.res.add_module("SingleRep", SingleRep(filter, params[0], drop_prob, block_size))
        filter = self.res[-1].filter
        for i, p in enumerate(params[1:]):
            self.res.add_module(f"Rep_{i}", Rep(filter, p, drop_prob, block_size))
            filter = self.res[-1].filter
        self.filter = filter
        self.ac = Mish()

        self.drop_block = DropBlock2D(drop_prob, block_size)

    def forward(self, x):
        return self.drop_block(self.ac(x + self.res(x)))

    def compute_time(self, st=True):
        if not self.training:
            if st:
                self.time = time.time()
            else:
                print(f"res time:{time.time() - self.time}")


class MultiSam(nn.Module):
    def __init__(self, in_channels, params):
        super(MultiSam, self).__init__()
        params = params.split(",")
        internal_neurons = in_channels // int(params[0])
        self.patchs = int(params[1])

        self.is_channel = int(params[2])

        self.is_spatial = int(params[3])

        kernel_size = int(params[4])

        self.down = nn.Conv2d(in_channels=in_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)

        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=in_channels, kernel_size=1, stride=1, bias=True)

        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels=self.patchs * 2, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2, stride=1, bias=True),
            nn.BatchNorm2d(1))

        self.filter = in_channels

    def forward(self, x):
        if self.is_channel:
            kernel_size = x.size(3)
            if not isinstance(kernel_size, int):
                kernel_size = kernel_size.item()

            kernel_size = kernel_size // self.patchs

            x1 = F.avg_pool2d(x, kernel_size=kernel_size)
            x1 = self.down(x1)
            x1 = F.relu(x1)
            x1 = self.up(x1)
            '''
            x2 = F.max_pool2d(x, kernel_size=kernel_size)
            x2 = self.down(x2)
            x2 = F.relu(x2)
            x2 = self.up(x2)'''

            out1 = torch.sigmoid(x1)  #+ x2)
            out1 = F.max_pool2d(out1, kernel_size=self.patchs)
            #out1 = out1.view(-1, self.filter, 1, 1)
            x = x * out1

        if self.is_spatial:
            x3 = x.chunk(self.patchs, dim=1)

            x_list = []
            for temp in x3:
                x_list.append(torch.max(temp, 1)[0].unsqueeze(1))
                x_list.append(torch.mean(temp, 1).unsqueeze(1))
            x3 = torch.cat(x_list, dim=1)

            x3 = torch.sigmoid(self.to_out(x3))

            x = x * x3
        return x


class AugFPN(nn.Module):
    def __init__(self, in_ch, params,ratio,is_Aug):
        super().__init__()
        ratio=ratio.split(',')
        ratio=[float(a) for a in ratio]   #[0.1, 0.2, 0.3]
        self.in_channels = in_ch
        self.out_channels = int(params)
        self.filter = self.out_channels
        self.is_Aug=False
        self.Conv=nn.Sequential(nn.Conv2d(self.in_channels,self.out_channels,1),
                                nn.BatchNorm2d(self.filter),
                                Mish())
        self.adaptive_pool_output_ratio = ratio
        self.high_lateral_conv = nn.Conv2d(self.in_channels, self.out_channels, 1)
        self.high_lateral_conv_attention = nn.Sequential(
            nn.Conv2d(self.out_channels * (len(self.adaptive_pool_output_ratio)), self.out_channels, 1), # 1x1卷积降维
            nn.ReLU(),
            nn.Conv2d(self.out_channels, len(self.adaptive_pool_output_ratio), 3, padding=1)) #3x3卷积降维
        if is_Aug:
            self.is_Aug=True

    def forward(self,x):
        y=self.Conv(x)  # y为FPN结构中x经过1x1通道降维后的结果，即x->(1x1conv)->y
        if self.is_Aug:
            h = x.size(2)
            w = x.size(3)
            AdapPool_Feature_maps=[]
            for i in range(len(self.adaptive_pool_output_ratio)):
                AdapPool_Features=F.adaptive_avg_pool2d(x,output_size=(max(1, int(h * self.adaptive_pool_output_ratio[i])),
                                                           max(1, int(w * self.adaptive_pool_output_ratio[i]))))
                AdapPool_Features=self.high_lateral_conv(AdapPool_Features)
                AdapPool_Features = F.upsample(AdapPool_Features, size=(h, w), mode='bilinear', align_corners=True)
                AdapPool_Feature_maps.append(AdapPool_Features)  # 生成不同尺度的特征图

            out = self.ASF(AdapPool_Feature_maps)
            return out+y
        else:
            return y

    def ASF(self,x):
        Concat_AdapPool_Features = torch.cat(x, dim=1)  # 拼接三个特征图
        fusion_weights = self.high_lateral_conv_attention(Concat_AdapPool_Features)
        fusion_weights = F.sigmoid(fusion_weights)  # Bx3xHxW 特征权重张量
        high_pool_fusion = 0
        for i in range(len(self.adaptive_pool_output_ratio)):
            high_pool_fusion += fusion_weights[:, i:i+1, :, :] * x[i] # 特征图权重融合

        return high_pool_fusion



class BasicConv(nn.Module):  # RFB的适配卷积

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, mish=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.mish = Mish() if mish else None  # 将relu替换为mish

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.mish is not None:
            x = self.mish(x)
        return x

class RFB(nn.Module):

    def __init__(self, in_ch, params, flag, scale=0.1):
        super(RFB, self).__init__()
        self.in_planes = in_ch
        self.out_planes = int(params)
        self.flag = flag  # 0-选择普通型，1-选择进阶型
        self.scale = scale
        self.filter = self.out_planes

        if not self.flag:  # 选择RFB普通模型
            inter_planes = self.in_planes // 8

            # 分支0，1x1卷积+3x3虫洞卷积,padding和dilation相同，才能保证输出size=输入size
            self.branch0 = nn.Sequential(
                BasicConv(self.in_planes, 2 * inter_planes, kernel_size=1, stride=1),  # 1x1卷积降维
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, dilation=1,
                          mish=False)
            )
            # 分支1，1x1卷积+3x3卷积+3x3虫洞卷积
            self.branch1 = nn.Sequential(
                BasicConv(self.in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3, dilation=3,
                          mish=False)
            )
            # 分支2，1x1卷积+3x3卷积+3x3卷积+3x3虫洞卷积
            self.branch2 = nn.Sequential(
                BasicConv(self.in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=1, padding=1),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5,
                          mish=False)
            )

            self.ConvLinear = BasicConv(6 * inter_planes, self.out_planes, kernel_size=1, stride=1, mish=False)

        else:  # 选择RFB_s模型
            inter_planes = self.in_planes // 4

            # 分支0，1x1卷积+3x3虫洞卷积
            self.branch0 = nn.Sequential(
                BasicConv(self.in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, mish=False)
            )
            # 分支1，1x1卷积+3x1卷积+3x3虫洞卷积
            self.branch1 = nn.Sequential(
                BasicConv(self.in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # 3x1卷积核
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, mish=False)
            )
            # 分支2，1x1卷积+1x3卷积+3x3虫洞卷积
            self.branch2 = nn.Sequential(
                BasicConv(self.in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=1, padding=(0, 1)),  # 1x3卷积核
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, mish=False)
            )
            # 分支3，1x1卷积+3x1卷积+1x3卷积+3x3虫洞卷积
            self.branch3 = nn.Sequential(
                BasicConv(self.in_planes, inter_planes // 2, kernel_size=1, stride=1),
                BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                # 1x3卷积核
                BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
                # 3x1卷积核
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, mish=False)
            )

            self.ConvLinear = BasicConv(4 * inter_planes, self.out_planes, kernel_size=1, stride=1, mish=False)

        if self.in_planes == self.out_planes:
            self.identity = True
        else:
            self.identity = False
            self.shortcut = BasicConv(self.in_planes, self.out_planes, kernel_size=1, stride=1, mish=False)

        self.mish = Mish()

    def forward(self, x):

        if not self.flag:
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            # 沿通道方向拼接concat
            out = torch.cat((x0, x1, x2), 1)
        else:
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            out = torch.cat((x0, x1, x2, x3), 1)

        # concat后过1x1降维卷积
        out = self.ConvLinear(out)
        if self.identity:
            out = out * self.scale + x
        else:
            short = self.shortcut(x)
            out = out * self.scale + short
        out = self.mish(out)

        return out

class SeparableConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm=True, activation=True):
        super(SeparableConvBlock, self).__init__()

        # 逐层卷积要求：in_ch=out_ch=group，in和out相同，结果的通道数就维持不变
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                        stride=1, padding=1, groups=in_channels, bias=False)

        # 逐点卷积为普通的1*1卷积
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm  # norm为True那么就是BN操作，BN中的输入channel就是PW卷积的输出channel
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation  # 如果activation为true就是用Mish激活函数
        if self.activation:
            self.mish = Mish()

    # 可分离卷积前向传播过程：先进行DW卷积，在进行PW卷积
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.mish(x)

        return x


class BiFPN(nn.Module):

    # params传入3个特征图的通道数
    def __init__(self, num_channels, params, froms, bfblock, rout,epsilon=1e-4, attention=True, is_cuda=False):

        super(BiFPN, self).__init__()
        self.epsilon = epsilon  # 防止分母为0的微小数
        num_channels = int(num_channels)
        params = params.split(',')
        params = [int(a) for a in params]
        self.bfblock = bfblock.split(',')  #模块循环次数
        self.rout = rout #输出分支数
        self.froms = froms.split(',')   # 字典的键为字符‘-1’，‘0’，‘1’,不是整形-1，0，1
        self.filter = num_channels

        # 融合过后的可分离卷积层，融合前后的通道数不变
        self.conv_m3 = SeparableConvBlock(num_channels, num_channels)
        self.conv_m4 = SeparableConvBlock(num_channels, num_channels)
        self.conv_p2 = SeparableConvBlock(num_channels, num_channels)
        self.conv_p3 = SeparableConvBlock(num_channels, num_channels)
        self.conv_p4 = SeparableConvBlock(num_channels, num_channels)
        self.conv_p5 = SeparableConvBlock(num_channels, num_channels)

        # 以下是3个上采样操作，对应c5in-p3out的上采样，使用最邻近插值
        self.c5tom4_upsample = nn.Upsample(scale_factor=2, mode='nearest')  # c5到m4的上采样
        self.m4tom3_upsample = nn.Upsample(scale_factor=2, mode='nearest')  # m4到m3的上采样
        self.m3top2_upsample = nn.Upsample(scale_factor=2, mode='nearest')  # m3到p2的上采样

        # 以下是3个下采样操作，对应p3out-p5out的下采样，使用最大池化，池化窗口3*3，步长2*2
        self.p2top3_downsample = nn.MaxPool2d(3, 2, 1)  # p3到p4的下采样
        self.p3top4_downsample = nn.MaxPool2d(3, 2, 1)  # p3到p4的下采样
        self.p4top5_downsample = nn.MaxPool2d(3, 2, 1)  # p4到p5的下采样

        self.mish = Mish()

        # 如果是第一个bifpn模块，将c3,c4,c5三个特征图分别做三次1x1卷积降维+BN操作得到c3in，c4in，c5in
        # 将4个特征图同时降维到统一维度，再进行后续操作，而不是在后续操作的同时进行降维
        # 1x1024x13x13->1x256x13x13
        self.c5_down_channel = nn.Sequential(
            nn.Conv2d(params[3], num_channels, kernel_size=1, bias=False),  # 512x52x52->75x52x52
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
        )
        # 1x512x26x26->1x256x26x26
        self.c4_down_channel = nn.Sequential(
            nn.Conv2d(params[2], num_channels, kernel_size=1, bias=False),  # 256x26x26->75x26x26
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
        )
        # 1x256x52x52->1x256x52x52
        self.c3_down_channel = nn.Sequential(
            nn.Conv2d(params[1], num_channels, kernel_size=1, bias=False),  # 128x13x13->75x13x13
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
        )
        # 1x128x104x104->1x256x104x104
        self.c2_down_channel = nn.Sequential(
            nn.Conv2d(params[0], num_channels, kernel_size=1, bias=False),  # 128x13x13->75x13x13
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
        )

        # Weight,m3、m4、p2、p5的融合分支为2个,都乘以对应的权重参数后再进行融合,初始化权重为两个1
        # 并且每个权值都需要经过一次Relu，保证每个权值都≥0
        self.m3_w = nn.Parameter(torch.ones(2, dtype=torch.float32.cuda() if is_cuda else torch.float32),
                                 requires_grad=True)
        self.m3_w_relu = nn.ReLU()
        self.m4_w = nn.Parameter(torch.ones(2, dtype=torch.float32.cuda() if is_cuda else torch.float32),
                                 requires_grad=True)
        self.m4_w_relu = nn.ReLU()
        self.p5_w = nn.Parameter(torch.ones(2, dtype=torch.float32.cuda() if is_cuda else torch.float32),
                                 requires_grad=True)
        self.p5_w_relu = nn.ReLU()
        self.p2_w = nn.Parameter(torch.ones(2, dtype=torch.float32.cuda() if is_cuda else torch.float32),
                                 requires_grad=True)
        self.p2_w_relu = nn.ReLU()

        # 只有p3out、p4out的融合输入为3个分支，初始化为三个1
        self.p3_w = nn.Parameter(torch.ones(3, dtype=torch.float32.cuda() if is_cuda else torch.float32),
                                 requires_grad=True)
        self.p3_w_relu = nn.ReLU()
        self.p4_w = nn.Parameter(torch.ones(3, dtype=torch.float32.cuda() if is_cuda else torch.float32),
                                 requires_grad=True)
        self.p4_w_relu = nn.ReLU()

        self.attention = attention

    def forward(self, x):  # 传入3张不同尺度下的特征图,以列表形式传入[x0,x1,x2]

        # 每组权重初始化为1，每个权重过relu激活函数，保证都大于等于0。attention为Ture就表示使用fast norm fusion机制，否则就不使用。
        # 当使用attention机制，进入_forward_fast_attention函数，判断当前的BiFPN是否是第一次进行BiFPN操作，为True则获取backbone中的3个特征图为c3，c4以及c5
        # c3，c4以及c5进行channel的调整得到c3in-c5in。
        # 如果不是第一次进行BiFPN操作，只需要获取上个BiFPN模块的输出即P3out-P5out 3个特征图

        if self.attention:
            for i in range(self.bfblock):
                x = self._forward_fast_attention(i,x)
        else:
            for i in range(self.bfblock):
                x = self._forward(x)

        if self.rout==1:
            x = x[-1]

        return x

    # 注意力融合机制
    def _forward_fast_attention(self, i, x):
        if i==0:  #说明就是第一次bifpn

            c2 = x[0][self.froms[0]]
            c3 = x[0][self.froms[1]]
            c4 = x[0][self.froms[2]]
            c5 = x[1]

            c2_in = self.c2_down_channel(c2)
            c3_in = self.c3_down_channel(c3)
            c4_in = self.c4_down_channel(c4)
            c5_in = self.c5_down_channel(c5)

        else:

            c2_in, c3_in, c4_in, c5_in = x

        # Weights: c5in + c4in -> m4
        m4_w = self.m4_w_relu(self.m4_w)
        weight = m4_w / (torch.sum(m4_w, dim=0) + self.epsilon)  # 对应快速融合公式

        # 先融合，再卷积，得到m4
        m4 = self.conv_m4(self.mish(weight[0] * c4_in + weight[1] * self.c5tom4_upsample(c5_in)))

        # Weights: c3in + m4 -> m3
        m3_w = self.m3_w_relu(self.m3_w)
        weight = m3_w / (torch.sum(m3_w, dim=0) + self.epsilon)

        # 先融合，再卷积，得到m3
        m3 = self.conv_m3(self.mish(weight[0] * c3_in + weight[1] * self.m4tom3_upsample(m4)))

        # Weights: c2in + m3 -> p2out
        p2_w = self.p2_w_relu(self.p2_w)
        weight = p2_w / (torch.sum(p2_w, dim=0) + self.epsilon)

        # 先融合，再卷积，得到p3out
        p2_out = self.conv_p2(self.mish(weight[0] * c2_in + weight[1] * self.m3top2_upsample(m3)))

        # Weights: c3in + m4 + m3 -> p3out
        p3_w = self.p3_w_relu(self.p3_w)
        weight = p3_w / (torch.sum(p3_w, dim=0) + self.epsilon)

        # 先融合，再卷积，得到p3out
        p3_out = self.conv_p3(
            self.mish(weight[0] * c3_in + weight[1] * m3 + weight[2] * self.p2top3_downsample(p2_out)))

        # Weights: c4in + m4 + p3 -> p4out
        p4_w = self.p4_w_relu(self.p4_w)
        weight = p4_w / (torch.sum(p4_w, dim=0) + self.epsilon)

        # 先融合，再卷积，得到p4out
        p4_out = self.conv_p4(
            self.mish(weight[0] * c4_in + weight[1] * m4 + weight[2] * self.p3top4_downsample(p3_out)))

        # Weights: c5in + p4 -> p5out
        p5_w = self.p5_w_relu(self.p5_w)
        weight = p5_w / (torch.sum(p5_w, dim=0) + self.epsilon)

        # 先融合，再卷积，得到p5out
        p5_out = self.conv_p5(self.mish(weight[0] * c5_in + weight[1] * self.p4top5_downsample(p4_out)))

        return p2_out, p3_out, p4_out, p5_out

    # 普通融合机制
    def _forward(self, i, x):

        if i==0:  #说明就是第一次bifpn
            c2 = x[0][self.froms[0]]
            c3 = x[0][self.froms[1]]
            c4 = x[0][self.froms[2]]
            c5 = x[1]

            c2_in = self.c2_down_channel(c2)
            c3_in = self.c3_down_channel(c3)
            c4_in = self.c4_down_channel(c4)
            c5_in = self.c5_down_channel(c5)

        else:
            c2_in, c3_in, c4_in, c5_in = x

        # c5in + c4in -> m4
        m4 = self.conv_m4(self.mish(c4_in + self.c5tom4_upsample(c5_in)))

        # c3in + m4 -> m3
        m3 = self.conv_m3(self.mish(c3_in + self.m4tom3_upsample(m4)))

        # c2in + m3 -> p2out
        p2_out = self.conv_p2(self.mish(c2_in + self.m3top2_upsample(m3)))

        # c3in + m4 + m3 -> p3out
        p3_out = self.conv_p3(self.mish(c3_in + m3 + self.p2top3_downsample(p2_out)))

        # c4in + m4 + p3 -> p4out
        p4_out = self.conv_p4(self.mish(c4_in + m4 + self.p3top4_downsample(p3_out)))

        # c5in + p4 -> p5out
        p5_out = self.conv_p5(self.mish(c5_in + self.p4top5_downsample(p4_out)))

        return p2_out, p3_out, p4_out, p5_out


class SCConv(nn.Module):
    def __init__(self, in_ch, out_ch, pooling_r):
        super(SCConv, self).__init__()

        # 平均池化下采样->k2卷积->上采样
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # k3卷积
        self.k3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # k4卷积
        self.k4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        # 上采样，size为指定采样到具体尺寸，scale为采样到当前尺寸的多少倍
        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4
        out = self.relu(out)

        return out

class SC(nn.Module):

    def __init__(self, in_ch, params):
        super(SC, self).__init__()

        params = int(params)
        self.pooling_r = params  # 自校正卷积的下采样倍率
        out_ch = int(in_ch // 2)
        self.filter = in_ch

        # BxCxHxW->BxC/2xHxW  1x1卷积通道降维
        self.x1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU(inplace=True))

        self.x2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU(inplace=True))

        self.k1 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU(inplace=True))

        self.scconv = SCConv(out_ch, out_ch, pooling_r=self.pooling_r)

        self.conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(in_ch))
        self.mish = Mish()

    def forward(self, x):
        residual = x

        x1 = self.x1(x)  # 1x1卷积降维->BN->relu
        x2 = self.x2(x)

        x1 = self.scconv(x1)  # 自校正卷积
        x2 = self.k1(x2)

        out = torch.cat((x1, x2), dim=1)
        out = self.conv(out)
        out += residual
        out = self.mish(out)

        return out


class CAtt(nn.Module):
    def __init__(self, inp, params, reduction=32):
        super(CAtt, self).__init__()
        self.inp = inp
        params = params.split(',')
        self.outp = int(params[0])
        self.filter=self.outp

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # kernelsize=(w,h)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, self.inp // reduction)

        self.conv1 = nn.Conv2d(self.inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = Mish()

        self.conv_h = nn.Conv2d(mip, self.outp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, self.outp, kernel_size=1, stride=1, padding=0)
        self.conv_x = nn.Conv2d(self.inp, self.outp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = self.conv_x(x)

        n, c, h, w = x.size()  # NCHW
        x_h = self.pool_h(x)  # NCH1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # NC1W->#NCW1

        y = torch.cat([x_h, x_w], dim=2)  ##NC(H+W)1
        y = self.conv1(y)  # 降维 C -> C//32,减少计算量
        y = self.bn1(y)
        y = self.act(y)

        # torch.split()作用将tensor分成块结构,将张量沿通道方向分为两份，一份通道数为H,一份为W
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # NCW1->#NC1W

        a_h = self.conv_h(x_h).sigmoid()  # 恢复维度
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out