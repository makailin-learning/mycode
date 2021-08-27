import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 转成CPU计算
def to_cpu(tensor):
    return tensor.detach().cpu()