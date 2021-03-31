import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
import torch

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_banch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches))) #normal distribution
    nn.init.constant_(conv.bias, 0) #init to a constant

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 9, stride=1):
        super().__init__()
        pad = int((kernel_size-1)/2)
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=(kernel_size,1), padding=(pad,0),stride=(stride,1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)
    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super().__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        #将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        # (net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，所以经过类型
        # 转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)),requires_grad=False)
        nn.init.constant_(self.PA, 1e-6) #用1e-6填充PA张量
        self.A = Variable(torch.from_numpy(A.astype(np.float32)),requires_grad= False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_c = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_c.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x:x
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()
