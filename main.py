import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable

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