import torch.nn as nn
from collections import OrderedDict
import torch

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)
                                                                                   # 定义 hwish  jiade
'''class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return x * (self.relu6(x+3)) / 6'''

def activation(act_type, inplace=False, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        #layer = HardSwish(inplace)                                               #    xiu gai

        layer = nn.LeakyReLU(neg_slope, inplace)

    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class CCALayer(nn.Module):
    def __init__(self, channel):
        super(CCALayer, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_du = nn.Sequential( nn.Conv2d(channel, channel, 1, padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y2 = self.max_pool(x)
        y = self.conv_du(y)
        y2 = self.conv_du(y2)
        return self.sigmoid (y + y2) * x
    
class CCALayer2(nn.Module):
    def __init__(self, channel):
        super(CCALayer2, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_du = nn.Sequential( nn.Conv2d(channel, channel, 1, padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y2 = self.max_pool(x)
        y = self.conv_du(y)
        y2 = self.conv_du(y2)
        return self.sigmoid (y + y2) 
class LCCALayer(nn.Module):
    def __init__(self, channel):
        super(LCCALayer, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.c3 = nn.Conv2d(channel, channel//4, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.c32 = nn.Conv2d(channel//4, channel, kernel_size=3, padding=(3 - 1) // 2 ,bias=False)
        self.act = activation('relu')
       
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.c32(self.c3(y))
       
        return self.sigmoid(y)
 
class HALayer(nn.Module):
    def __init__(self, channel, distillation_rate=0.5):
        super(HALayer, self).__init__()
        self.distilled_channels = int(channel * distillation_rate)

        self.hc1 = nn.Conv2d(channel, self.distilled_channels, 1, bias=True)
        
        self.c3 = nn.Conv2d(self.distilled_channels, self.distilled_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.c1 = nn.Conv2d(self.distilled_channels, self.distilled_channels, kernel_size=1,bias=False)
        self.c32 = nn.Conv2d(self.distilled_channels, self.distilled_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.act = activation('relu')

        self.c3f = nn.Conv2d(channel , channel , kernel_size=3, padding=(3 - 1) // 2, bias=False)

        self.ca = CCALayer2(self.distilled_channels)
    def forward(self, x):
        x_h = self.hc1(x)
        xup = self.c3(x_h) + self.c1(x_h)
        xup2 = self.ca(x_h) * xup
        
        xdo = self.act(self.c32(x_h))
        xf = torch.cat([xup , xdo], dim=1)
        xf = self.act(self.c3f(xf))
        
        return xf


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate1=0.25,distillation_rate2=0.5):
        super(IMDModule, self).__init__()
        self.distilled_channels1 = int(in_channels * distillation_rate1)  # 0.25
        self.ha = HALayer(in_channels)
        self.hac1 = conv_layer(in_channels, self.distilled_channels1, 1)

        self.distilled_channels2 = int(in_channels * distillation_rate2)  # 0.5
        self.ha1c1 = conv_layer(in_channels, self.distilled_channels2, 1)
        self.ha1 = HALayer(self.distilled_channels2)

        self.ha2c1 = conv_layer(self.distilled_channels2, self.distilled_channels1, 1)
        self.ha2 = HALayer(self.distilled_channels1)
            
        self.act = activation('relu')
        self.c3 = conv_layer(in_channels, in_channels, 3)
        self.ca = CCALayer(in_channels)

    def forward(self, input):
        outh = self.ha(input)
        outh_d = self.hac1(outh)
        
        outh1 = self.ha1c1(outh)
        outh1_d = self.ha1(outh1)
        
        outh2 = self.ha2c1(outh1_d)
        outh2_d = self.ha2(outh2)

        f_out = torch.cat([outh_d, outh1_d, outh2_d], dim=1)
        f_out = self.c3(f_out)
        f_out = self.ca(f_out) + input 
       
        return f_out
        
        
        
        
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)
