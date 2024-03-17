"""
    Basic Rep block structure, code from https://github.com/DingXiaoH/DiverseBranchBlock
    
    Reference:
    Diverse branch block: Building a convolution as an inception-like unit. 
    Xiaohan Ding, Xiangyu Zhang, Jungong Han, and Guiguang Ding. 
    In Proceedings ofthe IEEE/CVF Con ference on Computer Vision and Pattern Recognition, pages 10886-10895, 2021
"""

import math
import torch
import torch.nn as nn
from net.dbb_series.dbb_transforms import *

def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=False, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se


class RepBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, nonlinear=None, single_init=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        # if deploy:
        self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, groups=groups, bias=True)
        self.dbb_reparam.requires_grad_(False)
        # else:
        self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
        if groups < out_channels:
            self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                    padding=0, groups=groups)

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()
        
        self.attach_rate = 1.
        # self.only_raw = False

    def inversion(self, reset_bn=False):
        """
            inverse the fused_conv into 3x3_origin
        """
        self.deploy = False
        if reset_bn:
            self.dbb_origin.bn.running_mean.zero_()
            self.dbb_origin.bn.running_var.fill_(1.0)
            self.dbb_origin.bn.weight.data.fill_(1.0)
            self.dbb_origin.bn.bias.data = self.dbb_reparam.bias.data.clone().detach()
            self.dbb_origin.conv.weight.data = self.dbb_reparam.weight.data.clone().detach()
        else:
            # Keep running statistics of BN layer
            self.dbb_origin.bn.weight.data.fill_(1.0)
            std = (self.dbb_origin.bn.running_var + self.dbb_origin.bn.eps).sqrt().clone().detach()
            t = (self.dbb_origin.bn.weight / std).clone().detach().reshape(-1, 1, 1, 1)
            self.dbb_origin.conv.weight.data = (self.dbb_reparam.weight.data / t).clone().detach()
            # # _origin_bn_bias_data = self.bn.bias.data.clone().detach()
            self.dbb_origin.bn.bias.data = (self.dbb_reparam.bias.data + self.dbb_origin.bn.running_mean * self.dbb_origin.bn.weight / std).clone().detach()
            # # return torch.sum(torch.abs(_origin_bn_bias_data - self.bn.bias.data))

    def init_branch_weights(self):
        """
            init other branches after inversion.
        """
        if hasattr(self, "dbb_1x1"):
            n1 = self.dbb_1x1.conv.weight.size(0)
            init_range1 = .1 / math.sqrt(n1)
            self.dbb_1x1.conv.weight.data.uniform_(-init_range1, init_range1)
            self.dbb_1x1.bn.weight.data.fill_(1.0)
            self.dbb_1x1.bn.bias.data.zero_()
    
    def set_attach_rate(self, value):
        self.attach_rate = value
    

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0
        return transII_addbranch((k_origin, k_1x1), (b_origin, b_1x1))

    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias

    def forward(self, inputs):
        # if hasattr(self, 'dbb_reparam'):
        if self.deploy:
            return self.nonlinear(self.dbb_reparam(inputs))
        else:
            if hasattr(self, 'dbb_1x1'):
                return self.nonlinear(self.dbb_origin(inputs) + self.attach_rate * self.dbb_1x1(inputs))
            else:
                return self.nonlinear(self.dbb_origin(inputs))

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
    
    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


if __name__ == "__main__":
    net = RepBlock(3,6,3,1,1)
    net.eval()
    # net.turn_only_raw(True)
    
    x = torch.randn(1,3,4,4)
    print(net(x)[0][0])
    
    net.switch_to_deploy()
    print(net(x)[0][0])
    
    net.inversion()
    net.set_attach_rate(0.)
    print(net(x)[0][0])
