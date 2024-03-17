"""
    ConvBn blocks, code from https://github.com/DingXiaoH/DiverseBranchBlock    
    
    Reference:
    Diverse branch block: Building a convolution as an inception-like unit. 
    Xiaohan Ding, Xiangyu Zhang, Jungong Han, and Guiguang Ding. 
    In Proceedings ofthe IEEE/CVF Con ference on Computer Vision and Pattern Recognition, pages 10886-10895, 2021
"""


import torch
import torch.nn as nn
from .diversebranchblock import DiverseBranchBlock
from .acb import ACBlock
from .repblock import RepBlock
from .dbb_transforms import transI_fusebn


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        self.deploy = deploy
        # if deploy:
        self.rep_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        self.rep_conv.requires_grad_(False)
        # else:
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        
        self.attach_rate = 1.
        self.only_raw = False
    
    def inversion(self, reset_bn=False):
        """
            inverse the fused_conv into 3x3_origin
        """
        self.deploy = False
        if reset_bn:
            self.bn.running_mean.zero_()
            self.bn.running_var.fill_(1.0)
            self.bn.weight.data.fill_(1.0)
            self.bn.bias.data = self.rep_conv.bias.data.clone().detach()
            self.conv.weight.data = self.rep_conv.weight.data.clone().detach()
        else:
            # Keep running statistics of BN layer
            self.bn.weight.data.fill_(1.0)
            std = (self.bn.running_var + self.bn.eps).sqrt().clone().detach()
            t = (self.bn.weight / std).clone().detach().reshape(-1, 1, 1, 1)
            self.conv.weight.data = (self.rep_conv.weight.data / t).clone().detach()
            # _origin_bn_bias_data = self.bn.bias.data.clone().detach()
            self.bn.bias.data = (self.rep_conv.bias.data + self.bn.running_mean * self.bn.weight / std).clone().detach()
            # return torch.sum(torch.abs(_origin_bn_bias_data - self.bn.bias.data))

    def init_branch_weights(self):
        """
            init other branches after inversion.
            there's no more branches in this module.
        """
        pass
    
    def set_attach_rate(self, value):
        pass
    #     self.attach_rate = value
    
    # def turn_only_raw(self, value:bool):
    #     pass
    #     self.only_raw = value
    
    def forward(self, x):
        if self.deploy:
            return self.nonlinear(self.rep_conv(x))
        else:
            return self.nonlinear(self.bn(self.conv(x)))
        # if hasattr(self, 'bn'):
        #     return self.nonlinear(self.bn(self.conv(x)))
        # else:
        #     return self.nonlinear(self.conv(x))

    def switch_to_deploy(self, first=True):
        kernel, bias = transI_fusebn(self.conv.weight, self.bn)
        if first:
            # self.rep_conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
            #                 stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
            self.rep_conv.weight.data = kernel
            self.rep_conv.bias.data = bias
        else:
            self.rep_conv.weight.data = self.rep_conv.weight.data + kernel
            self.rep_conv.bias.data = self.rep_conv.bias.data + bias
        # for para in self.parameters():
        #     para.detach_()
        # self.__delattr__('conv')
        # self.__delattr__('bn')
        # self.conv = conv


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, IMPL='base', deploy=False):
    assert IMPL in ['base', 'DBB', 'ACB', 'REP']
    if IMPL == 'base' or kernel_size == 1 or kernel_size >= 7:
        blk_type = ConvBN
    elif IMPL == 'ACB':
        blk_type = ACBlock
    elif IMPL == 'DBB':
        blk_type = DiverseBranchBlock
    elif IMPL == 'REP':
        blk_type = RepBlock
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=groups, deploy=deploy)

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, IMPL='base', deploy=False):
    assert IMPL in ['base', 'DBB', 'ACB', 'REP']
    if IMPL == 'base' or kernel_size == 1 or kernel_size >= 7:
        blk_type = ConvBN
    elif IMPL == 'ACB':
        blk_type = ACBlock
    elif IMPL == 'DBB':
        blk_type = DiverseBranchBlock
    elif IMPL == 'REP':
        blk_type = RepBlock
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=groups, deploy=deploy, nonlinear=nn.ReLU())


def build_model(arch, num_classes, IMPL, deploy):
    if arch == 'ResNet-18':
        from resnet import create_Res18
        model = create_Res18(num_classes, IMPL, deploy)
    # elif arch == 'ResNet-50':
    #     from resnet import create_Res50
    #     model = create_Res50()
    # elif arch == 'MobileNet':
    #     from mobilenet import create_MobileNet
    #     model = create_MobileNet()
    # else:
    #     raise ValueError('TODO')
    return model
