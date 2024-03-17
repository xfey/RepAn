"""
    Asymmetric Convolution Block, code from https://github.com/DingXiaoH/DiverseBranchBlock
    
    Reference:
    Diverse branch block: Building a convolution as an inception-like unit. 
    Xiaohan Ding, Xiangyu Zhang, Jungong Han, and Guiguang Ding. 
    In Proceedings ofthe IEEE/CVF Con ference on Computer Vision and Pattern Recognition, pages 10886-10895, 2021
"""

import math
import torch.nn as nn
import torch.nn.init as init
import torch


class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,
                 use_affine=True, reduce_gamma=False, gamma_init=None, nonlinear=None):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        
        # if deploy:
        self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                    padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        self.fused_conv.requires_grad_(False)
        # else:
        self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=False,
                                        padding_mode=padding_mode)
        self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)


        if padding - kernel_size // 2 >= 0:
            #   Common use case. E.g., k=3, p=1 or k=5, p=2
            self.crop = 0
            #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
            hor_padding = [padding - kernel_size // 2, padding]
            ver_padding = [padding, padding - kernel_size // 2]
        else:
            #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
            #   Since nn.Conv2d does not support negative padding, we implement it manually
            self.crop = kernel_size // 2 - padding
            hor_padding = [0, padding]
            ver_padding = [padding, 0]

        self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                    stride=stride,
                                    padding=ver_padding, dilation=dilation, groups=groups, bias=False,
                                    padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                    stride=stride,
                                    padding=hor_padding, dilation=dilation, groups=groups, bias=False,
                                    padding_mode=padding_mode)
        self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
        self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

        if reduce_gamma:
            self.init_gamma(1.0 / 3)

        if gamma_init is not None:
            assert not reduce_gamma
            self.init_gamma(gamma_init)
        
        self.attach_rate = 1.
        # self.only_raw = False

    def set_attach_rate(self, value):
        self.attach_rate = value
    
    # def turn_only_raw(self, value:bool):
    #     self.only_raw = value

    def inversion(self, reset_bn=False):
        """
            inverse the fused_conv into 3x3_origin
        """
        self.deploy = False
        if reset_bn:
            self.square_bn.running_mean.zero_()
            self.square_bn.running_var.fill_(1.0)
            self.square_bn.weight.data.fill_(1.0)
            self.square_bn.bias.data = self.fused_conv.bias.data.clone().detach()
            self.square_conv.weight.data = self.fused_conv.weight.data.clone().detach()
        else:
            # Keep running statistics of BN layer
            self.square_bn.weight.data.fill_(1.0)
            std = (self.square_bn.running_var + self.square_bn.eps).sqrt().clone().detach()
            t = (self.square_bn.weight / std).clone().detach().reshape(-1, 1, 1, 1)
            self.square_conv.weight.data = (self.fused_conv.weight.data / t).clone().detach()
            # # _origin_bn_bias_data = self.bn.bias.data.clone().detach()
            self.square_bn.bias.data = (self.fused_conv.bias.data + self.square_bn.running_mean * self.square_bn.weight / std).clone().detach()
            # # return torch.sum(torch.abs(_origin_bn_bias_data - self.bn.bias.data))

    def init_branch_weights(self):
        """
            init other branches after inversion.
        """
        n1 = self.ver_conv.weight.size(0)
        init_range1 = .1 / math.sqrt(n1)
        self.ver_conv.weight.data.uniform_(-init_range1, init_range1)
        n2 = self.hor_conv.weight.size(0)
        init_range2 = .1 / math.sqrt(n2)
        self.hor_conv.weight.data.uniform_(-init_range2, init_range2)
        self.ver_bn.weight.data.fill_(1.0)
        self.ver_bn.bias.data.zero_()
        self.hor_bn.weight.data.fill_(1.0)
        self.hor_bn.bias.data.zero_()

    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        square_k, square_b = self._fuse_bn_tensor(self.square_conv, self.square_bn)
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        return square_k, hor_b + ver_b + square_b


    def switch_to_deploy(self, first=True):
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True
        if first:
            # self.fused_conv = nn.Conv2d(in_channels=self.square_conv.in_channels, out_channels=self.square_conv.out_channels,
            #                             kernel_size=self.square_conv.kernel_size, stride=self.square_conv.stride,
            #                             padding=self.square_conv.padding, dilation=self.square_conv.dilation, groups=self.square_conv.groups, bias=True,
            #                             padding_mode=self.square_conv.padding_mode)
            self.fused_conv.weight.data = deploy_k
            self.fused_conv.bias.data = deploy_b
        else:
            self.fused_conv.weight.data = self.fused_conv.weight.data + deploy_k
            self.fused_conv.bias.data = self.fused_conv.bias.data + deploy_b
        # self.__delattr__('square_conv')
        # self.__delattr__('square_bn')
        # self.__delattr__('hor_conv')
        # self.__delattr__('hor_bn')
        # self.__delattr__('ver_conv')
        # self.__delattr__('ver_bn')


    def init_gamma(self, gamma_value):
        init.constant_(self.square_bn.weight, gamma_value)
        init.constant_(self.ver_bn.weight, gamma_value)
        init.constant_(self.hor_bn.weight, gamma_value)
        print('init gamma of square, ver and hor as ', gamma_value)

    def single_init(self):
        init.constant_(self.square_bn.weight, 1.0)
        init.constant_(self.ver_bn.weight, 0.0)
        init.constant_(self.hor_bn.weight, 0.0)
        print('init gamma of square as 1, ver and hor as 0')

    def forward(self, input):
        if self.deploy:
            return self.nonlinear(self.fused_conv(input))
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop:-self.crop]
                hor_input = input[:, :, self.crop:-self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs = self.ver_conv(ver_input)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv(hor_input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            return self.nonlinear(square_outputs + self.attach_rate * (vertical_outputs + horizontal_outputs))

if __name__ == '__main__':
    N = 1
    C = 2
    H = 62
    W = 62
    O = 8
    groups = 4

    x = torch.randn(N, C, H, W)
    print('input shape is ', x.size())

    test_kernel_padding = [(3,1), (3,0), (5,1), (5,2), (5,3), (5,4), (5,6)]

    for k, p in test_kernel_padding:
        acb = ACBlock(C, O, kernel_size=k, padding=p, stride=1, deploy=False)
        acb.eval()
        # acb.turn_only_raw(True)
        # acb.set_attach_rate(1.)
        for module in acb.modules():
            if isinstance(module, nn.BatchNorm2d):
                nn.init.uniform_(module.running_mean, 0, 0.1)
                nn.init.uniform_(module.running_var, 0, 0.2)
                nn.init.uniform_(module.weight, 0, 0.3)
                nn.init.uniform_(module.bias, 0, 0.4)
        out = acb(x)
        acb.switch_to_deploy()
        deployout = acb(x)
        print('difference between the outputs of the training-time and converted ACB is')
        print(((deployout - out) ** 2).sum())
