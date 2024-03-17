"""
    ResNet with DBB block structure, code from https://github.com/DingXiaoH/DiverseBranchBlock
    
    Reference:
    Diverse branch block: Building a convolution as an inception-like unit. 
    Xiaohan Ding, Xiangyu Zhang, Jungong Han, and Guiguang Ding. 
    In Proceedings ofthe IEEE/CVF Con ference on Computer Vision and Pattern Recognition, pages 10886-10895, 2021
"""

import torch.nn as nn
import torch.nn.functional as F
from net.dbb_series.convnet_utils import conv_bn, conv_bn_relu, ConvBN
from net.dbb_series.diversebranchblock import DiverseBranchBlock
from net.dbb_series.acb import ACBlock
from .repblock import RepBlock


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, IMPL='base', deploy=False):
        super(BasicBlock, self).__init__()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv_bn(in_channels=in_planes, out_channels=self.expansion * planes, kernel_size=1, stride=stride, IMPL=IMPL, deploy=deploy)
        else:
            self.shortcut = nn.Identity()
        self.conv1 = conv_bn_relu(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, IMPL=IMPL, deploy=deploy)
        self.conv2 = conv_bn(in_channels=planes, out_channels=self.expansion * planes, kernel_size=3, stride=1, padding=1, IMPL=IMPL, deploy=deploy)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, IMPL='base', deploy=False):
        super(Bottleneck, self).__init__()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = conv_bn(in_planes, self.expansion*planes, kernel_size=1, stride=stride, IMPL=IMPL, deploy=deploy)
        else:
            self.shortcut = nn.Identity()

        self.conv1 = conv_bn_relu(in_planes, planes, kernel_size=1, IMPL=IMPL, deploy=deploy)
        self.conv2 = conv_bn_relu(planes, planes, kernel_size=3, stride=stride, padding=1, IMPL=IMPL, deploy=deploy)
        self.conv3 = conv_bn(planes, self.expansion*planes, kernel_size=1, IMPL=IMPL, deploy=deploy)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, width_multiplier=1, IMPL='base', deploy=False):
        super(ResNet, self).__init__()

        self.in_planes = int(64 * width_multiplier)
        if num_classes == 1000:
            self.stage0 = nn.Sequential()
            self.stage0.add_module('conv1', conv_bn_relu(in_channels=3, out_channels=self.in_planes, kernel_size=7, stride=2, padding=3,  IMPL=IMPL, deploy=deploy))
            self.stage0.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        elif num_classes in [10, 100]:
            self.stage0 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))

        self.stage1 = self._make_stage(block, int(64 * width_multiplier), num_blocks[0], stride=1, IMPL=IMPL, deploy=deploy)
        self.stage2 = self._make_stage(block, int(128 * width_multiplier), num_blocks[1], stride=2, IMPL=IMPL, deploy=deploy)
        self.stage3 = self._make_stage(block, int(256 * width_multiplier), num_blocks[2], stride=2, IMPL=IMPL, deploy=deploy)
        self.stage4 = self._make_stage(block, int(512 * width_multiplier), num_blocks[3], stride=2, IMPL=IMPL, deploy=deploy)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512*block.expansion*width_multiplier), num_classes)
    
    """Recursive-Rep function: set attach_rate \\lambda"""
    def set_attach_rate(self, value):
        for n,m in self.named_modules():
            if isinstance(m, DiverseBranchBlock) or isinstance(m, ACBlock) or isinstance(m, RepBlock):
                m.set_attach_rate(value)
    
    # """Recursive-Rep function: set if only raw outputs are used"""
    # def turn_only_raw(self, value):
    #     for n,m in self.named_modules():
    #         if isinstance(m, DiverseBranchBlock) or isinstance(m, ACBlock):
    #             m.turn_only_raw(value)

    """Recursive-Rep function: inversion all blocks"""
    def inversion_all(self, reset=False):
        for n,m in self.named_modules():
            if isinstance(m, DiverseBranchBlock) or isinstance(m, ACBlock) or isinstance(m, RepBlock):
                m.inversion(reset_bn=reset)
    
    """Recursive-Rep function: init other branches"""
    def init_branch_weights(self):
        for n,m in self.named_modules():
            if isinstance(m, DiverseBranchBlock) or isinstance(m, ACBlock) or isinstance(m, RepBlock):
                m.init_branch_weights()
    
    # =====================
    
    """Recursive-Rep method: inversion turn"""
    def inverse_turn_all(self, attach=1.):
        self.init_branch_weights()
        self.inversion_all()
        self.set_attach_rate(attach)
    
    """Recursive-Rep method: deploy"""
    def reparam(self):  # `first` is deprecated
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        # for n,m in self.named_modules():
        #     if isinstance(m, DiverseBranchBlock) or isinstance(m, ACBlock):
        #         m.switch_to_deploy()
    
    def weights(self, rep):
        for name, param in self.named_parameters():
            if (("reparam" in name) or ("rep_conv" in name) or ("bn" in name) or ("linear" in name)) == rep:
                yield param
                # yield name

    def _make_stage(self, block, planes, num_blocks, stride, IMPL='base', deploy=False):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride, IMPL=IMPL, deploy=deploy))
            self.in_planes = int(planes * block.expansion)
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def create_Res18(num_classes=1000, IMPL='DBB', deploy=False):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, width_multiplier=1, IMPL=IMPL, deploy=deploy)


def create_Res34(num_classes=1000, IMPL='DBB', deploy=False):
    return ResNet(BasicBlock, [3,4,6,3], num_classes, width_multiplier=1, IMPL=IMPL, deploy=deploy)


def create_Res50(num_classes=1000, IMPL='DBB', deploy=False):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, width_multiplier=1, IMPL=IMPL, deploy=deploy)


if __name__ == '__main__':
    net = create_Res18(1000, 'DBB', False)
    # for n,m in net.named_modules():
    #     if isinstance(m, DiverseBranchBlock) or isinstance(m, ACBlock):
    #         # m.turn_only_raw(value)
    #         print(n)
    for n,p in net.named_parameters():
        print(n)