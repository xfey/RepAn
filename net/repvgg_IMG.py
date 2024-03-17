import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


def reparam_func(layer):
    """[summary]
    Args:
        layer: Single RepVGG block
        Returns the reparamitrized weights
    """

    # 3x3 weight fuse
    std = (layer.bn_3.running_var + layer.bn_3.eps).sqrt()
    t = (layer.bn_3.weight / std).reshape(-1, 1, 1, 1)
    reparam_weight_3 = layer.conv_3.weight * t
    reparam_bias_3 = layer.bn_3.bias - layer.bn_3.running_mean * layer.bn_3.weight / std
    reparam_weight = reparam_weight_3
    reparam_bias = reparam_bias_3

    # 1x1 weight fuse
    std = (layer.bn_1.running_var + layer.bn_1.eps).sqrt()
    t = (layer.bn_1.weight / std).reshape(-1, 1, 1, 1)
    reparam_weight_1 = layer.conv_1.weight * t
    reparam_bias_1 = layer.bn_1.bias - layer.bn_1.running_mean * layer.bn_1.weight / std
    reparam_weight += F.pad(reparam_weight_1, [1, 1, 1, 1], mode="constant", value=0)
    reparam_bias += reparam_bias_1

    if layer.conv_3.weight.shape[0] == layer.conv_3.weight.shape[1]:
        # Check if in/out filters are equal. If not, skip the identity reparam
        if hasattr(layer, "bn_0"):
            # idx weight fuse - we only have access to bn_0
            std = (layer.bn_0.running_var + layer.bn_0.eps).sqrt()
            t = (layer.bn_0.weight / std).reshape(-1, 1, 1, 1)
            channel_shape = layer.conv_3.weight.shape
            idx_weight = (
                torch.eye(channel_shape[0], channel_shape[0])
                .unsqueeze(2)
                .unsqueeze(3)
                .to(layer.conv_3.weight.device)
            )
            reparam_weight_0 = idx_weight * t
            reparam_bias_0 = (
                layer.bn_0.bias - layer.bn_0.running_mean * layer.bn_0.weight / std
            )
            reparam_weight += F.pad(
                reparam_weight_0, [1, 1, 1, 1], mode="constant", value=0
            )
            reparam_bias += reparam_bias_0
    assert reparam_weight.shape == layer.conv_3.weight.shape
    return reparam_weight, reparam_bias


class PostRepVGGBlock(nn.Module):
    """Single RepVGG block."""

    def __init__(self, num_channels):
        super(PostRepVGGBlock, self).__init__()

        self.conv_3 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
            bias=False,
        )
        self.bn_3 = nn.BatchNorm2d(num_features=num_channels)

        self.conv_1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(1, 1),
            padding=0,
            stride=1,
            bias=False,
        )
        self.bn_1 = nn.BatchNorm2d(num_features=num_channels)

        self.bn_0 = nn.BatchNorm2d(num_features=num_channels)

        # Reparam conv block
        self.rep_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
            bias=True,
        )
        self.activation = nn.ReLU()
        
        self.reparam = False
        self.rep_conv.requires_grad_(False)
        self.attach_rate = 1.
    
    def init_branch_weights(self):
        # init.kaiming_normal_(self.conv_1.weight, mode='fan_out', nonlinearity='relu')
        
        n1 = self.conv_1.weight.size(0)
        init_range1 = .1 / math.sqrt(n1)
        self.conv_1.weight.data.uniform_(-init_range1, init_range1)
        
        self.bn_1.weight.data.fill_(1.0)
        self.bn_1.bias.data.zero_()
        self.bn_0.weight.data.fill_(1.0)
        self.bn_0.bias.data.zero_()
    
    def _reparam(self, first=True):
        """
            convs -> Rep_conv, turn to Rep mode
        """
        self.reparam = True
        with torch.no_grad():
            reparam_weight, reparam_bias = reparam_func(self)
            if first:
                self.rep_conv.weight.data = reparam_weight
                self.rep_conv.bias.data = reparam_bias
            else:
                # NOTE: the 'plus' operation will cause numerical explosion. 
                # [!] Manually init required.
                self.rep_conv.weight.data = reparam_weight + self.rep_conv.weight.data
                self.rep_conv.bias.data = reparam_bias + self.rep_conv.bias.data
    
    def inversion(self, reset_bn=False):
        """
            inverse the Rep_conv into 3x3_conv
            the mean & var of BN layer are unknown; set to 1+bias.
        """
        self.reparam = False
        if reset_bn:
            self.bn_3.running_mean.zero_()
            self.bn_3.running_var.fill_(1.0)
            self.bn_3.weight.data.fill_(1.0)
            self.bn_3.bias.data = self.rep_conv.bias.data.clone().detach()
            self.conv_3.weight.data = self.rep_conv.weight.data.clone().detach()
        else:
            # Keep running statistics of BN layer
            self.bn_3.weight.data.fill_(1.0)
            std = (self.bn_3.running_var + self.bn_3.eps).sqrt().clone().detach()
            t = (self.bn_3.weight / std).clone().detach().reshape(-1, 1, 1, 1)
            self.conv_3.weight.data = (self.rep_conv.weight.data / t).clone().detach()
            _origin_bn_bias_data = self.bn_3.bias.data.clone().detach()
            self.bn_3.bias.data = (self.rep_conv.bias.data + self.bn_3.running_mean * self.bn_3.weight / std).clone().detach()
            return torch.sum(torch.abs(_origin_bn_bias_data - self.bn_3.bias.data))

    def clear_rep_grad(self):
        self.rep_conv.zero_grad()
    
    def forward(self, x):
        if self.reparam:
            # with torch.no_grad():
            return self.activation(self.rep_conv(x))
        else:
            x_3 = self.bn_3(self.conv_3(x))
            x_1 = self.bn_1(self.conv_1(x))
            x_0 = self.bn_0(x)
            # return self.attach_rate * self.activation(x_3 + x_1 + x_0) + (not self.only310) * self.activation(self.rep_conv(x))
            return self.activation(x_3 + self.attach_rate * (x_1 + x_0))


class DownsamplePostRepVGGBlock(nn.Module):
    """Downsample RepVGG block. Comes at the end of a stage"""

    def __init__(self, in_channels, out_channels, stride=2):
        super(DownsamplePostRepVGGBlock, self).__init__()

        self.conv_3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn_3 = nn.BatchNorm2d(num_features=out_channels)

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=0,
            stride=stride,
            bias=False,
        )
        self.bn_1 = nn.BatchNorm2d(num_features=out_channels)

        # Reparam conv block
        self.rep_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=stride,
            bias=True,
        )
        self.activation = nn.ReLU()

        self.reparam = False
        self.rep_conv.requires_grad_(False)
        self.attach_rate = 1.
    
    def init_branch_weights(self):
        # init.kaiming_normal_(self.conv_1.weight, mode='fan_out', nonlinearity='relu')
        
        n1 = self.conv_1.weight.size(0)
        init_range1 = .1 / math.sqrt(n1)
        self.conv_1.weight.data.uniform_(-init_range1, init_range1)
        
        self.bn_1.weight.data.fill_(1.0)
        self.bn_1.bias.data.zero_()
    
    def _reparam(self, first=True):
        """
            convs -> Rep_conv, turn to Rep mode
        """
        self.reparam = True
        with torch.no_grad():
            reparam_weight, reparam_bias = reparam_func(self)
            if first:
                self.rep_conv.weight.data = reparam_weight
                self.rep_conv.bias.data = reparam_bias
            else:
                # NOTE: the 'plus' operation will cause numerical explosion. 
                # [!] Manually init required.
                self.rep_conv.weight.data = reparam_weight + self.rep_conv.weight.data
                self.rep_conv.bias.data = reparam_bias + self.rep_conv.bias.data
    
    def inversion(self, reset_bn=False):
        """
            inverse the Rep_conv into 3x3_conv
            the mean & var of BN layer are unknown; set to 1+bias.
        """
        self.reparam = False
        if reset_bn:
            self.bn_3.running_mean.zero_()
            self.bn_3.running_var.fill_(1.0)
            self.bn_3.weight.data.fill_(1.0)
            self.bn_3.bias.data = self.rep_conv.bias.data.clone().detach()
            self.conv_3.weight.data = self.rep_conv.weight.data.clone().detach()
        else:
            # Keep running statistics of BN layer
            self.bn_3.weight.data.fill_(1.0)
            std = (self.bn_3.running_var + self.bn_3.eps).sqrt().clone().detach()
            t = (self.bn_3.weight / std).clone().detach().reshape(-1, 1, 1, 1)
            self.conv_3.weight.data = (self.rep_conv.weight.data / t).clone().detach()
            _origin_bn_bias_data = self.bn_3.bias.data.clone().detach()
            self.bn_3.bias.data = (self.rep_conv.bias.data + self.bn_3.running_mean * self.bn_3.weight / std).clone().detach()
            return torch.sum(torch.abs(_origin_bn_bias_data - self.bn_3.bias.data))

    def clear_rep_grad(self):
        self.rep_conv.zero_grad()

    def forward(self, x):
        if self.reparam:
            # with torch.no_grad():
            return self.activation(self.rep_conv(x))
        else:
            x_3 = self.bn_3(self.conv_3(x))
            x_1 = self.bn_1(self.conv_1(x))
            # return self.attach_rate * self.activation(x_3 + x_1) + (not self.only310) * self.activation(self.rep_conv(x))
            return self.activation(x_3 + self.attach_rate * (x_1))


class RepVGGStage(nn.Module):
    """Single RepVGG stage. These are stacked together to form the full RepVGG architecture"""

    def __init__(self, in_channels, out_channels, N, stride=2):
        super(RepVGGStage, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.sequential = nn.Sequential(*[
            DownsamplePostRepVGGBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                stride=stride,
            )
        ] + [
            PostRepVGGBlock(num_channels=self.out_channels)
            for _ in range(0, N - 1)
        ])

    def forward(self, x):
        return self.sequential(x)

    def set_attach_rate(self, value):
        for block in self.sequential:
            block.attach_rate = value
    
    def init_branch_weights(self):
        for block in self.sequential:
            block.init_branch_weights()
    
    def clear_rep_grad(self):
        for stage in self.sequential:
            stage.clear_rep_grad()

    def inversion_all(self, reset=False):
        # [debug]
        if not reset:
            _results = torch.tensor([0.]).cuda()
            for block in self.sequential:
                _results = _results + block.inversion(reset_bn=reset)
            return _results
        else:
            for block in self.sequential:
                block.inversion(reset_bn=reset)

    def _reparam(self, first=True):
        for block in self.sequential:
            block._reparam(first)

    def _train(self):
        for block in self.sequential:
            block.reparam = False

    def switch_to_deploy(self):
        for block in self.sequential:
            block.reparam = True
            # delete old attributes
            if hasattr(block, "conv_3"):
                delattr(block, "conv_3")

            if hasattr(block, "conv_1"):
                delattr(block, "conv_1")

            if hasattr(block, "bn_1"):
                delattr(block, "bn_1")

            if hasattr(block, "bn_0"):
                delattr(block, "bn_0")

            if hasattr(block, "bn_3"):
                delattr(block, "bn_3")


class PostRepVGG(nn.Module):
    def __init__(
        self,
        filter_depth=[1, 2, 4, 14, 1],
        filter_list=[64, 64, 128, 256, 512],
        stride=[2, 2, 2, 2, 2],
        width_factor=[1, 1, 1, 1, 2.5],
        num_classes=1000,
    ):
        super(PostRepVGG, self).__init__()

        assert width_factor[0] == width_factor[1]   # align with the source code.
        
        filter_list[0] = min(64, 64 * width_factor[0])
        self.filter_depth = filter_depth

        # filter_list[1:] *= width_factor[1:]
        for i in range(1, len(filter_list)):
            filter_list[i] = int(filter_list[i] * width_factor[i])

        self.stages = nn.Sequential(*[
            RepVGGStage(
                in_channels=3,
                out_channels=int(filter_list[0]),
                N=filter_depth[0],
                stride=stride[0],
            )
        ] + [
            RepVGGStage(
                in_channels=int(filter_list[i - 1]),
                out_channels=filter_list[i],
                N=filter_depth[i],
                stride=stride[i],
            ) for i in range(1, len(filter_depth) - 1)
        ] + [
            RepVGGStage(
                in_channels=int(filter_list[-2]),
                out_channels=filter_list[-1],
                N=filter_depth[-1],
                stride=stride[-1],
            )
        ])

        self.fc = nn.Linear(in_features=int(filter_list[-1]), out_features=num_classes)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.partial_rep_flag = False


    def forward(self, x):
        x = self.stages(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def weights(self, rep:bool):
        for name, param in self.named_parameters():
            if (("rep_conv" in name) or ("fc" in name) or ("bn" in name)) == rep:
                yield param
    
    def init_branch_weights(self):
        for stage in self.stages:
            stage.init_branch_weights()

    def set_attach_rate(self, value):
        for stage in self.stages:
            stage.set_attach_rate(value)

    def inversion_layers(self, reset=False):
        # [debug]
        if not reset:
            _results = torch.tensor([0.]).cuda()
            for stage in self.stages:
                _results = _results + stage.inversion_all(reset=reset)
            return _results
        else:
            for stage in self.stages:
                stage.inversion_all(reset=reset)

    def _reparam(self, first=True):
        for stage in self.stages:
            stage._reparam(first)

    def _train(self):
        for stage in self.stages:
            stage._train()
    
    def _find_layer(self, index):
        # [1, 2, 4, 14, 1]; index starts from zero
        for stage in range(len(self.filter_depth)):
            if index > sum(self.filter_depth[:(stage+1)]) - 1:
                continue
            return stage, index - sum(self.filter_depth[:stage])
    
    """Method: clear grad of rep_conv"""
    def clear_rep_grad(self):
        for stage in self.stages:
            stage.clear_rep_grad()
    
    """Method: inversion turn for all"""
    def inverse_turn_all(self, attach=1.):
        self.init_branch_weights()
        self.inversion_layers()
        self.set_attach_rate(attach)
    
    """Method: inversion turn by block"""
    def inverse_turn_block(self, index, attach=1.):
        self.init_branch_weights()
        stage_idx, layer_idx = self._find_layer(index)
        self.stages[stage_idx].sequential[layer_idx].inversion()
        self.stages[stage_idx].sequential[layer_idx].attach_rate = attach
    
    """Method: inversion turn using array"""
    def inverse_turn_list(self, idxs, attach=1.):
        """
        Args:
            idxs: list like [0,1,1,0,1], where 1=activated and 0=froze
        """
        self.init_branch_weights()
        
        if sum(idxs) < len(idxs):
            self.partial_rep_flag = True
        for idx in range(len(idxs)):
            # turn 1 into activated, keep 0 stay same.
            if idxs[idx]:   # NOTE: opposite to func `reparam_turn_list`
                stage_idx, layer_idx = self._find_layer(idx)
                self.stages[stage_idx].sequential[layer_idx].inversion()
                self.stages[stage_idx].sequential[layer_idx].attach_rate = attach
    
    
    
    """Method: rep turn by block"""
    def reparam_turn_block(self, index):
        self.partial_rep_flag = True
        stage_idx, layer_idx = self._find_layer(index)
        if self.stages[stage_idx].sequential[layer_idx].reparam == False:
            self.stages[stage_idx].sequential[layer_idx]._reparam(first=True)
    
    """Method: reparam using array"""
    def reparam_turn_list(self, idxs:list):
        """
        Args:
            idxs: list like [0,1,1,0,1], where 1=activated and 0=froze
        """
        for idx in range(len(idxs)):
            # turn 0 into frozen, keep 1 stay same.
            if not idxs[idx]:   # NOTE: opposite to func `inverse_turn_list`
                self.partial_rep_flag = True
                stage_idx, layer_idx = self._find_layer(idx)
                if self.stages[stage_idx].sequential[layer_idx].reparam == False:
                    self.stages[stage_idx].sequential[layer_idx]._reparam(first=True)
    
    
    """Method: reset BN for calibration"""
    def reset_bn_calibration(self, train_loader, cali_batchsize=32):
        """
            reset running statistics for BN calibration.
            Args:
                cali_batchsize: Number of batches for calibration.
                max_cali_epoch: Number of times for reloading weights.
        """

        # Set attach_rate
        # _attach_rate = min(((cur_epoch + 1.) / (max_cali_epoch*0.8)), 1.)
        
        # Reset BN
        for n,m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d) and ("bn_3" in n):
                m.training = True
                m.momentum = None # cumulative moving average
                m.reset_running_stats()
        
        # Running calibration
        for cur_iter, (inputs, _) in enumerate(train_loader):
            if cur_iter >= cali_batchsize:
                break
            inputs = inputs.cuda()
            self.forward(inputs)
        
        # Reset inversion weights
        _results = self.inversion_layers()
        print(_results)
        
        # self.set_attach_rate(1.)

    # """Method: freeze grad by block"""
    # def freeze_grad_block(self, index):
    #     stage_idx, layer_idx = self._find_layer(index)
    #     for n,p in self.stages[stage_idx].sequential[layer_idx].named_parameters():
    #         p.requires_grad_(False)
    #     return stage_idx, layer_idx

    """Method: deploy model"""
    def deploy_model(self):
        self._reparam()
        for module in self.modules():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
    
    
    """Method: load checkpoints from official RepVGG"""
    def load_official_checkpoints(self, ckpt_path):
        # NOTE: no weights for `rep_conv` in official checkpoint files.
        new_ckpt = OrderedDict()
        raw_ckpt = torch.load(ckpt_path)
        
        def parse_module_name(name1:str, name2:str):
            name_map = {
                'rbr_dense.conv': 'conv_3',
                'rbr_dense.bn': 'bn_3',
                'rbr_1x1.conv': 'conv_1',
                'rbr_1x1.bn': 'bn_1',
                # 'rbr_identity': 'bn_0',
            }
            if name1 == 'rbr_identity':
                return 'bn_0'
            else:
                return name_map['.'.join([name1, name2])]
        
        def parse_key_name(name:str):
            name = name.split('.')
            # mode 0: stage0.rbr_dense.bn.weight
            if name[0] == 'stage0':
                _name_prefix = 'stages.0.sequential.0'
                _name_middle = parse_module_name(name[1], name[2])
                new_name = '.'.join([_name_prefix, _name_middle, name[-1]])
            elif name[0].startswith('stage'):
                _name_prefix = '.'.join(['stages', name[0][-1], 'sequential', name[1]])
                _name_middle = parse_module_name(name[2], name[3])
                new_name = '.'.join([_name_prefix, _name_middle, name[-1]])
            elif name[0] == 'linear':
                new_name = 'fc.' + name[-1]
            return new_name

        for k,v in raw_ckpt.items():
            new_ckpt[parse_key_name(k)] = v
        self.load_state_dict(new_ckpt, strict=False)    # ignore rep_conv series


# ----------------------------------------------


def RepVGG_A0(num_classes=1000):
    return PostRepVGG(
        filter_depth=[1, 2, 4, 14, 1],
        filter_list=[64, 64, 128, 256, 512],
        stride=[2, 2, 2, 2, 2],
        width_factor=[0.75] * 4 + [2.5],
        num_classes=num_classes,
    )


# used in this code.
def RepVGG_A1(num_classes=1000):
    return PostRepVGG(
        filter_depth=[1, 2, 4, 14, 1],
        filter_list=[64, 64, 128, 256, 512],
        stride=[2, 2, 2, 2, 2],
        width_factor=[1] * 4 + [2.5],
        num_classes=num_classes,
    )


def RepVGG_A2(num_classes=1000):
    return PostRepVGG(
        filter_depth=[1, 2, 4, 14, 1],
        filter_list=[64, 64, 128, 256, 512],
        stride=[2, 2, 2, 2, 2],
        width_factor=[1.5] * 4 + [2.75],
        num_classes=num_classes,
    )


def RepVGG_B0(num_classes=1000):
    return PostRepVGG(
        filter_depth=[1, 4, 6, 16, 1],
        filter_list=[64, 64, 128, 256, 512],
        # filter_list=[16, 16, 32, 64, 128],
        stride=[2, 2, 2, 2, 2],
        width_factor=[1] * 4 + [2.5],
        num_classes=num_classes,
    )


def RepVGG_B1(num_classes=1000):
    return PostRepVGG(
        filter_depth=[1, 4, 6, 16, 1],
        filter_list=[64, 64, 128, 256, 512],
        stride=[2, 2, 2, 2, 2],
        width_factor=[2] * 4 + [4],
        num_classes=num_classes,
    )


def RepVGG_B2(num_classes=1000):
    return PostRepVGG(
        filter_depth=[1, 4, 6, 16, 1],
        filter_list=[64, 64, 128, 256, 512],
        stride=[2, 2, 2, 2, 2],
        width_factor=[2.5] * 4 + [5],
        num_classes=num_classes,
    )


def RepVGG_B3(num_classes=1000):
    return PostRepVGG(
        filter_depth=[1, 4, 6, 16, 1],
        filter_list=[64, 64, 128, 256, 512],
        stride=[2, 2, 2, 2, 2],
        width_factor=[3] * 4 + [5],
        num_classes=num_classes,
    )


if __name__ == "__main__":
    x = torch.randn(1,3,32,32)


# #     model = RepVGG_A1()
# #     # QA
# #     model.eval()

# #     input = torch.randn(1, 3, 32, 32)

# #     out_train = model(input)
# #     deployed_model = deploy_model(model)
# #     # print(deployed_model)
# #     out_eval = deployed_model(input)

# #     print(((out_train - out_eval) ** 2).sum())
