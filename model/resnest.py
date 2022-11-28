""" Split Attention Conv2d (for ResNeSt Models)
Paper: `ResNeSt: Split-Attention Networks` - /https://arxiv.org/abs/2004.08955
Adapted from original PyTorch impl at https://github.com/zhanghang1989/ResNeSt
Modified for torchscript compat, performance, and consistency with timm by Ross Wightman
"""
import torch
import torch.nn.functional as F
from torch import nn

from .helpers import make_divisible


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix  # 每个基数组 (cardinality) 下划分的 splits 数 R
        self.cardinality = cardinality  # 基数组数 K

    def forward(self, x):
        # x.shape = (B, group_width) = (B, c)
        batch = x.size(0)
        if self.radix > 1:
            # (B, c) -> (B, K, R, c/(KR)) -> (B, R, K, c/(KR))
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            # 沿 splits 维度 R 计算 Softmax 用于为各 splits 加权
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    """Split-Attention (aka Splat)
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,
                 act_layer=nn.ReLU, norm_layer=None, drop_block=None, **kwargs):
        super(SplitAttn, self).__init__()

        out_channels = out_channels or in_channels  # group_width
        self.radix = radix  # 每个基数组 (cardinality) 下划分的 splits 数 R
        self.drop_block = drop_block

        # 因为 in_channels = out_channels = group_width
        # 所以 mid_chs = group_width * R
        # 表示每个 cardinal group 有 R 个 splits, 共有 mid_chs 个通道
        mid_chs = out_channels * radix

        if rd_channels is None:
            attn_chs = make_divisible(in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            in_channels, mid_chs, kernel_size, stride, padding, dilation,
            groups=groups * radix, bias=bias, **kwargs)

        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.act0 = act_layer(inplace=True)

        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer(inplace=True)

        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)  # 通道数 c' -> c

        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        # 特征图 x 包含了各 splits
        x = self.conv(x)
        x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)

        B, RC, H, W = x.shape  # RC = mid_chs = group_width (c) * self.radix (R)
        if self.radix > 1:
            # x.shape = (B, self.radix, group_width, H, W) 可见 x 是 R 个 splits 的集合
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            # 沿 radix 维对各 splits 按元素求和
            x_gap = x.sum(dim=1)  # x_gap.shape = (B, group_width, H, W)
        else:
            x_gap = x

        # 对 H, W 维取均值 (相当于全局平均池化)
        x_gap = x_gap.mean((2, 3), keepdim=True)  # x_gap.shape = (B, group_width) = (B, c)
        # Dense c' + BN + ReLU
        x_gap = self.fc1(x_gap)  # x_gap.shape = (B, attn_chs) = (B, c')
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)

        # Dense c
        x_attn = self.fc2(x_gap)  # x_gap.shape = (B, group_width) = (B, c)

        # r-Softmax
        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)

        # 按元素相乘计算各 splits 的注意力
        if self.radix > 1:
            # 此时 x.shape = (B, self.radix, RC // self.radix, H, W)
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn

        return out.contiguous()


class ResNestBottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, base_width=64, avd=False, avd_first=False, is_first=False,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(ResNestBottleneck, self).__init__()

        assert reduce_first == 1  # not supported
        assert attn_layer is None  # not supported
        assert aa_layer is None  # TODO not yet supported
        assert drop_path is None  # TODO not yet supported

        # group_width = planes * cardinality (K)
        group_width = int(planes * (base_width / 64.)) * cardinality
        first_dilation = first_dilation or dilation
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix  # 每个基数组 (cardinality) 下的 splits 数 R
        self.drop_block = drop_block

        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.act1 = act_layer(inplace=True)
        self.avd_first = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and avd_first else None

        # 每个基数组 (cardinality) 下的 splits 数 R
        if self.radix >= 1:
            self.conv2 = SplitAttn(
                group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, radix=radix, norm_layer=norm_layer, drop_block=drop_block)
            self.bn2 = nn.Identity()
            self.act2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)
            self.act2 = act_layer(inplace=True)

        self.avd_last = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and not avd_first else None

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x  # 跳跃/残差连接

        # Conv + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        if self.drop_block is not None:
            out = self.drop_block(out)
        out = self.act1(out)

        # AvgPool
        if self.avd_first is not None:
            out = self.avd_first(out)

        # SplitAttn + BN + ReLU
        out = self.conv2(out)
        out = self.bn2(out)
        if self.drop_block is not None:
            out = self.drop_block(out)
        out = self.act2(out)

        # AvgPool
        if self.avd_last is not None:
            out = self.avd_last(out)

        # Conv + BN
        out = self.conv3(out)
        out = self.bn3(out)
        if self.drop_block is not None:
            out = self.drop_block(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        # Res
        out += shortcut

        out = self.act3(out)

        return out