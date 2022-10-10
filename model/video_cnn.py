# coding: utf-8
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .CBAM import ChannelAttention, SpatialAttention
from .RFB import BasicRFB


class Conv_CBAM(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True,
                 bias=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_CBAM, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=p, groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.ca = ChannelAttention(c2)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


def conv3x3(in_planes, out_planes, stride=1):
    # return Conv_CBAM(in_planes, out_planes, k=3, s=stride,
    #                  p=1, bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    # return Conv_CBAM(in_planes, out_planes, k=1)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se

        if (self.se):
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.conv3 = conv1x1(planes, planes // 16)
            self.conv4 = conv1x1(planes // 16, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if (self.se):
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()

            out = out * w

        out = out + residual
        out = self.relu(out)

        return out


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=stride[0], padding=0),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))


    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, se=False):
        self.inplanes = 1
        super(ResNet, self).__init__()
        self.se = se
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.bn = nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        return x


class VideoCNN(nn.Module):
    def __init__(self, se=False):
        super(VideoCNN, self).__init__()

        # self.frontend3D = nn.Sequential(
        #     nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 3, 3), padding=(1, 3, 3), bias=False),
        #     # 32 * (88+2-5+1)/2 * 43 * 40+1-3+1=39
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(True),
        #     nn.Dropout3d(p=0.2),
        #     nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        #     # 32 * 43/2 * 21
        #
        #     nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
        #     # 64 * (21+2-5+1)/2 * 9 * 39+1-3+1=38
        #     nn.ReLU(True),
        #     nn.Dropout3d(p=0.2),
        #     nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        #     # 64 * 9/2 * 4
        #
        #     nn.Conv3d(64, 96, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
        #     # 64 * (21+2-5+1)/2 * 9 * 39+1-3+1=38
        #     nn.ReLU(True),
        #     nn.Dropout3d(p=0.2),
        #     nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        #
        # )
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.PReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        # resnet
        # self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)
        # self.resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], se=se)
        self.dropout = nn.Dropout(p=0.5)

        # backend_gru
        # initialize
        self._initialize_weights()

    def visual_frontend_forward(self, x):
        x = x.transpose(1, 2)
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        # x = self.resnet18(x)
        return x

    def forward(self, x):
        b, t = x.size()[:2]

        x = self.visual_frontend_forward(x)

        # x = self.dropout(x)
        # feat = x.view(b, -1, 512)

        # x = x.view(b, -1, 512)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
