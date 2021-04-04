import torch
import torch.nn as nn
from torch.nn import functional as F

import Non_local_embedded_gaussian as N
import resnet as R

import math


class ResNet_Encoder(nn.Module):
    def __init__(self, resnet_block=R.BasicBlock, num_resnet_blocks=[9, 9, 9], num_sa_block=0):
        super(ResNet_Encoder, self).__init__()

        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1, self.layer2, self.layer3 = self.ResEncoder_sa(
            num_resnet_blocks, num_sa_block, resnet_block)

        self.apply(R._weights_init)

    def ResEncoder_sa(self, num_resnet_blocks, num_sa_blocks, resnet_block):
        if num_sa_blocks < 27 and num_sa_blocks >= 18:
            return self._make_layer(resnet_block, 16, num_resnet_blocks[0] - num_sa_blocks + 18, stride=1),\
                False, False
        elif num_sa_blocks < 18 and num_sa_blocks >= 9:
            return self._make_layer(resnet_block, 16, num_resnet_blocks[0], stride=1),\
                self._make_layer(
                    resnet_block, 32, num_resnet_blocks[1] - num_sa_blocks + 9, stride=2), False
        elif num_sa_blocks < 9:
            return self._make_layer(resnet_block, 16, num_resnet_blocks[0], stride=1),\
                self._make_layer(resnet_block, 32, num_resnet_blocks[1], stride=2),\
                self._make_layer(
                    resnet_block, 64, num_resnet_blocks[2] - num_sa_blocks, stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        if not num_blocks:
            return None

        layers = []

        if stride:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if self.layer2:
            out = self.layer2(out)
        if self.layer3:
            out = self.layer3(out)

        return out


class NLB(nn.Module):
    '''
    Non-Local block
    '''

    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True):
        super(NLB, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.name = 'NLB'

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, EB=False, bn_layer=False):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        x_residual = x

        # proposed method
        if EB:
            x = EB(x)
            x = bn_layer(x)
            # rerlu addition
            x = nn.ReLU()(x)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        # divide by srt(dimension of query)
        f = f / math.sqrt(float(self.inter_channels))
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x_residual

        return z


class EB(nn.Module):
    '''
    Equivariant block (Average version)
    '''

    def __init__(self):
        super(EB, self).__init__()

        # self.gamma = nn.Parameter(torch.rand(1))
        # self._gamma = nn.Parameter(torch.tensor(-0.5))
        # self._lambda = nn.Parameter(torch.tensor(0.5))
        self._gamma = nn.Parameter(torch.zeros(1))
        self._lambda = nn.Parameter(torch.zeros(1))
        self.name = 'EB'

    def forward(self, x):
        '''
        out = x + gamma * avrpool(x)
        '''

        height = x.size(2)
        width = x.size(3)

        x_max = nn.AvgPool2d(kernel_size=(height, width))(x)

        out = torch.add(torch.sigmoid(self._lambda) * x,
                        torch.sigmoid(self._gamma) * x_max)

        return out


class Classifier_FC(nn.Module):
    def __init__(self, num_classes=10, in_channels=64):
        super(Classifier_FC, self).__init__()

        self.linear = nn.Linear(in_channels, num_classes)
        self.name = 'FCL'

    def forward(self, x):

        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


class ResNet(nn.Module):
    def __init__(self, _Encoder, _Classifier, num_resnet_blocks=[9, 9, 9], num_classes=10):
        super(ResNet, self).__init__()

        self.ResNet_Encoder = _Encoder(num_resnet_blocks=num_resnet_blocks)
        self.Classifier = _Classifier(num_classes=num_classes)
        self.name = 'ResNet'

    def forward(self, x):

        latent = self.ResNet_Encoder(x)
        out = self.Classifier(latent)

        return out


class Down_Conv(nn.Module):
    def __init__(self, in_planes, planes):
        super(Down_Conv, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3,
                              stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.name = 'Down_Conv'

    def forward(self, x, EB=False):
        return self.bn(self.conv(x))


class Self_Attention_res56(nn.Module):
    def __init__(self, _Encoder, _NLB, _Classifier, _Down_Conv, _num_sa_blocks, _Global_attribute=False):
        super(Self_Attention_res56, self).__init__()

        self.Encoder = _Encoder(num_sa_block=_num_sa_blocks)
        self.Classifier = _Classifier()

        self.NLB_list = self.make_sa_layer(_num_sa_blocks, _NLB, _Down_Conv)

        if _Global_attribute:
            if _num_sa_blocks > 18:
                self.bn3 = nn.BatchNorm2d(16)
                self.bn2 = nn.BatchNorm2d(32)
                self.bn1 = nn.BatchNorm2d(64)
                self.EB = EB()

            elif _num_sa_blocks > 9:
                self.bn2 = nn.BatchNorm2d(32)
                self.bn1 = nn.BatchNorm2d(64)
                self.EB = EB()

            else:
                self.bn1 = nn.BatchNorm2d(64)
                self.EB = EB()

        else:
            self.EB = False

    def make_sa_layer(self, num_sa_blocks, NLB, Down_Conv):
        NLB_list_tmp = []
        if num_sa_blocks < 9:
            for i in range(num_sa_blocks):
                NLB_list_tmp.append(NLB(64))
            return nn.ModuleList(NLB_list_tmp)

        elif num_sa_blocks < 18 and num_sa_blocks >= 9:
            for i in range(num_sa_blocks - 9):
                NLB_list_tmp.append(NLB(32))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(32, 64))
                NLB_list_tmp.append(NLB(64))
            return nn.ModuleList(NLB_list_tmp)

        elif num_sa_blocks < 27 and num_sa_blocks >= 18:
            for i in range(num_sa_blocks - 18):
                NLB_list_tmp.append(NLB(16))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(16, 32))
                NLB_list_tmp.append(NLB(32))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(32, 64))
                NLB_list_tmp.append(NLB(64))
            return nn.ModuleList(NLB_list_tmp)

        else:
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(3, 16))
                NLB_list_tmp.append(NLB(16))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(16, 32))
                NLB_list_tmp.append(NLB(32))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(32, 64))
                NLB_list_tmp.append(NLB(64))
            return nn.ModuleList(NLB_list_tmp)

    def forward(self, x):

        latent = self.Encoder(x)
        for i in range(len(self.NLB_list)):
            if self.NLB_list[i].name == 'NLB':
                if latent.size(1) == 16:
                    if self.EB:
                        latent = self.NLB_list[i](latent, self.EB, self.bn3)
                    else:
                        latent = self.NLB_list[i](latent)

                elif latent.size(1) == 32:
                    if self.EB:
                        latent = self.NLB_list[i](latent, self.EB, self.bn2)
                    else:
                        latent = self.NLB_list[i](latent)

                elif latent.size(1) == 64:
                    if self.EB:
                        latent = self.NLB_list[i](latent, self.EB, self.bn1)
                    else:
                        latent = self.NLB_list[i](latent)
            else:
                latent = self.NLB_list[i](latent)

        out = self.Classifier(latent)

        return out


class Self_Attention_full(nn.Module):
    def __init__(self, _NLB, _Classifier, _Down_Conv, _Global_attribute=False):
        super(Self_Attention_full, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)

        self.Classifier = _Classifier()

        self.NLB_list = self.make_sa_layer(_NLB, _Down_Conv)

        if _Global_attribute:

            self.bn3 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn1 = nn.BatchNorm2d(64)
            self.EB = EB()

        else:
            self.EB = False

    def make_sa_layer(self, NLB, Down_Conv):
        NLB_list_tmp = []

        for i in range(9):
            NLB_list_tmp.append(NLB(16))
        for i in range(9):
            if i == 0:
                NLB_list_tmp.append(Down_Conv(16, 32))
            NLB_list_tmp.append(NLB(32))
        for i in range(9):
            if i == 0:
                NLB_list_tmp.append(Down_Conv(32, 64))
            NLB_list_tmp.append(NLB(64))

        return nn.ModuleList(NLB_list_tmp)

    def forward(self, x):

        latent = F.relu(self.bn0(self.conv1(x)))

        for i in range(len(self.NLB_list)):
            if self.NLB_list[i].name == 'NLB':
                if latent.size(1) == 16:
                    if self.EB:
                        latent = self.NLB_list[i](latent, self.EB, self.bn3)
                    else:
                        latent = self.NLB_list[i](latent, self.EB)

                elif latent.size(1) == 32:
                    if self.EB:
                        latent = self.NLB_list[i](latent, self.EB, self.bn2)
                    else:
                        latent = self.NLB_list[i](latent)

                elif latent.size(1) == 64:
                    if self.EB:
                        latent = self.NLB_list[i](latent, self.EB, self.bn1)
                    else:
                        latent = self.NLB_list[i](latent)
            else:
                latent = self.NLB_list[i](latent)

        out = self.Classifier(latent)

        return out


class Self_Attention_res56_no_sharing(nn.Module):
    def __init__(self, _Encoder, _NLB, _Classifier, _Down_Conv, _num_sa_blocks, _Global_attribute=False):
        super(Self_Attention_res56_no_sharing, self).__init__()

        self.Encoder = _Encoder(num_sa_block=_num_sa_blocks)
        self.Classifier = _Classifier()

        self.NLB_list = self.make_sa_layer(_num_sa_blocks, _NLB, _Down_Conv)

        EB_list = []
        for i in range(_num_sa_blocks):
            EB_list.append(EB())
        self.EB_list = nn.ModuleList(EB_list)

        if _Global_attribute:
            if _num_sa_blocks > 18:
                self.bn3 = nn.BatchNorm2d(16)
                self.bn2 = nn.BatchNorm2d(32)
                self.bn1 = nn.BatchNorm2d(64)
                self.EB = True

            elif _num_sa_blocks > 9:
                self.bn2 = nn.BatchNorm2d(32)
                self.bn1 = nn.BatchNorm2d(64)
                self.EB = True

            else:
                self.bn1 = nn.BatchNorm2d(64)
                self.EB = True

        else:
            self.EB = False

    def make_sa_layer(self, num_sa_blocks, NLB, Down_Conv):
        NLB_list_tmp = []

        if num_sa_blocks < 9:
            for i in range(num_sa_blocks):
                NLB_list_tmp.append(NLB(64))
            return nn.ModuleList(NLB_list_tmp)

        elif num_sa_blocks < 18 and num_sa_blocks >= 9:
            for i in range(num_sa_blocks - 9):
                NLB_list_tmp.append(NLB(32))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(32, 64))
                NLB_list_tmp.append(NLB(64))
            return nn.ModuleList(NLB_list_tmp)

        elif num_sa_blocks < 27 and num_sa_blocks >= 18:
            for i in range(num_sa_blocks - 18):
                NLB_list_tmp.append(NLB(16))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(16, 32))
                NLB_list_tmp.append(NLB(32))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(32, 64))
                NLB_list_tmp.append(NLB(64))
            return nn.ModuleList(NLB_list_tmp)

        else:
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(3, 16))
                NLB_list_tmp.append(NLB(16))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(16, 32))
                NLB_list_tmp.append(NLB(32))
            for i in range(9):
                if i == 0:
                    NLB_list_tmp.append(Down_Conv(32, 64))
                NLB_list_tmp.append(NLB(64))
            return nn.ModuleList(NLB_list_tmp)

    def forward(self, x):

        latent = self.Encoder(x)
        j = 0

        for i in range(len(self.NLB_list)):
            if self.NLB_list[i].name == 'NLB':
                if latent.size(1) == 16:
                    if self.EB:
                        latent = self.NLB_list[i](
                            latent, self.EB_list[j], self.bn3)
                        j += 1
                    else:
                        latent = self.NLB_list[i](latent)

                elif latent.size(1) == 32:
                    if self.EB:
                        latent = self.NLB_list[i](
                            latent, self.EB_list[j], self.bn2)
                        j += 1
                    else:
                        latent = self.NLB_list[i](latent)

                elif latent.size(1) == 64:
                    if self.EB:
                        latent = self.NLB_list[i](
                            latent, self.EB_list[j], self.bn1)
                        j += 1
                    else:
                        latent = self.NLB_list[i](latent)
            else:
                latent = self.NLB_list[i](latent)

        out = self.Classifier(latent)

        return out


class Self_Attention_full_no_sharing(nn.Module):
    def __init__(self, _NLB, _Classifier, _Down_Conv, _Global_attribute=False):
        super(Self_Attention_full_no_sharing, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)

        self.Classifier = _Classifier()

        self.NLB_list = self.make_sa_layer(_NLB, _Down_Conv)

        if _Global_attribute:

            self.bn3 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn1 = nn.BatchNorm2d(64)
            EB_list = []
            for i in range(27):
                EB_list.append(EB())
            self.EB_list = nn.ModuleList(EB_list)
            self.EB = True

        else:
            self.EB = False

    def make_sa_layer(self, NLB, Down_Conv):
        NLB_list_tmp = []

        for i in range(9):
            NLB_list_tmp.append(NLB(16))
        for i in range(9):
            if i == 0:
                NLB_list_tmp.append(Down_Conv(16, 32))
            NLB_list_tmp.append(NLB(32))
        for i in range(9):
            if i == 0:
                NLB_list_tmp.append(Down_Conv(32, 64))
            NLB_list_tmp.append(NLB(64))

        return nn.ModuleList(NLB_list_tmp)

    def forward(self, x):

        latent = F.relu(self.bn0(self.conv1(x)))
        j = 0

        for i in range(len(self.NLB_list)):

            if self.NLB_list[i].name == 'NLB':
                if latent.size(1) == 16:
                    if self.EB:
                        latent = self.NLB_list[i](
                            latent, self.EB_list[j], self.bn3)
                        j += 1
                    else:
                        latent = self.NLB_list[i](latent)

                elif latent.size(1) == 32:
                    if self.EB:
                        latent = self.NLB_list[i](
                            latent, self.EB_list[j], self.bn2)
                        j += 1
                    else:
                        latent = self.NLB_list[i](latent)

                elif latent.size(1) == 64:
                    if self.EB:
                        latent = self.NLB_list[i](
                            latent, self.EB_list[j], self.bn1)
                        j += 1
                    else:
                        latent = self.NLB_list[i](latent)

            else:
                latent = self.NLB_list[i](latent)

        out = self.Classifier(latent)

        return out


def resnet20():
    return ResNet(ResNet_Encoder, Classifier_FC, [3, 3, 3])


def resnet32():
    return ResNet(ResNet_Encoder, Classifier_FC, [5, 5, 5])


def resnet44():
    return ResNet(ResNet_Encoder, Classifier_FC, [7, 7, 7])


def resnet56():
    return ResNet(ResNet_Encoder, Classifier_FC)


def self_attention_ResNet56(num_sa_block, global_attribute=False):
    return Self_Attention_res56(ResNet_Encoder, NLB, Classifier_FC, Down_Conv, num_sa_block, _Global_attribute=global_attribute)


def self_Attention_full(global_attribute=False):
    return Self_Attention_full(NLB, Classifier_FC, Down_Conv, _Global_attribute=global_attribute)


def self_attention_ResNet56_no_sharing(num_sa_block, global_attribute=False):
    return Self_Attention_res56_no_sharing(ResNet_Encoder, NLB, Classifier_FC, Down_Conv, num_sa_block, _Global_attribute=global_attribute)


def self_Attention_full_no_sharing(global_attribute=False):
    return Self_Attention_full_no_sharing(NLB, Classifier_FC, Down_Conv, _Global_attribute=global_attribute)
