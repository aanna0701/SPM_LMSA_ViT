import torch
import torch.nn as nn
from torch.nn import functional as F


import math


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

        self.fc1 = nn.Conv2d(3, 16, kernel_size=3,
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

        latent = F.relu(self.bn0(self.fc1(x)))

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

        self.fc1 = nn.Conv2d(3, 16, kernel_size=3,
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

        latent = F.relu(self.bn0(self.fc1(x)))
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
