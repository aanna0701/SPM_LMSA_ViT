'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# __all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Encoder(nn.Module):
    def __init__(self, resnet_block=R.BasicBlock, num_resnet_blocks=[9, 9, 9], num_sa_block=0):
        super(ResNet_Encoder, self).__init__()

        self.in_planes = 16

        self.fc1 = nn.Conv2d(3, 16, kernel_size=3,
                             stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1, self.layer2, self.layer3 = self.ResEncoder_sa(
            num_resnet_blocks, num_sa_block, resnet_block)

        self.apply(_weights_init)

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
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.layer1(out)
        if self.layer2:
            out = self.layer2(out)
        if self.layer3:
            out = self.layer3(out)

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
