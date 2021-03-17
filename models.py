import torch
import torch.nn as nn
from torch.nn import functional as F

import Non_local_embedded_gaussian as N
import resnet as R

class ResNet_NLB(nn.Module):
    def __init__(self, resnet_block, num_resnet_blocks, nlb_block, num_nlb_block, num_classes=10):
        super(ResNet_NLB, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(resnet_block, 16, num_resnet_blocks[0], stride=1)
        self.layer2 = self._make_layer(resnet_block, 32, num_resnet_blocks[1], stride=2)
        self.layer3 = self._make_layer(resnet_block, 64, num_resnet_blocks[2] - num_nlb_block, stride=2)
        self.layer_nlb = self._make_layer(N._NonLocalBlockND, 64, num_nlb_block, stride=False)
        self.linear = nn.Linear(64, num_classes)

        self.apply(R._weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        
        if stride:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion

        else:
            for i in range(num_blocks):
                layers.append(block(self.in_planes))
            

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer_nlb(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    
def resnet56():
    return R.ResNet(R.BasicBlock, [9, 9, 9])

def resnet56_nlb_1():
    return ResNet_NLB(R.BasicBlock, [9, 9, 9], N.NONLocalBlock2D, 1)

def resnet56_nlb_2():
    return ResNet_NLB(R.BasicBlock, [9, 9, 9], N.NONLocalBlock2D, 2)

def resnet56_nlb_3():
    return ResNet_NLB(R.BasicBlock, [9, 9, 9], N.NONLocalBlock2D, 3)

def resnet56_nlb_4():
    return ResNet_NLB(R.BasicBlock, [9, 9, 9], N.NONLocalBlock2D, 4)

def resnet56_nlb_5():
    return ResNet_NLB(R.BasicBlock, [9, 9, 9], N.NONLocalBlock2D, 5)

def resnet56_nlb_6():
    return ResNet_NLB(R.BasicBlock, [9, 9, 9], N.NONLocalBlock2D, 6)

def resnet56_nlb_9():
    return ResNet_NLB(R.BasicBlock, [9, 9, 9], N.NONLocalBlock2D, 9)