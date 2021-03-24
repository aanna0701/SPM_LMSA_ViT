import torch
import torch.nn as nn
from torch.nn import functional as F

import Self_Attention_with_Global_Attribute as S
import Non_local_embedded_gaussian as N
import resnet as R


class ResNet_Encoder(nn.Module):
    def __init__(self, resnet_block=R.BasicBlock, num_resnet_blocks=[9, 9, 9], num_sa_block=0):
        super(ResNet_Encoder, self).__init__()
        
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(resnet_block, 16, num_resnet_blocks[0], stride=1)
        self.layer2 = self._make_layer(resnet_block, 32, num_resnet_blocks[1], stride=2)
        self.layer3 = self._make_layer(resnet_block, 64, num_resnet_blocks[2] - num_sa_block, stride=2)
        
        self.apply(R._weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
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
        out = self.layer2(out)
        out = self.layer3(out)
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out
    
    
class SAB(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True):
        super(SAB, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

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
            
 
    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        
        return z

class EBA(nn.Module):
    def __init__(self):
        super(EBA, self).__init__()
        
        # self.gamma = nn.Parameter(torch.rand(1))
        self._gamma = nn.Parameter(torch.tensor(-1e1))
        self._lambda = nn.Parameter(torch.tensor(1e1))
        
    def forward(self, x):
        '''
        out = x + gamma * avrpool(x)
        '''

        height = x.size(2)
        width = x.size(3)
        
        x_max = nn.AvgPool2d(kernel_size=(height, width))(x)
        
        out = torch.add(torch.sigmoid(self._lambda) * x, torch.sigmoid(self._gamma) * x_max)
        # out = torch.add(x, torch.sigmoid(self.gamma) * x_max)
        # out = torch.add(x, x_max)

        return out
    
    
class Classifier_FC(nn.Module):
    def __init__(self, num_classes=10, in_channels=64):
        super(Classifier_FC, self).__init__()
        
        self.linear = nn.Linear(in_channels, num_classes)
        
    def forward(self, x):
        
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        return x

class ResNet(nn.Module):
    def __init__(self, _Encoder, _Classifier):        
        super(ResNet, self).__init__()
        
        self.ResNet_Encoder = _Encoder()
        self.Classifier = _Classifier()
                
    def forward(self, x):
        
        latent = self.ResNet_Encoder(x)
        out = self.Classifier(latent)
        
        return out
    
class Self_Attention(nn.Module):
    def __init__(self, _Encoder, _SAB, _Classifier, _num_sa_blocks, _inter_channels=64, _Global_attribute=False):
        super(Self_Attention, self).__init__()
        
        self.Encoder = _Encoder(num_sa_block=_num_sa_blocks)
        self.Classifier = _Classifier()
        self.SAB = _SAB(in_channels=_inter_channels)
        self.num_sa_blocks = _num_sa_blocks
        
        if _Global_attribute:
            self.EBA = EBA()
        else:
            self.EBA = None
        
        
    def forward(self, x):
       
        
        latent = self.Encoder(x)
              
        for i in range(self.num_sa_blocks):

            if self.EBA:
                latent = self.EBA(latent)
            latent = self.SAB(latent)
        
        out = self.Classifier(latent)
        
        return out
        
def resnet56():
    return ResNet(ResNet_Encoder, Classifier_FC)

def self_attention_ResNet56(num_sa_block, global_attribute=False):
    return Self_Attention(ResNet_Encoder, SAB, Classifier_FC, num_sa_block, _Global_attribute=global_attribute)
