import torch
import torch.nn as nn
from torch.nn import functional as F


class EB_2d(nn.Module):
    '''
    Equivariant block (Average version)
    '''

    def __init__(self):
        super(EB_2d, self).__init__()

        self._gamma = nn.Parameter(torch.zeros(1))
        self._lambda = nn.Parameter(torch.zeros(1))
        self.name = 'EB'

    def forward(self, x, bn):
        '''
        out = x + gamma * avrpool(x)
        '''

        height = x.size(2)
        width = x.size(3)

        x = bn(x)
        x_max = nn.AvgPool2d(kernel_size=(height, width))(x)

        out = torch.add(torch.sigmoid(self._lambda) * x,
                        torch.sigmoid(self._gamma) * x_max)

        return out


class Equivariant_Block(nn.Module):
    '''
    Equivariant block (Average version)
    '''

    def __init__(self):
        super(Equivariant_Block, self).__init__()

        self._gamma = nn.Parameter(torch.zeros(1))
        self._lambda = nn.Parameter(torch.zeros(1))
        self.name = 'EB'

    def forward(self, x):
        '''
        out = lambda * x + gamma * avrpool(x)

        [shape]
            x : (B, HW+1, C)
            x_tmp : (B, C, HW+1)
            x_pool : (B, C, 1)
            out_sum : (B, C, HW+1)
            out : (B, HW+1, C)
        '''

        x_tmp = x.permute(0, 2, 1)
        x_pool = F.adaptive_avg_pool1d(x_tmp, 1)

        out_sum = torch.add(torch.sigmoid(self._lambda) * x_tmp,
                            torch.sigmoid(self._gamma) * x_pool)
        out = out_sum.permute(0, 2, 1)

        return out


class Invariant_Block(nn.Module):
    '''
        y = sigma(sum_up(phi(x)))
        phi: transformation
        sigma: non-linear transformation 
    '''

    def __init__(self, phi, sigma):
        super(Invariant_Block, self).__init__()
        self.phi = phi
        self.sigma = sigma

    def forward(self, x):
        '''
            [shape]
            x : (B, HW, C)
            x_phi : (B, C, HW)
            x_sum : (B, C, 1)
            x_sigma : (B, 1, C)
        '''
        x_phi = self.phi(x).permute(0, 2, 1)
        x_sum = F.adaptive_avg_pool1d(x_phi)
        x_sigma = self.sigma(x_sum).permute(0, 2, 1)

        return x_sigma
