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


class EB_1d(nn.Module):
    '''
    Equivariant block (Average version)
    '''

    def __init__(self):
        super(EB_1d, self).__init__()

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
