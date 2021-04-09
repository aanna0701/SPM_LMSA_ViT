import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class MHSA(nn.Module):
    def __init__(self, in_channels, height=14, width=14, heads=4, relative_positional_enocoding=False):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.out = nn.Linear(in_channels, in_channels)

        self.RPE = relative_positional_enocoding

        if self.RPE:
            self.rel_w = nn.Parameter(torch.randn(
                [1, heads, in_channels // heads, 1, width]), requires_grad=True)
            self.rel_h = nn.Parameter(torch.randn(
                [1, heads, in_channels // heads, height, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, EB=False):
        '''
        [shapes]

            * C = heads * inter_dimension

            x : (B, HW+1, C)            
            q : (B, heads, HW+1, inter_dimension)
            k : (B, heads, HW+1, inter_dimension)
            v : (B, heads, HW+1, inter_dimension)
            similarity : (B, heads, HW+1, HW+1)
            out_tmp : (B, heads, HW+1, inter_dimension)
            out : (B, HW+1, C)

        '''

        B, n_tokens, C = x.size()

        inter_dimension = C // self.heads

        if EB:
            x = EB(x)
            x = nn.ReLU()(x)

        q = self.query(x).view(B, n_tokens, self.heads,
                               inter_dimension).permute(0, 2, 1, 3)
        k = self.key(x).view(B, n_tokens, self.heads,
                             inter_dimension).permute(0, 2, 1, 3)
        v = self.value(x).view(B, n_tokens, self.heads,
                               inter_dimension).permute(0, 2, 1, 3)

        similarity = torch.matmul(q, k.transpose(3, 2))
        similarity = similarity / math.sqrt(inter_dimension)

        if self.RPE:
            content_position = (self.rel_h + self.rel_w).view(1,
                                                              self.heads, inter_dimension, -1).permute(0, 1, 3, 2)
            content_position = torch.matmul(content_position, q)

            energy = similarity + content_position
            attention = self.softmax(energy)

        else:
            attention = self.softmax(similarity)

        out_tmp = torch.matmul(attention, v)
        out = out_tmp.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, n_tokens, -1)
        out = self.O(out)

        return out


class Positional_Embedding(nn.Module):
    def __init__(self, spatial_dimension, inter_channels):
        super(Positional_Embedding, self).__init__()
        '''
        [shape]
            self.PE : (1, HW+1, C)
        '''
        self.PE = nn.Parameter(torch.zeros(
            1, spatial_dimension, inter_channels))

    def forward(self, x):
        return x + self.PE


class Patch_Embedding(nn.Module):
    def __init__(self, patch_size, in_channels, inter_channels):
        super(Patch_Embedding, self).__init__()
        self.patch_embedding = nn.Conv2d(in_channels, inter_channels,
                                         kernel_size=patch_size, stride=patch_size)
        self._init_weights()

        self.cls_token = nn.Parameter(torch.zeros(1, inter_channels, 1))

    def _init_weights(self):
        nn.init.kaiming_normal_(
            self.patch_embedding.weight, nonlinearity='relu')
        nn.init.normal_(self.patch_embedding.bias, std=1e-6)

    def forward(self, x):
        '''
        [shapes]
            x : (B, C, H, W)
            out : (B, C', H, W)
            out_flat : (B, C', HW)
            out_concat : (B, C', HW+1)
            out : (B, HW+1, C')
        '''
        out = self.patch_embedding(x)
        out_flat = out.flatten(start_dim=2)
        out_concat = torch.cat((self.cls_token, out_flat))
        out = out_concat.permute(0, 2, 1)

        return out


class MLP(nn.Module):

    def __init__(self, in_channels, inter_channels=None):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        if inter_channels == None:
            self.inter_channels = self.in_channels * 4

        self.fc1 = nn.Linear(self.in_channels, self.inter_channels)
        self.fc2 = nn.Linear(self.in_channels, self.inter_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        '''
        [shape]
            x : (B, HW+1, C)
            x_inter : (B, 4*C, HW+1)
            x_out : (B, HW+1, C)
        '''
        x_inter = nn.ReLU()(self.fc1(x.permute(0, 2, 1)))
        x_out = self.fc2(x_inter)
        return x_out.permute(0, 2, 1)


class Transformer_Block(nn.Module):
    def __init__(self, in_channels, height, width, heads=8, relative_positional_enocoding=False):
        super(Transformer_Block, self).__init__()
        self.inter_channels = in_channels
        self.bn0 = nn.BatchNorm2d(self.inter_channels)
        self.bn1 = nn.BatchNorm2d(self.inter_channels)
        self.bn2 = nn.BatchNorm2d(self.inter_channels)
        self.MHSA = MHSA(in_channels, height, width, heads,
                         relative_positional_enocoding)
        self.MLP = MLP(in_channels)

    def forward(self, x, EB=False):
        if EB:
            x_inter1 = EB(x, self.bn0)
        else:
            x_inter1 = x

        x_inter1 = self.bn1(x_inter1)
        x_MHSA = self.MHSA(x_inter1)
        x_res1 = x + x_MHSA

        x_inter2 = self.bn2(x_res1)
        x_MLP = self.MLP(x_inter2)
        x_res2 = x_res1 + x_MLP

        return x_res2
