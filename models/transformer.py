import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from models.deepsets import Equivariant_Block, Invariant_Block


class MHSA(nn.Module):
    def __init__(self, in_channels, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.out = nn.Linear(in_channels, in_channels)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
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

        q = self.query(x).view(B, n_tokens, self.heads,
                               inter_dimension).permute(0, 2, 1, 3)
        k = self.key(x).view(B, n_tokens, self.heads,
                             inter_dimension).permute(0, 2, 1, 3)
        v = self.value(x).view(B, n_tokens, self.heads,
                               inter_dimension).permute(0, 2, 1, 3)

        similarity = torch.matmul(q, k.transpose(3, 2))
        similarity = similarity / math.sqrt(inter_dimension)

        attention = self.softmax(similarity)

        out_tmp = torch.matmul(attention, v)
        out = out_tmp.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, n_tokens, -1)
        out = self.out(out)

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
    def __init__(self, patch_size, in_channels, inter_channels, invariant_block=False):
        super(Patch_Embedding, self).__init__()
        self.patch_embedding = nn.Conv2d(in_channels, inter_channels,
                                         kernel_size=patch_size, stride=patch_size)
        self._init_weights()

        if not invariant_block:
            self.ib = False
            self.cls_token = nn.Parameter(torch.zeros(1, inter_channels, 1))
        else:
            self.ib = True
            self.phi = MLP(inter_channels)
            self.sigma = nn.GELU()
            self.cls_token = Invariant_Block(self.phi, self.sigma)

    def _init_weights(self):
        nn.init.kaiming_normal_(
            self.patch_embedding.weight)
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
        if not self.ib:
            cls_token = self.cls_token.expand(
                x.size(0), self.cls_token.size(1), self.cls_token.size(2))
        else:
            cls_token = self.cls_token(out_flat.permute(0, 2, 1)).permute(0, 2, 1).expand(
                x.size(0), self.cls_token.size(1), self.cls_token.size(2)
            )

        out_concat = torch.cat((cls_token, out_flat), dim=2)
        out = out_concat.permute(0, 2, 1)

        return out


class MLP(nn.Module):

    def __init__(self, in_channels, inter_channels=None):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        if inter_channels == None:
            self.inter_channels = self.in_channels * 4

        self.fc1 = nn.Linear(self.in_channels, self.inter_channels)
        self.fc2 = nn.Linear(self.inter_channels, self.in_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        '''
        [shape]
            x : (B, HW+1, C)
            x_inter : (B, HW+1, 4*C)
            x_out : (B, HW+1, C)
        '''

        x_inter = nn.GELU()(self.fc1(x))

        x_out = self.fc2(x_inter)

        return x_out


class Transformer_Block(nn.Module):
    def __init__(self, in_channels, heads=8):
        super(Transformer_Block, self).__init__()
        self.normalization1 = nn.LayerNorm(in_channels)
        self.normalization2 = nn.LayerNorm(in_channels)
        # self.normalization1 = nn.BatchNorm1d(in_channels)
        # self.normalization2 = nn.BatchNorm1d(in_channels)
        self.MHSA = MHSA(in_channels, heads)
        self.MLP = MLP(in_channels)

    def forward(self, x, _EB):
        '''
        [shape]
            x : (B, HW+1, C)
        '''

        x_inter1 = self.normalization1(x)

        if _EB:
            x_inter1 = _EB(x_inter1)

        x_MHSA = self.MHSA(x_inter1)
        x_res1 = x + x_MHSA

        x_inter2 = self.normalization2(x_res1)
        x_MLP = self.MLP(x_inter2)
        x_res2 = x_res1 + x_MLP

        return x_res2


class Classifier_1d(nn.Module):
    def __init__(self, num_classes=10, in_channels=64):
        super(Classifier_1d, self).__init__()

        self.linear = nn.Linear(in_channels, num_classes)
        self.name = 'Classifier'

    def forward(self, x):
        '''
        [shape]
            x : (B, HW+1, C)
            x[:, 0] : (B, C)
        '''

        x = self.linear(x[:, 0])

        return x


class ViT(nn.Module):
    def __init__(self, in_height, in_width, num_nodes, inter_dimension, depth, heads=8, num_classes=10, EB=False, IB=False):
        super(ViT, self).__init__()

        self.inter_dimension = inter_dimension
        self.heads = heads

        self.patch_embedding = Patch_Embedding(
            patch_size=int(math.sqrt((in_height * in_width) // num_nodes)), in_channels=3, inter_channels=inter_dimension, invariant_block=IB)
        self.positional_embedding = Positional_Embedding(
            spatial_dimension=num_nodes + 1, inter_channels=inter_dimension)
        self.classifier = Classifier_1d(
            num_classes=num_classes, in_channels=inter_dimension)
        self.transformers = self.make_layer(depth, Transformer_Block)

        if EB:
            self.EB = Equivariant_Block()
        else:
            self.EB = False

    def make_layer(self, num_blocks, block):
        layer_list = nn.ModuleList()
        for i in range(num_blocks):
            layer_list.append(
                block(self.inter_dimension, self.heads))

        return layer_list

    def forward(self, x):
        '''
        [shape]
            x : (B, 3, H, W)
            x_patch_embedded = (B, HW+1, C)
            x_tmp = (B, HW+1, C)
            x_out = (B, classes)
        '''
        x_patch_embedded = self.patch_embedding(x)

        x_tmp = self.positional_embedding(x_patch_embedded)
        for i in range(len(self.transformers)):
            x_tmp = self.transformers[i](x_tmp, self.EB)
        x_out = self.classifier(x_tmp)

        return x_out
