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
        out = F.dropout(out, 0.1)

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
        self.inter_channels = inter_channels

        if not invariant_block:
            self.ib = False
            self.cls_token = nn.Parameter(torch.zeros(1, 1, inter_channels))
        else:
            self.ib = True
            
            self.cls_token = GA_block(self.inter_channels)

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
            out_concat : (B, HW+1, C')
            out : (B, HW+1, C')
        '''
        out = self.patch_embedding(x)
        out_flat = out.flatten(start_dim=2)
        if not self.ib:
            cls_token = self.cls_token.expand(
                x.size(0), self.cls_token.size(1), self.cls_token.size(2))
        else:
            cls_token = self.cls_token(
                out_flat.permute(0, 2, 1))
            cls_token = cls_token.expand(
                x.size(0), cls_token.size(1), cls_token.size(2)
            )

        out_concat = torch.cat((cls_token, out_flat.permute(0, 2, 1)), dim=1)
        out = out_concat

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

        x_inter = F.gelu(self.fc1(x))
        x_inter = F.dropout(x_inter, 0.1)

        x_out = self.fc2(x_inter)
        x_out = F.dropout(x_out, 0.1)

        return x_out
    
class GA_block(nn.Module):
    '''
    phi : Equivariant block
    rho : MLP
    '''
    def __init__(self, in_channels):
        super(GA_block, self).__init__()
        self.phi = Equivariant_Block()
        self.rho = MLP(in_channels=in_channels)
        self.activation = F.gelu
    
    def forward(self, x):
        '''
            [shape]
            x : (B, HW, C)
            x_phi : (B, C, HW)
            x_sum : (B, C, 1)
            x_rho : (B, 1, C)
        '''
        x_phi = self.phi(x).permute(0, 2, 1)
        x_phi = self.activation(x_phi)
        x_sum = F.adaptive_avg_pool1d(x_phi, 1)
        x_rho = self.rho(x_sum.permute(0, 2, 1))
        
        return x_rho


class Transformer_Block(nn.Module):
    def __init__(self, in_channels, heads=8):
        super(Transformer_Block, self).__init__()
        self.normalization = nn.LayerNorm(in_channels)
        self.MHSA = MHSA(in_channels, heads)
        self.MLP = MLP(in_channels)

    def forward(self, x, _EB):
        '''
        [shape]
            x : (B, HW+1, C)
        '''

        
        x_inter1 = self.normalization(x)

        if _EB:
            x_inter1_spatial = x_inter1[:, 1:]
            x_inter1_cls = x_inter1[:, (0, )]
            x_inter1_spatial = _EB(x_inter1_spatial)
            x_inter1 = torch.cat((x_inter1_cls, x_inter1_spatial), dim=1)
            # x_inter1 = _EB(x_inter1)

        x_MHSA = self.MHSA(x_inter1)
        x_res1 = x + x_MHSA

        x_inter2 = self.normalization(x_res1)
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

class Pooling_layer(nn.Module):
    def __init__(self, in_channels):
        super(Pooling_layer, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels)
        self.fcl = nn.Linear(in_channels, 2*in_channels)
        
    def forward(self, x, EB):
        '''
        [shape]
            x : (B, HW+1, C)
            cls_token : (B, 1, C)
            feature_token : (B, HW, C)
            feature_token_reshape : (B, C, H, W)
            pool_feature_token : (B, (H/2)(W/2), 2*C)
            out : (B, (HW/2)+1, 2*C)
        '''
        cls_token = x[:, (0,)]
        feature_token = x[:, 1:]
        
        B, HW, C = feature_token.shape
        H = int(math.sqrt(HW))
        feature_token_reshape = feature_token.permute(0, 2, 1).contiguous()
        feature_token_reshape = feature_token_reshape.view((B, C, H, H))
                
        pool_feature_token = self.conv(feature_token_reshape)
        pool_feature_token = pool_feature_token.view((B, 2*C, -1)).permute(0, 2, 1)
        pool_cls_token = self.fcl(cls_token)
        
        out = torch.cat((pool_cls_token, pool_feature_token), dim=1)
  
        
        return out


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
        x_tmp = F.dropout(x_tmp, 0.1)
        for i in range(len(self.transformers)):
            x_tmp = self.transformers[i](x_tmp, self.EB)
        x_out = self.classifier(x_tmp)

        return x_out


class PiT(nn.Module):
    def __init__(self, in_height, in_width, num_nodes, inter_channels, num_blocks, heads=8, num_classes=10, EB=False, IB=False):
        super(PiT, self).__init__()

        self.in_channels = inter_channels
        self.heads = heads
        self.transformers = nn.ModuleList()

        self.patch_embedding = Patch_Embedding(
            patch_size=int(math.sqrt((in_height * in_width) // num_nodes)), in_channels=3, inter_channels=inter_channels, invariant_block=IB)
        self.positional_embedding = Positional_Embedding(
            spatial_dimension=num_nodes + 1, inter_channels=inter_channels)

        
        j = 0
        for i in range(len(num_blocks)):
            if not j+1 == len(num_blocks):
                self.make_layer(num_blocks[i], Transformer_Block, True)
                j += 1
                self.in_channels = 2 * self.in_channels
                self.heads = 2 * self.heads
            else:
                self.make_layer(num_blocks[i], Transformer_Block)
        
        self.classifier = Classifier_1d(
            num_classes=num_classes, in_channels=self.in_channels)        

        if EB:
            self.EB = Equivariant_Block()
        else:
            self.EB = False

    def make_layer(self, num_blocks, tr_block, pool_block=False):
        for i in range(num_blocks):
            self.transformers.append(
                tr_block(self.in_channels, self.heads))
        if pool_block:
            self.transformers.append(
                Pooling_layer(self.in_channels))


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
        x_tmp = F.dropout(x_tmp, 0.1)
        for i in range(len(self.transformers)):
            x_tmp = self.transformers[i](x_tmp, self.EB)
        x_out = self.classifier(x_tmp)

        return x_out