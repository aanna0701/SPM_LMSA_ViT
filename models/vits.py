import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from models.deepsets import Equivariant_Block


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
        return x.permute(0, 2, 1) + self.PE


class Patch_Embedding(nn.Module):
    def __init__(self, patch_size, in_channels, inter_channels):
        super(Patch_Embedding, self).__init__()
        self.patch_embedding = nn.Conv2d(in_channels, inter_channels,
                                         kernel_size=patch_size, stride=patch_size)
        self._init_weights()
        self.inter_channels = inter_channels

       
        
        


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
        
        
        

        


        return out_flat


class MLP(nn.Module):

    def __init__(self, in_channels, hidden_dim_ratio):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = self.in_channels * hidden_dim_ratio

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
    Class-token Embedding
    '''
    def __init__(self):
        super(GA_block, self).__init__()
    
    def forward(self, x):
        '''
            [shape]
            x : (B, HW+1, C)
            spatial_token : (B, HW, C)
            cls_token_in : (B, 1, C)
            weight : (B, 1, HW)
            weight_softmax : (B, 1, HW)
            cls_token_out : (B, 1, C)
            out : (B, HW+1, C)     
        '''
        spatial_token, cls_token_in = x[:, 1:], x[:, (0, )]
        weight = torch.matmul(cls_token_in, spatial_token.permute(0, 2, 1))
        weight_softmax = F.softmax(weight, dim=2)
        cls_token_out = torch.matmul(weight_softmax, spatial_token)
        cls_token_out = cls_token_out + cls_token_in
        
        out = torch.cat((cls_token_out, spatial_token), dim=1)
        
        return out




class Transformer_Block(nn.Module):
    def __init__(self, in_channels, heads=8, mlp_ratio=4):
        super(Transformer_Block, self).__init__()
        self.normalization = nn.LayerNorm(in_channels)
        self.MHSA = MHSA(in_channels, heads)
        self.MLP = MLP(in_channels, mlp_ratio)
        

    def forward(self, x, cls_token, GA_block):
        '''
        [shape]
            x : (B, HW, C)
        '''
        if not cls_token == None:
            x_in = torch.cat((cls_token, x), dim=1)
        else:
            x_in = x
        x_inter1 = self.normalization(x_in)


        x_MHSA = self.MHSA(x_inter1)
        x_res1 = x_in + x_MHSA
        x_inter2 = self.normalization(x_res1)
    
        if not cls_token == None:
       
            x_inter2 = GA_block(x_inter2)
            x_inter2 = self.normalization(x_inter2)

        
        x_MLP = self.MLP(x_inter2)
        x_res2 = x_inter2 + x_MLP

        if not cls_token == None:
            return x_res2[:, 1:], x_res2[:, (0, )]
        
        else:
            return x_res2
        

# class Transformer_Block(nn.Module):
#     def __init__(self, in_channels, heads=8, mlp_ratio=4):
#         super(Transformer_Block, self).__init__()
#         self.normalization = nn.LayerNorm(in_channels)
#         self.MHSA = MHSA(in_channels, heads)
#         self.MLP = MLP(in_channels, mlp_ratio)
        

#     def forward(self, x, cls_token, GA_block):
#         '''
#         [shape]
#             x : (B, HW, C)
#         '''
#         if not cls_token == None:
#             x_in = torch.cat((cls_token, x), dim=1)
#         else:
#             x_in = x
#         x_inter1 = self.normalization(x_in)


#         x_MHSA = self.MHSA(x_inter1)
#         x_res1 = x_in + x_MHSA
#         x_inter2 = self.normalization(x_res1)
    
#         if not cls_token == None:
       
#             x_inter2, cls_token = GA_block(x_inter2) 

        
#         x_MLP = self.MLP(x_inter2)
#         x_res2 = x_inter2 + x_MLP

#         if not cls_token == None:
#             return x_res2, cls_token
        
#         else:
#             return x_res2


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

        x = self.linear(x).squeeze()

        return x
    
class Classifier_2d(nn.Module):
    def __init__(self, num_classes=10, in_channels=64):
        super(Classifier_2d, self).__init__()

        self.linear = nn.Linear(in_channels, num_classes)
        self.name = 'Classifier'

    def forward(self, x):
        '''
        [shape]
            x : (B, HW, C)
            x_pool : (B, C, 1)
            out : (B, num_classes)
        '''
        x_pool = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1)

        out = self.linear(x_pool.permute(0, 2, 1)).squeeze()

        return out

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
    def __init__(self, in_height, in_width, num_nodes, inter_dimension, depth, mlp_ratio=4, heads=8, num_classes=10, GA=False):
        super(ViT, self).__init__()

        self.inter_dimension = inter_dimension
        self.heads = heads

        self.patch_embedding = Patch_Embedding(
            patch_size=int(math.sqrt((in_height * in_width) // num_nodes)), in_channels=3, inter_channels=inter_dimension)
        
        
        self.classifier = Classifier_1d(
        num_classes=num_classes, in_channels=inter_dimension)
        self.positional_embedding = Positional_Embedding(
        spatial_dimension=num_nodes, inter_channels=inter_dimension)
        
        self.GA = GA_block()
    
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.inter_dimension))
        
        self.transformers = self.make_layer(depth, Transformer_Block, mlp_ratio)


    def make_layer(self, num_blocks, block, mlp_ratio):
        layer_list = nn.ModuleList()
        for i in range(num_blocks):
            layer_list.append(
                block(self.inter_dimension, self.heads, mlp_ratio))

        return layer_list

    def forward(self, x):
        '''
        [shape]
            x : (B, 3, H, W)
            x_patch_embedded = (B, HW, C)
            x_tmp = (B, HW, C)
            x_out = (B, classes)
        '''
        x_patch_embedded = self.patch_embedding(x)
        x_tmp = self.positional_embedding(x_patch_embedded)
        cls_token = self.cls_token.expand(x_patch_embedded.size(0), self.cls_token.size(1), self.cls_token.size(2))

        
        x_tmp = F.dropout(x_tmp, 0.1)
        cls_token = F.dropout(cls_token, 0.1)
        
        for i in range(len(self.transformers)):
            x_tmp, cls_token = self.transformers[i](x_tmp, cls_token, self.GA)
                
        x_out = self.classifier(cls_token)

        return x_out

class ViT_w_o_token(nn.Module):
    def __init__(self, in_height, in_width, num_nodes, inter_dimension, depth, mlp_ratio=4, heads=8, num_classes=10):
        super(ViT_w_o_token, self).__init__()

        self.inter_dimension = inter_dimension
        self.heads = heads

        self.patch_embedding = Patch_Embedding(
            patch_size=int(math.sqrt((in_height * in_width) // num_nodes)), in_channels=3, inter_channels=inter_dimension)
        
        
        self.classifier = Classifier_2d(
        num_classes=num_classes, in_channels=inter_dimension)
        self.positional_embedding = Positional_Embedding(
        spatial_dimension=num_nodes, inter_channels=inter_dimension)

        
        self.transformers = self.make_layer(depth, Transformer_Block, mlp_ratio)


    def make_layer(self, num_blocks, block, mlp_ratio):
        layer_list = nn.ModuleList()
        for i in range(num_blocks):
            layer_list.append(
                block(self.inter_dimension, self.heads, mlp_ratio))

        return layer_list

    def forward(self, x):
        '''
        [shape]
            x : (B, 3, H, W)
            x_patch_embedded = (B, HW, C)
            x_tmp = (B, HW, C)
            x_out = (B, classes)
        '''
        x_patch_embedded = self.patch_embedding(x)
        x_tmp = self.positional_embedding(x_patch_embedded)

        
        x_tmp = F.dropout(x_tmp, 0.1)
        
        for i in range(len(self.transformers)):
            x_tmp = self.transformers[i](x_tmp, None, None)

        
        x_out = self.classifier(x_tmp)

        return x_out


class PiT(nn.Module):
    def __init__(self, in_height, in_width, num_nodes, inter_channels, num_blocks, heads=8, mlp_ratio=4,num_classes=10, EB=False, IB=False):
        super(PiT, self).__init__()

        self.in_channels = inter_channels
        self.heads = heads
        self.transformers = nn.ModuleList()

        self.patch_embedding = Patch_Embedding(
            patch_size=int(math.sqrt((in_height * in_width) // num_nodes)), in_channels=3, inter_channels=inter_channels, global_attribute=IB)
        self.positional_embedding = Positional_Embedding(
            spatial_dimension=num_nodes + 1, inter_channels=inter_channels)

        
        j = 0
        for i in range(len(num_blocks)):
            if not j+1 == len(num_blocks):
                self.make_layer(num_blocks[i], Transformer_Block, mlp_ratio, True)
                j += 1
                self.in_channels = 2 * self.in_channels
                self.heads = 2 * self.heads
            else:
                self.make_layer(num_blocks[i], Transformer_Block, mlp_ratio)
        
        self.classifier = Classifier_1d(
            num_classes=num_classes, in_channels=self.in_channels)        

        if EB:
            self.EB = Equivariant_Block()
        else:
            self.EB = False

    def make_layer(self, num_blocks, tr_block, mlp_ratio, pool_block=False):
        for i in range(num_blocks):
            self.transformers.append(
                tr_block(self.in_channels, self.heads, mlp_ratio))
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
    
