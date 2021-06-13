from sys import path
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class MHSA(nn.Module):
    def __init__(self, in_channels, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Linear(in_channels, in_channels, bias=False)
        self._init_weights(self.query)
        self.key = nn.Linear(in_channels, in_channels, bias=False)
        self._init_weights(self.key)
        self.value = nn.Linear(in_channels, in_channels, bias=False)
        self._init_weights(self.value)
        self.out = nn.Linear(in_channels, in_channels, bias=False)
        self._init_weights(self.out)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.edge = None
        
    def _init_weights(self,layer):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias:
            nn.init.normal_(layer.bias, std=1e-6)

    def forward(self, x, dropout=True):
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
        
        self.edge = similarity

        attention = self.softmax(similarity)

        out_tmp = torch.matmul(attention, v)
        out = out_tmp.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, n_tokens, -1)
        out = self.out(out)
        if dropout:
            out = self.dropout(out)
            
        return out


class Positional_Embedding(nn.Module):
    def __init__(self, spatial_dimension, inter_channels):
        super(Positional_Embedding, self).__init__()
        '''
        [shape]
            self.PE : (1, HW, C)
        '''
        self.PE = nn.Parameter(torch.zeros(
            1, spatial_dimension, inter_channels))
        

    def forward(self, x):
        return x.permute(0, 2, 1) + self.PE


# class Patch_Embedding(nn.Module):
#     def __init__(self, patch_size, in_channels, inter_channels, down_conv=False):
#         super(Patch_Embedding, self).__init__()
#         self.patch_embedding = nn.Conv2d(in_channels, inter_channels, kernel_size=patch_size, stride=patch_size, bias=False)
#         self._init_weights(self.patch_embedding)
#         self.inter_channels = inter_channels
#         self.down_conv = None
#         if down_conv:
#             self.down_conv = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=2, bias=False)
#             self._init_weights(self.down_conv)

#     def _init_weights(self,layer):
#         nn.init.kaiming_normal_(layer.weight)
#         if layer.bias:
#             nn.init.normal_(layer.bias, std=1e-6)

#     def forward(self, x):
#         '''
#         [shapes]
#             x : (B, C, H, W)
#             out : (B, C', H, W)
#             out_flat : (B, C', HW)
#             out_concat : (B, HW+1, C')
#             out : (B, HW+1, C')
#         ''' 
        
#         out = self.patch_embedding(x)
#         if self.down_conv:
#             out = self.down_conv(out)
#         out_flat = out.flatten(start_dim=2)
        
#         return out_flat

class Patch_Embedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, inter_channels, down_conv=False):
        super(Patch_Embedding, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.img_size = img_size
        # if not down_conv:
        #     self.conv = False
        #     self.patch_embedding = nn.Conv2d(in_channels, inter_channels, kernel_size=patch_size, stride=patch_size, bias=False)
        #     self._init_weights(self.patch_embedding)

        if not down_conv:
            self.conv = True
            self.patch_embedding = self.make_layer(2)
        
        else:
            self.conv = True
            # num_block = round(patch_size / 2)
            self.patch_embedding = self.make_layer(3)
        
                
        
    def make_layer(self, num_blocks):
        layer_list = nn.ModuleList()
        
        for i in range(num_blocks):
            
            conv_tmp = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=3, stride=2, bias=False)
            self.img_size = round((self.img_size - 3) / 2) + 1
            self._init_weights(conv_tmp)
            layer_list.append(
                conv_tmp
                )
            layer_list.append(
                nn.LayerNorm([self.inter_channels, self.img_size, self.img_size])
                )
                
            layer_list.append(
                nn.GELU()
                )
                
            self.in_channels = self.inter_channels
            
        return layer_list

    def _init_weights(self,layer):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias:
            nn.init.normal_(layer.bias, std=1e-6)

    def forward(self, x):
        '''
        [shapes]
            x : (B, C, H, W)
            out : (B, C', H, W)
            out_flat : (B, C', HW)
            out_concat : (B, HW+1, C')
            out : (B, HW+1, C')
        ''' 
        if not self.conv:
            out = self.patch_embedding(x)
        
        else:
            out = x
            for patch_embedding in self.patch_embedding:
                out = patch_embedding(out)
                
        
        out_flat = out.flatten(start_dim=2)
        
        return out_flat


class MLP(nn.Module):

    def __init__(self, in_channels, hidden_dim_ratio):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = self.in_channels * hidden_dim_ratio

        self.fc1 = nn.Linear(self.in_channels, self.inter_channels, bias=False)
        self._init_weights(self.fc1)
        self.fc2 = nn.Linear(self.inter_channels, self.in_channels, bias=False)
        self._init_weights(self.fc2)
        self.dropout = nn.Dropout(0.1)
        

    def _init_weights(self,layer):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias:
            nn.init.normal_(layer.bias, std=1e-6)

    def forward(self, x, dropout=True):
        '''
        [shape]
            x : (B, HW+1, C)
            x_inter : (B, HW+1, 4*C)
            x_out : (B, HW+1, C)
        '''

        x_inter = F.gelu(self.fc1(x))
        if dropout:
            x_inter = self.dropout(x_inter)

        x_out = self.fc2(x_inter)
        if dropout:
            x_out = self.dropout(x_out)

        return x_out
    

class GA_block(nn.Module):
    '''
    Class-token Embedding
    '''
    def __init__(self, in_size, in_channels):
        super(GA_block, self).__init__()
        self.in_dimension = in_channels
        self.avgpool = nn.AvgPool1d(in_size[0]-2)
        # self.maxpool = nn.MaxPool1d(in_size[0]-2)
        self.avgpool_2 = nn.AvgPool1d(in_size[0]-1)
        # self.maxpool_2 = nn.MaxPool1d(in_size[0]-1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        # self.mlp_nodes = nn.Linear(in_channels, in_channels, bias=False)
        self.linear1 = nn.Linear(in_channels, in_channels // 2, bias=False)
        self._init_weights(self.linear1)
        self.linear2 = nn.Linear(in_channels, in_channels // 2, bias=False)
        self._init_weights(self.linear2)
        self.linear4 = nn.Linear(in_channels // 2, in_channels, bias=False)
        self._init_weights(self.linear4)
        
    def _init_weights(self,layer):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias:
            nn.init.normal_(layer.bias, std=1e-6)
   
    def forward(self, x, cls_token, edge_per_node, pool=False):
        '''
            [shape]
            x : (B, HW+1, C)
            visual_token : (B, HW, C)
            cls_token_in : (B, 1, C)
            weight : (B, 1, HW)
            weight_softmax : (B, 1, HW)
            cls_token_out : (B, 1, C)
            out : (B, HW+1, C)
            
            edge_aggregation : (B, 1, C)
            channel_importance : (B, 1, C)
            nodes : (B, HW, C)
            channel_aggregation : (B, HW, 1)  
            node_importance : (B, HW, 1)
        '''
        nodes = x[:, 1:]    # (B, HW, C)
        
        edge_global = self.avgpool(edge_per_node.permute(0, 2, 1)).permute(0, 2, 1)   # (B, 1, C)
        node_global = self.avgpool(nodes.permute(0, 2, 1)).permute(0, 2, 1)     # (B, 1, C)
        
        edge_embed = self.linear1(edge_global)
        node_embed = self.linear2(node_global)
        
        # cat_global = torch.cat([edge_global, node_global], dim=1)   # (B, 2, C)
        # norm = torch.norm(cat_global, dim=2, keepdim=True, p=1)   # (B, 2, 1)
        # norm = norm.expand(norm.size(0), 2, nodes.size(2))  # (B, 2, C)
        # scale_edge = torch.div(cat_global, norm) * self.in_dimension
        
        
        # edge_global = scale_edge[:, (0,)] 
        # node_global = scale_edge[:, (1,)]
        
        channel_attention = edge_embed + node_embed
        # channel_attention = torch.mul(edge_embed,node_embed) # (B, 1, C)
        # channel_attention = torch.sum(cat_out, dim=1, keepdim=True) # (B, 1, C)
        
        
        
        channel_attention = self.sigmoid(self.linear4(channel_attention)) # (B, 1, C)
        # channel_attention = self.softmax(self.linear4(channel_attention)) # (B, 1, C)
       
        
        cls_token_out = cls_token + torch.mul(cls_token, channel_attention)    #(B, 1, C)
        # print(channel_attention[0])
        # print(cls_token_out[0])
        # print('\n\n====')

        # edge_aggregation = self.avgpool_2(edge_aggregation.permute(0, 2, 1)).permute(0, 2, 1)
        # channel_importance = self.softmax(edge_aggregation)
                
        # nodes = x[:, 1:]
        
        # channel_aggregation = torch.matmul(nodes, channel_importance.permute(0, 2, 1))
        # node_importance = self.softmax(channel_aggregation)
        

        # node_aggregation = torch.matmul(node_importance.permute(0, 2, 1), nodes)
        
        # # cls_token_out = cls_token + torch.mul(self.Linear(cls_token), weights)
        
        # cls_token_out = cls_token + self.sigmoid(node_aggregation)
        
                
        # print('edge_global: {}'.format(edge_global[0].norm(2)))
        # print('node_glboal: {}'.format(node_glboal[0].norm(2)))
        # print('scale_edge: {}'.format(scale_edge[0].norm(2)))
        # print('scale_node: {}'.format(scale_node[0].norm(2)))
        # print('edge_global_scaled: {}'.format(edge_global_scaled[0].norm(2)))
        # print('node_global_scaled: {}'.format(node_global_scaled[0].norm(2)))
        # print('nod_aggrechannel_attentiongation: {}'.format(channel_attention[0].norm(2)))
        # print('cls_token: {}\n\n'.format(cls_token[0].norm(2)))
        # print('cls_token_out: {}\n\n'.format(cls_token_out[0].norm(2)))
        # print('='*20)

        out = torch.cat((cls_token_out, x[:, 1:]), dim=1)
        
        
        return out



class Transformer_Block(nn.Module):
    def __init__(self, in_size, in_channels, heads=8, mlp_ratio=4, GA_flag=False):
        super(Transformer_Block, self).__init__()
        self.normalization_1 = nn.LayerNorm(in_size)
        if not GA_flag:
            self.normalization_2 = nn.LayerNorm(in_size)
        
        
        self.MHSA = MHSA(in_channels, heads)
        self.MLP = MLP(in_channels, mlp_ratio)
        self.MLP_MHSA = MLP(in_channels, mlp_ratio)
        # self.linear = nn.Linear(heads, 1, bias=False)
        # self._init_weights(self.linear)
        self.avgpool = nn.AvgPool1d(heads)
        
        if GA_flag:
            self.normalization_GA = nn.LayerNorm(in_size)
            self.GA = GA_block([in_size[0]+1, in_size[1]], in_channels)

        self.GA_flag = GA_flag
        
    def _init_weights(self,layer):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias:
            nn.init.normal_(layer.bias, std=1e-6)    

    def forward(self, x, cls_token, dropout=True):
        '''
        [shape]
            x : (B, HW, C)
            x_inter1 : (B, HW, C)
            x_MHSA : (B, HW, C)
            x_res1 : (B, HW, C)
            x_inter2 : (B, HW, C)
            x_MLP : (B, HW, C)
            x_res2 : (B, HW, C)
        '''
        if not cls_token == None:
            x_in = torch.cat((cls_token, x), dim=1)
        else:
            x_in = x
        x_inter1 = self.normalization_1(x_in)
        '''
            Node update
        '''
        x_MHSA = self.MHSA(x_inter1, dropout)
        x_res1 = x_in + x_MHSA
        if not self.GA_flag:
            x_inter2 = self.normalization_2(x_res1)

        else:       
            '''
                Global attribute update
            '''
            edge_per_node = x_MHSA[:, 1:, 1:]
            
            x_inter2 = self.GA(x_res1, x_inter1[:, (0,)], edge_per_node, dropout)
            x_inter2 = self.normalization_GA(x_inter2)
            
        
        x_MLP = self.MLP(x_inter2, dropout)
        x_res2 = x_inter2 + x_MLP

        if not cls_token == None:
            return x_res2[:, 1:], x_res2[:, (0, )]        
        else:
            return x_res2


class Classifier_1d(nn.Module):
    def __init__(self, num_classes=10, in_channels=64):
        super(Classifier_1d, self).__init__()

        self.linear = nn.Linear(in_channels, num_classes, bias=False)
        self._init_weights(self.linear)
        self.name = 'Classifier'
        
    def _init_weights(self,layer):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias:
            nn.init.normal_(layer.bias, std=1e-6)

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

        self.linear = nn.Linear(in_channels, num_classes, bias=False)
        self._init_weights(self.linear)
        self.name = 'Classifier'

    def _init_weights(self,layer):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias:
            nn.init.normal_(layer.bias, std=1e-6)

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

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, inter_dimension, depth, mlp_ratio=4, heads=8, num_classes=10, GA=False, dropout=True, in_channels=3):
        super(ViT, self).__init__()

        self.inter_dimension = inter_dimension
        self.heads = heads

        
        

        # if img_size > 32:
        #     self.patch_embedding = Patch_Embedding(patch_size=patch_size, in_channels=in_channels, inter_channels=inter_dimension, down_conv=True)
        #     self.num_nodes = (self.num_nodes - 3) // 2 + 1
        #     self.in_size = ((self.num_nodes)**2 + 1, self.in_size[1])
            
        # else:   
        self.patch_embedding = Patch_Embedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, inter_channels=inter_dimension, down_conv=img_size>32)
      
        self.dropout = dropout
        
        # if not img_size > 32:
        #     self.num_nodes = img_size // patch_size
            # self.in_size = ((self.num_nodes)**2 + 1, inter_dimension)
        
        if not img_size > 32:
            self.num_nodes = (img_size - 3) / 2 + 1
            self.num_nodes = int((self.num_nodes - 3) / 2 + 1)            
            self.in_size = ((self.num_nodes)**2 + 1, inter_dimension)
            
        else:
            self.num_nodes = (img_size - 3) / 2 + 1
            self.num_nodes = (self.num_nodes - 3) / 2 + 1
            self.num_nodes = int((self.num_nodes - 3) / 2 + 1)            
            self.in_size = ((self.num_nodes)**2 + 1, inter_dimension)
        
        self.classifier = Classifier_1d(
        num_classes=num_classes, in_channels=inter_dimension)
        self.positional_embedding = Positional_Embedding(spatial_dimension=self.in_size[0]-1, inter_channels=inter_dimension)
        
        self.dropout_layer = nn.Dropout(0.1)
        
    
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.inter_dimension))
        
        self.transformers = self.make_layer(depth, Transformer_Block, mlp_ratio, GA_flag=GA)
        
        self.final_cls_token = None
        
        self.layer_norm = nn.LayerNorm((1, self.inter_dimension))


    def make_layer(self, num_blocks, block, mlp_ratio, GA_flag):
        layer_list = nn.ModuleList()
        for i in range(num_blocks):
            layer_list.append(
                block(self.in_size, self.inter_dimension, self.heads, mlp_ratio, GA_flag))

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

        
        
        if self.dropout:
            x_tmp = self.dropout_layer(x_tmp)
            cls_token = self.dropout_layer(cls_token)
        
        for i in range(len(self.transformers)):
            x_tmp, cls_token = self.transformers[i](x_tmp, cls_token, self.dropout)
        
        self.final_cls_token = cls_token.squeeze(1)
                
        x_out = self.classifier(self.layer_norm(cls_token))

        return x_out
        
