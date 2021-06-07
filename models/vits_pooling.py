import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time


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
        self.edge = similarity
        similarity = similarity / math.sqrt(inter_dimension)

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
            self.PE : (1, HW+1, C)
        '''
        self.PE = nn.Parameter(torch.zeros(
            1, spatial_dimension, inter_channels))
        

    def forward(self, x):
        print(x.shape)
        print(self.PE.shape)
        return x.permute(0, 2, 1) + self.PE


class Patch_Embedding(nn.Module):
    def __init__(self, patch_size, in_channels, inter_channels):
        super(Patch_Embedding, self).__init__()
        self.patch_embedding = nn.Conv2d(in_channels, inter_channels,
                                         kernel_size=patch_size, stride=patch_size, padding=0, bias=False)
        self._init_weights(self.patch_embedding)
        self.inter_channels = inter_channels

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
        '''
        out = self.patch_embedding(x)
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


class MLP_GA(nn.Module):

    def __init__(self, in_channels, out_channels, compression_ratio=4):
        super(MLP_GA, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = self.in_channels // compression_ratio
        self.out_channels = out_channels

        self.fc1 = nn.Linear(self.in_channels, self.inter_channels, bias=False)
        self._init_weights(self.fc1)
        self.fc2 = nn.Linear(self.inter_channels, self.out_channels, bias=False)
        self._init_weights(self.fc2)
        self.dropout = nn.Dropout(0.1)

    def _init_weights(self,layer):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias:
            nn.init.normal_(layer.bias, std=1e-6)

    def forward(self, x):
        '''
        [shape]
            x : (B, HW+1, C)
            x_inter : (B, HW+1, 4*C)
            x_out : (B, HW+1, C)
        '''

        x_inter = F.gelu(self.fc1(x))
        # if dropout:
        #     x_inter = self.dropout(x_inter)

        x_out = self.fc2(x_inter)
        # if dropout:
        #     x_out = self.dropout(x_out)

        return x_out


# class Pooling_block(nn.Module):
#     '''
#     Class-token Embedding
#     '''
#     def __init__(self, in_size, in_channels, pooling_patch_size=4):
#         super(Pooling_block, self).__init__()
#         # self.mlp = MLP_GA(in_channels, in_channels, 4)
#         self.avgpool = nn.AvgPool1d(in_size[0]-2)
#         # self.maxpool = nn.MaxPool1d(in_size[0]-2)
#         self.avgpool_2 = nn.AvgPool1d(in_size[0]-1)
#         # self.maxpool_2 = nn.MaxPool1d(in_size[0]-1)
#         # self.softmax = nn.Softmax()
#         self.sigmoid = nn.Sigmoid()
#         # self.tanh = nn.Tanh()
#         # self.softmax_score = nn.Softmax(dim=1)
#         self.cls = nn.Linear(in_channels, in_channels * 2, bias=False)
#         self._init_weights(self.cls)
#         self.channel = nn.Linear(in_channels, in_channels * 2, bias=False)
#         self._init_weights(self.channel)
        
#         self.layernorm = nn.LayerNorm([in_size[0]-1, in_size[1]])
#         self.score = nn.Softmax2d()
        
#         self.in_channels = in_channels
#         self.HW = in_size[0]-2
#         self.pp = int(math.sqrt(pooling_patch_size))
#         self.weights = nn.Parameter(torch.ones(in_channels, self.pp+1, self.pp+1))
        
#     def _init_weights(self,layer):
#         nn.init.kaiming_normal_(layer.weight)
#         if layer.bias:
#             nn.init.normal_(layer.bias, std=1e-6)
        
#     def forward(self, x, edge):
#         '''
#             [shape]
#             x : (B, HW+1, C)
#             edge_aggregation : (B, 1, C)
#             node_aggregation : (B, 1, C)
#             channel_importance : (B, 1, C)
#             nodes : (B, HW, C)
#             channel_aggregation : (B, HW, 1)  
#             node_importance : (B, HW, 1)
#         '''
#         pp = self.pp
#         B, _, _ = x.shape
#         weights_fixed = self.weights
#         weights_fixed = weights_fixed.unsqueeze(0)
#         weights_fixed = weights_fixed.expand(B, self.in_channels, pp+1, pp+1)
        
        
#         nodes = x[:, 1:]
        
#         edge_global = self.avgpool_2(edge.permute(0, 2, 1)).permute(0, 2, 1)
#         node_glboal = self.avgpool(nodes.permute(0, 2, 1)).permute(0, 2, 1)
        
#         norm = torch.norm(torch.cat([edge_global, node_glboal], dim=1), dim=2, keepdim=True)
#         scale_edge = torch.div(norm[:, (1,)], norm[:, (0, )]+norm[:, (1, )])
#         scale_node = torch.div(norm[:, (0,)], norm[:, (0, )]+norm[:, (1, )])
        
#         edge_global_scaled = torch.mul(scale_edge, edge_global)
#         node_global_scaled = torch.mul(scale_node, node_glboal)
        
#         concat_total_global = torch.cat([edge_global_scaled, node_global_scaled], dim=1) 
#         sigmoid_total_global = self.sigmoid(concat_total_global)
        
#         channel_importance = torch.sum(sigmoid_total_global, dim=1, keepdim=True)
        
#         scores = torch.matmul(nodes, channel_importance.permute(0, 2, 1))

        
#         # cls_token = self.nonlinear_cls(layernorm[:, (0,)])
         
#         norm_scores = torch.norm(scores, dim=1, keepdim=True)
#         norm_scores = norm_scores.expand(B, self.HW, 1)
#         scores_norm = torch.div(scores, norm_scores)
        
#         concat_horizontal = torch.cat([nodes, scores_norm], dim=2)
#         concat_2d = concat_horizontal.permute(0, 2, 1).view(B, self.in_channels+1, int(math.sqrt(self.HW)), -1)
        
#         nodes_2d = concat_2d[:, :-1]
#         # nodes_2d_pad = F.pad(nodes_2d, (0, 1, 0, 1))
#         scores_2d = concat_2d[:, (-1,)]
#         scores_2d = scores_2d.expand(B, self.in_channels, scores_2d.size(2), scores_2d.size(3))
#         # scores_2d = self.sigmoid(scores_2d)
#         # scores_2d_pad = F.pad(scores_2d, (0, 1, 0, 1))

#         loop = int(math.sqrt(x.size(1))) // pp

#         nodes_pooled_tmp = []
#         for i in range(loop):
#             if i == loop-1:
#                 h = pp
#                 weights_fixed_h = weights_fixed[:, :, :-1]
                
#             else:
#                 h = pp+1
#                 weights_fixed_h = weights_fixed
#             for j in range(loop):
#                 if j == loop-1:
#                     w = pp
#                     weights_fixed_templete = weights_fixed_h[:, :, :, :-1]
#                 else:
#                     w = pp+1
#                     weights_fixed_templete = weights_fixed_h
                
#                 score_sliced = scores_2d[:, :, pp*i:pp*i+h, pp*j:pp*j+w]
#                 score_sliced = self.score(score_sliced)


#                 # weights_2d = self.score(score_sliced.flatten(start_dim=2))
#                 # weights_2d = weights_2d.view(B, 1, pp, pp)              
#                 # weights_2d.expand(B, self.in_channels, pp, -1)
                
#                 weights_2d = torch.mul(weights_fixed_templete, score_sliced) + weights_fixed_templete

                                
#                 weighted_2d = torch.mul(nodes_2d[:, :, pp*i:pp*i+h, pp*j:pp*j+w], weights_2d)
#                 weighted_2d = weighted_2d.view(B, self.in_channels, -1)
#                 weighted_sum = torch.sum(weighted_2d, dim=-1, keepdim=True).permute(0, 2, 1)
#                 weighted_channel_expand = self.channel(weighted_sum)
    
#                 nodes_pooled_tmp.append(weighted_channel_expand)
        
        
#         nodes_pooled = torch.cat(nodes_pooled_tmp, dim=1)
       
   

#         # out_nodes = self.out_nodes(nodes_pooled)
#         # out_cls = self.out_cls(cls_token)
        
#         out = torch.cat([self.cls(x[:, (0,)]), nodes_pooled], dim=1)
        
#         return out


# class Pooling_block(nn.Module):
#     '''
#     Class-token Embedding
#     '''
#     def __init__(self, in_size, in_channels, pooling_patch_size=4):
#         super(Pooling_block, self).__init__()
#         # self.mlp = MLP_GA(in_channels, in_channels, 4)
#         self.avgpool = nn.AvgPool1d(in_size[0]-2)
#         # self.maxpool = nn.MaxPool1d(in_size[0]-2)
#         self.avgpool_2 = nn.AvgPool1d(in_size[0]-1)
#         # self.maxpool_2 = nn.MaxPool1d(in_size[0]-1)
#         # self.softmax = nn.Softmax()
#         self.sigmoid = nn.Sigmoid()
#         # self.tanh = nn.Tanh()
#         # self.softmax_score = nn.Softmax(dim=1)
#         self.linear = nn.Linear(in_channels, in_channels, bias=False)
#         self._init_weights(self.linear)
#         self.out = nn.Linear(in_channels, in_channels * 2, bias=False)
#         self._init_weights(self.out)
        
#         self.layernorm = nn.LayerNorm([in_size[0]-1, in_size[1]])
#         self.score = nn.Softmax2d()
        
#         self.in_channels = in_channels
#         self.HW = in_size[0]-2
#         self.pp = int(math.sqrt(pooling_patch_size))
#         self.weights = nn.Parameter(torch.ones(in_channels, self.pp+1, self.pp+1))
        
#     def _init_weights(self,layer):
#         nn.init.kaiming_normal_(layer.weight)
#         if layer.bias:
#             nn.init.normal_(layer.bias, std=1e-6)
        
#     def forward(self, x, edge):
#         '''
#             [shape]
#             x : (B, HW+1, C)
#             edge_aggregation : (B, 1, C)
#             node_aggregation : (B, 1, C)
#             channel_importance : (B, 1, C)
#             nodes : (B, HW, C)
#             channel_aggregation : (B, HW, 1)  
#             node_importance : (B, HW, 1)
#         '''
#         pp = self.pp
#         B, _, _ = x.shape
#         weights_fixed = self.weights
#         weights_fixed = weights_fixed.unsqueeze(0)
#         weights_fixed = weights_fixed.expand(B, self.in_channels, pp+1, pp+1)
        
        
#         nodes = x[:, 1:]
        
#         edge_aggregation = self.avgpool_2(edge.permute(0, 2, 1)).permute(0, 2, 1)
#         node_aggregation = self.avgpool(nodes.permute(0, 2, 1)).permute(0, 2, 1)
        
#         concat_aggregation = torch.cat([edge_aggregation, node_aggregation], dim=1)
        
#         sigmoid_aggregation = self.sigmoid(concat_aggregation)
        
#         channel_importance = torch.sum(sigmoid_aggregation, dim=1, keepdim=True)
#         channel_importance = self.linear(channel_importance)
        
#         scores = torch.matmul(nodes, channel_importance.permute(0, 2, 1))

        
#         # cls_token = self.nonlinear_cls(layernorm[:, (0,)])
         
#         norm_scores = torch.norm(scores, dim=1, keepdim=True)
#         norm_scores = norm_scores.expand(B, self.HW, 1)
#         scores_norm = torch.div(scores, norm_scores)
        
#         concat_horizontal = torch.cat([nodes, scores_norm], dim=2)
#         concat_2d = concat_horizontal.permute(0, 2, 1).view(B, self.in_channels+1, int(math.sqrt(self.HW)), -1)
        
#         nodes_2d = concat_2d[:, :-1]
#         # nodes_2d_pad = F.pad(nodes_2d, (0, 1, 0, 1))
#         scores_2d = concat_2d[:, (-1,)]
#         scores_2d = scores_2d.expand(B, self.in_channels, scores_2d.size(2), scores_2d.size(3))
#         # scores_2d = self.sigmoid(scores_2d)
#         # scores_2d_pad = F.pad(scores_2d, (0, 1, 0, 1))

#         loop = int(math.sqrt(x.size(1))) // pp

#         nodes_pooled_tmp = []
#         for i in range(loop):
#             if i == loop-1:
#                 h = pp
#                 weights_fixed_h = weights_fixed[:, :, :-1]
                
#             else:
#                 h = pp+1
#                 weights_fixed_h = weights_fixed
#             for j in range(loop):
#                 if j == loop-1:
#                     w = pp
#                     weights_fixed_templete = weights_fixed_h[:, :, :, :-1]
#                 else:
#                     w = pp+1
#                     weights_fixed_templete = weights_fixed_h
                
#                 score_sliced = scores_2d[:, :, pp*i:pp*i+h, pp*j:pp*j+w]
#                 score_sliced = self.score(score_sliced)


#                 # weights_2d = self.score(score_sliced.flatten(start_dim=2))
#                 # weights_2d = weights_2d.view(B, 1, pp, pp)              
#                 # weights_2d.expand(B, self.in_channels, pp, -1)
                
#                 weights_2d = torch.mul(weights_fixed_templete, score_sliced) + weights_fixed_templete

                                
#                 weighted_2d = torch.mul(nodes_2d[:, :, pp*i:pp*i+h, pp*j:pp*j+w], weights_2d)
#                 weighted_2d = weighted_2d.view(B, self.in_channels, -1)
#                 weighted_sum = torch.sum(weighted_2d, dim=-1, keepdim=True).permute(0, 2, 1)
    
#                 nodes_pooled_tmp.append(weighted_sum)
        
        
#         nodes_pooled = torch.cat(nodes_pooled_tmp, dim=1)
       
   

#         # out_nodes = self.out_nodes(nodes_pooled)
#         # out_cls = self.out_cls(cls_token)
        
#         out = torch.cat([x[:, (0,)], nodes_pooled], dim=1)
        
#         return self.out(out)
    
  
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
        # self.softmax = nn.Softmax()
        # self.mlp_nodes = nn.Linear(in_channels, in_channels, bias=False)
        self.linear = nn.Linear(in_channels, in_channels, bias=False)
        self._init_weights(self.linear)
        
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
        
        edge_global = self.avgpool_2(edge_per_node.permute(0, 2, 1)).permute(0, 2, 1)   # (B, 1, C)
        node_glboal = self.avgpool(nodes.permute(0, 2, 1)).permute(0, 2, 1)     # (B, 1, C)
        
        norm = torch.norm(torch.cat([edge_global, node_glboal], dim=1), dim=2, keepdim=True, p=1)
        scale_edge = torch.div(norm[:, (1,)], norm[:, (0, )]+norm[:, (1, )])
        scale_node = torch.div(norm[:, (0,)], norm[:, (0, )]+norm[:, (1, )])
        
        
        
        edge_global_scaled = torch.mul(scale_edge, edge_global)
        node_global_scaled = torch.mul(scale_node, node_glboal)
        
        
        channel_attention = edge_global_scaled + node_global_scaled # (B, 1, C)
        
        
        
        channel_attention = self.sigmoid(channel_attention) # (B, 1, C)
       
        
        cls_token_out = cls_token + torch.mul(self.linear(cls_token), channel_attention)    #(B, 1, C)
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
            edge_per_node = x_MHSA
            
            x_inter2 = self.GA(x_res1, cls_token, edge_per_node, dropout)
            x_inter2 = self.normalization_GA(x_inter2)
            
        
        x_MLP = self.MLP(x_inter2, dropout)
        x_res2 = x_inter2 + x_MLP

        if not cls_token == None:
            return x_res2[:, 1:], x_res2[:, (0, )]        
        else:
            return x_res2


class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size,
                              stride=patch_size, padding=0, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x
    

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


class Pooling_layer(nn.Module):
    def __init__(self, in_channels):
        super(Pooling_layer, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels)
        self.fcl = nn.Linear(in_channels, 2*in_channels)
        
    def forward(self, x, cls_token, dropout):
        '''
        [shape]
            x : (B, HW+1, C)
            cls_token : (B, 1, C)
            feature_token : (B, HW, C)
            feature_token_reshape : (B, C, H, W)
            pool_feature_token : (B, (H/2)(W/2), 2*C)
            out : (B, (HW/2)+1, 2*C)
        '''
        cls_token = cls_token
        feature_token = x
        
        B, HW, C = feature_token.shape
        H = int(math.sqrt(HW))
        feature_token_reshape = feature_token.permute(0, 2, 1).contiguous()
        feature_token_reshape = feature_token_reshape.view((B, C, H, H))
                
        pool_feature_token = self.conv(feature_token_reshape)
        pool_feature_token = pool_feature_token.view((B, 2*C, -1)).permute(0, 2, 1)
        pool_cls_token = self.fcl(cls_token)
        
        
        return pool_feature_token, pool_cls_token


class ViT_pooling(nn.Module):
    def __init__(self, in_size, num_nodes, inter_dimension, num_blocks, mlp_ratio=4, heads=8, num_classes=10, GA=False, dropout=True, pooling='max'):
        super(ViT_pooling, self).__init__()

        self.in_channels = inter_dimension
        self.heads = heads
        

        self.in_size = (num_nodes + 1, inter_dimension)

        self.patch_embedding = Patch_Embedding(
            patch_size=int(math.sqrt((in_size * in_size) // num_nodes)), in_channels=3, inter_channels=inter_dimension)
        
        self.dropout = dropout
        
        self.classifier = Classifier_1d(
        num_classes=num_classes, in_channels=inter_dimension)
        self.positional_embedding = Positional_Embedding(
        spatial_dimension=num_nodes, inter_channels=inter_dimension)
        
        self.dropout_layer = nn.Dropout(0.1)
        
    
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.in_channels))
        
        self.transformers = nn.ModuleList()
        
        self.pooling = pooling

        j = 0
        for i in range(len(num_blocks)):
            if not j+1 == len(num_blocks):
                self.make_layer(num_blocks[i], Transformer_Block, mlp_ratio, GA, True)
                j += 1
                if pooling=='conv':
                    self.in_channels = 2 * self.in_channels
                    self.heads = 2 * self.heads
                    self.in_size = [((self.in_size[0]-1) // 4) + 1, self.in_size[1]*2]
                else:
                    self.in_size = [((self.in_size[0]-1) // 4) + 1, self.in_size[1]]
                
            else:
                self.make_layer(num_blocks[i], Transformer_Block, mlp_ratio, GA)
        
        self.classifier = Classifier_1d(
            num_classes=num_classes, in_channels=self.in_channels)        


    def make_layer(self, num_blocks, tr_block, mlp_ratio, GA_flag, pool_block=False):
        for _ in range(num_blocks):
            self.transformers.append(
                tr_block(self.in_size, self.in_channels, self.heads, mlp_ratio, GA_flag))
        if pool_block:

            if self.pooling == 'conv':
                self.transformers.append(
                    Pooling_layer(self.in_channels))


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
                
        x_out = self.classifier(cls_token)

        return x_out
        