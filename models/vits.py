import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from models.deepsets import Equivariant_Block


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
        return x.permute(0, 2, 1) + self.PE


class Patch_Embedding(nn.Module):
    def __init__(self, patch_size, in_channels, inter_channels):
        super(Patch_Embedding, self).__init__()
        self.patch_embedding = nn.Conv2d(in_channels, inter_channels,
                                         kernel_size=patch_size, stride=patch_size, bias=False)
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

    
# class GA_block(nn.Module):
#     '''
#     Class-token Embedding
#     '''
#     def __init__(self, in_size, in_channels):
#         super(GA_block, self).__init__()
#         # self.mlp = MLP_GA(in_channels, in_channels, 4)
#         self.in_dimension = in_channels
#         self.avgpool = nn.AvgPool1d(in_size[0]-2)
#         # self.maxpool = nn.MaxPool1d(in_size[0]-2)
#         self.avgpool_2 = nn.AvgPool1d(in_size[0]-1)
#         # self.maxpool_2 = nn.MaxPool1d(in_size[0]-1)
#         self.sigmoid = nn.Sigmoid()
#         # self.Linear = nn.Linear(in_channels, in_channels)
        
#     def forward(self, x, cls_token, edge_aggregation):
#         '''
#             [shape]
#             x : (B, HW+1, C)
#             visual_token : (B, HW, C)
#             cls_token_in : (B, 1, C)
#             weight : (B, 1, HW)
#             weight_softmax : (B, 1, HW)
#             cls_token_out : (B, 1, C)
#             out : (B, HW+1, C)     
#         '''
        
        
#         edge_aggregation_avgpool = self.avgpool_2(edge_aggregation.permute(0, 2, 1)).permute(0, 2, 1)
#         node_aggregation_avgpool = self.avgpool(x[:, 1:].permute(0, 2, 1)).permute(0, 2, 1)
#         norm_edge_aggreation = edge_aggregation_avgpool.norm(2)
#         norm_node_aggreation = node_aggregation_avgpool.norm(2)
#         norm_total = norm_edge_aggreation + norm_node_aggreation
    
#         # x_norm = self.normalization(torch.cat([edge_aggregation_avgpool, visual_token_avgpool], dim=1))      
#         # edge_aggregation_norm, visual_token_norm = x_norm[:, (0, )], x_norm[:, (1, )]
#         edge_aggregation_norm = (edge_aggregation_avgpool / norm_total) * norm_node_aggreation
#         node_aggregation_norm = (node_aggregation_avgpool / norm_total) * norm_edge_aggreation
        
#         # visual_toekn_avgpool = self.avgpool(visual_token_norm.permute(0, 2, 1)).permute(0, 2, 1)
        
#         # node_aggregation = torch.cat([visual_token_maxpool, visual_toekn_avgpool], dim=1)
#         # node_aggregation = self.mlp(node_aggregation)
        
#         # maxpool_embedding = node_aggregation[:, (0, )]
#         # avgpool_embedding = node_aggregation[:, (1, )]
#         # node_aggregation = maxpool_embedding + avgpool_embedding
        
#         total_aggregation = edge_aggregation_norm + node_aggregation_norm
#         weighting = self.sigmoid(total_aggregation)
                
        
#         total_aggregation_out = torch.mul(weighting, cls_token)
        

        
#         cls_token_out = cls_token + total_aggregation_out
        
#         # print('weighting: {}'.format(weighting[0].norm(2)))
#         # print('total_aggregation_out: {}'.format(total_aggregation_out[0].norm(2)))
#         # print('cls_token: {}'.format(cls_token[0].norm(2)))
#         # print('cls_token_out: {}'.format(cls_token_out[0].norm(2)))
#         # print('==========================================')
        
#         out = torch.cat((cls_token_out, x[:, 1:]), dim=1)
        
        
#         return out
    

class GA_block(nn.Module):
    '''
    Class-token Embedding
    '''
    def __init__(self, in_size, in_channels):
        super(GA_block, self).__init__()
        # self.mlp = MLP_GA(in_channels, in_channels, 4)
        self.in_dimension = in_channels
        # self.avgpool = nn.AvgPool1d(in_size[0]-2)
        # self.maxpool = nn.MaxPool1d(in_size[0]-2)
        self.avgpool_2 = nn.AvgPool1d(in_size[0]-1)
        # self.maxpool_2 = nn.MaxPool1d(in_size[0]-1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.c = nn.Linear(in_channels, in_channels, bias=False)
        # self.mlp_nodes = nn.Linear(in_channels, in_channels, bias=False)
        self.Linear = nn.Linear(in_channels, in_channels, bias=False)
        
    def forward(self, x, cls_token, edge_aggregation, pool=False):
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
        # nodes = x[:, 1:]
        
        # edge_aggregation = self.avgpool_2(edge_aggregation.permute(0, 2, 1)).permute(0, 2, 1)
        # node_aggregation = self.avgpool(nodes.permute(0, 2, 1)).permute(0, 2, 1)
        
        # edge_aggregation_sigmoid = self.sigmoid(edge_aggregation)
        # node_aggregation_sigmoid = self.sigmoid(node_aggregation)
        
        # channel_importance = edge_aggregation_sigmoid + node_aggregation_sigmoid
        # channel_importance = self.linear(channel_importance)
        
        # channel_aggregation = torch.matmul(nodes, channel_importance.permute(0, 2, 1))
        # node_importance = self.softmax(channel_aggregation)
        
        edge_aggregation = self.avgpool_2(edge_aggregation.permute(0, 2, 1)).permute(0, 2, 1)
        channel_importance = self.softmax(edge_aggregation)
                
        nodes = x[:, 1:]
        
        channel_aggregation = torch.matmul(nodes, channel_importance.permute(0, 2, 1))
        node_importance = self.softmax(channel_aggregation)
        
        node_aggregation = torch.matmul(node_importance.permute(0, 2, 1), nodes)
        weights = self.sigmoid(node_aggregation)
        
        cls_token_out = cls_token + torch.mul(self.Linear(cls_token), weights)
        
        # cls_token_out = cls_token + node_aggregation
        
                
        # print('edge_aggregation: {}'.format(channel_importance[0].norm(2)))
        # print('channel_aggregation: {}'.format(node_importance[0].norm(2)))
        # print('cls_token: {}'.format(cls_token[0].norm(2)))
        # print('nod_aggregation: {}'.format(node_aggregation[0].norm(2)))
        # print('cls_token_out: {}\n\n'.format(cls_token_out[0].norm(2)))
        # print('='*20)
    
        if pool:
        
            _, idx = torch.sort(channel_aggregation, dim=1)
        
            
        
            tmp = []
            for i in range(x.size(0)):
                node_sorted = nodes[i][idx[i].squeeze(-1).tolist()]
                tmp.append(node_sorted.unsqueeze(0))
                
            nodes_sorted = torch.cat(tmp, dim=0)
            nodes_pooled = nodes_sorted[:, nodes_sorted.size(1)//4*3:]
            
            
            out = torch.cat((cls_token_out, nodes_pooled), dim=1)
        
        else:
            out = torch.cat((cls_token_out, x[:, 1:]), dim=1)
        
        
        return out
    
# class GA_block(nn.Module):
#     '''
#     Class-token Embedding
#     '''
#     def __init__(self, in_channels, heads):
#         super(GA_block, self).__init__()

#         self.query = nn.Linear(in_channels, in_channels // 2, bias=False)
#         self._init_weights(self.query)
#         self.key = nn.Linear(in_channels, in_channels // 2, bias=False)
#         self._init_weights(self.key)
#         self.value = nn.Linear(in_channels, in_channels // 2, bias=False)
#         self._init_weights(self.value)
#         self.out = nn.Linear(in_channels // 2, in_channels, bias=False)
#         self._init_weights(self.out)
#         self.softmax = nn.Softmax(dim=-1)
#         self.inter_channels = in_channels 
#         self.heads = heads
        
#     def _init_weights(self,layer):
#         nn.init.kaiming_normal_(layer.weight)
#         if layer.bias:
#             nn.init.normal_(layer.bias, std=1e-6)
        
#     def forward(self, x, cls_token):
#         '''
#             [shape]
#             x : (B, HW+1, C)
#             visual_token : (B, HW, C)
#             cls_token_in : (B, 1, C)
#             weight : (B, 1, HW)
#             weight_softmax : (B, 1, HW)
#             cls_token_out : (B, 1, C)
#             out : (B, HW+1, C)     
#         '''

#         visual_token, cls_token_in = x[:, 1:], cls_token

#         q = self.query(cls_token)
#         k = self.key(visual_token)
#         v = self.value(visual_token)

#         B, _, C = q.size()

#         similarity = torch.matmul(q, k.permute(0, 2, 1))
#         similarity = similarity / math.sqrt(C)

#         attention =  self.softmax(similarity)

#         out_tmp = torch.matmul(attention, v)
        
#         out = self.out(out_tmp)
#         cls_token_out = out + cls_token_in
        
#         out = torch.cat((cls_token_out, visual_token), dim=1)
        
#         return out


class Transformer_Block(nn.Module):
    def __init__(self, in_size, in_channels, heads=8, mlp_ratio=4, GA_flag=False):
        super(Transformer_Block, self).__init__()
        self.normalization_1 = nn.LayerNorm(in_size)
        if not GA_flag:
            self.normalization_2 = nn.LayerNorm(in_size)
        
        
        self.MHSA = MHSA(in_channels, heads)
        self.MLP = MLP(in_channels, mlp_ratio)
        self.MLP_MHSA = MLP(in_channels, mlp_ratio)
        
        if GA_flag:
            self.normalization_GA = nn.LayerNorm(in_size)
            self.GA = GA_block([in_size[0]+1, in_size[1]], in_channels)

        self.GA_flag = GA_flag

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
            edge_aggregation = x_MHSA
            
            x_inter2 = self.GA(x_res1, cls_token, edge_aggregation, False)
            x_inter2 = self.normalization_GA(x_inter2)
            
        
        x_MLP = self.MLP(x_inter2, dropout)
        x_res2 = x_inter2 + x_MLP

        if not cls_token == None:
            return x_res2[:, 1:], x_res2[:, (0, )]        
        else:
            return x_res2

class Transformer_Block_pool(nn.Module):
    def __init__(self, in_size, in_channels, heads=8, mlp_ratio=4, GA_flag=False, ):
        super(Transformer_Block_pool, self).__init__()
        self.normalization_1 = nn.LayerNorm(in_size)
        if not GA_flag:
            self.normalization_2 = nn.LayerNorm(in_size)
        
        
        self.MHSA = MHSA(in_channels, heads)
        self.MLP = MLP(in_channels, mlp_ratio)
        self.MLP_MHSA = MLP(in_channels, mlp_ratio)
        
        if GA_flag:
            self.normalization_GA = nn.LayerNorm([(in_size[0]//4)+1, in_size[1]])
            self.GA = GA_block([in_size[0]+1, in_size[1]], in_channels)

        self.GA_flag = GA_flag

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
            edge_aggregation = x_MHSA
            
            x_inter2 = self.GA(x_res1, cls_token, edge_aggregation, True)
            x_inter2 = self.normalization_GA(x_inter2)
            
        
        x_MLP = self.MLP(x_inter2, dropout)
        x_res2 = x_inter2 + x_MLP

        if not cls_token == None:
            return x_res2[:, 1:], x_res2[:, (0, )]        
        else:
            return x_res2

# class Transformer_Block(nn.Module):
#     def __init__(self, in_size, in_channels, heads=8, mlp_ratio=4, GA_flag=False):
#         super(Transformer_Block, self).__init__()
#         self.normalization_1 = nn.LayerNorm(in_size)
#         self.normalization_2 = nn.LayerNorm(in_size)
        
        
#         self.MHSA = MHSA(in_channels, heads)
#         self.MLP = MLP(in_channels, mlp_ratio)
        
#         if GA_flag:
#             self.normalization_GA = nn.LayerNorm(in_size)
#             self.GA = GA_block([in_size[0]+1, in_size[1]], in_channels)

#         self.GA_flag = GA_flag

#     def forward(self, x, cls_token, dropout=True):
#         '''
#         [shape]
#             x : (B, HW, C)
#             x_inter1 : (B, HW, C)
#             x_MHSA : (B, HW, C)
#             x_res1 : (B, HW, C)
#             x_inter2 : (B, HW, C)
#             x_MLP : (B, HW, C)
#             x_res2 : (B, HW, C)
#         '''
#         if not cls_token == None:
#             x_in = torch.cat((cls_token, x), dim=1)
#         else:
#             x_in = x
#         x_inter1 = self.normalization_1(x_in)
#         '''
#             Node update
#         '''
#         x_MHSA = self.MHSA(x_inter1, dropout)
#         x_res1 = x_in + x_MHSA
#         if not self.GA_flag:
#             x_inter2 = self.normalization_2(x_res1)

#         else:       
#             '''
#                 Global attribute update
#             '''
#             edge_aggregation = x_MHSA[:, (0, )]
            
#             x_inter2 = self.GA(x_res1, cls_token, edge_aggregation)
#             x_inter2 = self.normalization_GA(x_inter2)
            
        
#         x_MLP = self.MLP(x_inter2, dropout)
#         x_res2 = x_inter2 + x_MLP

#         if not cls_token == None:
#             return x_res2[:, 1:], x_res2[:, (0, )]        
#         else:
#             return x_res2


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
    def __init__(self, in_height, in_width, num_nodes, inter_dimension, depth, mlp_ratio=4, heads=8, num_classes=10, GA=False, dropout=True):
        super(ViT, self).__init__()

        self.inter_dimension = inter_dimension
        self.heads = heads

        self.in_size = (num_nodes + 1, inter_dimension)

        self.patch_embedding = Patch_Embedding(
            patch_size=int(math.sqrt((in_height * in_width) // num_nodes)), in_channels=3, inter_channels=inter_dimension)
        
        self.dropout = dropout
        
        self.classifier = Classifier_1d(
        num_classes=num_classes, in_channels=inter_dimension)
        self.positional_embedding = Positional_Embedding(
        spatial_dimension=num_nodes, inter_channels=inter_dimension)
        
        self.dropout_layer = nn.Dropout(0.1)
        
    
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.inter_dimension))
        
        self.transformers = self.make_layer(depth, Transformer_Block, mlp_ratio, GA_flag=GA)


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
                
        x_out = self.classifier(cls_token)

        return x_out
        
