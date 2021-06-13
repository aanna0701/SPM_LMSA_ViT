from numpy import numarray
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from models.vits import *




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
    def __init__(self, img_size, patch_size, inter_dimension, num_blocks, mlp_ratio=4, heads=8, num_classes=10, GA=False, dropout=False, in_channels=3, down_conv=False):
        super(ViT_pooling, self).__init__()

        self.in_channels = inter_dimension
        self.heads = heads
        self.patch_embedding = Patch_Embedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, inter_channels=inter_dimension, down_conv=down_conv)
        
        self.dropout = dropout
        
        if not img_size > 32:
            self.num_nodes = img_size // 2
            self.num_nodes = self.num_nodes // 2            
            self.in_size = ((self.num_nodes)**2 + 1, inter_dimension)
            
        else:
            self.num_nodes = img_size // 2
            self.num_nodes = self.num_nodes // 2
            self.num_nodes = self.num_nodes // 2            
            self.in_size = ((self.num_nodes)**2 + 1, inter_dimension)
        
        self.classifier = Classifier_1d(num_classes=num_classes, in_channels=inter_dimension)
        self.positional_embedding = Positional_Embedding(spatial_dimension=self.in_size[0]-1, inter_channels=inter_dimension)
        
        self.dropout_layer = nn.Dropout(0.1)
        
    
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.in_channels))
        
        self.transformers = nn.ModuleList()
                
        

        j = 0
        for i in range(len(num_blocks)):
            if not j+1 == len(num_blocks):
                self.make_layer(num_blocks[i], Transformer_Block, mlp_ratio, GA, True)
                j += 1            
                self.in_channels = 2 * self.in_channels
                self.heads = 2 * self.heads
                self.num_nodes = self.num_nodes // 2
                self.in_size = ((self.num_nodes)**2 +1, self.in_channels)
                
            else:
                self.make_layer(num_blocks[i], Transformer_Block, mlp_ratio, GA)
        
        self.classifier = Classifier_1d(num_classes=num_classes, in_channels=self.in_channels)
        self.layer_norm = nn.LayerNorm((self.in_channels))


    def make_layer(self, num_blocks, tr_block, mlp_ratio, GA_flag, pool_block=False):
        for _ in range(num_blocks):
            self.transformers.append(
                tr_block(self.in_size, self.in_channels, self.heads, mlp_ratio, GA_flag))
        if pool_block:

            self.transformers.append(Pooling_layer(self.in_channels))


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
                
        x_out = self.classifier(self.layer_norm(cls_token))

        return x_out
        