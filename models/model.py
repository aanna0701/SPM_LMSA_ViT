
from models.nlb import *
from models.resnet_cifar import *
from models.vits import *


def resnet20():
    return ResNet(ResNet_Encoder, Classifier_2d, [3, 3, 3])


def resnet32():
    return ResNet(ResNet_Encoder, Classifier_2d, [5, 5, 5])


def resnet44():
    return ResNet(ResNet_Encoder, Classifier_2d, [7, 7, 7])


def resnet56():
    return ResNet(ResNet_Encoder, Classifier_2d)


def self_attention_ResNet56(num_sa_block, global_attribute=False):
    return Self_Attention_res56(ResNet_Encoder, NLB, Classifier_2d, Down_Conv, num_sa_block, _Global_attribute=global_attribute)


def self_Attention_full(global_attribute=False):
    return Self_Attention_full(NLB, Classifier_2d, Down_Conv, _Global_attribute=global_attribute)


def make_ViT(depth, channel, GA=False, cls_token=True, heads=4, dropout=True):
    if cls_token:
        return ViT(in_height=32, in_width=32, num_nodes=8*8, inter_dimension=channel, depth=depth, heads=heads, mlp_ratio=2, GA=GA, dropout=dropout)
    




def P_ViT_max(num_blocks, channel, GA=False, heads=4, dropout=True, pooling='max'):
    if num_blocks == 1:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=36, num_blocks=[2, 2, 2], heads=3, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 2:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=48, num_blocks=[2, 2, 2], heads=4, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 3:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=48, num_blocks=[3, 3, 3], heads=4, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 4:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=48, num_blocks=[4, 4, 4], heads=4, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 5:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=72, num_blocks=[4, 4, 4], heads=6, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)

def P_GiT_max(num_blocks, channel, GA=True, heads=4, dropout=True, pooling='max'):
    if num_blocks == 1:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=36, num_blocks=[2, 2, 2], heads=3, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 2:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=48, num_blocks=[2, 2, 2], heads=4, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 3:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=48, num_blocks=[3, 3, 3], heads=4, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 4:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=48, num_blocks=[4, 4, 4], heads=4, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 5:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=72, num_blocks=[4, 4, 4], heads=6, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)


def P_ViT_conv(num_blocks, channel, GA=False, heads=4, dropout=True, pooling='conv'):
    if num_blocks == 1:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=18, num_blocks=[1, 3, 2], heads=2, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 2:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=24, num_blocks=[1, 3, 2], heads=2, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 3:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=24, num_blocks=[2, 4, 3], heads=2, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 4:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=24, num_blocks=[3, 5, 4], heads=2, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 5:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=36, num_blocks=[2, 6, 4], heads=3, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)


def P_GiT_conv(num_blocks, channel, GA=True, heads=4, dropout=True, pooling='conv'):
    if num_blocks == 1:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=18, num_blocks=[1, 3, 2], heads=2, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 2:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=24, num_blocks=[1, 3, 2], heads=2, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 3:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=24, num_blocks=[2, 4, 3], heads=2, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 4:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=24, num_blocks=[2, 6, 4], heads=2, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)
    elif num_blocks == 5:
        return ViT_pooling(in_height=32, in_width=32, num_nodes=16*16, inter_dimension=36, num_blocks=[2, 6, 4], heads=3, mlp_ratio=2, GA=GA, dropout=dropout, pooling=pooling)



    
    