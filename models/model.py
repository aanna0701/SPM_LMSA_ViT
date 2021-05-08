
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


def ViT_Lite(depth, channel, GA=False, cls_token=True, heads=4, dropout=True):
    if cls_token:
        return ViT(in_height=32, in_width=32, num_nodes=8*8, inter_dimension=channel, depth=depth, heads=heads, mlp_ratio=2, GA=GA, dropout=dropout)
    




def PiT_Lite(channel, GA=False, heads=4):
    return PiT(in_height=32, in_width=32, num_nodes=16*16, inter_channels=channel, num_blocks=[1, 5, 1], heads=heads,  mlp_ratio=2, GA=GA)



    
    