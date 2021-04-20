import torch
import torch.nn as nn
from torch.nn import functional as F

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


def ViT_Lite_7(EB=False, IB=False):
    return ViT(in_height=32, in_width=32, num_nodes=8*8, inter_dimension=256, depth=7, heads=4, mlp_ratio=2, EB=EB, IB=IB)


def ViT_Lite_6(EB=False, IB=False):
    return ViT(in_height=32, in_width=32, num_nodes=8*8, inter_dimension=256, depth=6, heads=4, mlp_ratio=2,EB=EB, IB=IB)


def PiT_Lite_7(EB=False, IB=False):
    return PiT(in_height=32, in_width=32, num_nodes=16*16, inter_channels=128, num_blocks=[1, 5, 1], heads=2,  mlp_ratio=2, EB=EB, IB=IB)


def PiT_Lite_6(EB=False, IB=False):
    return PiT(in_height=32, in_width=32, num_nodes=16*16, inter_channels=128, num_blocks=[1, 4, 1], heads=2,  mlp_ratio=2, EB=EB, IB=IB)
