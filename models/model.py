import torch
import torch.nn as nn
from torch.nn import functional as F

from models.nlb import *
from models.resnet_cifar import *
from models.transformer import *


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


def ViT_Ti_cifar(EB=False, IB=False):
    if not EB:
        return ViT(in_height=32, in_width=32, num_nodes=8*8, inter_dimension=96, depth=12, heads=3, EB=EB, IB=IB)
    else:
        return ViT(in_height=32, in_width=32, num_nodes=8*8, inter_dimension=96, depth=12, heads=3, EB=EB, IB=IB)


def ViT_S_cifar(EB=False, IB=False):
    if not EB:
        return ViT(in_height=32, in_width=32, num_nodes=8*8, inter_dimension=192, depth=12, heads=6, EB=EB, IB=IB)
    else:
        return ViT(in_height=32, in_width=32, num_nodes=8*8, inter_dimension=192, depth=12, heads=6, EB=EB, IB=IB)


def ViT_B_cifar(EB=False, IB=False):
    if not EB:
        return ViT(in_height=32, in_width=32, num_nodes=8*8, inter_dimension=384, depth=12, heads=12, EB=EB, IB=IB)
    else:
        return ViT(in_height=32, in_width=32, num_nodes=8*8, inter_dimension=384, depth=12, heads=12, EB=EB, IB=IB)
