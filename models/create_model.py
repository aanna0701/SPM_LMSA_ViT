
from models.nlb import *
from models.resnet_cifar import *
from models.vits import *
from models.vits_pooling import *


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



def make_ViT(depth, channel, down_conv=False, GA=False, cls_token=True, heads=4, dropout=False, num_classes=10, img_size=32, patch_size=4, in_channels=3):
    if cls_token:
        return ViT(img_size=img_size, inter_dimension=channel, depth=depth, heads=heads, mlp_ratio=2, GA=GA, dropout=dropout, num_classes=num_classes, patch_size=patch_size, in_channels=in_channels, down_conv=down_conv)


def P_ViT_conv(version, img_size=32, patch_size=4, GA=False, num_classes=10, dropout=False, in_channels=3, down_conv=False):
    if version == 1:
        return ViT_pooling(img_size=img_size, patch_size=patch_size, inter_dimension=18, num_blocks=[1, 3, 2], heads=2, mlp_ratio=2, GA=GA, dropout=dropout, num_classes=num_classes, in_channels=in_channels, down_conv=down_conv)
    elif version == 2:
        return ViT_pooling(img_size=img_size, patch_size=patch_size, inter_dimension=24, num_blocks=[1, 3, 2], heads=2, mlp_ratio=2, GA=GA, dropout=dropout, num_classes=num_classes, in_channels=in_channels, down_conv=down_conv)
    elif version == 3:
        return ViT_pooling(img_size=img_size, patch_size=patch_size, inter_dimension=30, num_blocks=[1, 3, 2], heads=3, mlp_ratio=2, GA=GA, dropout=dropout, num_classes=num_classes, in_channels=in_channels, down_conv=down_conv)
    elif version == 4:
        return ViT_pooling(img_size=img_size, patch_size=patch_size, inter_dimension=36, num_blocks=[1, 3, 2], heads=3, mlp_ratio=2, GA=GA, dropout=dropout, num_classes=num_classes, in_channels=in_channels, down_conv=down_conv)
    elif version == 5:
        return ViT_pooling(img_size=img_size, patch_size=patch_size, inter_dimension=36, num_blocks=[2, 6, 4], heads=3, mlp_ratio=2, GA=GA, dropout=dropout, num_classes=num_classes, in_channels=in_channels, down_conv=down_conv)


def P_GiT_conv(version, img_size=32, patch_size=4, GA=True, num_classes=10, dropout=False, in_channels=3, down_conv=False):
    if version == 1:
        return ViT_pooling(img_size=img_size, patch_size=patch_size, inter_dimension=18, num_blocks=[1, 3, 2], heads=2, mlp_ratio=2, GA=GA, dropout=dropout, num_classes=num_classes, in_channels=in_channels, down_conv=down_conv)
    elif version == 2:
        return ViT_pooling(img_size=img_size, patch_size=patch_size, inter_dimension=24, num_blocks=[1, 3, 2], heads=2, mlp_ratio=2, GA=GA, dropout=dropout, num_classes=num_classes, in_channels=in_channels, down_conv=down_conv)
    elif version == 3:
        return ViT_pooling(img_size=img_size, patch_size=patch_size, inter_dimension=30, num_blocks=[1, 3, 2], heads=3, mlp_ratio=2, GA=GA, dropout=dropout, num_classes=num_classes, in_channels=in_channels, down_conv=down_conv)
    elif version == 4:
        return ViT_pooling(img_size=img_size, patch_size=patch_size, inter_dimension=36, num_blocks=[1, 3, 2], heads=3, mlp_ratio=2, GA=GA, dropout=dropout, num_classes=num_classes, in_channels=in_channels, down_conv=down_conv)
    elif version == 5:
        return ViT_pooling(img_size=img_size, patch_size=patch_size, inter_dimension=36, num_blocks=[2, 6, 4], heads=3, mlp_ratio=2, GA=GA, dropout=dropout, num_classes=num_classes, in_channels=in_channels, down_conv=down_conv)


