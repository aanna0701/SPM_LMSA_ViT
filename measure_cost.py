#!/usr/bin/env python

import logging as log
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

    

def main():

    img_size = 64
    patch_size = 8
    in_channels = 3

    GPU = 1
    
    from models.vit_pytorch.t2t import T2T_ViT

    model = T2T_ViT(img_size=img_size, num_classes=200)
        

    torch.cuda.set_device(GPU)
    from torchsummary import summary
    model.cuda(GPU)
    summary(model, (3, img_size, img_size))

    inputs = torch.randn(128, 3, img_size, img_size).cuda(GPU)
    # INIT LOGGERS

    # # Inference Time
    # import numpy as np
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 300
    # timings=np.zeros((repetitions,1))
    # #GPU-WARM-UP
    # for _ in range(10):
    #     _ = model(inputs)
    # # MEASURE PERFORMANCE
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter.record()
    #         _ = model(inputs)
    #         ender.record()
    #         # WAIT FOR GPU SYNC
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[rep] = curr_time
    # mean_syn = np.sum(timings) / repetitions
    # std_syn = np.std(timings)
    # print(mean_syn)
    
    # Throughput
    repetitions=100
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(inputs)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time
    Throughput =   (repetitions*128)/total_time
    print('Final Throughput:',Throughput)
        
    # # RF
    # from torchvision import models
    # from torchsummary import summary
    # vit = ViT(img_size=224, patch_size=16, dim=192)
    
    # resnet18 = models.resnet18(pretrained=False)
    # torch.cuda.set_device(GPU)
    # vit.cuda(GPU)
    # resnet18.cuda(GPU)
    
    # # summary(vit, (3, 224, 224))
    # # summary(resnet18, (3, 224, 224))
    # from torch_receptive_field import receptive_field
    # receptive_field_dict = receptive_field(vit, (3, 224, 224))
    # receptive_field_for_unit(receptive_field_dict, "2", (2,2))


if __name__ == '__main__':
        main()
