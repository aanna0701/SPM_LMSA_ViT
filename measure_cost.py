#!/usr/bin/env python

import torch
import torch.optim
import torch.utils.data
import torch

    

def main():

    img_size = 64
    patch_size = 8
    in_channels = 3

    GPU = 1
    
    # from models.vit_pytorch.swin import SwinTransformer

    # depths = [2, 6, 4]
    # num_heads = [3, 6, 12]
    # mlp_ratio = 2
    # window_size = 4
    # patch_size //= 2
        
    # model = SwinTransformer(img_size=img_size, window_size=window_size, patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=200)
     
    # from models.vit_pytorch.swin_cost import SwinTransformer
    from models.vit_pytorch.swin_cost import SwinTransformer

    patch_size = 8        

    depths = [2, 6, 4]
    num_heads = [3, 6, 12]
    mlp_ratio = 2
    window_size = 4
    patch_size //= 2
    
        
    model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=0, patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=200, is_base=False, num_trans=4)
      

        

    torch.cuda.set_device(GPU)
    from torchsummary import summary
    model.cuda(GPU)
    summary(model, (3, img_size, img_size))

    
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
    
    inputs = torch.randn(128, 3, img_size, img_size).cuda(GPU)
    repetitions=1000
    warmup = 200

    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions + warmup):
            if not rep < warmup:
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
