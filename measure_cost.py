#!/usr/bin/env python

import torch
import torch.optim
import torch.utils.data
import torch

    

def main():

    img_size = 64
    patch_size = 8

    GPU = 0
    

    # # '''
    # # SWIN
    # # '''    
    
    # from models.vit_pytorch.swin_flops import SwinTransformer

    # patch_size = 8        

    # depths = [2, 6, 4]
    # num_heads = [3, 6, 12]
    # mlp_ratio = 2
    # window_size = 4
    # patch_size //= 2
    
        
    # model_base = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=0, patch_size=patch_size, mlp_ratio=mlp_ratio, 
    #                         depths=depths, num_heads=num_heads, num_classes=200, is_base=True)
        
    # model_sl_wo_pool = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=0, patch_size=patch_size, mlp_ratio=mlp_ratio, 
    #                         depths=depths, num_heads=num_heads, num_classes=200, is_base=False, is_s_pool=False)
   
    # model_sl = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=0, patch_size=patch_size, mlp_ratio=mlp_ratio, 
    #                         depths=depths, num_heads=num_heads, num_classes=200, is_base=False)

        

    # torch.cuda.set_device(GPU)
    # from torchsummary import summary
    # def to_gpu_summary(m):        
    #     m.cuda(GPU)
    #     summary(m, (3, img_size, img_size))
        
    # # to_gpu_summary(model_base)
    # # to_gpu_summary(model_sl_wo_pool)
    # # to_gpu_summary(model_sl)
        
    
    # # FLOPs
    
    # print()
    # print('Swin Flops:',model_base.flops())
    # print('SL-Swin_wo_pool Flops:',model_sl_wo_pool.flops())
    # print('SL-Swin Flops:',model_sl.flops())
    
    # '''
    # ViT
    # '''    
    
    # from models.vit_pytorch.vit_flops import ViT

    # patch_size = 8        

    # dim = 192
    # depth = 9
    # heads = 12
    # mlp_dim_ratio = 2
    
        
    # model_base = ViT(img_size, patch_size, 200, dim, depth, heads, mlp_dim_ratio, pool = 'cls', channels = 3, 
    #              dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0., is_base=True)
        
    # model_sl = ViT(img_size, patch_size, 200, dim, depth, heads, mlp_dim_ratio, pool = 'cls', channels = 3, 
    #              dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0., is_base=False)

        

    # torch.cuda.set_device(GPU)
    # from torchsummary import summary
    # def to_gpu_summary(m):        
    #     m.cuda(GPU)
    #     summary(m, (3, img_size, img_size))
        
    # # to_gpu_summary(model_base)
    # # to_gpu_summary(model_sl)
        
    
    # # FLOPs
    
    # print()
    # print('ViT Flops:',model_base.flops())
    # print('SL-ViT Flops:',model_sl.flops())
    
    # '''
    # T2T
    # '''    
    
    # from models.vit_pytorch.t2t_flops import T2T_ViT

   
        
    # model_base = T2T_ViT(img_size=img_size, num_classes=200, drop_path_rate=0, is_base=True)
        
    # model_sl = T2T_ViT(img_size=img_size, num_classes=200, drop_path_rate=0, is_base=False)

        

    # torch.cuda.set_device(GPU)
    # from torchsummary import summary
    # def to_gpu_summary(m):        
    #     m.cuda(GPU)
    #     summary(m, (3, img_size, img_size))
        
    # # to_gpu_summary(model_base)
    # # to_gpu_summary(model_sl)
        
    
    # # FLOPs
    
    # print()
    # print('T2T Flops:',model_base.flops())
    # print('SL-T2T Flops:',model_sl.flops())
    
    
    # '''
    # CaiT
    # '''    
    
    # from models.vit_pytorch.cait_flops import CaiT

    # patch_size = 8        

    # if img_size == 64:
    #     patch_size = 8
    # elif img_size == 32:
    #     patch_size = 4
    # else:
    #     patch_size = 16
    
        
    # model_base = CaiT(img_size=img_size, patch_size = patch_size, num_classes=200, stochastic_depth=0, is_base=True)
        
    # model_sl = CaiT(img_size=img_size, patch_size = patch_size, num_classes=200, stochastic_depth=0, is_base=False)

        

    # torch.cuda.set_device(GPU)
    # from torchsummary import summary
    # def to_gpu_summary(m):        
    #     m.cuda(GPU)
    #     summary(m, (3, img_size, img_size))
        
    # # to_gpu_summary(model_base)
    # # to_gpu_summary(model_sl)
    # print()    
    
    # # FLOPs
    
    
    # print('CaiT Flops:',model_base.flops())
    # print('SL-CaiT Flops:',model_sl.flops())
    
    
    
    # '''
    # PiT
    # '''    
    
    # from models.vit_pytorch.pit_flops import PiT
       

    # if img_size == 32:
    #     patch_size = 2
    # elif img_size > 32:
    #     patch_size = 4
    

    # channel = 96
    # heads = (2, 4, 8)
    # depth = (2, 6, 4)
    
    # dim_head = channel // heads[0]
    
        
    # model_base = PiT(img_size=img_size, patch_size = patch_size, num_classes=200, dim=channel, 
    #                 mlp_dim_ratio=2, depth=depth, heads=heads, dim_head=dim_head, dropout=0, 
    #                 stochastic_depth=0, is_base=True)
        
    # model_sl_wo_pool = PiT(img_size=img_size, patch_size = patch_size, num_classes=200, dim=channel, 
    #                 mlp_dim_ratio=2, depth=depth, heads=heads, dim_head=dim_head, dropout=0, 
    #                 stochastic_depth=0, is_base=False, is_s_pool=False)
        
    # model_sl = PiT(img_size=img_size, patch_size = patch_size, num_classes=200, dim=channel, 
    #                 mlp_dim_ratio=2, depth=depth, heads=heads, dim_head=dim_head, dropout=0, 
    #                 stochastic_depth=0, is_base=False)

        

    # torch.cuda.set_device(GPU)
    # from torchsummary import summary
    # def to_gpu_summary(m):        
    #     m.cuda(GPU)
    #     summary(m, (3, img_size, img_size))
        
    # # to_gpu_summary(model_base)
    # # to_gpu_summary(model_sl_wo_pool)
    # # to_gpu_summary(model_sl)
        
    
    # # ViT FLOPs
    
    # print()
    # print('PiT Flops:',model_base.flops())
    # print('SL-PiT_wo_pool Flops:',model_sl_wo_pool.flops())
    # print('SL-PiT Flops:',model_sl.flops())
    
    
    
    from models.conv_cifar_pytoch.resnet import resnet56
    
    model_res56 = resnet56(num_classes=200)
   
    from models.conv_cifar_pytoch.resnet import resnet110
    
    model_res100 = resnet110(num_classes=200)
        
        
    from models.conv_cifar_pytoch.efficientnet import EfficientNetB0
    
    model_effi = EfficientNetB0(num_classes=200)
    
    
    # # Throughput
    
    # inputs = torch.randn(128, 3, img_size, img_size).cuda(GPU)
    # repetitions=1000
    # warmup = 200
    # model_effi.cuda(GPU)
    # total_time = 0
    # with torch.no_grad():
    #     for rep in range(repetitions + warmup):
    #         if not rep < warmup:
    #             starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #             starter.record()
    #             model_effi(inputs)
    #             ender.record()
    #             torch.cuda.synchronize()
    #             curr_time = starter.elapsed_time(ender)/1000
    #             total_time += curr_time
    # Throughput =   (repetitions*128)/total_time
    # print('Final Throughput:',Throughput)
        

    # CNN flops
    from utils.flops_counter import get_model_complexity_info
    
    with torch.cuda.device(GPU):
        net = model_res56
        macs, params = get_model_complexity_info(net, (3, 64, 64), as_strings=True,
                                                print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            
        net = model_res100
        macs, params = get_model_complexity_info(net, (3, 64, 64), as_strings=True,
                                                print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            
        net = model_effi
        macs, params = get_model_complexity_info(net, (3, 64, 64), as_strings=True,
                                                print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            
if __name__ == '__main__':
        main()
