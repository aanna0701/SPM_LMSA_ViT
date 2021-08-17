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



    
def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
    
class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)    

class ViT(nn.Module):
    def __init__(self, img_size, patch_size,dim, channels = 3):
        super().__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_size // patch_size) ** 2
        patch_dim = (channels) * patch_size ** 2
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        #     nn.Linear(patch_dim, dim),
        # )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.to_patch_embedding = nn.Conv2d(3, dim, patch_size, patch_size)
       
    def forward(self, img):
        x = self.to_patch_embedding(img)
       
        return x

class SPE(nn.Module):
    def __init__(self, img_size, patch_size,dim, channels = 3):
        super().__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_size // patch_size) ** 2
        patch_dim = (channels+4) * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            PatchShifting(patch_size),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
       
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
       
        return x

class SPE_rgb(nn.Module):
    def __init__(self, img_size, patch_size,dim, channels = 3):
        super().__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_size // patch_size) ** 2
        patch_dim = (channels)*5 * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            PatchShifting_rgb(patch_size),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
       
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
       
        return x

class Conv1(nn.Module):
    def __init__(self, img_size, patch_size,dim, channels = 3):
        super().__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_size // patch_size) ** 2
        
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, 16, 8, 4),
            Rearrange('b c h w -> b (h w) c')
        )
        
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
       
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
       
        return x

class Conv2(nn.Module):
    def __init__(self, img_size, patch_size,dim, channels = 3):
        super().__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_size // patch_size) ** 2
        
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, 10, 8, 1),
            Rearrange('b c h w -> b (h w) c')
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
       
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
       
        return x

class Conv3(nn.Module):
    def __init__(self, img_size, patch_size,dim, channels = 3):
        super().__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_size // patch_size) ** 2
        
          
        self.conv1 = nn.Conv2d(channels, dim, 3, 2, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 2, 1)
        self.conv3 = nn.Conv2d(dim, dim, 3, 2, 1)
        self.rearrange = Rearrange('b c h w -> b (h w) c')

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
       
    def forward(self, img):
        # x = self.to_patch_embedding(img)
        
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.rearrange(x)
        
        
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
       
        return x

class T2T(nn.Module):
    def __init__(self, img_size, patch_size,dim, channels = 3):
        super().__init__()
        layers = []
        layer_dim = channels
        output_image_size = img_size

        for i, (kernel_size, stride) in enumerate([(7, 4), (3, 2)]):
            layer_dim *= kernel_size ** 2
            is_first = i == 0
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)
            num_patches = output_image_size ** 2
            
            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                nn.Unfold(kernel_size = kernel_size, stride = stride, padding = stride // 2),
                Rearrange('b c n -> b n c'),
                Transformer(dim = layer_dim, num_patches=num_patches, heads = 1, depth = 1, dim_head = 64, mlp_dim = 64),
            ])
            
        num_patches = output_image_size ** 2

        layers.append(nn.Linear(layer_dim, dim))
        self.to_patch_embedding = nn.Sequential(*layers)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
       
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
       
        return x

class G_Attention(nn.Module):
    def __init__(self, dim, num_patches=64, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        
    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # channel_agg = self.g_block(v)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        scores = self.attend(dots)
        

        out = einsum('b h i j, b h j d -> b h i d', scores, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.hidden_states = {}
        self.scale = {}
        self.h = []
        

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, G_Attention(dim, num_patches = num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            if i == 0:
                self.h = []   
            x = self.drop_path(attn(x)) + x
            # self.h.append(attn.fn.avg_h)           
            # self.hidden_states[str(i)] = attn.fn.value
      
            x = self.drop_path(ff(x)) + x         
            # self.scale[str(i)] = attn.fn.scale
        return x
    
    
class PatchShifting(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.shift = patch_size // 2

    def forward(self, x):
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        
        x_pad = x_pad.mean(dim=1, keepdim = True)
        
        x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
               
        x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)
        
        
        return x_cat
    
class PatchShifting_rgb(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.shift = patch_size // 2

    def forward(self, x):
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        
        # x_pad = x_pad.mean(dim=1, keepdim = True)
        
        x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
               
        x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)
        
        
        return x_cat

    

def main():

    img_size = 64
    patch_size = 8
    in_channels = 3

    GPU = 1
    # model_names = ['ViT', 'SPE', 'SPE_rgb','16-8-4', '10-8-1', '3-2-1x3', 'T2T']
    # models = []
    # # models.append(ViT(img_size, patch_size, 192, in_channels))
    # # models.append(SPE(img_size, patch_size, 192, in_channels))
    # # models.append(SPE_rgb(img_size, patch_size, 192, in_channels))
    # models.append(Conv1(img_size, patch_size, 192, in_channels))
    # # models.append(Conv2(img_size, patch_size, 192, in_channels))
    # # models.append(Conv3(img_size, patch_size, 192, in_channels))
    # # models.append(T2T(img_size, patch_size, 192, in_channels))




        

    # torch.cuda.set_device(GPU)

    # # for i, model in enumerate(models):
    # #     from torchsummary import summary
    # #     model.cuda(GPU)
    # #     print(f'\n{model_names[i]} Memory cost')
    # #     summary(model, (3, img_size, img_size))

    # # for i, model in enumerate(models):
    # #     from torchsummaryX import summary
    # #     model.cuda(GPU)
    # #     print(f'\n{model_names[i]} FLOPs')
    # #     summary(model, torch.zeros((1, 3, img_size, img_size)).cuda(GPU))
        
    # for i, model in enumerate(models):
    #     from torch.profiler import profile, record_function, ProfilerActivity 
    #     model.cuda(GPU)
    #     inputs = torch.randn(1, 3, img_size, img_size).cuda(GPU)
        
    #     print(f'\n{model_names[i]} Infer time')
        
    #     for i in range(5):
            
    #         with profile(activities=[
    #         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #             with record_function("model_inference"):
    #                 model(inputs)
                                
            
    #         print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        
    # RF
    from torchvision import models
    from torchsummary import summary
    vit = ViT(img_size=224, patch_size=16, dim=192)
    
    resnet18 = models.resnet18(pretrained=False)
    torch.cuda.set_device(GPU)
    vit.cuda(GPU)
    resnet18.cuda(GPU)
    
    # summary(vit, (3, 224, 224))
    # summary(resnet18, (3, 224, 224))
    from torch_receptive_field import receptive_field
    receptive_field_dict = receptive_field(vit, (3, 224, 224))
    receptive_field_for_unit(receptive_field_dict, "2", (2,2))


if __name__ == '__main__':
        main()
