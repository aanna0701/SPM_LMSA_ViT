# import torch
# from torch import nn
# from einops import rearrange
# from einops.layers.torch import Rearrange
# import math
# from .Coord import CoordLinear

# class ShiftedPatchTokenization(nn.Module):
#     def __init__(self, in_dim, dim, merging_size=2, exist_class_t=False, is_pe=False, is_Coord=False):
#         super().__init__()
        
#         self.exist_class_t = exist_class_t
        
#         self.patch_shifting = PatchShifting(merging_size)
        
#         patch_dim = (in_dim*5) * (merging_size**2) 
#         if exist_class_t:
#             self.class_linear = nn.Linear(in_dim, dim)

#         self.is_pe = is_pe
        
#         self.merging = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
#             nn.LayerNorm(patch_dim),
#             nn.Linear(patch_dim, dim) if not is_Coord else CoordLinear(patch_dim, dim, exist_cls_token=False)
#         )

#     def forward(self, x):
        
#         if self.exist_class_t:
#             visual_tokens, class_token = x[:, 1:], x[:, (0,)]
#             reshaped = rearrange(visual_tokens, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
#             out_visual = self.patch_shifting(reshaped)
#             out_visual = self.merging(out_visual)
#             out_class = self.class_linear(class_token)
#             out = torch.cat([out_class, out_visual], dim=1)
        
#         else:
#             out = x if self.is_pe else rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
#             out = self.patch_shifting(out)
#             out = self.merging(out)    
        
#         return out
        
# class PatchShifting(nn.Module):
#     def __init__(self, patch_size):
#         super().__init__()
#         self.shift = int(patch_size * (1/2))
        
#     def forward(self, x):
     
#         x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
#         # if self.is_mean:
#         #     x_pad = x_pad.mean(dim=1, keepdim = True)
        
#         """ 4 cardinal directions """
#         #############################
#         # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
#         # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
#         # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
#         # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
#         # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1) 
#         #############################
        
#         """ 4 diagonal directions """
#         # #############################
#         x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
#         x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
#         x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
#         x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
#         x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1) 
#         # #############################
        
#         """ 8 cardinal directions """
#         #############################
#         # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
#         # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
#         # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
#         # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
#         # x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
#         # x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
#         # x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
#         # x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
#         # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2, x_lu, x_ru, x_lb, x_rb], dim=1) 
#         #############################
        
#         # out = self.out(x_cat)
#         out = x_cat
        
#         return out

import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import math
from .Coord import CoordLinear

class ShiftedPatchTokenization(nn.Module):
    def __init__(self, in_dim, dim, merging_size=2, exist_class_t=False, is_pe=False, is_Coord=False):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.exist_class_t = exist_class_t
        self.merging_size = merging_size
        # self.num_patches = num_patches
        # self.num_patches = self.num_patches // (merging_size**2)
        
        self.patch_shifting = PatchShifting(merging_size)
        
        patch_dim = (in_dim*5) * (merging_size**2) 
        self.patch_dim = patch_dim
        if exist_class_t:
            self.class_linear = nn.Linear(in_dim, dim)

        self.is_pe = is_pe
        self.is_Coord = is_Coord
    
        # self.merging = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim) if not is_Coord else CoordLinear(patch_dim, dim, exist_cls_token=False)
        # )
    
        self.merging = nn.ModuleList()
        
        self.merging.append(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size))
        self.merging.append(nn.LayerNorm(patch_dim))
        self.merging.append(nn.Linear(patch_dim, dim) if not is_Coord else CoordLinear(patch_dim, dim, exist_cls_token=False))
        

    def forward(self, x, H, W):
        
        if self.exist_class_t:
            visual_tokens, class_token = x[:, 1:], x[:, (0,)]
            reshaped = rearrange(visual_tokens, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
            out_visual = self.patch_shifting(reshaped)
            out_visual = self.merging(out_visual)
            out_class = self.class_linear(class_token)
            out = torch.cat([out_class, out_visual], dim=1)
        
        else:
            out = x if self.is_pe else rearrange(x, 'b (h w) d -> b d h w', h=H)
            out = self.patch_shifting(out)
            # out = self.merging(out)    
            for i, layer in enumerate(self.merging):
                out = layer(out) if not (i == len(self.merging)-1 and self.is_Coord) else layer(out, H//self.merging_size, W//self.merging_size)
                # if i ==0 :
                #     out = out.transpose(1, 2)
        
        if self.is_pe:
            out = out.transpose(1, 2).view(-1, self.dim, H//self.merging_size, W//self.merging_size)
        
        return out
    
    # def flops(self):
    #     flops = 0
        
    #     flops += self.num_patches * self.patch_dim
        
    #     if self.exist_class_t:
    #         flops += self.in_dim * self.dim
            
    #     if self.is_Coord:
    #         flops += self.num_patches * (self.patch_dim+2) * self.dim
    #     else:
    #         flops += self.num_patches * self.patch_dim * self.dim
            
    #     return flops
        

        
        
class PatchShifting(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.shift = int(patch_size * (1/2))
        
    def forward(self, x):
     
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        # if self.is_mean:
        #     x_pad = x_pad.mean(dim=1, keepdim = True)
        
        """ 4 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1) 
        #############################
        
        """ 4 diagonal directions """
        # #############################
        x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
        x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
        x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
        x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
        x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1) 
        # #############################
        
        """ 8 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
        # x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
        # x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
        # x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2, x_lu, x_ru, x_lb, x_rb], dim=1) 
        #############################
        
        # out = self.out(x_cat)
        out = x_cat
        
        return out