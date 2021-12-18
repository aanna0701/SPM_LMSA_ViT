import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from math import sqrt
from einops.layers.torch import Rearrange
import math
from einops import rearrange, repeat



class AffineNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_trans=4):
        super().__init__()
        self.in_dim = in_dim
        self.n_trans = n_trans
        n_output = 6*self.n_trans
            
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, n_output)
        )
        
        self.pre_linear = nn.Conv2d(self.in_dim, hidden_dim, (1, 1))
        self.post_linear = nn.Conv2d(hidden_dim, self.in_dim, (1, 1))

        self.theta = list()
    def forward(self, x):
        if len(x.size()) == 3:
            x = rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
        
        x = self.pre_linear(x)
        out = self.post_linear(x)
        out = rearrange(out, 'b d h w -> b (h w) d')
 
        
        return out
    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, patch_size, dim, out_dim):
        super().__init__()
        
        self.merging = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2 = patch_size)
        self.dim = dim
        self.patch_dim = dim * (patch_size ** 2)
        self.reduction = nn.Linear(self.patch_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(self.patch_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        
        x = rearrange(x, 'b (h w) c -> b h w c', h = int(math.sqrt(L)))
        x = self.merging(x)
        
        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops
    
     

class STT(nn.Module):
    def __init__(self, img_size=224, patch_size=2, in_dim=3, embed_dim=96, depth=2, heads=4, type='PE', 
                 init_eps=0., init_noise=[1e-3, 1e-3], merging_size=4, no_init=False):
        super().__init__()
        assert type in ['PE', 'Pool'], 'Invalid type!!!'

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size // patch_size
        self.in_dim = in_dim
        
        
        if type == 'PE':
            self.input = nn.Conv2d(3, in_dim, (3, 3), 2, 1)
            self.rearrange = Rearrange('b c h w -> b (h w) c')         
            self.affine_net = AffineNet(in_dim, in_dim)

        else:
            self.input = nn.Identity()
            self.rearrange = nn.Identity()
            self.affine_net = AffineNet(self.num_patches, depth, in_dim, in_dim, heads)    

        self.patch_merge = PatchMerging(patch_size//2, in_dim, embed_dim)

            
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # nn.init.xavier_normal_(m.weight)
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        
        
        x = self.input(x)
        
        affine = self.affine_net(x)
        
        x = self.rearrange(x)
        
        out = x + affine
        
        out = self.patch_merge(out)
        
        return out
    