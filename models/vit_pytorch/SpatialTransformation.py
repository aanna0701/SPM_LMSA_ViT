import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops.layers.torch import Rearrange
import math
from einops import rearrange, repeat


def exists(val):
    return val is not None
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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

class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.mask = torch.eye(num_patches, num_patches)
        self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        
        self.scale = nn.Parameter(self.scale*torch.ones(heads))
        

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        context = x if not exists(context) else torch.cat((x, context), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        """ LMSA """
        ############################
        scale = self.scale
        dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((x.size(0), self.heads, 1, 1)))

        dots[:, :,:, 0] = -987654321
        ###########################
        
        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax
        attn = self.attend(dots)        
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        
        
    def forward(self, x, context = None):

        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x    

class AffineNet(nn.Module):
    def __init__(self, num_patches, depth, in_dim, hidden_dim, heads, n_trans=4, patch_size=None):
        super().__init__()
        self.in_dim = in_dim
        self.n_trans = n_trans
        n_output = 6*self.n_trans
        self.patch_size = patch_size
        self.param_transformer = Transformer(self.in_dim, num_patches, depth, heads, hidden_dim//heads, self.in_dim)
        
        if patch_size is not None:       
            self.rearrange = Rearrange('b c (h p_h) (w p_w) -> b (h w) (c p_h p_w)', p_h=patch_size, p_w=patch_size)
  
        else:
            self.rearrange = nn.Identity()
            
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, n_output, bias=False),
            nn.Tanh()
        )
        
        self.transformation = Affine()

        self.theta = list()
    def forward(self, param_token, x, init, scale=None):
        # print(x.shape)
        param_token = repeat(param_token, '() n d -> b n d', b = x.size(0))
        param_attd = self.param_transformer(param_token, self.rearrange(x))
        param = self.mlp_head(param_attd[:, 0])
        param_list = torch.chunk(param, self.n_trans, dim=-1)
        
        out = []
        theta = []
        if len(x.size()) == 3:
            x = rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
        for i in range(self.n_trans):
            if scale is not None:
                out.append(self.transformation(x, param_list[i], init[i], scale[i]))
            else:
                out.append(self.transformation(x, param_list[i], init[i]))
            theta.append(self.transformation.theta)
                
        out = torch.cat(out, dim=1)
        out = out.view(out.size(0), out.size(1), -1).transpose(1, 2)
        self.theta = theta
        
        
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
    
    
# class PatchMerging(nn.Module):
#     r""" Patch Merging Layer.
#     Args:
#         input_resolution (tuple[int]): Resolution of input feature.
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     def __init__(self, patch_size, dim, out_dim):
#         super().__init__()
        
#         self.dim = dim
#         self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
#         self.norm = nn.LayerNorm(4 * dim)

#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         B, L, C = x.shape

#         x = rearrange(x, 'b (h w) c -> b h w c', h = int(math.sqrt(L)))

#         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
#         x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
#         x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
#         x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
#         x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        
#         x = self.norm(x)
#         x = self.reduction(x)

#         return x

#     def extra_repr(self) -> str:
#         return f"input_resolution={self.input_resolution}, dim={self.dim}"

#     def flops(self):
#         H, W = self.input_resolution
#         flops = H * W * self.dim
#         flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
#         return flops
    


class Affine(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super().__init__()
        
        self.theta = None
        self.mode = padding_mode
        
    def forward(self, x, theta, init, scale=None):
        print('========')
        print(scale)
        print(theta[0])
        
        
        if scale is not None:
            theta = torch.mul(theta, scale)
        
        init = torch.reshape(init.unsqueeze(0), (1, 2, 3)).expand(x.size(0), -1, -1) 
        theta = torch.reshape(theta, (theta.size(0), 2, 3))    
        theta = theta + init 
        self.theta = theta    
        
        print(theta[0])
        
        grid = F.affine_grid(theta, x.size())
        
        return F.grid_sample(x, grid, padding_mode=self.mode)
     

class STiT(nn.Module):
    def __init__(self, img_size=224, patch_size=2, in_dim=3, embed_dim=96, depth=2, heads=4, type='PE', 
                 init_eps=0., init_noise=[1e-3, 1e-3]):
        super().__init__()
        assert type in ['PE', 'Pool'], 'Invalid type!!!'

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size // patch_size
        self.in_dim = in_dim
        
        
        if type == 'PE':
            self.input = Rearrange('b c h w -> b (h w) c')
            merge_size = 4
            pt_dim = 3 
            self.affine_net = AffineNet(self.num_patches, depth, pt_dim * (merge_size**2), 64, heads, patch_size=merge_size)
            self.param_token = nn.Parameter(torch.rand(1, 1, pt_dim * (merge_size**2)))
        else:
            self.input = nn.Identity()
            pt_dim = in_dim    
            self.affine_net = AffineNet(self.num_patches, depth, pt_dim, pt_dim, heads)
    
            self.param_token = nn.Parameter(torch.rand(1, 1, pt_dim))
                      
        if not init_eps == 0.:
            self.scale_list = nn.ParameterList()  
            for _ in range(4):
                self.scale_list.append(nn.Parameter(torch.zeros(1, 6).fill_(init_eps)))
    
        else: self.scale_list = None  
        
        self.init_list = list()
        for i in range(4):
            self.init_list.append(self.make_init(i, self.num_patches, init_noise=init_noise).cuda(torch.cuda.current_device()))
  
        self.patch_merge = PatchMerging(patch_size, pt_dim*5, embed_dim)
    
        self.theta = None    
            
        self.apply(self._init_weights)

    def make_init(self, n, num_patches, init_noise=[0, 0]):                
        
            ratio = np.random.normal(1/num_patches, init_noise[0], size=2)
            ratio_scale_a = np.random.normal(1, init_noise[1], size=2)
            ratio_scale_b = np.random.normal(0, init_noise[1], size=2)
            ratio_x = float((math.cos(n * math.pi))*ratio[0])
            ratio_y = float((math.sin(((n//2) * 2 + 1) * math.pi / 2))*ratio[1])             
    
            out = torch.tensor([float(ratio_scale_a[1]), float(ratio_scale_b[0]), ratio_x, 
                                float(ratio_scale_b[1]), float(ratio_scale_a[0]), ratio_y])
        
            return out
    
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
        
        affine = self.affine_net(self.param_token, x, self.init_list, self.scale_list)
        self.theta = self.affine_net.theta
        x = self.input(x)
        out = torch.cat([x, affine], dim=-1)      
        out = self.patch_merge(out)
        
        return out