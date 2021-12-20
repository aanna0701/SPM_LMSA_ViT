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
from utils.coordconv import CoordConv, CoordLinear

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
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_LSA=False):
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
        self.is_LSA = is_LSA
        self.scale = nn.Parameter(self.scale*torch.ones(heads))
        

    def forward(self, x, context):
        b, n, _, h = *x.shape, self.heads
        
        if not self.is_LSA:
            context = torch.cat((x, context), dim = 1)
        else:    
            context = context

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if not self.is_LSA:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((x.size(0), self.heads, 1, 1)))

        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax
        attn = self.attend(dots)        
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0., is_LSA=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_LSA=is_LSA)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
                
    def forward(self, x, context = None):

        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x    

class AffineNet(nn.Module):
    def __init__(self, num_patches, depth, in_dim, hidden_dim, heads, n_trans=4, merging_size=2, is_LSA=False):
        super().__init__()
        self.in_dim = in_dim
        self.n_trans = n_trans
        n_output = 6*self.n_trans
        # self.param_transformer = Transformer(self.in_dim*(patch_size**2), num_patches, depth, heads, hidden_dim//heads, self.in_dim)
        self.param_transformer = Transformer(hidden_dim, num_patches, depth, heads, hidden_dim//heads, self.in_dim*2, is_LSA=is_LSA)       

        self.depth_wise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, merging_size, merging_size, groups=hidden_dim),
            Rearrange('b c h w -> b (h w) c')
        )
            
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_output)
        )
        
        self.transformation = Affine()
        self.pre_linear = nn.Conv2d(self.in_dim, hidden_dim, (1, 1))
        self.post_linear = nn.Conv2d(hidden_dim, self.in_dim, (1, 1))

        self.theta = list()
    def forward(self, param_token, x, init, scale=None):
        # print(x.shape)
        if len(x.size()) == 3:
            x = rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1)))) 
            
        x = self.pre_linear(x)
        param_token = repeat(param_token, '() n d -> b n d', b = x.size(0))
        param_attd = self.param_transformer(param_token, self.depth_wise_conv(x))
        param = self.mlp_head(param_attd[:, 0])
        param_list = torch.chunk(param, self.n_trans, dim=-1)
        
        out = []
        theta = []       
        
        x = torch.chunk(x, self.n_trans, dim=1)
        for i in range(self.n_trans):
            if scale is not None:
                out.append(self.transformation(x[i], param_list[i], init, scale[i]))
            else:
                out.append(self.transformation(x[i], param_list[i], init))
            theta.append(self.transformation.theta)
                
        out = torch.cat(out, dim=1)
        out = self.post_linear(out)
        
        out = rearrange(out, 'b d h w -> b (h w) d')
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
    

class Affine(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super().__init__()
        
        self.theta = None
        self.mode = padding_mode
        
    def forward(self, x, theta, init, scale=None):
        # print('========')
        # print(scale)
        # print(theta[0])     
        
        theta = F.tanh(theta)
        if scale is not None:
            theta = torch.mul(theta, scale)
        
        init = torch.reshape(init.unsqueeze(0), (1, 2, 3)).expand(x.size(0), -1, -1) 
        theta = torch.reshape(theta, (theta.size(0), 2, 3))    
        theta = theta + init 
        self.theta = theta    
   
        # print(theta[0])
        
        grid = F.affine_grid(theta, x.size())
        
        return F.grid_sample(x, grid, padding_mode=self.mode)
     

class STT(nn.Module):
    def __init__(self, img_size=224, patch_size=2, in_dim=3, pa_dim=64, embed_dim=96, depth=2, heads=4, type='PE', 
                 init_eps=0., is_LSA=False, merging_size=4):
        super().__init__()
        assert type in ['PE', 'Pool'], 'Invalid type!!!'

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size // patch_size
        self.in_dim = in_dim
        
        
        if type == 'PE':
            in_dim = pa_dim
            self.input = nn.Conv2d(3, in_dim, 3, 2, 1)
            self.rearrange = Rearrange('b c h w -> b (h w) c')     
            self.patch_merge = PatchMerging(patch_size//2, in_dim, embed_dim)    
        else:
            self.input = nn.Identity()
            self.rearrange = nn.Identity()
            self.patch_merge = PatchMerging(patch_size, in_dim, embed_dim)
            
        self.affine_net = AffineNet(self.num_patches, depth, in_dim, pa_dim, heads, merging_size=merging_size, is_LSA=is_LSA)
        self.param_token = nn.Parameter(torch.rand(1, 1, pa_dim))
                      
        if not init_eps == 0.:
            self.scale_list = nn.ParameterList()  
            for _ in range(4):
                self.scale_list.append(nn.Parameter(torch.zeros(1, 6).fill_(init_eps)))
    
        else: self.scale_list = None  
        
        self.init = self.make_init().cuda(torch.cuda.current_device())

        self.theta = None    
            
        self.apply(self._init_weights)

    def make_init(self,):                
        out = torch.tensor([1, 0, 0,
                            0, 1, 0])
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
        
        x = self.input(x)
        affine = self.affine_net(self.param_token, x, self.init, self.scale_list)
        self.theta = self.affine_net.theta
        x = self.rearrange(x)
        out = x + affine
        out = self.patch_merge(out)
        
        return out
    
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.layers import trunc_normal_
# import numpy as np
# import torch
# from torch import nn, einsum
# import torch.nn.functional as F
# from math import sqrt
# from einops.layers.torch import Rearrange
# import math
# from einops import rearrange, repeat
# from utils.coordconv import CoordConv, CoordLinear

# def exists(val):
#     return val is not None
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x):
#         return self.net(x)

# class Attention(nn.Module):
#     def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.to_q = nn.Linear(dim, inner_dim, bias = False)
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

#         self.attend = nn.Softmax(dim = -1)

#         self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
#         self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         )
       
#         self.scale = nn.Parameter(self.scale*torch.ones(heads))
        

#     def forward(self, x, context = None):
#         b, n, _, h = *x.shape, self.heads
        
        
#         context = x if not exists(context) else torch.cat((x, context), dim = 1)

#         qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

#         # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
#         """ LMSA """
#         ############################
#         scale = self.scale
#         dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((x.size(0), self.heads, 1, 1)))

#         dots[:, :, :, 0] = -987654321
#         ###########################
        
#         dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax
#         attn = self.attend(dots)        
#         attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)


# class Transformer(nn.Module):
#     def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         self.layer_dropout = layer_dropout

#         for ind in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
#             ]))
        
        
#     def forward(self, x, context = None):

#         for attn, ff in self.layers:
#             x = attn(x, context = context) + x
#             x = ff(x) + x
#         return x    

# class AffineNet(nn.Module):
#     def __init__(self, num_patches, depth, in_dim, hidden_dim, heads, n_trans=4, merging_size=2):
#         super().__init__()
#         self.in_dim = in_dim
#         self.n_trans = n_trans
#         n_output = 6*self.n_trans
#         # self.param_transformer = Transformer(self.in_dim*(patch_size**2), num_patches, depth, heads, hidden_dim//heads, self.in_dim)
#         self.param_transformer = Transformer(self.in_dim, num_patches, depth, heads, self.in_dim//heads, self.in_dim*2)
       

#         self.depth_wise_conv = nn.Sequential(
#             nn.Conv2d(self.in_dim, self.in_dim, merging_size, merging_size, groups=self.in_dim),
#             Rearrange('b c h w -> b (h w) c')
#         )
            
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(self.in_dim),
#             nn.Linear(self.in_dim, n_output)
#         )
        
#         self.transformation = Affine()
#         self.pre_linear = nn.Conv2d(self.in_dim, hidden_dim, (1, 1))
#         self.post_linear = nn.Conv2d(hidden_dim, self.in_dim, (1, 1))

#         self.theta = list()
#     def forward(self, param_token, x, init, scale=None):
#         # print(x.shape)
#         param_token = repeat(param_token, '() n d -> b n d', b = x.size(0))
#         param_attd = self.param_transformer(param_token, self.depth_wise_conv(x))
#         param = self.mlp_head(param_attd[:, 0])
#         param_list = torch.chunk(param, self.n_trans, dim=-1)
        
#         out = []
#         theta = []
#         if len(x.size()) == 3:
#             x = rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
        
#         x = self.pre_linear(x)
#         x = torch.chunk(x, self.n_trans, dim=1)
#         for i in range(self.n_trans):
#             if scale is not None:
#                 out.append(self.transformation(x[i], param_list[i], init, scale[i]))
#             else:
#                 out.append(self.transformation(x[i], param_list[i], init))
#             theta.append(self.transformation.theta)
                
#         out = torch.cat(out, dim=1)
#         out = self.post_linear(out)
#         out = rearrange(out, 'b d h w -> b (h w) d')
#         self.theta = theta
        
        
#         return out
    
# class PatchMerging(nn.Module):
#     r""" Patch Merging Layer.
#     Args:
#         input_resolution (tuple[int]): Resolution of input feature.
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     def __init__(self, patch_size, dim, out_dim):
#         super().__init__()
        
#         self.merging = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2 = patch_size)
#         self.dim = dim
#         self.patch_dim = dim * (patch_size ** 2)
#         self.reduction = nn.Linear(self.patch_dim, out_dim, bias=False)
#         self.norm = nn.LayerNorm(self.patch_dim)

#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         B, L, C = x.shape
        
#         x = rearrange(x, 'b (h w) c -> b h w c', h = int(math.sqrt(L)))
#         x = self.merging(x)
        
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
    

# class Affine(nn.Module):
#     def __init__(self, padding_mode='zeros'):
#         super().__init__()
        
#         self.theta = None
#         self.mode = padding_mode
        
#     def forward(self, x, theta, init, scale=None):
#         print('========')
#         print(scale)
#         print(theta[0])
        
        
#         theta = F.tanh(theta)
#         if scale is not None:
#             theta = torch.mul(theta, scale)
        
#         init = torch.reshape(init.unsqueeze(0), (1, 2, 3)).expand(x.size(0), -1, -1) 
#         theta = torch.reshape(theta, (theta.size(0), 2, 3))    
#         theta = theta + init 
#         self.theta = theta    
   
#         print(theta[0])
        
#         grid = F.affine_grid(theta, x.size())
        
#         return F.grid_sample(x, grid, padding_mode=self.mode)
     

# class STT(nn.Module):
#     def __init__(self, img_size=224, patch_size=2, in_dim=3, embed_dim=96, depth=2, heads=4, type='PE', 
#                  init_eps=0., init_noise=[1e-3, 1e-3], merging_size=4):
#         super().__init__()
#         assert type in ['PE', 'Pool'], 'Invalid type!!!'

#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = img_size // patch_size
#         self.in_dim = in_dim
        
        
#         if type == 'PE':
#             self.input = nn.Conv2d(3, in_dim, (3, 3), 2, 1)
#             self.rearrange = Rearrange('b c h w -> b (h w) c')         
#             self.affine_net = AffineNet(self.num_patches, depth, in_dim, in_dim, heads, merging_size=merging_size)
#             self.param_token = nn.Parameter(torch.rand(1, 1, in_dim))
#         else:
#             self.input = nn.Identity()
#             self.rearrange = nn.Identity()
#             self.affine_net = AffineNet(self.num_patches, depth, in_dim, in_dim, heads)    
#             self.param_token = nn.Parameter(torch.rand(1, 1, in_dim))
                      
#         if not init_eps == 0.:
#             self.scale_list = nn.ParameterList()  
#             for _ in range(4):
#                 self.scale_list.append(nn.Parameter(torch.zeros(1, 6).fill_(init_eps)))
    
#         else: self.scale_list = None  
        
#         self.init = self.make_init().cuda(torch.cuda.current_device())

#         self.patch_merge = PatchMerging(patch_size//2, in_dim, embed_dim)
#         self.theta = None    
            
#         self.apply(self._init_weights)

#     def make_init(self,):                
#         out = torch.tensor([1, 0, 0,
#                             0, 1, 0])
#         return out

#     def _init_weights(self, m):
#         if isinstance(m, (nn.Linear, nn.Conv2d)):
#             # nn.init.xavier_normal_(m.weight)
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.LayerNorm)):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
        
#         x = self.input(x)
#         affine = self.affine_net(self.param_token, x, self.init, self.scale_list)
#         self.theta = self.affine_net.theta
#         x = self.rearrange(x)
#         out = x + affine
#         out = self.patch_merge(out)
        
#         return out
    