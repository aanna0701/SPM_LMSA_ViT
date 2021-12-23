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
from utils_.coordconv import CoordConv, CoordLinear

def exists(val):
    return val is not None

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.fn = fn        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)    
    def flops(self):
        flops = 0
        flops += self.fn.flops()
        flops += self.dim        
        return flops 
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)    
    def flops(self):
        flops = 0
        flops += self.dim * self.hidden_dim
        flops += self.dim * self.hidden_dim
        
        return flops
    
class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_LSA=False, is_coord=False):
        super().__init__()
        self.inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim  = dim
        self.num_patches = num_patches
        self.is_coord = is_coord
        self.to_q = nn.Linear(self.dim, self.inner_dim, bias = False)        
        if not self.is_coord:
            self.to_kv = nn.Linear(self.dim, self.inner_dim * 2, bias = False)
        else:
            if is_LSA:
                self.to_kv = CoordLinear(self.dim, self.inner_dim * 2, exist_cls_token=False, bias = False)
            else:
                self.to_kv = CoordLinear(self.dim, self.inner_dim * 2, bias = False)
        
        self.attend = nn.Softmax(dim = -1)
        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))        
        self.to_out = nn.Sequential(
                nn.Linear(self.inner_dim, self.dim),
                nn.Dropout(dropout))
        self.is_LSA = is_LSA
        if self.is_LSA:
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
    def flops(self):
        flops = 0
        flops += self.dim * self.inner_dim 
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 2 * self.num_patches
        else:
            if self.is_LSA:
                flops += (self.dim+2) * self.inner_dim * 2 * self.num_patches
            else:    
                flops += (self.dim+2) * self.inner_dim * 2 * self.num_patches
                flops += self.dim * self.inner_dim * 2 
        flops += self.inner_dim * self.num_patches
        flops += self.inner_dim * self.num_patches
        flops += self.num_patches   # scaling
        flops += self.num_patches   # pre-mix
        flops += self.num_patches   # post-mix
        flops += self.inner_dim * self.dim
        
        return flops
    
class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0., is_LSA=False, is_coord=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, 
                                       is_LSA=is_LSA, is_coord=is_coord)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
                
    def forward(self, x, context):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x  
    
    def flops(self):
        flops = 0        
        for (attn, ff) in self.layers:       
            flops += attn.flops()
            flops += ff.flops()
        
        return flops
    
class AffineNet(nn.Module):
    def __init__(self, num_patches, depth, in_dim, hidden_dim, heads, n_trans=4, merging_size=2, is_LSA=False, is_coord=False):
        super().__init__()
        self.in_dim = in_dim
        self.n_trans = n_trans
        self.n_output = 6*self.n_trans
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.merging_size = merging_size
        self.is_coord = is_coord
        # self.param_transformer = Transformer(self.in_dim*(patch_size**2), num_patches, depth, heads, hidden_dim//heads, self.in_dim)
        self.param_transformer = Transformer(self.in_dim, self.num_patches//(self.merging_size**2), depth, heads, self.in_dim//heads, self.in_dim*2, is_LSA=is_LSA, is_coord=is_coord)       
        self.depth_wise_conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.in_dim, self.merging_size, self.merging_size, groups=self.in_dim),
            Rearrange('b c h w -> b (h w) c')
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.n_output)
        )  
        self.transformation = Affine()
        if not self.is_coord:
            self.pre_linear = nn.Conv2d(self.in_dim, self.hidden_dim, (1, 1))
            self.post_linear = nn.Conv2d(self.hidden_dim, self.in_dim, (1, 1))
        else:
            self.pre_linear = CoordConv(self.in_dim, self.hidden_dim, 1)
            self.post_linear = CoordConv(self.hidden_dim, self.in_dim, 1)
        
        self.theta = list()
        
    def forward(self, param_token, x, init, scale=None):
        # print(x.shape)
        if len(x.size()) == 3:
            x = rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1)))) 
        param_token = repeat(param_token, '() n d -> b n d', b = x.size(0))
        param_attd = self.param_transformer(param_token, self.depth_wise_conv(x))
        param = self.mlp_head(param_attd[:, 0])
        param_list = torch.chunk(param, self.n_trans, dim=-1)
        
        out = []
        theta = []       
        
        x = self.pre_linear(x)
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
    
    def flops(self):
        flops = 0
        flops += (self.merging_size**2)*self.num_patches*self.hidden_dim    # depth-wise conv
        flops += self.param_transformer.flops()                 # parameter-transformer
        flops += self.hidden_dim + self.hidden_dim*self.n_output    # mlp head
        if not self.is_coord:
            flops += self.num_patches*self.in_dim*self.hidden_dim    # pre-linear
            flops += self.num_patches*self.in_dim*self.hidden_dim   # post-linear
        else:
            flops += self.num_patches*(self.in_dim+2)*self.hidden_dim    # pre-linear
            flops += self.num_patches*self.in_dim*(self.hidden_dim+2)   # post-linear
        
        return flops    
    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, num_patches, patch_size, dim, out_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.merging = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2 = patch_size)
        self.dim = dim
        self.out_dim = out_dim
        self.patch_dim = dim * (patch_size ** 2)
        self.reduction = nn.Linear(self.patch_dim, self.out_dim, bias=False)
        self.norm = nn.LayerNorm(self.patch_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = rearrange(x, 'b (h w) c -> b h w c', h = int(math.sqrt(self.num_patches)))
        x = self.merging(x)        
        x = self.norm(x)
        x = self.reduction(x)

        return x

    def flops(self):
        flops = 0
        flops += (self.num_patches//(self.patch_size**2))*self.patch_dim*self.out_dim
        flops += (self.num_patches//(self.patch_size**2))*self.patch_dim
        
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
    def __init__(self, img_size=224, patch_size=2, in_dim=3, embed_dim=96, depth=2, heads=4, type='PE', 
                 init_eps=0., is_LSA=False, merging_size=4, is_coord=False):
        super().__init__()
        assert type in ['PE', 'Pool'], 'Invalid type!!!'

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size**2
        self.in_dim = in_dim
        self.type = type
        self.is_coord = is_coord
        
        if self.type == 'PE':
            # self.input = nn.Conv2d(3, self.in_dim, 3, 2, 1) if not is_coord else CoordConv(3, self.in_dim, 3, 2, 1)
            if not is_coord:
                self.input = nn.Conv2d(3, self.in_dim, 7, 4, 2)
            else:
                self.input = nn.Sequential(
                    nn.Unfold(kernel_size=7, stride = 4, padding = 2),
                    Rearrange('b c n -> b n c'),
                    nn.LayerNorm(3*(7**2)),
                    CoordLinear(3*(7**2), self.in_dim, exist_cls_token=False)
                )

            # self.rearrange = Rearrange('b c h w -> b (h w) c')     
            self.affine_net = AffineNet(self.num_patches//16, depth, self.in_dim, self.in_dim, heads, merging_size=merging_size, is_LSA=is_LSA, is_coord=is_coord)
            self.patch_merge = PatchMerging(self.num_patches//16, patch_size//4, self.in_dim, embed_dim) 
           
        else:
            self.input = nn.Identity()
            self.rearrange = nn.Identity()
            self.affine_net = AffineNet(self.num_patches, depth, self.in_dim, self.in_dim*2, heads, merging_size=merging_size, is_LSA=is_LSA, is_coord=is_coord)
            self.patch_merge = PatchMerging(self.num_patches, patch_size, self.in_dim*2, embed_dim)
        self.param_token = nn.Parameter(torch.rand(1, 1, self.in_dim))
                      
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
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        
        x = self.input(x)
        affine = self.affine_net(self.param_token, x, self.init, self.scale_list)
        self.theta = self.affine_net.theta
        # x = self.rearrange(x)
        out = x + affine
        out = self.patch_merge(out)
        
        return out
    
    def flops(self):
        flops = 0
        if self.type=='PE':
            # flops_input = (3**2)*3*self.in_dim*((self.img_size//2)**2)
            if not self.is_coord:
                flops_input = (3**2)*3*self.in_dim*((self.img_size//2)**2)
            else:    
                flops_input = 3*(2**2)*((self.img_size//2)**2) + 3*(2**2)*self.in_dim*((self.img_size//2)**2)
        else:
            flops_input = 0
        flops += flops_input
        flops += self.affine_net.flops()   
        flops += self.patch_merge.flops() 
        
        return flops
    
    
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
    