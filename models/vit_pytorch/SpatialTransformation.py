import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import torch
from torch import nn, einsum
import torch.nn.functional as F

import math
from einops import rearrange, repeat

ALPHA = 1

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
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., if_patch_attn=False):
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
        self.if_patch_attn = if_patch_attn

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        context = x if not exists(context) else torch.cat((x, context), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        
        
        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax
        attn = self.attend(dots)        
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0., if_patch_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, if_patch_attn=if_patch_attn)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        
        
    def forward(self, x, context = None):

        for attn, ff in self.layers:
            
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x
    
"""    
class Localisation(nn.Module):
    def __init__(self, img_size, n_tokenize,in_dim=16, n_trans=4, type_trans='trans'):
        super().__init__()
        self.in_dim = in_dim
        
        if img_size == 32:
            
            self.layers0 = nn.Sequential(
                nn.Conv2d(3, self.in_dim, 3, 2, 1, bias=False)
            )     
        
            img_size //= 2
            
        elif img_size == 64:
            self.layers0 = nn.Sequential(
                nn.Conv2d(3, self.in_dim, 7, 4, 2, bias=False)
            )     
        
            img_size //= 4
        
        self.layers1 = self.make_layer(self.in_dim, self.in_dim*2)
        self.in_dim *= 2
        img_size //= 2
        
        
        if type_trans=='trans':
            n_output = 2*n_trans
        elif type_trans=='affine':
            n_output = 6*n_trans
        elif type_trans=='rigid':
            n_output = 3*n_trans
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, n_output)
        )
        
        self.num_transform = n_trans
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.in_dim))
        self.cls_transformer = Transformer(self.in_dim, img_size**2, 2, 4, 16, 128)

        
        self.apply(self._init_weights)

        
    def make_layer(self, in_dim, hidden_dim):
        layers = nn.ModuleList([])
    
        layers.append(nn.Conv2d(in_dim, hidden_dim, 3, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.GELU())
            
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
    
        feature1 = self.layers0(x)
        feature2 = self.layers1(feature1)
        
        out = rearrange(feature2, 'b c h w -> b (h w) c')
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = x.size(0))
        cls_attd = self.cls_transformer(cls_tokens, out)
        out = self.mlp_head(cls_attd[:, 0])
        
        # out = torch.chunk(out, self.n_tokenize, -1)

        
        return out
        
"""       

"""
class Localisation(nn.Module):
    def __init__(self, img_size, n_tokenize,in_dim=16, n_trans=4, type_trans='trans'):
        super().__init__()
        self.in_dim = in_dim
        
        if img_size == 32:
            
            self.layers0 = nn.Sequential(
                nn.Conv2d(3, self.in_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.GELU()
            )     
        
            img_size //= 2
            
        elif img_size == 64:
            self.layers0 = nn.Sequential(
                nn.Conv2d(3, self.in_dim, 7, 4, 2, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.GELU()
            )     
        
            img_size //= 4
        
        self.layers1 = self.make_layer(self.in_dim, self.in_dim*2)
        self.in_dim *= 2
        img_size //= 2
        
        # self.layers2 = self.make_layer(self.in_dim, self.in_dim*2)
        # self.in_dim *= 2
        # img_size //= 2
        
        if type_trans=='trans':
            n_output = 2*n_trans
        elif type_trans=='affine':
            n_output = 6*n_trans
        elif type_trans=='rigid':
            n_output = 3*n_trans
        
        # self.n_tokenize = n_tokenize 
        # n_output *= n_tokenize
            
        self.mlp_head = nn.Sequential(
            nn.Linear(self.in_dim * (img_size**2), 64, bias=False),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, n_output, bias=False),
            nn.LayerNorm(n_output)
        )
        self.num_transform = n_trans
        
        
        self.apply(self._init_weights)

        
    def make_layer(self, in_dim, hidden_dim):
        layers = nn.ModuleList([])
    
        layers.append(nn.Conv2d(in_dim, hidden_dim, 3, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.GELU())
            
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
    
        feature1 = self.layers0(x)
        feature2 = self.layers1(feature1)
        
        out = feature2.view(feature2.size(0), -1)
        out = self.mlp_head(out)
        
        # out = torch.chunk(out, self.n_tokenize, -1)

        
        return out
"""


class Localisation(nn.Module):
    def __init__(self, img_size, n_tokenize,in_dim=16, n_trans=4, type_trans='trans'):
        super().__init__()
        self.in_dim = in_dim
        
        if img_size == 32:
            
            self.layers0 = nn.Sequential(
                nn.Conv2d(3, self.in_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.GELU()
            )     
        
            img_size //= 2
            
        elif img_size == 64:
            self.layers0 = nn.Sequential(
                nn.Conv2d(3, self.in_dim, 7, 4, 2, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.GELU()
            )     
        
            img_size //= 4
        
        self.layers1 = self.make_layer(self.in_dim, self.in_dim*2)
        self.in_dim *= 2
        img_size //= 2
        
        # self.layers2 = self.make_layer(self.in_dim, self.in_dim*2)
        # self.in_dim *= 2
        # img_size //= 2
        
        if type_trans=='trans':
            n_output = 2*n_trans
        elif type_trans=='affine':
            n_output = 6*n_trans
        elif type_trans=='rigid':
            n_output = 3*n_trans
        
        # self.n_tokenize = n_tokenize 
        # n_output *= n_tokenize
            
        self.mlp_head = nn.Sequential(
            nn.Linear(self.in_dim * (img_size**2), 64, bias=False),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, n_output, bias=False),
            nn.LayerNorm(n_output),
            nn.Tanh()
        )
        self.num_transform = n_trans
        
        
        self.apply(self._init_weights)

        
    def make_layer(self, in_dim, hidden_dim):
        layers = nn.ModuleList([])
    
        layers.append(nn.Conv2d(in_dim, hidden_dim, 3, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.GELU())
            
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
    
        feature1 = self.layers0(x)
        feature2 = self.layers1(feature1)
        
        out = feature2.view(feature2.size(0), -1)
        out = self.mlp_head(out)
        
        # out = torch.chunk(out, self.n_tokenize, -1)

        
        return out
        
class Affine(nn.Module):
    def __init__(self, adaptive=False, constant=False):
        super().__init__()
        
        self.constant = adaptive
        self.theta = None
        self.init = None
            
        self.constant_tmp = 1 if not self.constant > 0. else constant
        
        
    def forward(self, x, theta, init, epoch=None, const=None):
        
        if not self.constant > 0.:            
            constant = 1
            
        elif const is not None:
            constant = const
                
        else:
            if epoch is not None:
                constant = self.constant * epoch         
                constant = 1 - math.exp(-constant)
                self.constant_tmp = constant
                
            else:
                constant = self.constant_tmp 
        # print(theta[0])
        # theta = theta + init
        theta = theta * constant + init * (1-constant)
        self.theta = theta    
        self.init = init    
        
        theta = torch.reshape(theta, (theta.size(0), 2, 3))        
        # print('========')
       
        print(theta[0])
        
        grid = F.affine_grid(theta, x.size())
        
        return F.grid_sample(x, grid)
    

# class Rigid(nn.Module):
#     def __init__(self, constant=5e1, adaptive=False):
#         super().__init__()
#         self.tmp1 = torch.tensor([[0, 0, 1],[0, 0, 1]]).cuda(torch.cuda.current_device())
#         self.tmp2 = torch.tensor([[1, 0, 0],[0, 1, 0]]).cuda(torch.cuda.current_device())
#         self.tmp3 = torch.tensor([[0, -1, 0],[1, 0, 0]]).cuda(torch.cuda.current_device())

            
#         self.constant = adaptive
#         self.theta = None
#         self.constant_tmp = 1
#         self.is_adaptive = adaptive
        
#     def forward(self, x, theta,  patch_size, epoch=None, train=False):
        
#         if not train or not self.is_adaptive:
#             constant = 1
                
#         else:
#             if epoch is not None:
#                 constant = self.constant * epoch         
#                 constant = 1 - math.exp(-constant)
#                 self.constant_tmp = constant
                
#             else:
#                 constant = self.constant_tmp 

#         # print(constant)

        
#         theta = theta * constant 
#         theta = theta.unsqueeze(-1)
                
#         angle = theta[:, (0,)]
#         angle = angle * math.pi
#         trans = theta[:, 1:]
        
#         cos = torch.cos(angle)
#         sin = torch.sin(angle)
     
#         mat_cos = torch.mul(cos, self.tmp2.expand(x.size(0), 2, 3))
#         mat_sin = torch.mul(sin, self.tmp3.expand(x.size(0), 2, 3))
#         mat_trans = torch.mul(trans, self.tmp1.expand(x.size(0), 2, 3))
        
#         theta = mat_cos + mat_sin + mat_trans
#         self.theta = theta
        
        
#         grid = F.affine_grid(theta.expand(x.size(0), 2, 3), x.size())
        
#         return F.grid_sample(x, grid)
    