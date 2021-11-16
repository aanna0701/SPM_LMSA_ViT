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
        
        self.scale = nn.Parameter(self.scale*torch.ones(heads))
        
        self.if_patch_attn = if_patch_attn

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
    

class Localisation(nn.Module):
    def __init__(self, img_size,in_dim=16, n_trans=4, type_trans='affine'):
        super().__init__()
        self.in_dim = in_dim
        
            
        self.layers0 = nn.Sequential(
            nn.Conv2d(3, self.in_dim, 3, 2, 1)
        )     
    
        img_size //= 2
        
        
        self.layers1 = self.make_layer(self.in_dim, self.in_dim*2)
        self.in_dim *= 2
        img_size //= 2
        
        if img_size == 64:
            self.layers2 = self.make_layer(self.in_dim, self.in_dim*2)
            self.in_dim *= 2
            img_size //= 2
        
        else:
            self.layer2 = None
        
        if type_trans == 'affine':
            n_output = 6*n_trans
        
        elif type_trans == 'trans_scale':
            n_output = 3*n_trans
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, n_output, bias=False)
        )
        
        
        self.num_transform = n_trans
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.in_dim))
        self.cls_transformer = Transformer(self.in_dim, img_size**2, 2, 4, 16, 128)

        
        self.apply(self._init_weights)

        
    def make_layer(self, in_dim, hidden_dim):
        layers = nn.ModuleList([])
    
        layers.append(nn.Conv2d(in_dim, hidden_dim, 3, 2, 1, bias=False))
            
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            # nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
    
        feature1 = self.layers0(x)
        out = self.layers1(feature1)
        
        out = self.layers2(out) if self.layer2 is not None else out
        
        out = rearrange(out, 'b c h w -> b (h w) c')
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = x.size(0))
        cls_attd = self.cls_transformer(cls_tokens, out)
        out = self.mlp_head(cls_attd[:, 0])
        
        return out
        


class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale
     
class Affine(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super().__init__()
        
        self.theta = None
        self.mode = padding_mode
        
    def forward(self, x, theta, init, scale=None):
        
        # theta = torch.mul(theta, self.scale) + init
        theta = theta + init if scale is None else torch.mul(theta, scale) + init
        # theta = theta + init if scale is None else torch.mul(theta, scale) + torch.mul(init, (1-scale))
        # theta = theta 
        self.theta = theta
        
        theta = torch.reshape(theta, (theta.size(0), 2, 3))        
        print('========')
        print(scale)
        print(theta[0])
        
        grid = F.affine_grid(theta, x.size())
        
        return F.grid_sample(x, grid, padding_mode=self.mode)
     
class Trans_scale(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super().__init__()
        
        self.mode = padding_mode
        self.trans = torch.tensor([[0, 0, 1], [0, 0, 1]]).cuda(torch.cuda.current_device())
        self.scaling = torch.tensor([[1, 0, 0], [0, 1, 0]]).cuda(torch.cuda.current_device())
        
    def forward(self, x, theta, init, scale=None):
        
        
        print('========')
        print(scale)
        
        # trans = torch.mul(self.trans, theta[:, 1:].unsqueeze(-1))
        # scaling = torch.mul(self.scaling, theta[:, 0].unsqueeze(-1).expand(-1, 2).unsqueeze(-1))
        
        if scale is not None:
            scale = scale.expand(x.size(0), -1).unsqueeze(-1)
            trans = torch.mul(self.trans, torch.mul(theta[:, 1:].unsqueeze(-1), scale[:,1:]))
            scaling = torch.mul(self.scaling, torch.mul(theta[:, 0].unsqueeze(-1).expand(-1, 2).unsqueeze(-1), scale[:, (0,)]))
        
        else:
            trans = torch.mul(self.trans, theta[:, 1:].unsqueeze(-1))
            scaling = torch.mul(self.scaling, theta[:, 0].unsqueeze(-1).expand(-1, 2).unsqueeze(-1))
        
        theta = trans + scaling
        init = torch.reshape(init.unsqueeze(0), (1, 2, 3)).expand(x.size(0), -1, -1) 
        
        
        
        print(theta[0])
        
        # theta = torch.mul(theta, self.scale) + init
        # theta = theta + init if scale is None else torch.mul(theta, scale) + init
        theta = theta + init 
        # theta = theta + init if scale is None else torch.mul(theta, scale) + torch.mul(init, (1-scale))
        # theta = theta 
        self.theta = theta
        
        # theta = torch.reshape(theta, (theta.size(0), 2, 3))        
        
        print(theta[0])
        
        grid = F.affine_grid(theta, x.size())
        
        return F.grid_sample(x, grid, padding_mode=self.mode)
     
     
     