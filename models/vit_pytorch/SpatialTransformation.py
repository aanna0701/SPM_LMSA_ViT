
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
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
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_LSA=False):
        super().__init__()
        self.inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim  = dim
        self.num_patches = num_patches
        self.to_q = nn.Linear(self.dim, self.inner_dim, bias = False) 
        self.to_kv = nn.Linear(self.dim, self.inner_dim * 2, bias = False)
        
        self.attend = nn.Softmax(dim = -1)
        self.mix_heads_pre_attn = nn.Parameter(torch.zeros(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.zeros(heads, heads))        
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
        flops += self.dim * self.inner_dim * 2 * self.num_patches
        flops += self.inner_dim * self.num_patches
        flops += self.inner_dim * self.num_patches
        flops += self.num_patches   # scaling
        flops += self.num_patches   # pre-mix
        flops += self.num_patches   # post-mix
        flops += self.inner_dim * self.dim
        
        return flops
    
class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0., is_LSA=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_LSA=is_LSA)),
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
    def __init__(self, num_patches, depth, in_dim, hidden_dim, heads, n_trans=4, down_sizing=2, is_LSA=False):
        super().__init__()
        self.in_dim = in_dim
        self.n_trans = n_trans
        self.n_output = 6*self.n_trans
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.down_sizing = down_sizing
        # self.param_transformer = Transformer(self.in_dim*(patch_size**2), num_patches, depth, heads, hidden_dim//heads, self.in_dim)
        self.param_transformer = Transformer(self.in_dim, self.num_patches//(self.down_sizing**2), depth, heads, self.in_dim//heads, self.in_dim*2, is_LSA=is_LSA)       
        self.depth_wise_conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.in_dim, self.down_sizing, self.down_sizing, groups=self.in_dim),
            Rearrange('b c h w -> b (h w) c')
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.n_output)
        )  
        self.transformation = Affine()
        self.pre_linear = nn.Conv2d(self.in_dim, self.hidden_dim, (1, 1))
        self.post_linear = nn.Conv2d(self.hidden_dim, self.in_dim, (1, 1))
    
        self.theta = list()
        
    def forward(self, param_token, x, init, scale=None):
        # print(x.shape)
        if len(x.size()) == 3:
            x = rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1)))) 
        param_token = repeat(param_token, '() n d -> b n d', b = x.size(0))
        param_attd = self.param_transformer(param_token, self.depth_wise_conv(x))
        param = self.mlp_head(param_attd[:, 0])
        param_list = torch.chunk(param.unsqueeze(1), self.n_trans, dim=-1)
        self.theta = param_list
        
        out = []
        # theta = []       
        
        x = self.pre_linear(x)
        x = torch.chunk(x, self.n_trans, dim=1)
        for i in range(self.n_trans):
            if scale is not None:
                out.append(self.transformation(x[i], param_list[i], init, scale[i]))
            else:
                out.append(self.transformation(x[i], param_list[i], init))
            # theta.append(self.transformation.theta)            
        out = torch.cat(out, dim=1)
        out = self.post_linear(out)        
        out = rearrange(out, 'b d h w -> b (h w) d')
        
        
        return out
    
    def flops(self):
        flops = 0
        flops += (self.down_sizing**2)*self.num_patches*self.hidden_dim    # depth-wise conv
        flops += self.param_transformer.flops()                 # parameter-transformer
        flops += self.in_dim + self.in_dim*self.n_output    # mlp head
        flops += self.num_patches*self.in_dim*self.hidden_dim    # pre-linear
        flops += self.num_patches*self.in_dim*self.hidden_dim   # post-linear
        
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
        flops += (self.num_patches//(self.patch_size**2))*(self.patch_dim+2)*self.out_dim
        flops += (self.num_patches//(self.patch_size**2))*self.patch_dim
        
        return flops
    
class Affine(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super().__init__()
        
        self.theta = None
        self.mode = padding_mode
        
    def forward(self, x, theta, init, scale=None):
        print('========')
        print(scale)
        print(theta[0])     
        self.theta = theta 
        # theta = F.sigmoid(theta)
        theta = F.normalize(theta, dim=(-1))
        if scale is not None:
            theta = torch.mul(theta, scale)
        
        init = torch.reshape(init.unsqueeze(0), (1, 2, 3)).expand(x.size(0), -1, -1) 
        theta = torch.reshape(theta, (theta.size(0), 2, 3))    
        theta = theta + init 
           
        print(theta[0])
        
        grid = F.affine_grid(theta, x.size())
        
        return F.grid_sample(x, grid, padding_mode=self.mode)
    
class STT(nn.Module):
    def __init__(self, img_size=224, patch_size=2, in_dim=3, embed_dim=96, depth=2, heads=4, type='PE', 
                 init_eps=0., is_LSA=False, down_sizing=4, n_trans=4, exist_cls_token=False):
        super().__init__()
        assert type in ['PE', 'Pool'], 'Invalid type!!!'

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size**2
        self.in_dim = in_dim
        self.type = type
        self.exist_cls_token = exist_cls_token
        
        if self.type == 'PE':
            if not img_size >= 224:
                if patch_size > 2:
                    self.input = nn.Conv2d(3, self.in_dim, 3, 2, 1)
                    self.rearrange = Rearrange('b c h w -> b (h w) c')      
                    self.affine_net = AffineNet(self.num_patches//4, depth, self.in_dim, self.in_dim, heads, down_sizing=down_sizing, 
                                                is_LSA=is_LSA, n_trans=n_trans)
                    self.patch_merge = PatchMerging(self.num_patches//4, patch_size//2, self.in_dim, embed_dim) 
                else:
                    self.input = nn.Conv2d(3, self.in_dim, 3, 1, 1)
                    self.rearrange = Rearrange('b c h w -> b (h w) c')      
                    self.affine_net = AffineNet(self.num_patches, depth, self.in_dim, self.in_dim, heads, down_sizing=down_sizing, 
                                                is_LSA=is_LSA, n_trans=n_trans)
                    self.patch_merge = PatchMerging(self.num_patches, patch_size, self.in_dim, embed_dim) 
            else:
                self.input = nn.Conv2d(3, self.in_dim, 7, 4, 2)
                self.rearrange = Rearrange('b c h w -> b (h w) c')      
                self.affine_net = AffineNet(self.num_patches//16, depth, self.in_dim, self.in_dim, heads, down_sizing=down_sizing, 
                                            is_LSA=is_LSA, n_trans=n_trans)
                self.patch_merge = PatchMerging(self.num_patches//16, patch_size//4, self.in_dim, embed_dim)   
           
        else:
            self.input = nn.Identity()
            self.rearrange = nn.Identity()
            self.affine_net = AffineNet(self.num_patches, depth, self.in_dim, self.in_dim, heads, down_sizing=down_sizing, 
                                        is_LSA=is_LSA, n_trans=n_trans)
            self.patch_merge = PatchMerging(self.num_patches, 2, self.in_dim, self.in_dim*2)
            self.cls_proj = nn.Linear(self.in_dim, self.in_dim*2) if exist_cls_token else None 
        
        self.param_token = nn.Parameter(torch.zeros(1, 1, self.in_dim))
                      
        if init_eps is not None:
            self.scale_list = nn.ParameterList()  
            for _ in range(n_trans):
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
        
        if self.type=='Pool':
            if self.cls_proj is not None:
                cls = x[:, (0, )]
                x = x[:, 1:]
                cls = self.cls_proj(cls)
            
        x = self.input(x)
        affine = self.affine_net(self.param_token, x, self.init, self.scale_list)
        self.theta = self.affine_net.theta
        x = self.rearrange(x)
        out = x + affine
        out = self.patch_merge(out)
        
        if self.type=='Pool':
            if self.cls_proj is not None:
                out = torch.cat([cls, out], dim=1)
        
        return out
    
    def flops(self):
        flops = 0
        if self.type=='PE':
            if self.img_size < 224:
                if self.patch_size > 2:
                    flops_input = (3**2)*3*self.in_dim*((self.img_size//2)**2)
                else:
                    flops_input = (3**2)*3*self.in_dim*((self.img_size)**2)
            else:
                flops_input = (7**2)*3*self.in_dim*((self.img_size//16)**2)
        else:
            if self.cls_proj is not None:
                flops_input = self.in_dim * self.in_dim*2
            else:
                flops_input = 0      
        flops += flops_input
        flops += self.affine_net.flops()   
        flops += self.patch_merge.flops() 
        
        
        return flops
    