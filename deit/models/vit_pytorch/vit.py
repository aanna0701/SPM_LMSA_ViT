# import torch
# from torch import nn, einsum
# from utils_.drop_path import DropPath
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
# from .SpatialTransformation import STT
# from utils_.coordconv import CoordLinear
# from utils_.iRPE import build_rpe, get_rpe_config
# # helpers
# def pair(t):
#     return t if isinstance(t, tuple) else (t, t)

# # classes

# def init_weights(m):
#     if isinstance(m, (nn.Linear, nn.Conv2d)):
#         nn.init.xavier_normal_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
# class PreNorm(nn.Module):
#     def __init__(self, num_tokens, dim, fn):
#         super().__init__()
#         self.dim = dim
#         self.num_tokens = num_tokens
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), ** kwargs)
#     def flops(self):
#         flops = 0        
#         flops += self.fn.flops()
#         flops += self.dim * (self.num_tokens+1)        
#         return flops    
# class FeedForward(nn.Module):
#     def __init__(self, dim, num_patches, hidden_dim, dropout = 0., is_coord=False):
#         super().__init__()
#         self.dim = dim
#         self.hidden_dim = hidden_dim
#         self.num_patches = num_patches
#         self.is_coord = is_coord
        
#         if self.is_coord:
#             self.net = nn.Sequential(
#                 CoordLinear(dim, hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 CoordLinear(hidden_dim, dim),
#                 nn.Dropout(dropout)
#             )            
#         else:
#             self.net = nn.Sequential(
#                 nn.Linear(dim, hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden_dim, dim),
#                 nn.Dropout(dropout)
#             )            
#     def forward(self, x):
#         return self.net(x)
    
#     def flops(self):
#         flops = 0
#         if not self.is_coord:
#             flops += self.dim * self.hidden_dim * (self.num_patches+1)
#             flops += self.dim * self.hidden_dim * (self.num_patches+1)
#         else:
#             flops += (self.dim+2) * self.hidden_dim * self.num_patches
#             flops += self.dim * self.hidden_dim
#             flops += self.dim * (self.hidden_dim+2) * self.num_patches
#             flops += self.dim * self.hidden_dim
        
#         return flops

# class Attention(nn.Module):
#     def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_coord=False, is_LSA=False, is_rpe=False):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)
#         self.is_coord = is_coord
#         self.num_patches = num_patches
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.dim = dim
#         self.inner_dim = inner_dim
#         self.attend = nn.Softmax(dim = -1)
#         self.to_qkv = CoordLinear(self.dim, self.inner_dim * 3, bias = False) if self.is_coord else nn.Linear(self.dim, self.inner_dim * 3, bias = False)
#         init_weights(self.to_qkv)
        
#         self.is_rpe = is_rpe
#         self.rpe_q, self.rpe_k, self.rpe_v = None, None, None
#         if is_rpe:
#             self.is_rpe = is_rpe
#             rpe_config = get_rpe_config(
#                 ratio=1.9,
#                 method="product",
#                 mode='ctx',
#                 shared_head=True,
#                 skip=1,
#                 rpe_on='qkv',
#             )
#             self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config, head_dim=dim_head, num_heads=heads)
        
#         if is_coord:
#             self.to_out = nn.Sequential(
#                 CoordLinear(self.inner_dim, self.dim),
#                 nn.Dropout(dropout)
#             ) if project_out else nn.Identity()            
#         else:            
#             self.to_out = nn.Sequential(
#                 nn.Linear(self.inner_dim, self.dim),
#                 nn.Dropout(dropout)
#             ) if project_out else nn.Identity()
            
#         if is_LSA:
#             self.scale = nn.Parameter(self.scale*torch.ones(heads))    
#             self.mask = torch.eye(self.num_patches+1, self.num_patches+1)
#             self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
#         else:
#             self.mask = None

#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

#         if not self.is_rpe:
#             if self.mask is None:
#                 dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            
#             else:
#                 scale = self.scale
#                 dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
#                 dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

#         else:
#             if self.mask is None:
#                 q = q * self.scale
#                 dots = einsum('b h i d, b h j d -> b h i j', q, k)
#                 # image relative position on keys
#                 if self.rpe_k is not None:
#                     dots += self.rpe_k(q)
                    
#                 if self.rpe_q is not None:
#                     dots += self.rpe_q(k * self.scale).transpose(2, 3)
            
#             else:
#                 scale = self.scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1))
#                 q = torch.mul(q, scale)
#                 dots = einsum('b h i d, b h j d -> b h i j', q, k)
                
#                 if self.rpe_k is not None:
#                     dots += self.rpe_k(q)
                
#                 if self.rpe_q is not None:
#                     dots += self.rpe_q(torch.mul(k, scale)).transpose(2, 3)
#                 dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321
        
#         attn = self.attend(dots)
#         out = einsum('b h i j, b h j d -> b h i d', attn, v) 
        
#         if self.rpe_v is not None:
#             out += self.rpe_v(attn)
            
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
    
#     def flops(self):
#         flops = 0
#         if not self.is_coord:
#             flops += self.dim * self.inner_dim * 3 * (self.num_patches+1)
#         else:
#             flops += (self.dim+2) * self.inner_dim * 3 * self.num_patches  
#             flops += self.dim * self.inner_dim * 3  
            
#         flops += self.inner_dim * ((self.num_patches+1)**2)
#         flops += self.inner_dim * ((self.num_patches+1)**2)
#         if not self.is_coord:
#             flops += self.inner_dim * self.dim * (self.num_patches+1)
#         else:
#             flops += (self.inner_dim+2) * self.dim * self.num_patches
#             flops += self.inner_dim * self.dim
        
#         return flops

# class Transformer(nn.Module):
#     def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0., 
#                  is_coord=False, is_LSA=False, is_rpe=False):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         self.scale = {}

#         for i in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_coord=is_coord, is_LSA=is_LSA, is_rpe=is_rpe)),
#                 PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout = dropout, is_coord=is_coord))
#             ]))            
#         self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    
#     def forward(self, x):
#         for i, (attn, ff) in enumerate(self.layers):       
#             x = self.drop_path(attn(x)) + x
#             x = self.drop_path(ff(x)) + x            
#             self.scale[str(i)] = attn.fn.scale
#         return x
    
#     def flops(self):
#         flops = 0        
#         for (attn, ff) in self.layers:       
#             flops += attn.flops()
#             flops += ff.flops()
        
#         return flops
# class ViT(nn.Module):
#     def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, channels = 3, 
#                  dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0., pe_dim=128, is_coord=False, is_LSA=False,
#                  is_base=True, eps=0., down_sizing=2, n_trans=8, STT_head=4, STT_depth=1, is_rpe=False, is_ape=False):
#         super().__init__()
#         image_height, image_width = pair(img_size)
#         patch_height, patch_width = pair(patch_size)
#         self.num_patches = (image_height // patch_height) * (image_width // patch_width)
#         self.patch_dim = channels * patch_height * patch_width
#         self.dim = dim
#         self.num_classes = num_classes
#         self.is_base = is_base
#         self.is_coord = is_coord
#         self.is_ape = is_ape
#         if self.is_base:
#             self.to_patch_embedding = nn.Sequential(
#                 Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#                 nn.Linear(self.patch_dim, self.dim)
#             )
            
#         else:
#             self.to_patch_embedding = STT(img_size=img_size, patch_size=patch_size, in_dim=pe_dim, embed_dim=dim, type='PE', heads=STT_head, depth=STT_depth
#                                            ,init_eps=eps, is_LSA=True, down_sizing=down_sizing, n_trans=n_trans)
                    
#         if is_ape:
#             self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
            
#         self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
#         self.dropout = nn.Dropout(emb_dropout)
#         self.transformer = Transformer(self.dim, self.num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout, 
#                                        stochastic_depth, is_coord=is_coord, is_LSA=is_LSA, is_rpe=is_rpe)

#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(self.dim),
#             nn.Linear(self.dim, self.num_classes)
#         )
#         self.theta = None
#         self.scale = None   
        
#         self.apply(init_weights)

#     def forward(self, img):
#         # patch embedding
#         theta = list()
#         scale = list()
#         x = self.to_patch_embedding(img)
            
#         if not self.is_base:     
#             theta.append(self.to_patch_embedding.theta)   
#             scale.append(self.to_patch_embedding.scale_list)
        
#         b, n, _ = x.shape
        
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
      
#         x = torch.cat((cls_tokens, x), dim=1)
#         if self.is_ape:
#             x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)

#         x = self.transformer(x)    
        
#         self.theta = theta 
#         self.scale = scale
        
#         return self.mlp_head(x[:, 0])

#     def flops(self):
#         flops = 0
        
#         if self.is_base:
#             # if self.is_coord:
#             #     flops_pe = self.num_patches * (self.patch_dim+2) * self.dim
#             # else:
#             flops_pe = self.num_patches * self.patch_dim * self.dim 
#         else:
#             flops_pe = self.to_patch_embedding.flops()        
#         flops += flops_pe        
#         flops += self.transformer.flops()           
#         flops += self.dim               # layer norm
#         flops += self.dim * self.num_classes    # linear
        
#         return flops

import torch
from torch import nn, einsum
from utils_.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .SPT import ShiftedPatchTokenization
from .Coord import CoordLinear
# helpers
 
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), ** kwargs)
 
class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout = 0., is_Coord=False):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.is_Coord = is_Coord
        
        if not is_Coord:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )            
        else:
            self.net = nn.Sequential(
                CoordLinear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                CoordLinear(hidden_dim, dim),
                nn.Dropout(dropout)
            )            
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_LSA=False, is_Coord=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim = -1)
        self. is_Coord = is_Coord
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias = False) if not is_Coord else CoordLinear(self.dim, self.inner_dim * 3, bias = False)
        init_weights(self.to_qkv)
        
        if not is_Coord:
            self.to_out = nn.Sequential(
                nn.Linear(self.inner_dim, self.dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity() 
        else:
            self.to_out = nn.Sequential(
                CoordLinear(self.inner_dim, self.dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
            
        if is_LSA:
            self.scale = nn.Parameter(self.scale*torch.ones(heads))    
            self.mask = torch.eye(self.num_patches+1, self.num_patches+1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if self.mask is None:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v) 
            
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    def flops(self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches+1)
        else:
            flops += (self.dim+2) * self.inner_dim * 3 * self.num_patches  
            flops += self.dim * self.inner_dim * 3  


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0., is_LSA=False, is_Coord=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_LSA=is_LSA, is_Coord=is_Coord)),
                PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout = dropout, is_Coord=is_Coord if not i == depth-1 else False))
            ]))            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):       
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x            
            self.scale[str(i)] = attn.fn.scale
        return x

class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, channels = 3, 
                 dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0.,is_LSA=False, is_SPT=False, is_Coord=False):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.num_classes = num_classes
        self.is_Coord = is_Coord
       
        if not is_SPT:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(self.patch_dim, self.dim)
            )
            
        else:
            self.to_patch_embedding = ShiftedPatchTokenization(3, self.dim, patch_size, is_pe=True, is_Coord=is_Coord)
        
        if not is_Coord:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
            
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(self.dim, self.num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout, 
                                       stochastic_depth, is_LSA=is_LSA, is_Coord=is_Coord)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )
        
        self.apply(init_weights)

    def forward(self, img):
        # patch embedding
        
        x = self.to_patch_embedding(img)
            
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
      
        x = torch.cat((cls_tokens, x), dim=1)
        if not self.is_Coord:
            x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)      
        
        return self.mlp_head(x[:, 0])


