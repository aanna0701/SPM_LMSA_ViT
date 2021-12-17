import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.relative_norm_residuals import compute_relative_norm_residuals
from .SpatialTransformation import STT
from utils.coordconv import CoordLinear
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
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout = 0., is_coord=False):
        super().__init__()
        
        if is_coord:
            self.net = nn.Sequential(
                CoordLinear(num_patches, dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                CoordLinear(num_patches, hidden_dim, dim),
                nn.Dropout(dropout)
            )
            
        else:
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
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_coord=False, is_LSA=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = CoordLinear(num_patches, dim, inner_dim * 3, bias = False) if is_coord else nn.Linear(dim, inner_dim * 3, bias = False)
        init_weights(self.to_qkv)
        

        if is_coord:
            self.to_out = nn.Sequential(
                CoordLinear(num_patches, inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
            
        else:            
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
            
        if is_LSA:
            self.scale = nn.Parameter(self.scale*torch.ones(heads))    
            self.mask = torch.eye(num_patches+1, num_patches+1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None
        
        self.value = 0
        self.avg_h = None
        self.cls_l2 = None

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
    
        
        # self.value = compute_relative_norm_residuals(v, out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0., is_coord=False, is_LSA=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.hidden_states = {}
        self.scale = {}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_coord=is_coord, is_LSA=is_LSA)),
                PreNorm(dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout = dropout, is_coord=is_coord))
            ]))            
            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):       
            x = self.drop_path(attn(x)) + x
            self.hidden_states[str(i)] = attn.fn.value
            x = self.drop_path(ff(x)) + x
            
            self.scale[str(i)] = attn.fn.scale
        return x

class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, pool = 'cls', channels = 3, 
                 dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0., pe_dim=64, is_coord=False, is_LSA=False,
                 is_base=True, eps=0., no_init=False, init_noise=[1e-3, 1e-3], merging_size=4):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        if is_base:
           if is_coord:    
                self.to_patch_embedding = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                    CoordLinear(num_patches, patch_dim, dim, exist_cls_token=False)
                )   
           else:
               self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(patch_dim, dim)
            )
            
        else:
            self.to_patch_embedding = STT(img_size=img_size, patch_size=patch_size, in_dim=pe_dim, embed_dim=dim, type='PE',
                                           init_eps=eps, init_noise=init_noise, merging_size=merging_size ,no_init=no_init)
        
            
        if not is_coord:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout, stochastic_depth, is_coord=is_coord, is_LSA=is_LSA)

        self.pool = pool
        self.to_latent = nn.Identity()


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

 
        self.is_base = is_base
        self.is_coord = is_coord
        self.theta = None
        self.scale = None   
        
        self.apply(init_weights)


    def forward(self, img):
        # patch embedding
        
        x = self.to_patch_embedding(img)
            
        if not self.is_base and not self.is_coord:        
            self.theta = self.to_patch_embedding.theta
            self.scale = self.to_patch_embedding.scale_list
        
        b, n, _ = x.shape

        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
      
        x = torch.cat((cls_tokens, x), dim=1)
        if not self.is_coord:
            x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        
        
        return self.mlp_head(x)


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
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         # self.scale = nn.Parameter(self.scale*torch.ones(heads))

#         self.attend = nn.Softmax(dim = -1)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
#         init_weights(self.to_qkv)
        
 

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
        
        
#         self.mask = torch.eye(num_patches+1, num_patches+1)
#         self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
#         self.inf = float('-inf')
        
#         self.value = 0
#         self.avg_h = None
#         self.cls_l2 = None

#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
      

#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
#         # scale = self.scale
#         # dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
    
        
#         # dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf

#         attn = self.attend(dots)

#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
    
        
#         self.value = compute_relative_norm_residuals(v, out)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

# class Transformer(nn.Module):
#     def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         self.hidden_states = {}
#         self.scale = {}

#         for i in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
#                 PreNorm(dim, FeedForward(dim, dim * mlp_dim_ratio, dropout = dropout))
#             ]))            
            
#         self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
#     def forward(self, x):
#         for i, (attn, ff) in enumerate(self.layers):       
#             x = self.drop_path(attn(x)) + x
#             self.hidden_states[str(i)] = attn.fn.value
#             x = self.drop_path(ff(x)) + x
            
#             self.scale[str(i)] = attn.fn.scale
#         return x

# class ViT(nn.Module):
#     def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, pool = 'cls', channels = 3, 
#                  dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0., pe_dim=64,
#                  is_base=True, eps=0., no_init=False, init_noise=[1e-3, 1e-3], merging_size=4):
#         super().__init__()
#         image_height, image_width = pair(img_size)
#         patch_height, patch_width = pair(patch_size)

#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

#         if is_base:
        
#             self.to_patch_embedding = nn.Sequential(
#                 Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#                 nn.Linear(patch_dim, dim)
#             )
            
#         else:
#             self.to_patch_embedding = STT(img_size=img_size, patch_size=patch_size, in_dim=pe_dim, embed_dim=dim, type='PE',
#                                            init_eps=eps, init_noise=init_noise, merging_size=merging_size ,no_init=no_init)
            

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)
#         self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout, stochastic_depth)

#         self.pool = pool
#         self.to_latent = nn.Identity()


#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )

 
#         self.is_base = is_base
#         self.theta = None
#         self.scale = None   
        
#         self.apply(init_weights)


#     def forward(self, img):
#         # patch embedding
        
#         x = self.to_patch_embedding(img)
            
#         if not self.is_base:        
#             self.theta = self.to_patch_embedding.theta
#             self.scale = self.to_patch_embedding.scale_list
        
#         b, n, _ = x.shape

        
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
      
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)

#         x = self.transformer(x)

#         x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

#         x = self.to_latent(x)
        
        
#         return self.mlp_head(x)


