import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.relative_norm_residuals import compute_relative_norm_residuals
import math

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


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))

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
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5        
        # self.scale = nn.Parameter(torch.rand(heads))
        # self.scale = nn.Parameter(torch.rand(1))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        init_weights(self.to_qkv)
        
 

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        
        self.mask = torch.eye(num_patches+1, num_patches+1)
        self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        self.inf = float('-inf')
        
        self.value = 0
        self.avg_h = None
        self.cls_l2 = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
      

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # scale = self.scale
        # dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
    
        
        # dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
    
        
        self.value = compute_relative_norm_residuals(v, out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.hidden_states = {}
        self.scale = {}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))            
            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):       
            x = self.drop_path(attn(x)) + x
            self.hidden_states[str(i)] = attn.fn.value
            x = self.drop_path(ff(x)) + x
            
            self.scale[str(i)] = attn.fn.scale
        return x

class BottleneckTransformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.hidden_states = {}
        self.scale = {}
        self.activation = nn.GELU()

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, nn.Linear(dim, dim // 2)),
                PreNorm(dim // 2, Attention(dim // 2, num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim // 2, nn.Linear(dim // 2, dim))
            ]))            
            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for i, (contr, attn, expand) in enumerate(self.layers):    
            residuals = x
            x = self.activation(self.drop_path(contr(x))) 
            x = self.activation(self.drop_path(attn(x))) + x
            x = self.activation(self.drop_path(expand(x))) + residuals
            
        return x

class GiT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., stochastic_depth=0.):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = 15 * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        self.to_patch_embedding = nn.Sequential(
            PatchShifting(patch_size),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
            Transformer(dim, num_patches, 2, 1, 64, 64*2, 0),
            PatchMerging(dim, dim*2, 2, is_pe=True)
        )

        dim *= 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # heads *= 2
        # self.transformer0 = nn.Sequential(
        #     PatchMerging(dim, 2),
        #     Transformer(dim, num_patches, depth[0], heads, dim_head, mlp_dim, dropout, stochastic_depth)
        # )
        # dim *= 2
        # heads *= 2
        # self.transformer1 = nn.Sequential(
        #     PatchMerging(2),
        #     Transformer(dim, num_patches, depth[1], heads, dim_head, mlp_dim, dropout, stochastic_depth)
        # )
        
        self.transformer = []
        for i in range(len(depth)):
            if i+1 != len(depth):                
                num_patches //= 4
                self.transformer.append(Transformer(dim, num_patches, depth[i], heads, dim_head, dim*2, dropout, stochastic_depth)) 
                self.transformer.append(PatchMerging(dim, dim*2, 2))  
                heads *= 2
                dim *= 2 
            else:
                num_patches //= 4
                self.transformer.append(Transformer(dim, num_patches, depth[i], heads, dim_head, dim*2, dropout, stochastic_depth)) 
                
        
        self.transformer = nn.Sequential(*self.transformer)
        
        
        self.pool = pool
        self.to_latent = nn.Identity()


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.final_cls_token = None
        
        self.apply(init_weights)


    def forward(self, img):
        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)


        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        
        self.final_cls_token = x
        
        return self.mlp_head(x)

class PatchMerging(nn.Module):
    def __init__(self, in_dim, dim, merging_size=2, is_pe=False):
        super().__init__()
        
        self.is_pe = is_pe
        
        patch_dim = in_dim * (merging_size**2)        
        
        self.merging = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
            nn.Linear(patch_dim, dim)
        )
        
        if not is_pe:
            self.class_linear = nn.Linear(in_dim, dim)

    def forward(self, x):
        
        _, n, _ = x.size()
        h = int(math.sqrt(n))
        
        if not self.is_pe:
            visual_tokens, class_token = x[:, 1:], x[:, (0,)]
            reshaped = rearrange(visual_tokens, 'b (h w) d -> b d h w', h=h)
            out_visual = self.merging(reshaped)
            out_class = self.class_linear(class_token)
            out = torch.cat([out_class, out_visual], dim=1)
        
        else:
            reshaped = rearrange(x, 'b (h w) d -> b d h w', h=h)
            out = self.merging(reshaped)
        
        return out
    
class PatchShifting(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.shift = int(patch_size * (1/2))

    def forward(self, x):
        
        # x = x.mean(dim=1, keepdim = True)

        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        
        # x_pad = x_pad.mean(dim=1, keepdim = True)
        
        x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
               
        x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)
        
        
        return x_cat
    
# class PatchShifting(nn.Module):
#     def __init__(self, patch_size):
#         super().__init__()
#         self.shift = int(patch_size * (1/2))

#     def forward(self, x):
        
#         # x = x.mean(dim=1, keepdim = True)

#         x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        
#         # x_pad = x_pad.mean(dim=1, keepdim = True)
        
#         x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
#         x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
#         x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
#         x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
               
#         x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)
        
        
#         return x_cat
    
