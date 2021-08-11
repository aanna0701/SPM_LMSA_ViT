import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

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
        self.scale = nn.Parameter(self.scale*torch.ones(heads))
        
        self.mask = torch.eye(num_patches+1, num_patches+1) 
        self.mask = torch.nonzero((self.mask == 1), as_tuple=False) 
        self.inf = float('-inf')

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # scale = self.scale
        # dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
        # dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf
        
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

from utils.drop_path import DropPath
class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for attn, ff in self.layers:
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
        return x

class DeepViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim=396, depth=16, heads=12, mlp_dim=1188, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., stochastic_depth=0.):
        super().__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_size // patch_size) ** 2
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
        
        
        # patch_dim = (channels+4) * patch_size ** 2
        # self.to_patch_embedding = nn.Sequential(
        #     PatchShifting(patch_size),
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        #     nn.Linear(patch_dim, dim),
        # )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim, dropout, stochastic_depth=stochastic_depth)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class PatchShifting(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.shift = patch_size // 2

    def forward(self, x):
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        
        x_pad = x_pad.mean(dim=1, keepdim = True)
        
        x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
               
        x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)
        
        
        return x_cat