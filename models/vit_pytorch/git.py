import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

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
        
        self.linear1 = nn.Linear(dim, hidden_dim)
        self._init_weights(self.linear1)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self._init_weights(self.linear2)
        
        self.net = nn.Sequential(
            self.linear1,
            nn.GELU(),
            nn.Dropout(dropout),
            self.linear2,
            nn.Dropout(dropout)
        )
        
    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)      
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self._init_weights(self.to_qkv)
        
        self.to_out = nn.Linear(inner_dim, dim)
        self._init_weights(self.to_out)

        self.to_out = nn.Sequential(
            self.to_out,
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()        
        
        self.lam = nn.Parameter(torch.zeros(1))
        self.gam = nn.Parameter(torch.zeros(1))
        
    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        
        # nodes = x[:, 1:]
        # global_attribute = x[:, (0, )]
        
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        out_nodes = self.to_out(out)

        # print(node_aggregation[0].norm(2))
        
        # print(f'gam:{self.gam}')
        
        return out_nodes

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class GiT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'mean', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., stochastic_depth=0.):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        
        self.linear_to_path = nn.Linear(patch_dim, dim)
        self._init_weights(self.linear_to_path)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            self.linear_to_path,
        )
        
        # self.linear_cls_tokens = nn.Linear(num_patches, 1)
        # self._init_weights(self.linear_cls_tokens)
        # self.to_cls_tokens = nn.Sequential(
        #     Rearrange('b n c -> b c n'),
        #     self.linear_cls_tokens,
        #     Rearrange('b c n -> b n c')
        # )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, stochastic_depth)

        self.pool = pool
        self.to_latent = nn.Identity()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        nn.linear_mlp_head = nn.Linear(dim, num_classes) if pool == 'cls' else nn.Linear(dim+1, num_classes)
        self._init_weights(nn.linear_mlp_head)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim) if pool == 'cls' else nn.LayerNorm(dim+1),
            nn.linear_mlp_head
        )

    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  

    def forward(self, img):
        # pathc embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        # cls_tokens = self.to_cls_tokens(x)
        
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        # x = x.mean(dim = 1) if self.pool == 'mean' else read_out(x)
        x = read_out(x).mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def read_out(x):
    x_max = x.max(dim=-1, keepdim=True)[0]
    out = torch.cat([x, x_max], dim=-1)
    return out