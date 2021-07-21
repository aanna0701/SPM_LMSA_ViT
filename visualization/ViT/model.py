import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
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
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0.):
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
        self.mask = torch.eye(num_patches, num_patches)
        self.mask = (self.mask == 1).nonzero()
        self.inf = float('-inf')
        self.score = None
        
    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf

        attn = self.attend(dots)
        self.score = attn

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.hidden_states = {}

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
        
        self.scores = []
    def forward(self, x):
        self.hidden_states['patch_embedding'] = x[:, 1:].mean(dim=-1)
        for i, (attn, ff) in enumerate(self.layers):            
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
            
            self.hidden_states[str(i)] = x[:, 1:].mean(dim=-1)
            self.scores.append(attn.fn.score)
            
        return x

class Model(nn.Module):
    def __init__(self, *, img_size=32, patch_size=4, num_classes=100, dim=192, depth=9, heads=12, mlp_dim=384, pool = 'cls', channels = 3, dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0.1):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.linear_to_path = nn.Linear(patch_dim, dim)
        self._init_weights(self.linear_to_path)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            self.linear_to_path,
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim, dropout, stochastic_depth)

        self.pool = pool
        self.to_latent = nn.Identity()
        
        nn.linear_mlp_head = nn.Linear(dim, num_classes)
        self._init_weights(nn.linear_mlp_head)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.linear_mlp_head
        )
        
        self.final_cls_token = None
        self.scores = None
        self.featuremaps = None

    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  

    def forward(self, img):
        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        
        x = self.transformer(x)
        
        self.featuremaps = self.transformer.hidden_states
        self.scores = self.transformer.scores

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        
        self.final_cls_token = x
        
        return self.mlp_head(x)

class patch_shifting(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        # x_l = torch.cat([torch.nn.pad(x, ), x[:, :, :, 1:]], dim=-1)
        # x_r = torch.cat([x[:, :, :, :-1], self.w_pad], dim=-1)
        # x_t = torch.cat([self.h_pad, x[:, :, 1:]], dim=-2)
        # x_b = torch.cat([x[:, :, :-1], self.h_pad], dim=-2)
        
        # print(x_l.shape)
        # print(x_r.shape)
        # print(x_t.shape)
        # print(x_b.shape)
        
        x_pad = torch.nn.functional.pad(x, (1, 1, 1, 1))


        x_l = x_pad[:, :, 1:-1, :-2]
        x_r = x_pad[:, :, 1:-1, 2:]
        x_t = x_pad[:, :, :-2, 1:-1]
        x_b = x_pad[:, :, 2:, 1:-1]
        
        x_cat = torch.cat([x, x_l, x_r, x_t, x_b], dim=1)
        
        return x_cat
        