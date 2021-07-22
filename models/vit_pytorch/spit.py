from math import sqrt
from utils.drop_path import DropPath
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num

def conv_output_size(image_size, kernel_size, stride, padding = 0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

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
        
        # self.scale = nn.Parameter(torch.rand(heads))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self._init_weights(self.to_qkv) 
        
        self.to_out = nn.Linear(inner_dim, dim)
        self._init_weights(self.to_out)
        
        self.to_out = nn.Sequential(
            self.to_out,
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.mask = torch.eye(num_patches+1, num_patches+1)
        self.mask = (self.mask == 1).nonzero()
        self.inf = float('-inf')
        
    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # scale = self.scale
        # dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
    
        # dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        self.hidden_states = {}
        self.scale = {}
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches=num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
                     
            self.scale[str(i)] = attn.fn.scale
        return x

# pooling layer

class Pool(nn.Module):
    def __init__(self, dim, num_patches):
        super().__init__()
        
        
        self.cls_ff = nn.Linear(dim, dim * 2)
        self._init_weights(self.cls_ff)
        
        self.patch_shifting = Patch_shifting(dim*5,dim)
        
        self.unfold = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 2, p2 = 2)
        
        self.tokens_ff = nn.Linear(dim*4, dim*2, bias=False)
        self._init_weights(self.tokens_ff)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim*2))
        
    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  

    def forward(self, x):
        cls_token, tokens = x[:, :1], x[:, 1:]

        cls_token = self.cls_ff(cls_token)
        tokens = rearrange(tokens, 'b (h w) c -> b c h w', h = int(sqrt(tokens.shape[1])))
        tokens_shift_cat = self.patch_shifting(tokens)
        tokens = self.unfold(tokens_shift_cat)
        tokens = self.tokens_ff(tokens)

        return torch.cat((cls_token, tokens), dim = 1) + self.pos_embedding

# main class

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class SPiT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0., stochastic_depth=0.):
        super().__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert isinstance(depth, tuple), 'depth must be a tuple of integers, specifying the number of blocks before each downsizing'
        heads = cast_tuple(heads, len(depth))

        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)        
        patch_dim = 3 * patch_height * patch_width

        self.linear_to_path = nn.Linear(patch_dim, dim)
        self._init_weights(self.linear_to_path)
        self.to_patch_embedding = nn.Sequential(
            Patch_shifting(),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            self.linear_to_path
        )


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        layers = []
        

        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)
            
            layers.append(Transformer(dim, num_patches, layer_depth, layer_heads, dim_head, dim*2, dropout, stochastic_depth))

            if not_last:
                num_patches = num_patches // 4
                layers.append(Pool(dim, num_patches))
                dim *= 2
                

        self.layers = nn.Sequential(*layers)

        nn.linear_mlp_head = nn.Linear(dim, num_classes)
        self._init_weights(nn.linear_mlp_head)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.linear_mlp_head
        )


    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.layers(x)

        return self.mlp_head(x[:, 0])
    

class Patch_shifting(nn.Module):
    def __init__(self, dim_in=15, dim_out=3):
        super().__init__()
        self.linear_to_patch = nn.Conv2d(dim_in, dim_out, 1)
        self._init_weights(self.linear_to_patch)   
        
    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)       

    def forward(self, x):
        
        x_pad = torch.nn.functional.pad(x, (1, 1, 1, 1))
        
        x_l = x_pad[:, :, 1:-1, :-2]
        x_r = x_pad[:, :, 1:-1, 2:]
        x_t = x_pad[:, :, :-2, 1:-1]
        x_b = x_pad[:, :, 2:, 1:-1]
        
        x_cat = torch.cat([x, x_l, x_r, x_t, x_b], dim=1)
        
        return self.linear_to_patch(x_cat)
        