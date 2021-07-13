import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils.relative_norm_residuals import compute_relative_norm_residuals

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

# class G_Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim = -1)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
#         self._init_weights(self.to_qkv)
        
#         self.to_out = nn.Linear(inner_dim, dim)
#         self._init_weights(self.to_out)

#         self.to_out = nn.Sequential(
#             self.to_out,
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
        
#         # self.g_block = G_Block(dim, inner_dim, heads, dropout)
#         self.g_block = G_Block(dim_head, dropout)
        
#     def _init_weights(self,layer):
#         nn.init.xavier_normal_(layer.weight)
#         if layer.bias is not None:
#             nn.init.zeros_(layer.bias)  

#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
#         # global_attribute = self.g_block(x)

#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

#         scores = self.attend(dots)

#         out = einsum('b h i j, b h j d -> b h i d', scores, v)
#         global_attribute = self.g_block(out)
#         out = out + global_attribute
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
    
# class G_Attention(nn.Module):
#     def __init__(self, dim, num_patches=64, heads = 8, dim_head = 64, dropout = 0., ver=1):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim = -1)
#         self.to_qkv = nn.Linear(dim, inner_dim, bias = False)
#         self._init_weights(self.to_qkv)
        
#         self.to_out = nn.Linear(inner_dim, dim)
#         self._init_weights(self.to_out)

#         self.to_out = nn.Sequential(
#             self.to_out,
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
        
#         # self.g_block = G_Block(dim, inner_dim, heads, dropout)
#         self.g_block = G_Block(dim, dim_head, dropout)
#         self.ver = ver
#         self.mask = torch.eye(num_patches, num_patches)
#         self.mask = self.mask.unsqueeze(dim=0).unsqueeze(dim=0)
#         self.gelu = nn.GELU()
        
#     def _init_weights(self,layer):
#         nn.init.xavier_normal_(layer.weight)
#         if layer.bias is not None:
#             nn.init.zeros_(layer.bias)  

#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         mask = self.mask.expand(b, h, *self.mask[0, 0].size())
        
#         qkv = self.to_qkv(x)
#         qkv = rearrange(qkv, 'b n (h d) -> b h n d', h = h)

#         dots = einsum('b h i d, b h j d -> b h i j', qkv, qkv) * self.scale
#         dots = self.gelu(dots)
#         # dots[:, :].fill_diagonal_(float('-inf'))
#         dots[mask == 1] = float('-inf')        

#         scores = self.attend(dots)

#         out = einsum('b h i j, b h j d -> b h i d', scores, qkv)
#         global_attribute = self.g_block(x)
#         out = out + global_attribute
        
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
        
#         return out


class G_Attention(nn.Module):
    def __init__(self, dim, num_patches=64, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        # self.scale = dim_head ** -0.5
        self.scale = nn.Parameter(torch.rand(heads))
        # self.scale = nn.Parameter(torch.rand(1))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self._init_weights(self.to_qkv)
        
        self.to_out = nn.Linear(inner_dim, dim)
        self._init_weights(self.to_out)

        self.to_out = nn.Sequential(
            self.to_out,
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        # self.g_block = G_Block(dim, inner_dim, heads, dropout)
        self.g_block = G_Block(dim, dim_head, dropout)
        self.mask = torch.eye(num_patches+1, num_patches+1)
        self.mask = (self.mask == 1).nonzero()
        self.inf = float('-inf')
        self.value = 0
        
    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  

    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # global_attribute = self.g_block(x)

        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        scale = self.scale
        dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((x.size(0), self.heads, 1, 1)))
    
        dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf
        
        
        scores = self.attend(dots)
        

        out = einsum('b h i j, b h j d -> b h i d', scores, v)
        # global_attribute = self.g_block(x)
        # out = out + global_attribute
        self.value = compute_relative_norm_residuals(v, out)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out
    
class G_Block(nn.Module):
    def __init__(self, dim, dim_head, dropout):
        super().__init__()
        
        self.to_phi = nn.Linear(dim, dim)
        self._init_weights(self.to_phi)
        self.dim_head = dim_head

        self.to_phi = nn.Sequential(
            self.to_phi,
            nn.Dropout(dropout)
        )
        
        # self.rho = nn.GELU()
        self.rho = nn.Identity()

    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  
    
    def forward(self, x):
        phi = self.to_phi(x)
        phi = rearrange(phi, 'b n (h d) -> b h n d', d=self.dim_head)
        pool = phi.mean(dim=-2, keepdim=True)
        rho = self.rho(pool)
        
        return rho
    
# class G_Block(nn.Module):
#     def __init__(self, dim, inner_dim, heads, dropout):
#         super().__init__()
        
#         self.to_phi = nn.Linear(dim, inner_dim)
#         self._init_weights(self.to_phi)

#         self.to_phi = nn.Sequential(
#             self.to_phi,
#             nn.Dropout(dropout)
#         )
        
#         self.rho = nn.GELU()
#         self.heads = heads

#     def _init_weights(self,layer):
#         nn.init.xavier_normal_(layer.weight)
#         if layer.bias is not None:
#             nn.init.zeros_(layer.bias)  
    
#     def forward(self, x):
#         phi = self.to_phi(x)
#         phi = rearrange(phi, 'b n (h d) -> b h n d', h = self.heads)
#         pool, _ = phi.max(dim=-2, keepdim=True)
#         rho = self.rho(pool)
        
#         return rho

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.hidden_states = {}
        self.scale = {}

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, G_Attention(dim, num_patches = num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):    
            input_ = x
            x = self.drop_path(attn(x))           
            self.hidden_states[str(i)] = attn.fn.value
            x = x + input_
            x = self.drop_path(ff(x)) + x         
            self.scale[str(i)] = attn.fn.scale
        return x

class GiT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., stochastic_depth=0.):
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

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim, dropout, stochastic_depth)
        
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
        # patch embedding
        # x = self.to_patch_embedding(img)
        # b, n, _ = x.shape

        # x += self.pos_embedding[:, :n]
        # x = self.dropout(x)

        # x = self.transformer(x)
        # x = self.read_out(x)
        
        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x =  x[:, 0]

        # x = self.to_latent(x)
        
        
        return self.mlp_head(x)
    
    def read_out(self, x):
        maxpool, _ = x.max(dim=-1, keepdim = True)
        cat = torch.cat([x, maxpool], dim=-1)
        out = cat.mean(dim=-2)
        
        return out