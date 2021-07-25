import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
# from utils.positional_encoding import positionalencoding2d

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

class G_Attention(nn.Module):
    def __init__(self, dim, num_patches=64, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        # self.scale = nn.Parameter(torch.rand(heads))
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
        # self.g_block = G_Block(dim, num_patches)
        self.mask = torch.eye(num_patches+1, num_patches+1)
        self.mask = (self.mask == 1).nonzero()
        self.inf = float('-inf')
        self.value = 0
        self.entropy = HLoss()
        self.avg_h = None
        self.num_nodes = num_patches
        
    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # channel_agg = self.g_block(v)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # scale = self.scale
        # dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((x.size(0), self.heads, 1, 1)))
    
        # dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf
        
        
        scores = self.attend(dots)
        

        out = einsum('b h i j, b h j d -> b h i d', scores, v)
        
        # self.avg_h = self.entropy(scores) / self.num_nodes
        
        # out = torch.cat([out, channel_agg], dim=-1)

        # global_attribute = self.g_block(x)
        # out = out + global_attribute
        # self.value = compute_relative_norm_residuals(v, out[:, :, :, :-1])
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out = self.g_block(out)
        out = self.to_out(out)
        
        return out
    
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
        
    def forward(self, x):
        log = torch.log(x + 1e-12)
        # log[:, :, self.mask[:, 0], self.mask[:, 1]] = 0.
        info = torch.mul(log, x)
        h = -1.0*torch.sum(info, dim=-1)
        h = h.mean(dim=-1)
        
        return h
    
    
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
        self.h = []
        

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, G_Attention(dim, num_patches = num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            if i == 0:
                self.h = []   
            x = self.drop_path(attn(x)) + x
            # self.h.append(attn.fn.avg_h)           
            # self.hidden_states[str(i)] = attn.fn.value
      
            x = self.drop_path(ff(x)) + x         
            # self.scale[str(i)] = attn.fn.scale
        return x

class GiT(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., stochastic_depth=0.):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        # patch_dim = (channels)*5 * patch_height * patch_width
        patch_dim = (channels+4) * patch_height * patch_width


        self.linear_to_path = nn.Linear(patch_dim, dim)
        # self.linear_to_path = nn.Conv2d(3*5, dim, 4, 4)
        self._init_weights(self.linear_to_path)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # self.to_patch_embedding = nn.Sequential(
        #     PatchShifting(),
        #     self.linear_to_path,
        #     Rearrange('b c h w -> b (h w) c')
        # )
        
        self.to_patch_embedding = nn.Sequential(
            PatchShifting(),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            self.linear_to_path
        )
    
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     self.linear_to_path,
        # )
    
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim, dropout, stochastic_depth)
        
        nn.linear_mlp_head = nn.Linear(dim, num_classes)
        self._init_weights(nn.linear_mlp_head)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.linear_mlp_head
        )
        
        self.h_loss = 0.
        
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
        x += self.pos_embedding[:, :n+1]
        x = self.dropout(x)

        x = self.transformer(x)
        
        # entropy = torch.cat(self.transformer.h, dim=1)
        
        # self.h_loss = entropy.mean(dim=-1)
        # print(self.h_loss)

        x =  x[:, 0]

        # x = self.to_latent(x)
        
        
        return self.mlp_head(x)
    
# class PatchShifting(nn.Module):
#     def __init__(self):
#         super().__init__()


#     def forward(self, x):
#         # x_l = torch.cat([torch.nn.pad(x, ), x[:, :, :, 1:]], dim=-1)
#         # x_r = torch.cat([x[:, :, :, :-1], self.w_pad], dim=-1)
#         # x_t = torch.cat([self.h_pad, x[:, :, 1:]], dim=-2)
#         # x_b = torch.cat([x[:, :, :-1], self.h_pad], dim=-2)
        
#         # print(x_l.shape)
#         # print(x_r.shape)
#         # print(x_t.shape)
#         # print(x_b.shape)
        
#         x_pad = torch.nn.functional.pad(x, (2, 2, 2, 2))
        
#         x_pad = x_pad.mean(dim=1, keepdim = True)
        
#         x_l = x_pad[:, :, 2:-2, 1:-3]
#         x_r = x_pad[:, :, 2:-2, 3:-1]
#         x_t = x_pad[:, :, 1:-3, 2:-2]
#         x_b = x_pad[:, :, 3:-1, 2:-2]
#         x_l2 = x_pad[:, :, 2:-2, :-4]
#         x_r2 = x_pad[:, :, 2:-2, 4:]
#         x_t2 = x_pad[:, :, :-4, 2:-2]
#         x_b2 = x_pad[:, :, 4:, 2:-2]
        
               
#         x_cat = torch.cat([x, x_l, x_r, x_t, x_b, x_l2, x_r2, x_t2, x_b2], dim=1)
        
        
#         return x_cat
    
    
class PatchShifting(nn.Module):
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
        
        x_pad = torch.nn.functional.pad(x, (2, 2, 2, 2))
        
        x_pad = x_pad.mean(dim=1, keepdim = True)
        # x_pad = x_pad
        
        x_l = x_pad[:, :, 2:-2, :-4]
        x_r = x_pad[:, :, 2:-2, 4:]
        x_t = x_pad[:, :, :-4, 2:-2]
        x_b = x_pad[:, :, 4:, 2:-2]
        
               
        x_cat = torch.cat([x, x_l, x_r, x_t, x_b], dim=1)
        
        
        return x_cat
    