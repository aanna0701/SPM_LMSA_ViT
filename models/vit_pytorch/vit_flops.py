import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
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
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        
        return self.fn(self.norm(x), **kwargs)
    
    def flops(self):
        flops = 0
        
        flops += self.fn.flops()
        flops += self.dim * self.num_tokens
        
        return flops
    


class FeedForward(nn.Module):
    def __init__(self, num_tokens, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
  
        
    def forward(self, x):
        return self.net(x)    
    
    def flops(self):
        flops = 0
        
        flops += self.dim * self.hidden_dim * self.num_tokens
        flops += self.dim * self.hidden_dim * self.num_tokens
        
        return flops

class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_base=True):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.num_patches = num_patches
        
        self.is_base = is_base
        if not self.is_base:
            self.scale = nn.Parameter(self.scale*torch.ones(heads))
            self.mask = torch.eye(num_patches, num_patches)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
            self.inf = float('-inf')

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        init_weights(self.to_qkv)        
 

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
      
        if self.is_base:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))           
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)
    
    def flops(self):
        flops = 0
        
        flops += self.dim * self.inner_dim * 3 * self.num_patches
        flops += self.inner_dim * (self.num_patches**2)
        flops += self.inner_dim * (self.num_patches**2)
        flops += self.inner_dim * self.dim * self.num_patches
        
        return flops


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0., is_base=True):
        super().__init__()
        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_base=is_base)),
                PreNorm(num_patches, dim, FeedForward(num_patches, dim, dim * mlp_dim_ratio, dropout = dropout))
            ]))            
            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for (attn, ff) in self.layers:       
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
            
        return x
    
    def flops(self):
        flops = 0
        
        for (attn, ff) in self.layers:       
            flops += attn.flops()
            flops += ff.flops()
        
        return flops

    

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, pool = 'cls', channels = 3, 
                 dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0., is_base=True):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        
        self.is_base = is_base
        
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.num_classes = num_classes

        if self.is_base:
        
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(self.patch_dim, self.dim)
            )
            
        else:
            self.to_patch_embedding = ShiftedPatchMerging(img_size, 3, self.dim, patch_size, is_pe=True)
            

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.num_patches += 1
        self.transformer = Transformer(dim, self.num_patches, depth, heads, dim_head, mlp_dim_ratio, 
                                       dropout, stochastic_depth, is_base=is_base)

        self.pool = pool
        self.to_latent = nn.Identity()


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )



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
        
        return self.mlp_head(x)

    def flops(self):
        flops = 0
        
        flops_pe = self.num_patches * self.patch_dim * self.dim if self.is_base else self.to_patch_embedding.flops()
        flops += flops_pe
        
        flops += self.transformer.flops()   
        
        flops += self.dim               # layer norm
        flops += self.dim * self.num_classes    # linear
        
        return flops



class ShiftedPatchMerging(nn.Module):
    def __init__(self, img_size, in_dim, dim, merging_size=2, exist_class_t=False, is_pe=True):
        super().__init__()
        
        self.exist_class_t = exist_class_t
        self.is_pe = is_pe
        self.token_length = img_size // merging_size
        self.patch_shifting = PatchShifting(merging_size)
        self.in_dim = in_dim
        self.dim = dim
        self.patch_dim = (in_dim*5) * (merging_size**2)
        self.class_linear = nn.Linear(self.in_dim, self.dim)
    
        self.merging = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.dim, bias=False)
        )

    def forward(self, x):
        
        if self.exist_class_t:
            visual_tokens, class_token = x[:, 1:], x[:, (0,)]
            reshaped = rearrange(visual_tokens, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
            out_visual = self.patch_shifting(reshaped)
            out_visual = self.merging(out_visual)
            out_class = self.class_linear(class_token)
            out = torch.cat([out_class, out_visual], dim=1)
        
        else:
            x = x if self.is_pe else rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
            out = self.patch_shifting(x)
            out = self.merging(out)    
        
        return out
    
    def flops(self):
        flops = 0
        L = self.token_length**2
        
        if self.exist_class_t:
            flops += self.in_dim * self.dim   # class-token linear      
        
        flops += L * self.patch_dim               # layer norm
        flops += L * self.patch_dim * self.dim    # linear
        
        return flops

    
class PatchShifting(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.shift = int(patch_size * (1/2))
        
    def forward(self, x):
     
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        #############################
        
        """ 4 diagonal directions """
        # #############################
        x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
        x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
        x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
        x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
        x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1) 
        # #############################
        
        out = x_cat
        
        return out