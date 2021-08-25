import math
import torch
from torch import nn, einsum

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
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_pe=False, is_last=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5        
        # if not is_pe:
        #     self.scale = nn.Parameter(self.scale*torch.ones(heads))
 

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)        

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        num_patches = num_patches -1 if is_pe else num_patches
        self.mask = torch.eye(num_patches+1, num_patches+1) 
        self.mask = torch.nonzero((self.mask == 1), as_tuple=False) 
        self.inf = float('-inf')
        self.is_pe = is_pe
        

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
      

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # if self.is_pe:
        #     dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # else:
        #     scale = self.scale
        #     dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
        #     dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # self.value = compute_relative_norm_residuals(v, out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out) + v.squeeze() if self.is_pe else self.to_out(out)
        # out = self.to_out(out)
        return out

from utils.drop_path import DropPath
class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., stochastic_depth=0., is_pe=False):
        super().__init__()
        self.layers = nn.ModuleList([])

        for i in range(depth):
            transformer = nn.ModuleList([])
            transformer.append(PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_pe=is_pe)))            
            dim = dim_head * heads
            transformer.append(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))   
            self.layers.append(transformer)       
        
        self.is_pe = is_pe
        
        self.layers = nn.ModuleList(self.layers)
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):       
            x = attn(x) if self.is_pe else self.drop_path(attn(x)) + x
            # x = self.drop_path(attn(x)) + x
            x = ff(x) + x if self.is_pe else self.drop_path(ff(x)) + x
        return x


def exists(val):
    return val is not None


def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

# main class

class T2TViT(nn.Module):
    def __init__(self, *, image_size, num_classes, dim=256, depth = 12, heads = 4, mlp_dim_ratio = 2, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., transformer = None, t2t_layers = ((3, 2), (3, 2)), stochastic_depth=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        layers = []
        layer_dim = channels
        output_image_size = image_size


        for i, (kernel_size, stride, padding) in enumerate(t2t_layers):
            if i == 0:
                layer_dim *= kernel_size ** 2
                in_dim = channels
            else:
                layer_dim = 64 * (kernel_size ** 2)
                in_dim = 64
            
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, padding)
            num_patches = output_image_size ** 2
            
            # layers.extend([
            #     RearrangeImage() if not is_first else nn.Identity(),
            #     nn.Unfold(kernel_size = kernel_size, stride = stride, padding = padding),
            #     Rearrange('b c n -> b n c'),
            #     Transformer(dim = layer_dim, num_patches=num_patches, heads = 1, depth = 1, dim_head = 64, mlp_dim = 64, dropout = dropout, is_pe=True) if not is_last else nn.Identity(),
            # ])
            
            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                PatchMerging(in_dim, 64, stride, True),
                Transformer(dim = 64, num_patches=num_patches, heads = 1, depth = 1, dim_head = 64, mlp_dim = 64, dropout = dropout, is_pe=True) if not is_last else nn.Identity(),
            ])
            
        num_patches = output_image_size ** 2

        # layers.append(nn.Linear(layer_dim, dim))
        layers.append(nn.Linear(64, dim))
        self.to_patch_embedding = nn.Sequential(*layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, output_image_size ** 2 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim_ratio)]), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, dim * mlp_dim_ratio, dropout, stochastic_depth=stochastic_depth)
        else:
            self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.apply(init_weights)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            

class PatchMerging(nn.Module):
    def __init__(self, in_dim, dim, merging_size=2, is_pe=False):
        super().__init__()
        
        self.is_pe = is_pe
        patch_dim = in_dim * (merging_size**2)
  
    
        self.patch_shifting = PatchShifting(merging_size, in_dim, in_dim*2, True)
        patch_dim = (in_dim*2) * (merging_size**2) 
    
        self.merging = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        
        out = self.patch_shifting(x)
        out = self.merging(out)
        return out
    
class PatchShifting(nn.Module):
    def __init__(self, patch_size, in_dim, out_dim, is_mean=False):
        super().__init__()
        self.shift = int(patch_size * (1/2))
        self.is_mean = is_mean
        self.out = nn.Conv2d(in_dim*5, out_dim, 1)
        
    def forward(self, x):
     
        # x = x.mean(dim=1, keepdim = True)

        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        # if self.is_mean:
        #     x_pad = x_pad.mean(dim=1, keepdim = True)
        
        x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
               
        x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)
        
        out = self.out(x_cat)
        
        return out
    