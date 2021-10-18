import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.relative_norm_residuals import compute_relative_norm_residuals

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
    def __init__(self, dim, hidden_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
  
        
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        init_weights(self.to_qkv)
        
 

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        
        
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

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
    
        
        self.value = compute_relative_norm_residuals(v, out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)
    

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.hidden_states = {}
        self.scale = {}

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head)),
                PreNorm(dim, FeedForward(dim, dim * mlp_dim_ratio))
            ]))            
            
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):       
            x = attn(x) + x
            x = ff(x) + x
            
        return x

class Localisation(nn.Module):
    def __init__(self, *, img_size, patch_size=4, dim=96, depths=[1, 4, 2], heads=[3, 6, 9], mlp_dim_ratio=2, num_transformation=4 , dim_head = 16):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = 3 * patch_height * patch_width


        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        
        self.trans_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.y_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.rotation_token = nn.Parameter(torch.randn(1, 1, dim))
        
        img_size //= patch_size
        
        self.layers = nn.ModuleList([])

        for ind, (depth, heads) in enumerate(zip(depths, heads)):
            not_last = ind < (len(depths) - 1)
            
            self.layers.append(Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim_ratio))

            if not_last:
                
                self.layers.append(Pool(img_size, dim))
                img_size //= 2
                dim *= 2

        self.layers = nn.Sequential(*self.layers)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_transformation),
            nn.Tanh()
        )
        
        self.final_cls_token = None
        
        self.apply(init_weights)


    def forward(self, img):
        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        trans_token = repeat(self.trans_token, '() n d -> b n d', b = b)
        
        x = torch.cat((trans_token, x), dim=1)
        x += self.pos_embedding
        x = self.layers(x)

        x = x[:, (0,)]
        
        return self.mlp_head(x)
    

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, img_size, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.img_size = img_size
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H = self.img_size
        W = H
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"img_size={self.img_size}, dim={self.dim}"

    def flops(self):
        H, W = self.img_size
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

import math
    
class Pool(nn.Module):
    def __init__(self, img_size, dim):
        super().__init__()
        self.downsample = PatchMerging(img_size, dim)
        self.cls_ff = nn.Linear(dim, dim * 2)
        


    def forward(self, x):
        trans_tokens, visual_tokens = x[:, (0,)], x[:, 1:]

        trans_tokens = self.cls_ff(trans_tokens)

        visual_tokens = self.downsample(visual_tokens)

        return torch.cat((trans_tokens, visual_tokens), dim = 1)