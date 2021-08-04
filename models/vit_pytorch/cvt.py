import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper methods

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# classes

class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, num_patches, heads = 8, dim_head = 64, dropout = 0.,):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5
        # self.scale = nn.Parameter(torch.rand(heads))

        self.attend = nn.Softmax(dim = -1)

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding = padding, stride = kv_proj_stride, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
        
        # self.mask = torch.eye(num_patches+1, num_patches+1)
        # self.mask = (self.mask == 1).nonzero()
        # self.inf = float('-inf')

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = h), (q, k, v))


        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        
        # scale = self.scale
        # dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
    
  
        # dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

from utils.drop_path import DropPath
class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, num_patches, dim_head = 64, mlp_mult = 4, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout = dropout, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for attn, ff in self.layers:
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
        return x

class CvT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        img_size=32,
        s1_emb_dim = 64,
        s1_emb_kernel = 7,
        s1_emb_stride = 4,
        s1_proj_kernel = 7,
        s1_kv_proj_stride = 4,
        s1_heads = 1,
        s1_depth = 1,
        s1_mlp_mult = 4,
        s2_emb_dim = 192,
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        s2_heads = 3,
        s2_depth = 2,
        s2_mlp_mult = 4,
        s3_emb_dim = 384,
        s3_emb_kernel = 3,
        s3_emb_stride = 2,
        s3_proj_kernel = 3,
        s3_kv_proj_stride = 2,
        s3_heads = 6,
        s3_depth = 10,
        s3_mlp_mult = 4,
        dropout = 0.,
        patch_size = 3,
        stochastic_depth=0.
    ):
        super().__init__()

        dim = 3
        layers = []

        
        num_patches = conv_output_size(img_size, patch_size, round(patch_size/2), 0)
        layers.append(nn.Sequential(
                nn.Conv2d(dim, s1_emb_dim, kernel_size = s1_emb_kernel, padding = 1, stride = s1_emb_stride),
                LayerNorm(s1_emb_dim),
                Transformer(dim = s1_emb_dim, proj_kernel = s1_proj_kernel, kv_proj_stride = s1_kv_proj_stride, depth = s1_depth, heads = s1_heads, mlp_mult = s1_mlp_mult, dropout = dropout, num_patches=num_patches, stochastic_depth=stochastic_depth)
            ))
        
        dim = s1_emb_dim
        num_patches = conv_output_size(num_patches, s2_emb_kernel, s2_emb_stride, (s2_emb_kernel // 2))
        layers.append(nn.Sequential(
                nn.Conv2d(dim, s2_emb_dim, kernel_size = s2_emb_kernel, padding = (s2_emb_kernel // 2), stride = s2_emb_stride),
                LayerNorm(s2_emb_dim),
                Transformer(dim = s2_emb_dim, proj_kernel = s2_proj_kernel, kv_proj_stride = s2_kv_proj_stride, depth = s2_depth, heads = s2_heads, mlp_mult = s2_mlp_mult, dropout = dropout, num_patches=num_patches, stochastic_depth=stochastic_depth)
            ))
        dim = s2_emb_dim
        
        num_patches = conv_output_size(num_patches, s3_emb_kernel, s3_emb_stride, (s3_emb_kernel // 2))
        layers.append(nn.Sequential(
                nn.Conv2d(dim, s3_emb_dim, kernel_size = s3_emb_kernel, padding = (s3_emb_kernel // 2), stride = s3_emb_stride),
                LayerNorm(s3_emb_dim),
                Transformer(dim = s3_emb_dim, proj_kernel = s3_proj_kernel, kv_proj_stride = s3_kv_proj_stride, depth = s3_depth, heads = s3_heads, mlp_mult = s3_mlp_mult, dropout = dropout, num_patches=num_patches, stochastic_depth=stochastic_depth)
            ))
        dim = s3_emb_dim

        self.layers = nn.Sequential(
            
            *layers,
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


def conv_output_size(image_size, kernel_size, stride, padding = 0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)
