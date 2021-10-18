from math import sqrt
from utils.drop_path import DropPath
import torch
from torch import nn, einsum
import torch.nn.functional as F
from .SpatialTransformation import Translation
from .localisation import Localisation
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
    def __init__(self, dim, patch_size, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.scale = nn.Parameter(self.scale*torch.ones(heads))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
 
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.mask = torch.eye(patch_size+1, patch_size+1)
        self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        self.inf = float('-inf')
        

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        scale = self.scale
        dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
    
        dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf
        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, patch_size, depth, heads, dim_head, mlp_dim, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        self.hidden_states = {}
        self.scale = {}
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, patch_size=patch_size, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
                     
            self.scale[str(i)] = attn.fn.scale
        return x

# depthwise convolution, for pooling

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
  
        
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_out, dim_out, kernel_size = 1, bias = bias)
        ) 
        
        
        
    def forward(self, x):
        return self.net(x)

# pooling layer

class Pool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.downsample = DepthWiseConv2d(dim, dim * 2, kernel_size = 3, stride = 2, padding = 1)
        self.cls_ff = nn.Linear(dim, dim * 2)
        


    def forward(self, x):
        cls_token, tokens = x[:, :1], x[:, 1:]

        cls_token = self.cls_ff(cls_token)

        tokens = rearrange(tokens, 'b (h w) c -> b c h w', h = int(sqrt(tokens.shape[1])))
        tokens = self.downsample(tokens)
        tokens = rearrange(tokens, 'b c h w -> b (h w) c')

        return torch.cat((cls_token, tokens), dim = 1)

# main class

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PiT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, dim_head = 64, dropout = 0., emb_dropout = 0., stochastic_depth=0., is_base=True):
        super(PiT, self).__init__()
        heads = cast_tuple(heads, len(depth))

        if is_base:
            " Base "
            output_size = conv_output_size(img_size, patch_size*2, patch_size)
            
            
            self.to_patch_embedding = nn.Sequential(
                nn.Conv2d(3, dim, patch_size*2, patch_size),
                Rearrange('b c h w -> b (h w) c')
            )
        
        
        else:
            self.to_patch_embedding = ShiftedPatchMerging(3, dim, patch_size, is_pe=True)

        
        output_size = img_size // patch_size
        
        num_patches = output_size ** 2
        
        
        

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList([])
        self.pool_idx = []

        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)
            
            self.layers.append(Transformer(dim, output_size, layer_depth, layer_heads, dim_head, dim*mlp_dim_ratio, dropout, stochastic_depth))

            if not_last:
                if is_base:
                    " Base "
                    self.layers.append(Pool(dim))
                
                else:
                    "SPE"
                    self.layers.append(ShiftedPatchMerging(dim, dim*2 ,2, exist_class_t=True, is_pe=False))
                    self.pool_idx.append(ind+1) if ind == 0 else self.pool_idx.append(ind+2)
                
                dim *= 2
                output_size = conv_output_size(output_size, 3, 2, 1)
      
        # self.layers = nn.Sequential(*layers)


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.is_SPT = False
        if not is_base:
            from .swin import SwinTransformer
            # self.localisation = SwinTransformer(img_size=img_size, window_size=4, drop_path_rate=0, patch_size=patch_size*2, mlp_ratio=2, depths=[2, 6, 4], num_heads=[2, 4, 8], num_classes=8, embed_dim=32)
            self.localisation = Localisation(img_size=img_size)
            self.is_SPT = True
        
        self.theta = None
        
        self.apply(init_weights)

    def forward(self, img):
        self.theta = self.localisation(img) if self.is_SPT else None
        x = self.to_patch_embedding(img) if not self.is_SPT else self.to_patch_embedding(img, self.theta)
        b, n, _ = x.shape
        


        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        # x = self.layers(x, theta)
        for i, layer in enumerate(self.layers):
            
            x = layer(x, self.theta) if i in self.pool_idx else layer(x)

        return self.mlp_head(x[:, 0])
    
class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

import math
    
class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))
    
class ShiftedPatchMerging(nn.Module):
    def __init__(self, in_dim, dim, merging_size=2, exist_class_t=False, is_pe=False):
        super().__init__()
        
        self.exist_class_t = exist_class_t
        

        self.transformation = SpatialTransformation()
        
        patch_dim = (in_dim*5) * (merging_size**2) 
        self.class_linear = nn.Linear(in_dim, dim)

        
        self.is_pe = is_pe
        
        if is_pe:
            self.merging = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim)
            )
            
        else:
            self.merging = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
                nn.LayerNorm(in_dim * (merging_size**2) ),
                nn.Linear(in_dim * (merging_size**2) , dim)
            )
        
        self.channel_mix = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, in_dim, 1, bias=False)            
        )
        # self.localisation = Localisation(in_dim, img_size, 4)

    def forward(self, x, theta):
        theta = theta
        num_patches = x.size(1) if len(x.size()) == 3 else x.size(-1)**2
        
        if self.exist_class_t:
            visual_tokens, class_token = x[:, 1:], x[:, (0, )]
            
            num_patches = visual_tokens.size(1)
            reshaped = rearrange(visual_tokens, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
            # theta = self.localisation(reshaped)
            out_visual = self.transformation(reshaped, num_patches, theta, 4)
            out_visual = self.channel_mix(out_visual) + reshaped
            
            out_visual = self.merging(out_visual)
            out_class = self.class_linear(class_token)
            out = torch.cat([out_class, out_visual], dim=1)
        
        else:
            num_patches = x.size(1)
            out = x if self.is_pe else rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
            
            # theta = self.localisation(out)
            out = self.transformation(out, num_patches, theta, 4, pe=True)
            out = self.merging(out)
    
        
        return out

      
class SpatialTransformation(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.transformation = Translation()
        # self.unit = [
        #     self.make_unit([[1],
        #                    [1]]),
        #     self.make_unit([[1],
        #                    [-1]]),
        #     self.make_unit([[-1],
        #                    [1]]),
        #     self.make_unit([[-1],
        #                    [-1]]),
        #     ]
        
    def make_unit(self, x):
        return torch.tensor(x)
                
    def forward(self, x, num_patches, theta, num_transform, pe=False):   
        
        theta = (2*theta)/num_patches
        
        if pe:
            out = [x]
            theta_list = torch.chunk(theta/num_patches, num_transform, dim=-1)
            
            
            for i in range(num_transform):
                out.append(self.transformation(x, theta_list[i]))
            out = torch.cat(out, dim=1)
                    
        else:
            chunks = torch.chunk(x, num_transform+1, dim=1)
            out = [chunks[-1]]
            theta_list = torch.chunk(theta/num_patches, num_transform, dim=-1)
            
            
            for i in range(num_transform):
                out.append(self.transformation(chunks[i], theta_list[i]))
                
            out = torch.cat(out, dim=1)
        
        return out
    
    
    
    
# class ShiftedPatchMerging(nn.Module):
#     def __init__(self, in_dim, dim, merging_size=2, exist_class_t=False, is_pe=False):
#         super().__init__()
        
#         self.exist_class_t = exist_class_t
        
#         self.patch_shifting = PatchShifting(merging_size) if not is_pe else PatchShifting(merging_size, 0.25)
        
#         patch_dim = (in_dim*5) * (merging_size**2) 
#         self.class_linear = nn.Linear(in_dim, dim)

        
#         self.is_pe = is_pe
        
#         if is_pe:
#             self.merging = nn.Sequential(
#                 Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
#                 nn.LayerNorm(patch_dim),
#                 nn.Linear(patch_dim, dim)
#             )
            
#         else:
#             self.merging = nn.Sequential(
#                 Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
#                 nn.LayerNorm(in_dim * (merging_size**2) ),
#                 nn.Linear(in_dim * (merging_size**2) , dim)
#             )
        
#         self.channel_mix = nn.Sequential(
#             nn.BatchNorm2d(in_dim),
#             nn.Conv2d(in_dim, in_dim, 1, bias=False)            
#         )

#     def forward(self, x):
        
#         # if self.exist_class_t:
#         #     visual_tokens, class_token = x[:, 1:], x[:, (0,)]
#         #     reshaped = rearrange(visual_tokens, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
#         #     out_visual = self.patch_shifting(reshaped)
#         #     out_visual = self.merging(out_visual)
#         #     out_class = self.class_linear(class_token)
#         #     out = torch.cat([out_class, out_visual], dim=1)
        
#         # else:
#         #     out = x if self.is_pe else rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
#         #     out = self.patch_shifting(out)
#         #     out = self.merging(out)
    
        
#         if self.exist_class_t:
#             visual_tokens, class_token = x[:, 1:], x[:, (0,)]
#             reshaped = rearrange(visual_tokens, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
            
#             out_visual = self.patch_shifting(reshaped)
#             out_visual = self.channel_mix(out_visual) + reshaped
            
#             out_visual = self.merging(out_visual)
#             out_class = self.class_linear(class_token)
#             out = torch.cat([out_class, out_visual], dim=1)
        
#         else:
#             out = x if self.is_pe else rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
#             out = self.patch_shifting(out, pe=True)
#             out = self.merging(out)
    
        
#         return out

    
# class PatchShifting(nn.Module):
#     def __init__(self, patch_size, shift_ratio=0.5):
#         super().__init__()
#         # self.shift = int(patch_size * shift_ratio)
#         self.shift = patch_size // 2
        
#     def forward(self, x, pe=False):
     
#         x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        
#         if pe:
#             """ 4 diagonal directions """
#             # #############################
#             x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
#             x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
#             x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
#             x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
#             x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1) 
#             # #############################
#             out = x_cat
        
#         else:
#             """ 4 diagonal directions """
#             # #############################
#             _, C, _, _ = x_pad.size()
    
#             # x[:, :C//4] = x_pad[:, :C//4, :-self.shift*2, :-self.shift*2]
#             # x[:, C//4:C//2] = x_pad[:, C//4:C//2, :-self.shift*2, self.shift*2:]
#             # x[:, C//2:(C//4)*3] = x_pad[:, C//2:(C//4)*3, self.shift*2:, :-self.shift*2]
#             # x[:, (C//4)*3:] = x_pad[:, (C//4)*3:, self.shift*2:, self.shift*2:]
#             x[:, :C//5] = x_pad[:, :C//5, :-self.shift*2, :-self.shift*2]
#             x[:, C//5:(C//5)*2] = x_pad[:, C//5:(C//5)*2, :-self.shift*2, self.shift*2:]
#             x[:, (C//5)*2:(C//5)*3] = x_pad[:, (C//5)*2:(C//5)*3, self.shift*2:, :-self.shift*2]
#             x[:, (C//5)*3:(C//5)*4] = x_pad[:, (C//5)*3:(C//5)*4:, self.shift*2:, self.shift*2:]
#             # #############################
#             out = x
        
#         return out