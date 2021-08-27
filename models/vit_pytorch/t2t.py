import math
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
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_pe=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5        
        # self.scale = nn.Parameter(torch.rand(heads))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        init_weights(self.to_qkv)
        
        print(num_patches)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        
        self.mask = torch.eye(num_patches+1, num_patches+1) if not is_pe else torch.eye(num_patches, num_patches)
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
        
        # scale = self.scale
        # dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
    
        
        # dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
    
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio=2, dropout = 0., stochastic_depth=0., is_pe=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.hidden_states = {}
        self.scale = {}
        self.is_pe = is_pe

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_pe=is_pe)),
                PreNorm(dim, FeedForward(dim, dim * mlp_dim_ratio, dropout = dropout))
            ]))            
            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
        
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):       
            x = self.drop_path(attn(x)) + x if not self.is_pe else attn(x) + x
            self.hidden_states[str(i)] = attn.fn.value
            x = self.drop_path(ff(x)) + x if not self.is_pe else ff(x) + x
            
            self.scale[str(i)] = attn.fn.scale
        return x
# helpers

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
    def __init__(self, *, image_size, num_classes, dim=256, depth = 8, heads = 4, mlp_dim_ratio = 2, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., t2t_layers = ((7, 4), (3, 2), (3, 2)), stochastic_depth=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        layers = []
        layer_dim = channels
        output_image_size = image_size

        ''' Base '''
        
        for i, (kernel_size, stride) in enumerate(t2t_layers):
            layer_dim *= kernel_size ** 2
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)
            num_patches= output_image_size ** 2

            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                nn.Unfold(kernel_size = kernel_size, stride = stride, padding = stride // 2),
                Rearrange('b c n -> b n c'),
                Transformer(dim = layer_dim, num_patches=num_patches ,heads = 1, depth = 1, dim_head = layer_dim, mlp_dim_ratio = 1, dropout = dropout, is_pe=True) if not is_last else nn.Identity(),
            ])
        
        ''' SPM '''
        
        # for i, (kernel_size, stride) in enumerate(t2t_layers):
        #     in_dim = 3 if i == 0 else dim
        #     layer_dim = 3 * (kernel_size ** 2) if i == 0 else 64 * 2
        #     is_first = i == 0
        #     is_last = i == (len(t2t_layers) - 1)
        #     output_image_size = output_image_size // stride
        #     num_patches= output_image_size ** 2
        #     layers.extend([
        #         RearrangeImage() if not is_first else nn.Identity(),
        #         ShiftedPatchMerging(in_dim, dim, stride),
        #         Transformer(dim = dim, num_patches=num_patches ,heads = 1, depth = 1, dim_head = 64, mlp_dim_ratio = 1, dropout = dropout, is_pe=True) if not is_last else nn.Identity(),
        #     ])

        layers.append(nn.Linear(layer_dim, dim))
        self.to_patch_embedding = nn.Sequential(*layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, output_image_size ** 2 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        
        self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, stochastic_depth=stochastic_depth)
        
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n+1]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    
class ShiftedPatchMerging(nn.Module):
    def __init__(self, in_dim, dim, merging_size=2, exist_class_t=False):
        super().__init__()
        
        self.exist_class_t = exist_class_t
        
        self.patch_shifting = PatchShifting(merging_size)
        
        patch_dim = (in_dim*5) * (merging_size**2) 
        self.class_linear = nn.Linear(in_dim, dim)
    
        self.merging = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
            nn.Linear(patch_dim, dim)
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
            out = self.patch_shifting(x)
            out = self.merging(out)
    
        
        return out

    
class PatchShifting(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.shift = int(patch_size * (1/2))
        
    def forward(self, x):
     
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        # if self.is_mean:
        #     x_pad = x_pad.mean(dim=1, keepdim = True)
        
        x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
               
        x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)
        
        # out = self.out(x_cat)
        out = x_cat
        
        return out
    