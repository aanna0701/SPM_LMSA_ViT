from random import randrange
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def exists(val):
    return val is not None

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes

class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6
            
        self.dim = dim

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale
    
    def flops(self):
        flops = 0
        
        flops += self.fn.flops()
        
        return flops

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
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., if_patch_attn=False, is_base=True):
        super().__init__()
        self.inner_dim = dim_head *  heads
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_patches = num_patches
        self.is_base = is_base
        if not is_base:
            self.scale = nn.Parameter(self.scale*torch.ones(heads))
            self.mask = torch.eye(num_patches, num_patches)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)

        self.to_q = nn.Linear(dim, self.inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, self.inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.if_patch_attn = if_patch_attn

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        context = x if not exists(context) else torch.cat((x, context), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if self.is_base:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        else:
            """ LMSA """
            ############################
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((x.size(0), self.heads, 1, 1)))

            
            if self.if_patch_attn:
                dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -1e-9
            else:
                dots[:, :,:, 0] = -1e-9
            ###########################
                
        
        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax
        attn = self.attend(dots)        
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    def flops(self):
        flops = 0
        
        if self.if_patch_attn:
            flops += self.dim * self.inner_dim * 3 * self.num_patches
            flops += self.inner_dim * (self.num_patches**2)
            flops += self.inner_dim * (self.num_patches**2)
            flops += self.inner_dim * self.dim * self.num_patches
        
        else:
            flops += self.dim * self.inner_dim 
            flops += self.dim * self.inner_dim * 2 * (self.num_patches+1)
            flops += self.inner_dim * self.num_patches
            flops += self.inner_dim * self.num_patches
            flops += self.inner_dim * self.dim      
        
        return flops


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0., stochastic_depth=0., if_patch_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                LayerScale(dim, PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, if_patch_attn=if_patch_attn)), depth = ind + 1),
                LayerScale(dim, PreNorm(num_patches, dim, FeedForward(num_patches, dim, mlp_dim, dropout = dropout)), depth = ind + 1)
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x, context = None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            
            x = self.drop_path(attn(x, context = context)) + x
            x = self.drop_path(ff(x)) + x
        return x
    
    
    def flops(self):
        flops = 0
        
        for (attn, ff) in self.layers:       
            flops += attn.flops()
            flops += ff.flops()
        
        return flops
    
    
from utils.drop_path import DropPath

class CaiT(nn.Module):
    def __init__(
        self,
        *,
        img_size,
        patch_size,
        num_classes,
        dim=192,
        depth=24,
        cls_depth=2,
        heads=4,
        mlp_dim=384,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        layer_dropout = 0.,
        stochastic_depth = 0.,
        is_base=True
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.dim = dim
        self.num_classes = num_classes
        self.is_base = is_base
        
        if is_base:
            """ Base """
            #########################
            patch_dim = 3 * patch_size ** 2
            
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
                nn.Linear(patch_dim, dim),
            )
            self.pe_flops = patch_dim * dim * num_patches
            #########################
        else:  
            """ SPM """
            #########################
            self.to_patch_embedding = ShiftedPatchMerging(img_size, 3, dim, patch_size, exist_class_t=False, is_pe=True)
            #########################
            
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.patch_transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim, dropout, layer_dropout, stochastic_depth=stochastic_depth, if_patch_attn=True)
        self.cls_transformer = Transformer(dim, num_patches, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout, stochastic_depth=stochastic_depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.apply(init_weights)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.patch_transformer(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = self.cls_transformer(cls_tokens, context = x)

        return self.mlp_head(x[:, 0])
    
    def flops(self):
        flops = 0
        
        flops_pe = self.pe_flops if self.is_base else self.to_patch_embedding.flops()
        flops += flops_pe
        
        flops += self.patch_transformer.flops()   
        flops += self.cls_transformer.flops()   
        
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
    

def pair(t):
    return t if isinstance(t, tuple) else (t, t)
    