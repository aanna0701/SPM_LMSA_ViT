from random import randrange
import torch
from torch import nn, einsum
from .SpatialTransformation import STT
from utils.coordconv import CoordLinear, AddCoords1D
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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
    def __init__(self, num_tokens, dim, hidden_dim, dropout = 0., is_coord=False, if_patch_attn=True):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.is_coord = is_coord
        self.if_patch_attn = if_patch_attn
        if not if_patch_attn:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )
        else:
            if is_coord:
                self.net = nn.Sequential(
                    CoordLinear(dim, hidden_dim, exist_cls_token=(not self.if_patch_attn)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    CoordLinear(hidden_dim, dim, exist_cls_token=(not self.if_patch_attn)),
                    nn.Dropout(dropout)
                )
            else:
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
        if not self.if_patch_attn:
            flops += self.dim * self.hidden_dim
            flops += self.dim * self.hidden_dim
        else:
            if self.is_coord:
                flops += (self.dim+2) * self.hidden_dim * self.num_tokens
                flops += self.dim * (self.hidden_dim+2) * self.num_tokens
            else:
                flops += self.dim * self.hidden_dim * self.num_tokens
                flops += self.dim * self.hidden_dim * self.num_tokens
            
        return flops

class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., if_patch_attn=False, is_LSA=False, is_coord=False):
        super().__init__()
        self.inner_dim = dim_head *  heads
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_patches = num_patches
        self.is_LSA = is_LSA
        self.is_coord = is_coord
        if is_LSA:
            self.scale = nn.Parameter(self.scale*torch.ones(heads))
            self.mask = torch.eye(num_patches, num_patches)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)

        
        if not is_coord:    
            self.to_q = nn.Linear(dim, self.inner_dim, bias = False)
            self.to_kv = nn.Linear(dim, self.inner_dim * 2, bias = False)
        else:    
            if if_patch_attn:
                self.to_q = CoordLinear(dim, self.inner_dim, bias = False, exist_cls_token=False)
                self.to_kv = CoordLinear(dim, self.inner_dim * 2, bias = False, exist_cls_token=False)
            else:
                self.to_q = nn.Linear(dim, self.inner_dim, bias = False)
                self.to_kv = CoordLinear(dim, self.inner_dim * 2, bias = False)
            

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

        if not self.is_LSA:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((x.size(0), self.heads, 1, 1)))

            if self.if_patch_attn:
                dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -1e-9
            else:
                dots[:, :,:, 0] = -1e-9
                
        
        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax
        attn = self.attend(dots)        
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    def flops(self):
        flops = 0
        
        if self.if_patch_attn:
            if not self.is_coord:
                flops += self.dim * self.inner_dim * 3 * self.num_patches
            else:    
                flops += (self.dim+2) * self.inner_dim * 3 * self.num_patches
                
            flops += self.inner_dim * (self.num_patches**2)
            flops += self.inner_dim * (self.num_patches**2)
            flops += self.inner_dim * self.dim * self.num_patches
        
        else:
            if not self.is_coord:
                flops += self.dim * self.inner_dim 
                flops += self.dim * self.inner_dim * 2 * (self.num_patches+1)
            else:
                flops += self.dim * self.inner_dim 
                flops += (self.dim+2) * self.inner_dim * 2 * (self.num_patches+1)
                
            flops += self.inner_dim * self.num_patches
            flops += self.inner_dim * self.num_patches
            flops += self.inner_dim * self.dim      
        
        return flops


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0., stochastic_depth=0., if_patch_attn=False, is_LSA=False, is_coord=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                LayerScale(dim, PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, if_patch_attn=if_patch_attn, is_LSA=is_LSA, is_coord=is_coord)), depth = ind + 1),
                LayerScale(dim, PreNorm(num_patches, dim, FeedForward(num_patches, dim, mlp_dim, dropout = dropout, is_coord=is_coord, if_patch_attn=if_patch_attn)), depth = ind + 1)
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
        is_base=True, pe_dim=128, is_coord=False, is_LSA=False, 
        eps=0., merging_size=2, n_trans=8, STT_head=4, 
        STT_depth=1, is_ape=False
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.dim = dim
        self.num_classes = num_classes
        self.is_base = is_base
        self.is_ape = is_ape
        
        if is_coord:
            self.coord = AddCoords1D()
        
        if is_base:
            patch_dim = 3 * patch_size ** 2
            
            # if is_coord:    
            #     self.to_patch_embedding = nn.Sequential(
            #         Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            #         CoordLinear(patch_dim, dim, exist_cls_token=False)
            #     ) 
            #     self.pe_flops = (patch_dim+2) * dim * num_patches
            
            # else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
                nn.Linear(patch_dim, dim),
            )
            self.pe_flops = patch_dim * dim * num_patches
        
        else:  
            self.to_patch_embedding = STT(img_size=img_size, patch_size=patch_size, in_dim=pe_dim, embed_dim=dim, type='PE', heads=STT_head, depth=STT_depth
                                           ,init_eps=eps, is_LSA=True, merging_size=merging_size, n_trans=n_trans)
       
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        if is_ape:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.patch_transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim, dropout, layer_dropout, stochastic_depth=stochastic_depth, if_patch_attn=True, is_LSA=is_LSA, is_coord=is_coord)
        self.cls_transformer = Transformer(dim, num_patches, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout, stochastic_depth=stochastic_depth, is_LSA=is_LSA, is_coord=is_coord)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.apply(init_weights)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        if not self.is_base:        
            self.theta = self.to_patch_embedding.theta
            self.scale = self.to_patch_embedding.scale_list
        
        b, n, _ = x.shape
        if self.is_ape:
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



    