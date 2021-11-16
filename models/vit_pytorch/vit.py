import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.relative_norm_residuals import compute_relative_norm_residuals
import math
from .SpatialTransformation import Localisation, Affine, Trans_scale
import numpy as np
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
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.scale = nn.Parameter(self.scale*torch.ones(heads))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        init_weights(self.to_qkv)
        
 

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        
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
      

        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        scale = self.scale
        dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
    
        
        dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
    
        
        self.value = compute_relative_norm_residuals(v, out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.hidden_states = {}
        self.scale = {}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, dim * mlp_dim_ratio, dropout = dropout))
            ]))            
            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):       
            x = self.drop_path(attn(x)) + x
            self.hidden_states[str(i)] = attn.fn.value
            x = self.drop_path(ff(x)) + x
            
            self.scale[str(i)] = attn.fn.scale
        return x

class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, pool = 'cls', channels = 3, 
                 dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0.,
                 is_base=True, n_trans=4, is_learn=True, \
                 init_type='aistats', eps=0., padding_mode='zeros', type_trans='affine'):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        if is_base:
        
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(patch_dim, dim),
            )
            
        else:
            self.to_patch_embedding = ShiftedPatchTokenization(img_size//patch_size, 3, dim, patch_size, is_learn=is_learn, 
                                                               init_type=init_type, eps=eps, padding_mode=padding_mode, 
                                                               type_trans=type_trans)
            

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout, stochastic_depth)

        self.pool = pool
        self.to_latent = nn.Identity()


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.n_trans = n_trans
        self.is_learn = is_learn
 
        if not is_base:
            if self.is_learn:
                self.localisation = Localisation(img_size=img_size, n_trans=n_trans, type_trans=type_trans)

        self.theta = None
        self.scale = 0    
        
        self.final_cls_token = None
        
        self.apply(init_weights)


    def forward(self, img):
        # patch embedding
        
        if not self.is_learn:
            x = self.to_patch_embedding(img)
            
        else:
                  
            theta = self.localisation(img)
            theta = torch.chunk(theta, self.n_trans, dim=1)
         
            x = self.to_patch_embedding(img, theta)  
        
            self.theta = self.to_patch_embedding.theta
            self.scale = self.to_patch_embedding.scale
        
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        
        self.final_cls_token = x
        
        return self.mlp_head(x)



    
class ShiftedPatchTokenization(nn.Module):
    def __init__(self, num_patches, in_dim, dim, merging_size=2, exist_class_t=False, is_learn=False, n_trans=4, init_type='aistats', eps=0., padding_mode='zeros', type_trans='affine'):
        super().__init__()
        self.exist_class_t = exist_class_t
        
        self.is_learn = is_learn
        
        if is_learn:      
            self.patch_shifting = SpatialTransformation_learn(num_patches, init_type=init_type, eps=eps, padding_mode=padding_mode, type_trans=type_trans)
            
        else:           
            self.patch_shifting = SpatialTransformation_fix(merging_size) 
            
        patch_dim = (in_dim*(1+n_trans)) * (merging_size**2) 
        
        self.merging = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )
        
        self.theta = None
        self.scale = None
        # print(self.merging)
        
        
    def forward(self, x, theta=None):
        out = x if len(x.size()) == 4 else rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
        
        if self.is_learn:
            
            out = self.patch_shifting(out, theta)
        else:
            out = self.patch_shifting(out)
        out = self.merging(out)
      
        self.theta = self.patch_shifting.theta
        self.scale = self.patch_shifting.scale

        return out

    
class SpatialTransformation_fix(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
      
        self.shift = patch_size // 2
        self.is_learn = False
        self.theta = None
      
    def forward(self, x):
   
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        # if self.is_mean:
        #     x_pad = x_pad.mean(dim=1, keepdim = True)
      
        """ 4 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1) 
        #############################
      
        """ 4 diagonal directions """
        # #############################
        x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
        x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
        x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
        x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
        x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1) 
        # #############################
      
        """ 8 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
        # x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
        # x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
        # x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2, x_lu, x_ru, x_lb, x_rb], dim=1) 
        #############################
      
        # out = self.out(x_cat)
        out = x_cat
        
        return out
    
    
class SpatialTransformation_learn(nn.Module):
    def __init__(self, num_patches, init_type='aistats', eps=0., padding_mode='zeros', type_trans='affine'):
        super().__init__()
        
        self.is_learn = True
        if type_trans == 'affine':
            self.transformation = Affine(padding_mode=padding_mode)
        elif type_trans == 'trans_scale':
            self.transformation = Trans_scale(padding_mode=padding_mode)
        self.init = list()
        
        for i in range(4):
            self.init.append(self.make_init(i, num_patches, init_type=init_type).cuda(torch.cuda.current_device()))
        
        self.theta = list()
        
        init_eps = eps
        
        if not eps == 0.:
            self.scale = nn.ParameterList()
            for i in range(4):
                self.scale.append(nn.Parameter(torch.zeros(1, 1).fill_(init_eps)))
        else: self.scale = None
                
          
    def make_init(self, n, num_patches, init_type='aistats'):
        
        # ratio1 = torch.normal(1/num_patches, 1e-1).item()
        # ratio2 = torch.normal(1/num_patches, 1e-1).item()
        ratio = np.random.normal(1/num_patches, 1e-2, size=2)
        ratio_scale = float(np.random.normal(1, 1e-2))
        ratio_x = float((math.cos(n * math.pi))*ratio[0])
        ratio_y = float((math.sin(((n//2) * 2 + 1) * math.pi / 2))*ratio[1])
        
        if init_type == 'aistats':
            tmp = torch.tensor([ratio_scale, 0, ratio_x, 0, ratio_scale, ratio_y])
            # tmp = torch.tensor([1, 0, (math.cos(n * math.pi)), 0, 1, (math.sin(((n//2) * 2 + 1) * math.pi / 2))])
        elif init_type == 'identity':
            tmp = torch.tensor([1, 0, 0, 0, 1, 0])
            
        print(tmp)
        
        return tmp
        
                
    def forward(self, x, theta_list):   
        
        # print(theta[0])
        
        out = [x]
        tmp = list()
        
        
        for i in range (len(theta_list)):
            if self.scale is None:
                out.append(self.transformation(x, theta_list[i], self.init[i]))
            
            else:
                out.append(self.transformation(x, theta_list[i], self.init[i], self.scale[i]))
                
            tmp.append(self.transformation.theta)
            
        self.theta = tmp
        
        out = torch.cat(out, dim=1)
        
        
       
        return out