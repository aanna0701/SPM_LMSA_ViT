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

class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_last=False):
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
        self.mask = torch.eye(num_patches+1, num_patches+1)
        self.mask = (self.mask == 1).nonzero()
        self.inf = float('-inf')
        
        self.value = 0
        self.entropy = HLoss()
        self.avg_h = None
        self.l2 = NSLoss()
        self.cls_l2 = None
        self.is_last = is_last
        
    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  

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
    
        if self.is_last:
            self.cls_l2 = self.l2(attn[:, :, 0])
            self.avg_h = self.entropy(attn[:, :, 0])
        
        # self.value = compute_relative_norm_residuals(v, out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.hidden_states = {}
        self.scale = {}
        self.h = None
        self.cls_l2 = None
        is_last = False

        for i in range(depth):
            if i == depth-1 :
                is_last = True
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_last=is_last)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))            
            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):       
            x = self.drop_path(attn(x)) + x
            # self.hidden_states[str(i)] = attn.fn.value
            self.h = attn.fn.avg_h         
            self.cls_l2 = attn.fn.cls_l2        
            x = self.drop_path(ff(x)) + x
            self.scale[str(i)] = attn.fn.scale
        return x

class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., stochastic_depth=0.):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.linear_to_path = nn.Linear(patch_dim, dim)
        self._init_weights(self.linear_to_path)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            self.linear_to_path,
        )


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim, dropout, stochastic_depth)

        self.pool = pool
        self.to_latent = nn.Identity()
        
        nn.linear_mlp_head = nn.Linear(dim, num_classes)
        self._init_weights(nn.linear_mlp_head)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.linear_mlp_head
        )
        
        self.final_cls_token = None

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
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        
        self.l2_loss = self.transformer.cls_l2.mean()
        
        self.h_loss = self.transformer.h.mean()

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        
        self.final_cls_token = x
        
        return self.mlp_head(x)

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
        
    def forward(self, x):
        log = torch.log(x + 1e-6)
        # log[:, :, self.mask[:, 0], self.mask[:, 1]] = 0.
        info = torch.mul(log, x)
        h = -1.0*torch.sum(info, dim=-1)
        h = h.mean(dim=-1)
        
        return h
    
class NSLoss(nn.Module):
    def __init__(self):
        super(NSLoss, self).__init__()
        
    def forward(self, x):
        # x[:, :, self.mask[:, 0], self.mask[:, 1]] = 0.
        l2 = torch.linalg.norm(x, dim=-1, ord=2)
        return l2
        # max = torch.linalg.norm(x, dim=-1, ord=float('inf'))
        return max