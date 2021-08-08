import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.transforms as transforms
import math
# from utils.positional_encoding import positionalencoding2d

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

class G_Attention(nn.Module):
    def __init__(self, dim, num_patches=64, heads = 8, dim_head = 64, dropout = 0.):
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
        
        # self.g_block = G_Block(dim, inner_dim, heads, dropout)
        # self.g_block = G_Block(dim, num_patches)
        self.mask = torch.eye(num_patches+1, num_patches+1)
        self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        self.inf = float('-inf')
        self.value = 0
        self.entropy = HLoss()
        self.avg_h = None
        self.num_nodes = num_patches
        
    def _init_weights(self,layer):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # channel_agg = self.g_block(v)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # #scale = self.scale
        # dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((x.size(0), self.heads, 1, 1)))
    
        # dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf
        
        
        scores = self.attend(dots)
        

        out = einsum('b h i j, b h j d -> b h i d', scores, v)
        
        # self.avg_h = self.entropy(scores) / self.num_nodes
        
        # out = torch.cat([out, channel_agg], dim=-1)

        # global_attribute = self.g_block(x)
        # out = out + global_attribute
        # self.value = compute_relative_norm_residuals(v, out[:, :, :, :-1])
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out = self.g_block(out)
        out = self.to_out(out)
        
        return out
    
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
        
    def forward(self, x):
        log = torch.log(x + 1e-12)
        # log[:, :, self.mask[:, 0], self.mask[:, 1]] = 0.
        info = torch.mul(log, x)
        h = -1.0*torch.sum(info, dim=-1)
        h = h.mean(dim=-1)
        
        return h
    
    
# class G_Block(nn.Module):
#     def __init__(self, dim, inner_dim, heads, dropout):
#         super().__init__()
        
#         self.to_phi = nn.Linear(dim, inner_dim)
#         self._init_weights(self.to_phi)

#         self.to_phi = nn.Sequential(
#             self.to_phi,
#             nn.Dropout(dropout)
#         )
        
#         self.rho = nn.GELU()
#         self.heads = heads

#     def _init_weights(self,layer):
#         nn.init.xavier_normal_(layer.weight)
#         if layer.bias is not None:
#             nn.init.zeros_(layer.bias)  
    
#     def forward(self, x):
#         phi = self.to_phi(x)
#         phi = rearrange(phi, 'b n (h d) -> b h n d', h = self.heads)
#         pool, _ = phi.max(dim=-2, keepdim=True)
#         rho = self.rho(pool)
        
#         return rho

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., stochastic_depth=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.hidden_states = {}
        self.scale = {}
        self.h = []
        

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, G_Attention(dim, num_patches = num_patches, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            if i == 0:
                self.h = []   
            x = self.drop_path(attn(x)) + x
            # self.h.append(attn.fn.avg_h)           
            # self.hidden_states[str(i)] = attn.fn.value
      
            x = self.drop_path(ff(x)) + x         
            # self.scale[str(i)] = attn.fn.scale
        return x

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))
    
def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

class GiT(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., stochastic_depth=0.):
        super().__init__()
        layers = []
        layer_dim = channels
        output_image_size = img_size

        for i, (kernel_size, stride) in enumerate([(7, 4), (3, 2)]):
            layer_dim *= kernel_size ** 2
            is_first = i == 0
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)
            num_patches = output_image_size ** 2
            
            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                nn.Unfold(kernel_size = kernel_size, stride = stride, padding = stride // 2),
                Rearrange('b c n -> b n c'),
                Transformer(dim = layer_dim, num_patches=num_patches, heads = 1, depth = 1, dim_head = 64, mlp_dim = 64, dropout = dropout),
            ])
            
        num_patches = output_image_size ** 2

        layers.append(nn.Linear(layer_dim, dim))
        self.to_patch_embedding = nn.Sequential(*layers)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim, dropout, stochastic_depth)
        
        nn.linear_mlp_head = nn.Linear(dim, num_classes)
        self._init_weights(nn.linear_mlp_head)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.linear_mlp_head
        )
        
        self.h_loss = 0.
        
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
        x += self.pos_embedding[:, :n+1]
        x = self.dropout(x)

        x = self.transformer(x)
        
        # entropy = torch.cat(self.transformer.h, dim=1)
        
        # self.h_loss = entropy.mean(dim=-1)
        # print(self.h_loss)

        x =  x[:, 0]

        # x = self.to_latent(x)
        
        
        return self.mlp_head(x)
    
# class PatchShifting(nn.Module):
#     def __init__(self, patch_size):
#         super().__init__()
#         self.shift = patch_size // 2

#     def forward(self, x):
#         # x_l = torch.cat([torch.nn.pad(x, ), x[:, :, :, 1:]], dim=-1)
#         # x_r = torch.cat([x[:, :, :, :-1], self.w_pad], dim=-1)
#         # x_t = torch.cat([self.h_pad, x[:, :, 1:]], dim=-2)
#         # x_b = torch.cat([x[:, :, :-1], self.h_pad], dim=-2)
        
#         # print(x_l.shape)
#         # print(x_r.shape)
#         # print(x_t.shape)
#         # print(x_b.shape)
#         x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        
#         x_pad = x_pad.mean(dim=1, keepdim = True)
#         # x_pad = transforms.Grayscale()
        
#         x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
#         x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
#         x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
#         x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
               
#         x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)
        
        
#         return x_cat
    