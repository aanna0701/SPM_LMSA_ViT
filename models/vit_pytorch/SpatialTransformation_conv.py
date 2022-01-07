import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import math
from einops import rearrange, repeat

    
class AffineNet(nn.Module):
    def __init__(self, num_patches, in_dim, hidden_dim, n_trans=4, merging_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.n_trans = n_trans
        self.n_output = 6*self.n_trans
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.merging_size = merging_size
        self.param_transformer = nn.Sequential(            
            nn.Conv2d(in_dim, self.in_dim, 3, 2, 1),
            nn.Conv2d(self.in_dim,in_dim,  3, 2, 1),
            Rearrange('b c h w -> b (c h w)'),
        )
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.in_dim*(self.num_patches//16)),
            nn.Linear(self.in_dim*(self.num_patches//16), self.n_output)
        )  
        self.transformation = Affine()
        self.pre_linear = nn.Conv2d(self.in_dim, self.hidden_dim, (1, 1))
        self.post_linear = nn.Conv2d(self.hidden_dim, self.in_dim, (1, 1))
    
        self.theta = list()
        
    def forward(self, x, init, scale=None):
        param_attd = self.param_transformer(x)
        param = self.mlp_head(param_attd)
        param_list = torch.chunk(param, self.n_trans, dim=-1)
        
        out = []
        theta = []       
        
        x = self.pre_linear(x)
        x = torch.chunk(x, self.n_trans, dim=1)
        for i in range(self.n_trans):
            if scale is not None:
                out.append(self.transformation(x[i], param_list[i], init, scale[i]))
            else:
                out.append(self.transformation(x[i], param_list[i], init))
            theta.append(self.transformation.theta)            
        out = torch.cat(out, dim=1)
        out = self.post_linear(out)        
        out = rearrange(out, 'b d h w -> b (h w) d')
        self.theta = theta
        
        return out
    
    def flops(self):
        flops = 0
        # flops += (self.merging_size**2)*self.num_patches*self.hidden_dim    # depth-wise conv
        flops += 9*self.in_dim*(self.num_patches//4)*self.in_dim*2                # parameter-transformer
        flops += 9*self.in_dim*self.in_dim*2*(self.num_patches//16)                # parameter-transformer
        flops += self.in_dim*(self.num_patches//16) + self.in_dim*self.n_output*(self.num_patches//16)    # mlp head
        flops += self.num_patches*self.in_dim*self.hidden_dim    # pre-linear
        flops += self.num_patches*self.in_dim*self.hidden_dim   # post-linear
        
        return flops    
    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, num_patches, patch_size, dim, out_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.merging = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2 = patch_size)
        self.dim = dim
        self.out_dim = out_dim
        self.patch_dim = dim * (patch_size ** 2)
        self.reduction = nn.Linear(self.patch_dim, self.out_dim, bias=False)
        self.norm = nn.LayerNorm(self.patch_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = rearrange(x, 'b (h w) c -> b h w c', h = int(math.sqrt(self.num_patches)))
        x = self.merging(x)     
        x = self.norm(x)
        x = self.reduction(x)

        return x

    def flops(self):
        flops = 0
        flops += (self.num_patches//(self.patch_size**2))*self.patch_dim*self.out_dim
        flops += (self.num_patches//(self.patch_size**2))*self.patch_dim
        
        return flops
    
class Affine(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super().__init__()
        
        self.theta = None
        self.mode = padding_mode
        
    def forward(self, x, theta, init, scale=None):
        # print('========')
        # print(scale)
        # print(theta[0])     
        
        theta = F.tanh(theta)
        if scale is not None:
            theta = torch.mul(theta, scale)
        
        init = torch.reshape(init.unsqueeze(0), (1, 2, 3)).expand(x.size(0), -1, -1) 
        theta = torch.reshape(theta, (theta.size(0), 2, 3))    
        theta = theta + init 
        self.theta = theta    
   
        # print(theta[0])
        
        grid = F.affine_grid(theta, x.size())
        
        return F.grid_sample(x, grid, padding_mode=self.mode)
    
class STT(nn.Module):
    def __init__(self, img_size=224, patch_size=2, in_dim=3, embed_dim=96, depth=2, heads=4, type='PE', 
                 init_eps=0., is_LSA=False, merging_size=4, n_trans=4):
        super().__init__()
        assert type in ['PE', 'Pool'], 'Invalid type!!!'

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size**2
        self.in_dim = in_dim
        self.type = type
        
        if self.type == 'PE':
            if not img_size >= 224:
                self.input = nn.Conv2d(3, self.in_dim, 3, 2, 1)
                self.rearrange = Rearrange('b c h w -> b (h w) c')      
                self.affine_net = AffineNet(self.num_patches//4, self.in_dim, self.in_dim, n_trans=n_trans)
                self.patch_merge = PatchMerging(self.num_patches//4, patch_size//2, self.in_dim, embed_dim)
            else:
                self.input = nn.Conv2d(3, self.in_dim, 7, 4, 2)
                self.rearrange = Rearrange('b c h w -> b (h w) c')      
                self.affine_net = AffineNet(self.num_patches//16, depth, self.in_dim, self.in_dim, heads, merging_size=merging_size, 
                                            is_LSA=is_LSA, n_trans=n_trans)
                self.patch_merge = PatchMerging(self.num_patches//16, patch_size//4, self.in_dim, embed_dim)   
           
        else:
            self.input = nn.Identity()
            self.rearrange = nn.Identity()
            self.affine_net = AffineNet(self.num_patches, depth, self.in_dim, self.in_dim, heads, merging_size=merging_size, 
                                        is_LSA=is_LSA, n_trans=n_trans)
            self.patch_merge = PatchMerging(self.num_patches, 2, self.in_dim, self.in_dim*2)
        self.param_token = nn.Parameter(torch.rand(1, 1, self.in_dim))
                      
        if not init_eps == 0.:
            self.scale_list = nn.ParameterList()  
            for _ in range(n_trans):
                self.scale_list.append(nn.Parameter(torch.zeros(1, 6).fill_(init_eps)))
        else: self.scale_list = None  
        
        self.init = self.make_init().cuda(torch.cuda.current_device())
        self.theta = None                
        self.apply(self._init_weights)

    def make_init(self,):                
        out = torch.tensor([1, 0, 0,
                            0, 1, 0])
        return out

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # nn.init.xavier_normal_(m.weight)
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        
        x = self.input(x)
        affine = self.affine_net(x, self.init, self.scale_list)
        self.theta = self.affine_net.theta
        x = self.rearrange(x)
        out = x + affine
        out = self.patch_merge(out)
        
        return out
    
    def flops(self):
        flops = 0
        if self.type=='PE':
            # flops_input = (3**2)*3*self.in_dim*((self.img_size//2)**2)
            flops_input = (3**2)*3*self.in_dim*((self.img_size)**2)
            
        else:
            flops_input = 0
        flops += flops_input
        flops += self.affine_net.flops()   
        flops += self.patch_merge.flops() 
        
        return flops
    