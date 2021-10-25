import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math

ALPHA = 1

class Localisation(nn.Module):
    def __init__(self, img_size, n_tokenize,in_dim=16, n_trans=4, type_trans='trans'):
        super().__init__()
        self.in_dim = in_dim
        
        self.layers0 = nn.Sequential(
            nn.Conv2d(3, self.in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.GELU()
        )
        
        img_size //= 2
        
        self.layers1 = self.make_layer(self.in_dim, self.in_dim*2)
        self.in_dim *= 2
        img_size //= 2
        
        # self.layers2 = self.make_layer(self.in_dim, self.in_dim*2)
        # self.in_dim *= 2
        # img_size //= 2
        
        if type_trans=='trans':
            n_output = 2*n_trans
        elif type_trans=='affine':
            n_output = 4*n_trans
        elif type_trans=='rigid':
            n_output = 3*n_trans
        
        # self.n_tokenize = n_tokenize 
        # n_output *= n_tokenize
            
        self.mlp_head = nn.Sequential(
            nn.Linear(self.in_dim * (img_size**2), 64, bias=False),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, n_output, bias=False),
            nn.LayerNorm(n_output),
            nn.Tanh()
        )
        self.num_transform = n_trans
        
        self.apply(self._init_weights)

        
    def make_layer(self, in_dim, hidden_dim):
        layers = nn.ModuleList([])
    
        layers.append(nn.Conv2d(in_dim, hidden_dim, 3, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.GELU())
            
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
    
        feature1 = self.layers0(x)
        feature2 = self.layers1(feature1)
        
        out = feature2.view(feature2.size(0), -1)
        out = self.mlp_head(out)
        
        # out = torch.chunk(out, self.n_tokenize, -1)

        
        return out
        
        

class Translation(nn.Module):
    def __init__(self, constant=5e1, adaptive=False):
        super(Translation, self).__init__()
        self.tmp1 = torch.tensor([[0, 0, 1],[0, 0, 1]]).cuda(torch.cuda.current_device())
        self.tmp2 = torch.tensor([[1, 0, 0],[0, 1, 0]]).cuda(torch.cuda.current_device())
        
        self.constant = constant
        self.theta = None
        self.constant_tmp = 1
        self.is_adaptive = adaptive

        
    def forward(self, x, theta, patch_size, epoch=None, train=False):
        
        if not train or not self.is_adaptive:
            constant = 1
                
        else:
            if epoch is not None:
                constant = self.constant * epoch         
                constant = 1 - math.exp(-constant)
                self.constant_tmp = constant
                
            else:
                constant = self.constant_tmp 
              
        
        theta = theta * constant
        theta = theta.unsqueeze(-1)
        theta = torch.mul(theta, self.tmp1)
        theta = theta + self.tmp2.expand(x.size(0), 2, 3)
        
        # print(theta[0])
        
        # print(theta[2])
        
        
        grid = F.affine_grid(theta, x.size())
        
        return F.grid_sample(x, grid)
    

class Affine(nn.Module):
    def __init__(self, constant=5e1, adaptive=False):
        super().__init__()
        
        self.constant = constant
        self.theta = None
        self.constant_tmp = 1
        self.is_adaptive = adaptive
        
        
    def forward(self, x, theta,  patch_size, epoch=None, train=False):
        
        if not train or not self.is_adaptive:
            constant = 1
                
        else:
            if epoch is not None:
                constant = self.constant * epoch         
                constant = 1 - math.exp(-constant)
                self.constant_tmp = constant
                
            else:
                constant = self.constant_tmp 
            
        theta = theta * constant    
        
        cos = torch.cat([theta[:, (0, )], theta[:, (1, )]], dim=-1).unsqueeze(-1)
        sin = torch.cat([-theta[:, (1, )], theta[:, (0, )]], dim=-1).unsqueeze(-1)
        xy = theta[:, 2:].unsqueeze(-1)
        
        theta = torch.cat([cos, sin, xy], dim=-1)
        
        # theta = torch.reshape(theta, (theta.size(0), 2, 3))        
            
        
        grid = F.affine_grid(theta, x.size())
        
        return F.grid_sample(x, grid)
    

# class Affine(nn.Module):
#     def __init__(self, constant=5e1, adaptive=False):
#         super().__init__()
        
#         self.constant = constant
#         self.theta = None
#         self.constant_tmp = 1
#         self.is_adaptive = adaptive
        
        
#     def forward(self, x, theta,  patch_size, epoch=None, train=False):
        
#         if not train or not self.is_adaptive:
#             constant = 1
                
#         else:
#             if epoch is not None:
#                 constant = self.constant * epoch         
#                 constant = 1 - math.exp(-constant)
#                 self.constant_tmp = constant
                
#             else:
#                 constant = self.constant_tmp 
            
#         theta = theta * constant    
        
#         theta = torch.reshape(theta, (theta.size(0), 2, 3))        
            
#         print(theta[0])
        
#         grid = F.affine_grid(theta, x.size())
        
#         return F.grid_sample(x, grid)
    
    

class Rigid(nn.Module):
    def __init__(self, constant=5e1, adaptive=False):
        super().__init__()
        self.tmp1 = torch.tensor([[0, 0, 1],[0, 0, 1]]).cuda(torch.cuda.current_device())
        self.tmp2 = torch.tensor([[1, 0, 0],[0, 1, 0]]).cuda(torch.cuda.current_device())
        self.tmp3 = torch.tensor([[0, -1, 0],[1, 0, 0]]).cuda(torch.cuda.current_device())

            
        self.constant = constant
        self.theta = None
        self.constant_tmp = 1
        self.is_adaptive = adaptive
        
    def forward(self, x, theta,  patch_size, epoch=None, train=False):
        
        if not train or not self.is_adaptive:
            constant = 1
                
        else:
            if epoch is not None:
                constant = self.constant * epoch         
                constant = 1 - math.exp(-constant)
                self.constant_tmp = constant
                
            else:
                constant = self.constant_tmp 

        # print(constant)

        
        theta = theta * constant 
        theta = theta.unsqueeze(-1)
                
        angle = theta[:, (0,)]
        angle = angle * math.pi
        trans = theta[:, 1:]
        
        cos = torch.cos(angle)
        sin = torch.sin(angle)
     
        mat_cos = torch.mul(cos, self.tmp2.expand(x.size(0), 2, 3))
        mat_sin = torch.mul(sin, self.tmp3.expand(x.size(0), 2, 3))
        mat_trans = torch.mul(trans, self.tmp1.expand(x.size(0), 2, 3))
        
        theta = mat_cos + mat_sin + mat_trans
        self.theta = theta
        
        
        grid = F.affine_grid(theta.expand(x.size(0), 2, 3), x.size())
        
        return F.grid_sample(x, grid)
    