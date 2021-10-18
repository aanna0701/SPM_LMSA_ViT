import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_



class Localisation(nn.Module):
    def __init__(self, img_size, in_dim=16, n_trans=4, type_trans='trans'):
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
            n_output = 6*n_trans
            
        self.mlp_head = nn.Sequential(
            nn.Linear(self.in_dim * (img_size**2), 64, bias=False),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, n_output)
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
        
        return out
        

class Translation(nn.Module):
    def __init__(self):
        super(Translation, self).__init__()
        self.tmp1 = torch.tensor([[0, 0, 1],[0, 0, 1]]).cuda(torch.cuda.current_device())
        self.tmp2 = torch.tensor([[1, 0, 0],[0, 1, 0]]).cuda(torch.cuda.current_device())
        
      
        
    def forward(self, x,  theta):
        theta = torch.mul(theta, self.tmp1)
        theta = theta + self.tmp2.expand(x.size(0), 2, 3)
        
        # print(theta[0])
        
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        
        return F.grid_sample(x, grid, align_corners=True)
    

class Affine(nn.Module):
    def __init__(self):
        super().__init__()
              
        
    def forward(self, x,  theta):
                
        
        theta = torch.reshape(theta, (theta.size(0), 2, 3))
        
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        
        return F.grid_sample(x, grid, align_corners=True)
    
    

class Rotation(nn.Module):
    def __init__(self, angle=0.):
        super().__init__()
        self.angle = nn.Parameter(torch.tensor([angle], dtype=torch.float))
        
    def forward(self, x):
        
        cos = torch.cos(self.angle)
        sin = torch.sin(self.angle)
        
        mat_cos = torch.mul(cos, torch.tensor([[1, 0, 0],
                          [0, 1, 0]]))
        mat_sin = torch.mul(sin, torch.tensor([[0, -1, 0],
                                [1, 0, 0]]))
        theta = mat_cos + mat_sin
        
        grid = F.affine_grid(theta.expand(x.size(0), 2, 3), x.size(), align_corners=True)
        return F.grid_sample(x, grid, align_corners=True)
    

class Rigid(nn.Module):
    def __init__(self, angle=0.):
        super().__init__()
        self.angle = nn.Parameter(torch.tensor([angle], dtype=torch.float))
        
    def forward(self, x):
        
        cos = torch.cos(self.angle)
        sin = torch.sin(self.angle)
        
        mat_cos = torch.mul(cos, torch.tensor([[1, 0, 0],
                          [0, 1, 0]]))
        mat_sin = torch.mul(sin, torch.tensor([[0, -1, 0],
                                [1, 0, 0]]))
        theta = mat_cos + mat_sin
        
        grid = F.affine_grid(theta.expand(x.size(0), 2, 3), x.size(), align_corners=True)
        return F.grid_sample(x, grid, align_corners=True)