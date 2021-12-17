import torch
import torch.nn as nn
from math import sqrt
from einops import rearrange

class AddCoords2D(nn.Module):
    
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        
        
        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords2D(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret
    
    
class AddCoords1D(nn.Module):
    
    def __init__(self, num_patches, with_r=False):
        super().__init__()
        self.with_r = with_r
        self.spatial_dim = int(sqrt(num_patches))
        self.xx = nn.Parameter(torch.rand(1, 1, self.spatial_dim))
        self.yy = nn.Parameter(torch.rand(1, self.spatial_dim, 1))
        
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """

        batch_size, n, _ = input_tensor.size()
        # t_dim = int(sqrt(n)) 
        # xx_channel = torch.arange(t_dim).repeat(1, t_dim, 1)
        # yy_channel = xx_channel.clone().transpose(1, 2)
        xx_channel = self.xx.repeat(1, self.spatial_dim, 1)
        yy_channel = self.yy.repeat(1, 1, self.spatial_dim)

        # xx_channel = xx_channel.float() / (self.spatial_dim - 1)
        # yy_channel = yy_channel.float() / (self.spatial_dim - 1)

        # xx_channel = xx_channel * 2 - 1
        # yy_channel = yy_channel * 2 - 1
        
        print(xx_channel)
        print(yy_channel)

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        
        input_tensor = rearrange(input_tensor, 'b (h w) d -> b d h w', h = self.spatial_dim)     
                

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)
      
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        ret = rearrange(ret, 'b d h w -> b (h w) d')

        return ret


class CoordLinear(nn.Module):

    def __init__(self, num_patches, in_channels, out_channels, bias=True, with_r=False, exist_cls_token=True, **kwargs):
        super().__init__()
        self.addcoords = AddCoords1D(num_patches, with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.linear = nn.Linear(in_size, out_channels, bias=bias)
        
        self.exist_cls_token = exist_cls_token
        if exist_cls_token:
            self.cls_linear = nn.Linear(in_channels, out_channels, bias=bias) 


    def forward(self, x):
        if self.exist_cls_token:
            cls_token = self.cls_linear(x[:, (0,)]) # (b, 1, d')
            ret = self.addcoords(x[:, 1:])  # (b, n, d+2) or (b, n, d+3)
            ret = self.linear(ret)          # (b, n, d')
            out = torch.cat([cls_token, ret], dim=1)    # (b, n+1, d')
        else:
            ret = self.addcoords(x) # (b, n, d+2) or (b, n, d+3)
            ret = self.linear(ret)  # (b, n, d')
            out = ret   
        
        return out 