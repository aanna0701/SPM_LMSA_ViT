import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F

def CosineSimiliarity(x):
    
    norm = torch.norm(x)
    
    sim = einsum('b n, d n -> b d', x/norm, x/norm)
    mask = 1 - torch.eye(x.size(0)).cuda(torch.cuda.current_device())
    
    masked_sim = torch.mul(sim, mask)
    
    norm = torch.norm(masked_sim) / 2
            
    return norm.unsqueeze(-1)
