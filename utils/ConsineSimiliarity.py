import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import math

def CosineSimiliarity(x):
    
    avg = x.mean(dim=0, keepdim= True)
    x_ = x - avg
    
    norm = torch.norm(x_, dim=-1, keepdim= True)
    
    norm = torch.div(x_, norm)
    
    sim = einsum('b n, d n -> b d', norm, norm)
    mask = 1 - torch.eye(x.size(0)).cuda(torch.cuda.current_device())
    
    masked_sim = torch.mul(sim, mask)
    
    norm = torch.norm(masked_sim) / math.sqrt(2)
            
    return norm.unsqueeze(-1)

# def CosineSimiliarity(x):
    
#     norm = torch.norm(x, dim=-1, keepdim= True)
    
#     norm = torch.div(x, norm)
    
#     sim = einsum('b n, d n -> b d', norm, norm)
#     mask = 1 - torch.eye(x.size(0)).cuda(torch.cuda.current_device())
    
#     masked_sim = torch.mul(sim, mask)
    
#     norm = torch.norm(masked_sim) / math.sqrt(2)
            
#     return norm.unsqueeze(-1)

def MeanVector(x):
    
    avg = x.mean(dim=0, keepdim= True)
    norm = torch.norm(avg, keepdim= True)
    
    sim = einsum('b n, d n -> b d', norm, norm)
    mask = 1 - torch.eye(x.size(0)).cuda(torch.cuda.current_device())
    
    masked_sim = torch.mul(sim, mask)
    
    norm = torch.norm(masked_sim) / math.sqrt(2)
    
    return norm.unsqueeze(-1)

def Identity(x):
    
    identity = torch.tensor([1, 0, 0, 1, 0, 0]).unsqueeze(0)
    
    diff = x - identity.cuda(torch.cuda.current_device())
    
    norm = torch.norm(diff)
    
    return norm