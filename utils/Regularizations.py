import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import math

def CosineSimiliarity(x):
    loss = 0
    iden = torch.tensor([[1, 0, 0, 0, 1, 0]]).cuda(torch.cuda.current_device())     
    x = torch.cat(x, dim=1)
    iden = iden.expand(x.size(0), *iden.size())
    matrix = torch.cat([iden, x], dim=1) 
    norm = torch.norm(matrix, dim=-1, keepdim = True)
    matrix = torch.div(matrix, norm)
    similiarity = einsum('b n c, b l c -> b n l', matrix, matrix)
    similiarity = torch.exp(similiarity[:, :, 1:])
    positive = similiarity[:, (0,)]
    negative = similiarity[:, 1:]
    mask = 1 - torch.eye(negative.size(-1)).cuda(torch.cuda.current_device())
    negative = torch.mul(negative, mask)
    
    numer = positive + torch.sum(negative, dim=1, keepdim = True)
    results = -torch.log(torch.div(positive, numer))
    results = torch.mean(results)
            
    return results

def Identity(x):
        
    length = len(x)
    
    iden = torch.tensor([[1, 0, 0], [0, 1, 0]]).cuda(torch.cuda.current_device())
    iden = iden.expand(*x.size())
    
    res = x - iden
    
    loss = torch.norm(res, dim=(1, 2))
    loss = torch.square(loss)
    loss = torch.mean(loss)
                
    return loss

# def CosineSimiliarity(x):
    
#     norm = torch.norm(x, dim=-1, keepdim= True)
    
#     norm = torch.div(x, norm)
    
#     sim = einsum('b n, d n -> b d', norm, norm)
#     mask = 1 - torch.eye(x.size(0)).cuda(torch.cuda.current_device())
    
#     masked_sim = torch.mul(sim, mask)
    
#     norm = torch.norm(masked_sim) / math.sqrt(2)
            
#     return norm.unsqueeze(-1)

# def MeanVector(x):
    
#     avg = x.mean(dim=0, keepdim= True)
#     norm = torch.norm(avg, keepdim= True)
    
#     sim = einsum('b n, d n -> b d', norm, norm)
#     mask = 1 - torch.eye(x.size(0)).cuda(torch.cuda.current_device())
    
#     masked_sim = torch.mul(sim, mask)
    
#     norm = torch.norm(masked_sim)
    
#     return norm.unsqueeze(-1)

# def Identity(x):
    
#     identity = torch.tensor([1, 0, 0, 1, 0, 0]).unsqueeze(0)
    
#     diff = x - identity.cuda(torch.cuda.current_device())
    
#     norm = torch.norm(diff)
    
#     return norm