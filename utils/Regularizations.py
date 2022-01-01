import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import math

def CosineSimiliarity(x):
    # print(x)
    loss = 0
    
    for x in x:
        
        cat = torch.cat(x, dim=1)
        batch, num, _ = cat.size()
        iden = torch.tensor([[1, 0, 0, 0, 1, 0]]).cuda(torch.cuda.current_device())
        iden = iden.expand(batch, *iden.size())
        cat_iden = torch.cat([iden, cat], dim=1)
        norm = torch.norm(cat_iden, dim=-1, keepdim = True)
        x_norm = torch.div(cat_iden, norm)
        similiarity = einsum('b n c, b l c -> b n l', x_norm, x_norm)
        exp = torch.exp(similiarity[:, :, 1:])
        positive = 2*exp[:, (0,)]
        negative = exp[:, 1:]
        mask = 1 - torch.eye(num).cuda(torch.cuda.current_device())
        n = torch.mul(negative, mask)
        
        sum = torch.sum(n, dim=1, keepdim = True)
        numer = positive + sum
        div = torch.div(positive, numer)
        results = -torch.log(div)
        results = torch.sum(results, dim=-1)
        results = torch.mean(results)
        loss = loss + results
            
    return loss

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