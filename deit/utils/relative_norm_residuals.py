import torch
# import jax.numpy as jnp
# import jax
# import functools



def l1_matrix_norm(x):
    return x.abs().sum(axis=-2 % x.ndim).max(axis=-1).values

def linf_matrix_norm(x):
    return l1_matrix_norm(x.transpose(-2, -1))

def composite_norm(x):
    # return torch.sqrt(l1_matrix_norm(x) * linf_matrix_norm(x))
    return l1_matrix_norm(x)

def compute_residuals(inputs, hidden_states):
    # low = hidden_states.mean(-2, keepdim= True)
    # u, s, vh = torch.linalg.svd(hidden_states, full_matrices=False)
    # print(u[:, :, :1].shape)
    # print(s[:, :1].shape)
    # print(vh[:, :1 ,:].shape)
    # mean = hidden_states[:, (2, )]
    # low = einsum("b i j,b j,b j k -> b i k", u[:, :, :1], s[:, :1], vh[:, :1 ,:])
    residuals = hidden_states - inputs.mean(dim=-2, keepdim = True).expand(inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3))
    return residuals

# def compute_residuals(hidden_states):
#     # low = hidden_states.mean(-2, keepdim= True)
#     u, s, vh = torch.linalg.svd(hidden_states, full_matrices=False)
#     # print(u[:, :, :1].shape)
#     # print(s[:, :1].shape)
#     # print(vh[:, :1 ,:].shape)
#     # mean = hidden_states[:, (2, )]
#     low = einsum("b i j,b j,b j k -> b i k", u[:, :, :1], s[:, :1], vh[:, :1 ,:])
#     residuals = hidden_states - low
#     return residuals

def compute_relative_norm_residuals(inputs, hidden_states):
    residuals = compute_residuals(inputs, hidden_states)
    return composite_norm(residuals) / composite_norm(inputs)

# def compute_relative_norm_residuals(hidden_states):
#     residuals = compute_residuals(hidden_states)
#     return composite_norm(residuals) / composite_norm(hidden_states)

# def compute_rank(hidden_states):
#     print('compute rank')
#     rank_batch = torch.matrix_rank(hidden_states)
#     rank_avg = torch.mean(rank_batch.float())
#     return rank_avg.item()