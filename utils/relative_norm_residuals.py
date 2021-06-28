import torch
# import jax.numpy as jnp
# import jax
# import functools
import numpy as np


# def compute_low_rank(x, k=1):
#     U, s, Vh = jax.vmap(jnp.linalg.svd)(x)
#     return jnp.einsum("ij,j,jk->ik", U[:, :k], s[:k], Vh[:k ,:])

# def l1_matrix_norm(x):
#     return x.abs().sum(axis=-2 % x.ndim).max(axis=-1).values

# def linf_matrix_norm(x):
#     return l1_matrix_norm(x.transpose(-2, -1))

# def composite_norm(x):
#     return torch.sqrt(l1_matrix_norm(x) * linf_matrix_norm(x))

# def compute_residuals(hidden_states):
#     rank_one = jax.jit(jax.vmap(jax.vmap(functools.partial(compute_low_rank, k=1))))(jnp.array(hidden_states))
#     rank_one = torch.tensor(np.array(rank_one))
#     residuals = hidden_states - rank_one
#     return residuals

# def compute_relative_norm_residuals(hidden_states):
#     residuals = compute_residuals(hidden_states)
#     return composite_norm(residuals) / composite_norm(hidden_states)

def compute_rank(hidden_states):
    print('compute rank')
    rank_batch = torch.matrix_rank(hidden_states)
    rank_avg = torch.mean(rank_batch.float())
    return rank_avg.item()