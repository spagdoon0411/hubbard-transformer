import torch
from model.param_embedding import ParamEmbedding

# A parameter embedding takes in a buffer of shape (seq, batch) and
# pushes the parameters out into an embedding dimension, producing an
# output of dimension (seq, batch, embed).

# TODO: double-check this module for correct embedding.

n_params = 10
target_dim = 16

pe = ParamEmbedding(
    n_params=n_params,
    target_dim=target_dim,
)

batch = 32

# (seq, batch)
test_params = torch.randn(n_params, batch)
logits = pe(test_params)
assert logits.shape == (n_params, batch, target_dim)
