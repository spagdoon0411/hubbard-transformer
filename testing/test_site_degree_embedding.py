import torch
from model.site_degree_embedding import SiteDegreeEmbedding
from model.param_embedding import SimpleParamEmbedding
from model.token_embedding import OccupationSpinEmbedding
from model.position_encoding import PositionEncoding

n_params = 2
embed_dim = 32
input_token_dims = [10, 2]
einops_rearrange = "o sp -> (o sp)"
batch = 16
n_tokens = 25
max_len = 100

# Initialization should not explode
he = SiteDegreeEmbedding(
    n_params=n_params,
    embed_dim=embed_dim,
    input_token_dims=input_token_dims,
    input_token_rearrange=einops_rearrange,
    param_embedding=SimpleParamEmbedding,
    token_embedding=OccupationSpinEmbedding,
    position_encoding=PositionEncoding,
)

test_params = torch.randn(n_params, batch)
test_occupations = torch.randn(n_tokens, batch, *input_token_dims)

logits = he(test_params, test_occupations)

assert logits.shape == (n_tokens + n_params, batch, embed_dim)
