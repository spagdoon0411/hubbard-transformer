from model.sampling import Sampling
from model.site_degree_embedding import SiteDegreeEmbedding
from model.param_embedding import SimpleParamEmbedding
from model.token_embedding import OccupationSpinEmbedding
from model.position_encoding import PositionEncoding
from model.hubbard_deembedding import HubbardDeembedding
import torch

n_params = 10
n_tokens = 27
additional_tokens = 5
embed_dim = 32
batch = 20
input_token_dims = [10, 2]
input_token_rearrange = "o sp -> (o sp)"

deembedding_function = HubbardDeembedding(
    embed_dim=embed_dim,
    target_token_dims=input_token_dims,
)

embedding_function = SiteDegreeEmbedding(
    n_params=n_params,
    embed_dim=embed_dim,
    input_token_dims=input_token_dims,
    input_token_rearrange=input_token_rearrange,
    param_embedding=SimpleParamEmbedding,
    token_embedding=OccupationSpinEmbedding,
    position_encoding=PositionEncoding,
)

# This layer takes parameters and tokens, expanding the current token buffer
# with the number of tokens requested
sampling = Sampling(
    embed_dim=embed_dim,
    embedding_function=embedding_function,
    deembedding_function=deembedding_function,
)

test_params = torch.randn(n_params, batch)
test_occupations = torch.randn(n_tokens, batch, *input_token_dims)

expanded_tokens = sampling.sample(
    params=test_params,  # type: ignore
    tokens=test_occupations,  # type: ignore
    up_to=n_tokens + additional_tokens,
)

assert expanded_tokens.shape == (
    n_tokens + additional_tokens,
    batch,
    *input_token_dims,  # type: ignore
)

new_tokens = expanded_tokens[n_tokens : n_tokens + additional_tokens, :, :, :]
assert torch.all(new_tokens.sum(dim=-2) == 1)
