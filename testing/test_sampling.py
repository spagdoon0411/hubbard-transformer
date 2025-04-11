from model.sampling import Sampling
from model.site_degree_embedding import SiteDegreeEmbedding
from model.param_embedding import SimpleParamEmbedding
from model.token_embedding import OccupationSpinEmbedding
from model.position_encoding import PositionEncoding
from model.hubbard_deembedding import HubbardDeembedding
import torch
import torch.nn.functional as F
from torchtyping import TensorType
import einops as ein

n_params = 10
n_tokens = 27
additional_tokens = 5
embed_dim = 32
batch = 20
input_token_dims = [50, 2]
input_token_rearrange = "o sp -> (o sp)"
target_part_num = 100

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
    particle_number=100,
    embedding_function=embedding_function,
    deembedding_function=deembedding_function,
)

test_params = torch.randn(n_params, batch)


# Generate some dummy occupations
max_occ = 50
occ, spin = input_token_dims
test_occupations = torch.rand(n_tokens, batch, *input_token_dims)
test_occupations = (test_occupations * (max_occ + 1)).to(torch.int64)
test_occupations = torch.argmax(test_occupations, dim=-2)
test_occupations = F.one_hot(test_occupations, num_classes=occ)
test_occupations = ein.rearrange(
    test_occupations,
    "s b sp o -> s b o sp",
)

test_occupations = test_occupations.to(dtype=torch.float32)

# Did we create realistic one-hot occupations?
assert torch.all(test_occupations.sum(dim=-2) == 1)

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

part_nums = torch.argmax(expanded_tokens, dim=-2).sum(dim=0).sum(dim=-1)
assert torch.all(part_nums == target_part_num)

new_tokens = expanded_tokens[n_tokens : n_tokens + additional_tokens, :, :, :]
assert torch.all(new_tokens.sum(dim=-2) == 1)
