from model.model import HubbardWaveFunction
import torch
import torch.nn.functional as F

n_params = 5
batch = 16
n_tokens = 25
input_token_dims = [2, 2]
embed_dim = 32
n_heads = 2
n_layers = 2
dim_feedforward = 64
particle_number = 19
max_len = 100


test_params = torch.randn(n_params, batch)
test_occupations = torch.randn(n_tokens, batch, *input_token_dims)
test_occupations_idx = test_occupations.argmax(dim=-2)
test_occupations = F.one_hot(test_occupations_idx, num_classes=input_token_dims[0])
test_occupations = test_occupations.to(dtype=torch.float32)

model = HubbardWaveFunction(
    embed_dim=embed_dim,
    n_heads=n_heads,
    n_layers=n_layers,
    dim_feedforward=dim_feedforward,
    particle_number=particle_number,
    max_len=max_len,
)

prob, phase = model(test_params, test_occupations)

assert prob.shape == (n_tokens, batch, input_token_dims[1])
assert phase.shape == (n_tokens, batch, input_token_dims[1])

sample_seq, sample_batch = 10, 20

test_sample_params = torch.randn(n_params)
chains = model.sample(
    num_chains=sample_batch,
    chain_length=sample_seq,
    params=test_sample_params,  # type: ignore
)

assert chains.shape == (sample_seq, sample_batch, *input_token_dims)
