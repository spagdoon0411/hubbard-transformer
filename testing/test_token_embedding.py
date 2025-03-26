import torch
from model.token_embedding import OccupationSpinEmbedding

# An occupation spin embedding takes in a buffer of shape (s, b, o1, ..., on)
# and returns a buffer of shape (s, b, e) where e is the model embedding dimension.

input_token_dims = [10, 17, 234, 2]
output_token_dims = 32  # 32 embedding dimensions

ose = OccupationSpinEmbedding(
    input_token_dims=input_token_dims,
    output_token_dims=output_token_dims,
    einops_pattern="o sp a j -> (o a j sp)",
)

seq, batch = (5, 3)
occupations = torch.randn(seq, batch, *input_token_dims)
logits = ose(occupations)

assert logits.shape == (seq, batch, output_token_dims)
