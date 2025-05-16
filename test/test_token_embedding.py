import pytest
import torch
from model.token_embedding import OccupationSpinEmbedding


@pytest.fixture
def setup_occupation_spin_embedding():
    input_token_dims = [10, 17, 234, 2]
    output_token_dims = 32  # 32 embedding dimensions

    ose = OccupationSpinEmbedding(
        input_token_dims=input_token_dims,
        output_token_dims=output_token_dims,
        einops_pattern="o sp a j -> (o a j sp)",
    )

    return ose, input_token_dims, output_token_dims


def test_occupation_spin_embedding(setup_occupation_spin_embedding):
    ose, input_token_dims, output_token_dims = setup_occupation_spin_embedding

    seq, batch = (5, 3)
    occupations = torch.randn(seq, batch, *input_token_dims)
    logits = ose(occupations)

    assert logits.shape == (seq, batch, output_token_dims)
