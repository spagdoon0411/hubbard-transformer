import torch
from model.param_embedding import SimpleParamEmbedding
import pytest

n_params = 10
target_dim = 16
batch = 32


@pytest.fixture
def param_embedding():
    return SimpleParamEmbedding(
        n_params=n_params,
        target_dim=target_dim,
    )


def test_param_embedding(param_embedding):
    # (seq, batch)
    test_params = torch.randn(n_params, batch)
    logits = param_embedding(test_params)
    assert logits.shape == (n_params, batch, target_dim)
