import pytest
from model.position_encoding import PositionEncoding
from model.site_degree_embedding import SiteDegreeEmbedding
from model.param_embedding import SimpleParamEmbedding
from model.token_embedding import OccupationSpinEmbedding
from model.hubbard_deembedding import HubbardDeembedding
from model.sampling import Sampling

import torch.nn as nn
import torch
import einops as ein
import ipdb  # TODO: remove before push


@pytest.fixture()
def sampling_module(request):
    """
    Produces a sampling module with reasonable hyperparameters.
    """
    particle_number = request.param["particle_number"]
    embed_dim = 32
    n_heads = 2
    n_layers = 2
    dim_feedforward = 64
    dropout = 0.1
    activation = "relu"
    max_len = 100
    n_params = 5
    token_dims = (2, 2)  # occupation, spin
    input_token_rearrange = "o sp -> (o sp)"
    wavelen_fact = 1e6

    embedding = SiteDegreeEmbedding(
        n_params=n_params,
        embed_dim=embed_dim,
        input_token_dims=token_dims,
        input_token_rearrange=input_token_rearrange,
        param_embedding=SimpleParamEmbedding,
        token_embedding=OccupationSpinEmbedding,
        position_encoding=PositionEncoding,
        max_len=max_len,
        wavelen_fact=wavelen_fact,
    )

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=n_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )

    logit_norm = nn.LayerNorm(
        embed_dim,
    )

    transformer_encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=n_layers,
        mask_check=True,
        norm=logit_norm,
    )

    deembedding = HubbardDeembedding(
        embed_dim=embed_dim,
        target_token_dims=(2, 2),  # occupation, spin
    )

    sampling_mask = torch.triu(
        torch.ones(
            max_len,
            max_len,
        )
        * -torch.inf,
        diagonal=1,
    )

    sampling = Sampling(
        embed_dim=embed_dim,
        particle_number=particle_number,
        embedding_function=embedding,
        deembedding_function=deembedding,
        transformer_encoder=transformer_encoder,
        mask=sampling_mask,
    )

    return {
        "sampling": sampling,
    }


@pytest.mark.parametrize(
    "sampling_module, function_params",
    [
        ({"particle_number": None}, {"random_seed": 42}),
        ({"particle_number": 4}, {"random_seed": 37}),
    ],
    indirect=["sampling_module"],
)
def test_sample_simple(sampling_module, function_params):
    """
    Ensures sampling without a particle number requirement produces
    chains meeting basic constraints (dimension, , etc.).
    """
    random_seed: int = function_params["random_seed"]

    torch.manual_seed(random_seed)

    tokens = torch.zeros((0, 16, 2, 2))  # (sequence, batch, occupation, spin)
    params = torch.rand((5, 16))  # (n_params, batch)

    sampling = sampling_module["sampling"]

    tokens, _ = sampling.sample(
        params=params,
        tokens=tokens,
        up_to=10,  # Sample up to 10 tokens
    )

    assert tokens.shape == (10, 16, 2, 2), (
        "Sampled tokens should have shape (sequence, batch, occupation, spin)"
    )

    assert torch.all(tokens.sum(dim=-2) == 1), (
        "Sampled tokens are not one-hot across the occupation dimension"
    )
