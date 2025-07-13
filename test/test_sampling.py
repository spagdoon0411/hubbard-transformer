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

    sampling_mask = torch.tril(
        torch.ones(
            max_len,
            max_len,
        )
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
def test_generate_samples(sampling_module, function_params):
    """
    Ensures the next tokens produced by
    :func:`Sampling._generate_samples` reflect the single-token
    distribution passed.
    """
    random_seed: int = function_params["random_seed"]

    torch.manual_seed(random_seed)

    sampling: Sampling = sampling_module["sampling"]

    # Single-token probability distribution
    prob_dist = torch.rand(16, 2, 2)  # (batch, occ, spin)
    prob_dist /= prob_dist.sum(
        dim=-2, keepdim=True
    )  # Normalize over the occupation axis, not the spin axis

    all_samples = torch.zeros((3000, 16, 2, 2))  # Preallocate space for samples
    for i in range(3000):
        samples, log_prob = sampling._generate_samples(prob_dist)  # (16, 2, 2), (16, 2)
        all_samples[i] = samples

    # Do samples reflect the probability distributions?
    sum = all_samples.sum(dim=0) / 3000  # (16, 2, 2)

    kl_div = ein.einsum(
        torch.nn.functional.kl_div(
            sum.log(),
            prob_dist.log(),
            reduction="none",
            log_target=True,
        ),  # Computes pointwise KL divergence terms
        "b o sp -> ",
    )

    assert kl_div < 0.01, (
        "KL divergence between sampled and original distribution is too high"
    )


@pytest.mark.parametrize(
    "sampling_module, function_params",
    [
        ({"particle_number": None}, {"random_seed": 42}),
        ({"particle_number": 4}, {"random_seed": 37}),
    ],
    indirect=["sampling_module"],
)
def test_generate_samples_simple(sampling_module, function_params):
    """
    Ensure the samples generated meet basic constraints (e.g.,
    one-hot along the occupation axis).
    """

    random_seed: int = function_params["random_seed"]

    torch.manual_seed(random_seed)

    sampling: Sampling = sampling_module["sampling"]

    # Mock a single-token batched probability distribution
    prob_dist = torch.rand(16, 2, 2)  # (batch, occ, spin)
    prob_dist /= prob_dist.sum(
        dim=-2, keepdim=True
    )  # Normalize over the occupation axis, not the spin axis

    for _ in range(1000):
        samples, log_prob = sampling._generate_samples(prob_dist)

        # Ensure samples are one-hot along the occupation axis
        # The outer loop is necessary to validate this.
        assert torch.all(samples.sum(dim=-2) == 1), (
            "Samples are not one-hot across the occupation dimension"
        )

        # Ensure samples have the correct shape
        assert samples.shape == (16, 2, 2), (
            "Samples should have shape (batch, occupation, spin)"
        )

        # Ensure log_prob has the correct shape
        assert log_prob.shape == (16, 2), (
            "Log probabilities should have shape (batch, spin)"
        )


@pytest.mark.parametrize(
    "sampling_module, function_params",
    [
        ({"particle_number": None}, {"random_seed": 42}),
        ({"particle_number": 4}, {"random_seed": 37}),
    ],
    indirect=["sampling_module"],
)
def test_correct_log_probs(sampling_module, function_params):
    """
    Probs returned from sampling one token agree with probs passed in
    the distribution sampled from.
    """
    random_seed: int = function_params["random_seed"]

    torch.manual_seed(random_seed)

    sampling: Sampling = sampling_module["sampling"]

    # Mock a single-token batched probability distribution
    prob_dist = torch.rand(16, 2, 2)  # (batch, occ, spin)
    prob_dist /= prob_dist.sum(
        dim=-2, keepdim=True
    )  # Normalize over the occupation axis, not the spin axis

    for _ in range(3000):
        samples, log_prob = sampling._generate_samples(prob_dist)

        # Obtain prob entries corresponding to the token that was sampled
        # from the distribution
        sampled_idx = samples.argmax(dim=-2)  # (b sp)
        sampled_idx = ein.repeat(
            sampled_idx,
            "b sp -> b 1 sp",
        )
        probs_from_dist = ein.rearrange(
            torch.gather(
                input=prob_dist,  # (b o sp)
                dim=1,
                index=sampled_idx,  # (b 1 sp) containing occ idx
            ),
            "b 1 sp -> b sp",
        )
        # probs_from_dist[b][o][sp] = prob_dist[b][sampled_idx[b][o][sp]][sp]

        log_prob_from_dist = probs_from_dist.log()

        assert torch.allclose(log_prob_from_dist, log_prob), (
            "Log probs and input distribution probs don't agree for a sampled token"
        )


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
