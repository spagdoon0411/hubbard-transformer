import pytest
from model.model import HubbardWaveFunction
from utils.dummy_data import create_occupations, create_uniform_params
import numpy as np
import random

import torch
import einops as ein

# Display sampling distributions
DISPLAY_PLOTS = False


@pytest.fixture()
def model(request):
    """
    A full HubbardWaveFunction model that samples from the grand canonical
    distribution.
    """

    n_heads = 2
    embed_dim = 32
    n_layers = 2
    dim_feedforward = 64
    particle_number = None
    max_len = 100
    diag = {}

    model = HubbardWaveFunction(
        embed_dim=embed_dim,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        particle_number=particle_number,
        max_len=max_len,
        n_heads=n_heads,
        diag=diag,
    )

    return {
        "model": model,
    }


def test_autoregressive(model):
    """
    Ensures the model meets autoregressive properties: the probability of
    sampling a token at position i depends only on the tokens prior to i.
    """

    model = model["model"]
    occupations = create_occupations(s=10, b=32, sp=2, o=2)
    params = create_uniform_params(n_params=5, b=32)

    partial_occ = occupations[:7, :, :]

    full_mask = model.encoder_mask[: 5 + 10, : 5 + 10]
    partial_mask = model.encoder_mask[: 5 + 7, : 5 + 7]

    model.eval()  # Necessary because of dropout random state

    # Embedding is independent of logits
    full_logits = model.embedding(params=params, tokens=occupations)
    partial_logits = model.embedding(params=params, tokens=partial_occ)
    assert torch.allclose(full_logits[: 5 + 7, :, :], partial_logits), (
        "Embedding produces different logits for full and partial sequences"
    )

    # Attention updates preserve causality constraints
    full_logits = model.transformer_encoder(full_logits, mask=full_mask)
    partial_logits = model.transformer_encoder(partial_logits, mask=partial_mask)
    assert torch.allclose(full_logits[: 5 + 7, :, :], partial_logits), (
        "Transformer encoder produces different logits for full and partial sequences"
    )

    # Deembedding produces the same probabilities
    full_prob = model.deembedding(full_logits, calculate_phase=False)
    partial_prob = model.deembedding(partial_logits, calculate_phase=False)
    assert torch.allclose(full_prob[: 5 + 7, :, :], partial_prob), (
        "Deembedding produces different probabilities for full and partial sequences"
    )


def test_single_prob_agreement(model):
    """
    Ensures that the probabilities resulting from de-embedding a final logit
    independently of the previous tokens agree with the probability the final
    logit would have if it were in a sequence with the previous tokens.
    """

    model = model["model"]
    model.eval()  # Necessary because of dropout random state

    tokens = create_occupations(s=5, b=1, o=2, sp=2)  # (s, b, o, sp)
    parameters = create_uniform_params(n_params=5, b=1)  # (n_params, b)

    mask_size = parameters.shape[0] + tokens.shape[0]
    mask = model.encoder_mask[:mask_size, :mask_size]

    # Embed the entire chain and compute probabilities
    logits = model.embedding(params=parameters, tokens=tokens)
    logits = model.transformer_encoder(logits, mask=mask)

    # De-embed all logits
    prob = model.deembedding(logits, calculate_phase=False)

    # De-embed just the last logit
    single_prob = model.deembedding(logits[-1, :, :], calculate_phase=False)

    assert torch.allclose(prob[-1, :, :], single_prob), (
        "Probabilities from deembedding do not agree with independent deembedding"
    )


@pytest.mark.parametrize(
    "seed", [42, 65, 123, 456, 789, 101112, 131415, 161718, 192021, 222324]
)
def test_sampling_model_agree_single(model, seed):
    """
    Ensures the probabilities that appear when producing a single token
    agree with the probabilities that appear for the token when the token
    is embedded in the full chain.

    Tests with multiple seeds to check for consistency across different
    random initializations.
    """

    torch.manual_seed(seed)  # For reproducibility
    model = model["model"]
    model.eval()

    occupations = create_occupations(s=5, b=1, o=2, sp=2)  # (s, b, o, sp)
    parameters = create_uniform_params(n_params=5, b=1)  # (n_params, b)

    # (s b o sp), (b sp)
    next_token, log_prob = model.sampling._sample_one_more_token(
        params=parameters,
        more_tokens=occupations,
    )
    prob_sampling = log_prob.exp()

    # Obtain model probabilities
    logits = model.embedding(params=parameters, tokens=occupations)
    mask_size = parameters.shape[0] + occupations.shape[0]
    mask = model.encoder_mask[:mask_size, :mask_size]
    logits = model.transformer_encoder(logits, mask=mask)
    prob_dist = model.deembedding(logits, calculate_phase=False)[-1, :, :]  # (b, sp, o)

    # prob_model[b][sp][i] = prob_dist[b][sp][index[i]] for i = 1
    prob_model = torch.gather(
        prob_dist,  # (b sp o)
        dim=-1,
        index=next_token.argmax(-2).unsqueeze(-1),  # (b, sp, 1)
    ).squeeze(-1)  # (b, sp)

    try:
        assert torch.allclose(prob_sampling, prob_model), (
            f"Probabilities from sampling do not agree with probabilities "
            f"from embedding for seed {seed}"
        )
    except AssertionError as e:
        print(f"FAILED: Seed {seed} - {e}")
        raise


@pytest.mark.parametrize(
    "seed", [42, 65, 123, 456, 789, 101112, 131415, 161718, 192021, 222324]
)
def test_sampling_model_agree_forward(model, seed):
    """
    Ensures the probabilities that appear when producing a single token
    agree with the probabilities that appear for the token when the token
    is embedded in the full chain.
    """

    torch.manual_seed(seed)  # For reproducibility

    model = model["model"]
    model.eval()

    occupations = create_occupations(s=5, b=1, o=2, sp=2)  # (s, b, o, sp)
    parameters = create_uniform_params(n_params=5, b=1)  # (n_params, b)

    # (s b o sp), (b sp)
    next_token, log_prob = model.sampling._sample_one_more_token(
        params=parameters,
        more_tokens=occupations,
    )
    prob_sampling = log_prob.exp()
    occupations = torch.cat(
        (occupations, next_token.unsqueeze(0)),  # Add the sampled token
        dim=0,
    )

    prob_model, _ = model.forward(params=parameters, tokens=occupations)
    prob_model = prob_model[-1, :, :]  # (b, sp, o)

    assert torch.allclose(prob_sampling, prob_model), (
        "Probabilities from sampling do not agree with probabilities from embedding"
    )


def test_full_sampling_dist_converges():
    """
    Ensures that sampling a large number of chains from the transformer
    produces a distribution that converges to the expected distribution,
    measured with a KL-div threshold.
    """
    pass


def test_sampling_embedding_prob_agreement():
    """
    Determine whether the probabilities that appear as tokens are sampled
    produce a chain probability that agrees with the probabilty we
    would obtain if we embedded the whole chain using the embedding-
    attention-dembedding flow.
    """
    pass


@pytest.mark.parametrize("seed", [13, 37, 40123])
# @pytest.mark.skip(reason="Long test that should be run manually")
def test_sampled_vs_wavefunction_distribution(model, seed):
    """
    Tests whether sampled chains converge to the wave function distribution.

    Samples many chains from the model and compares the empirical distribution
    to the calculated wave function probabilities.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model["model"]
    model.eval()

    # Parameters for the test
    num_sites = 4
    sample_size = 10000
    params = create_uniform_params(n_params=5, b=1).squeeze(-1)  # (n_params,)

    # Calculate the wave function distribution
    basis_psi, basis, _ = model.compute_basis_information(
        num_sites=num_sites,
        params=params,
    )

    # Should converge to this distribution, from the model
    basis_dist = basis_psi.abs() ** 2  # (s, b, sp)

    # Sample chains from the model
    samples, log_probs = model.sample(
        num_chains=sample_size,
        chain_length=num_sites,
        params=params,
    )

    # Convert samples to string representation for comparison
    from utils.logging import chains_to_strings, chain_strings_to_integers

    basis_strs = chains_to_strings(basis)
    samples_strs = chains_to_strings(samples)
    basis_ints = chain_strings_to_integers(basis_strs)
    samples_ints = chain_strings_to_integers(samples_strs)

    samples_ints_unique, counts = samples_ints.unique(
        return_counts=True
    )  # (n_unique,), (n_unique,)

    # Add missing basis states to the sample basis state counts as zeros
    for i in range(basis_ints.shape[0]):
        if basis_ints[i] not in samples_ints_unique:
            samples_ints_unique = torch.cat(
                (samples_ints_unique, torch.tensor([basis_ints[i]]))
            )
            counts = torch.cat((counts, torch.tensor([0])))

    rev_sort = torch.argsort(samples_ints_unique, descending=True)
    samples_ints_unique = samples_ints_unique[rev_sort]
    counts = counts[rev_sort]

    samples_dist = counts / sample_size

    assert torch.all(samples_ints_unique == basis_ints), (
        f"Sampled basis states do not match the original basis states "
        f"after sorting for seed {seed}"
    )

    # Calculate KL divergence between sampled and wave function distributions
    kl_div = ein.einsum(
        torch.nn.functional.kl_div(
            samples_dist.log(),
            basis_dist.flatten().log(),
            reduction="none",
            log_target=True,
        ),
        "b -> ",
    )

    if DISPLAY_PLOTS:
        import matplotlib.pyplot as plt

        plt.bar(
            range(len(samples_ints_unique)),
            samples_dist.detach().numpy(),
            label="Sampled Distribution",
            alpha=0.5,
        )
        plt.bar(
            range(len(basis_ints)),
            basis_dist.detach().numpy().flatten(),
            label="Wave Function Distribution",
            alpha=0.5,
        )
        plt.title(
            f"Sampled vs Wave Function Distribution, KL Divergence: {kl_div.item():.4f}"
        )
        plt.legend()
        plt.show()
