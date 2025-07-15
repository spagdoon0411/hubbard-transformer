import pytest
from model.model import HubbardWaveFunction

import torch
import einops as ein

import ipdb


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


def test_sampling_model_agreement(model):
    """
    Products of probabilities that appear during sampling should agree
    with products of probs that we obtain by embedding the chain,
    performing attention, and then de-embedding every token in the chain
    to probabilities.
    """

    model: HubbardWaveFunction = model["model"]

    up_to = 10
    batch_size = 32

    params = ein.repeat(
        torch.randn(5),
        "p -> p b",
        b=batch_size,
    )

    # Chains sampled
    token_buf = torch.zeros(up_to, batch_size, 2, 2)  # s b o sp
    log_prob_buf = torch.zeros(up_to, batch_size)  # s b
    raw_prob_buf = torch.zeros(up_to, batch_size, 2)  # s b sp
    for i in range(up_to):
        # Samples a pair of spins, obtains the log probs associated with
        # each spin, and takes the product of probs by summing along the
        # spin axis
        next_token, log_probs, prob_dist = model.sampling._sample_one_more_token(
            params=params,  # type: ignore
            more_tokens=token_buf[:i, :, :],  # type: ignore
            return_raw_probs=True,
        )

        assert next_token.shape == (batch_size, 2, 2), (
            "Next token should have shape (batch, occupation, spin)"
        )

        assert log_probs.shape == (batch_size,), (
            "Log probabilities should have shape (batch,)"
        )

        # index into the next prob buf with the sampled token

        token_buf[i] = next_token
        log_prob_buf[i] = log_probs

        # Gather probs from the occupation that was sampled.
        occ_idx = next_token.argmax(dim=-2)  # (b, sp)
        occ_idx = ein.repeat(
            occ_idx,
            "b sp -> b 1 sp",
        )

        # b o sp -> b sp, indexing using indices along o
        token_prob = ein.rearrange(
            torch.gather(
                prob_dist,
                dim=-2,
                index=occ_idx,
            ),
            "b 1 sp -> b sp",
        )

        raw_prob_buf[i] = token_prob

    # Embeds all of the tokens in a parallelized manner, mapping them
    # to logits, then performs attention updates on all of them together
    # and passes all of the logits through the probability head.

    # Probabilities of the tokens actually sampled
    model_probs, _ = model.forward(
        tokens=token_buf,  # (s b o sp)
        params=params,  # (p b)
    )  # (s b sp)

    log_model_probs = model_probs.log()
    log_model_probs = ein.einsum(
        log_model_probs,
        "s b sp -> s b",
    )

    ipdb.set_trace()

    avg_diff = torch.abs(log_prob_buf - log_model_probs).mean()

    assert torch.allclose(log_prob_buf, log_model_probs, atol=1e-5), (
        f"Log probabilities from sampling and model should agree; average difference is {avg_diff.item()}"
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
