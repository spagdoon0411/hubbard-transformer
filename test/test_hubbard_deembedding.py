import pytest
import torch
from model.hubbard_deembedding import HubbardDeembedding

# Test parameters
(seq, batch, embed) = (12, 16, 8)
possible_spins = 32
max_occ = 2
test_logits = torch.randn(seq, batch, embed)


@pytest.fixture
def deembedding():
    """
    Fixture to initialize the HubbardDeembedding instance.
    """
    return HubbardDeembedding(
        embed_dim=embed,
        target_token_dims=[max_occ, possible_spins],
    )


def test_prob_dist(deembedding):
    """
    Model outputs a valid probability distribution
    with the correct dimensions and constraints.
    """
    prob_dist = deembedding(test_logits)
    assert prob_dist.shape == (seq, batch, max_occ, possible_spins)

    # Check if the probabilities sum to 1 along the last axis
    assert torch.allclose(prob_dist.sum(dim=-1), torch.ones(seq, batch, max_occ))


def test_single_logit(deembedding):
    """
    Passing a single logit produces the correct output shape.
    """
    prob_dist = deembedding(test_logits[0, :, :])
    assert prob_dist.shape == (batch, max_occ, possible_spins)

    # Check if the probabilities sum to 1 along the last axis
    assert torch.allclose(prob_dist.sum(dim=-1), torch.ones(batch, max_occ))


def test_phases(deembedding):
    """
    Model correctly computes phases and calculates psi.
    """
    prob, phase = deembedding(test_logits, calculate_phase=True)
    assert prob.shape == (seq, batch, max_occ, possible_spins)
    assert phase.shape == (seq, batch, max_occ, possible_spins)

    idx = prob.argmax(dim=-2)  # s b sp
    prob = prob.gather(-2, idx.unsqueeze(-2)).squeeze(-2)  # s b sp
    phase = phase.gather(-2, idx.unsqueeze(-2)).squeeze(-2)  # s b sp
    psi = deembedding.compute_psi(prob, phase)

    assert psi.shape == (seq, batch, possible_spins)
