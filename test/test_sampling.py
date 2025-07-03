from model.hamiltonian import HubbardHamiltonian
from model.model import HubbardWaveFunction
import pytest
import einops as ein
import torch


@pytest.fixture
def model_hamiltonian():
    ham = HubbardHamiltonian(t=1.0, U=2.0)

    h_model = HubbardWaveFunction(
        embed_dim=32,
        n_heads=2,
        n_layers=2,
        dim_feedforward=64,
        particle_number=4,
        max_len=10,
    )

    params = torch.tensor(
        [
            1.0,  # t
            2.0,  # U
            32,  # embed dim
            4,  # particle number
            5,  # number of params
        ]
    )

    data = {
        "ham": ham,
        "h_model": h_model,
        "params": params,
    }

    return data


@pytest.fixture()
def some_samples(model_hamiltonian):
    h_model = model_hamiltonian["h_model"]
    params = model_hamiltonian["params"]
    sample_size = 1000

    samples, log_probs = h_model.sample(
        num_chains=sample_size,
        chain_length=4,
        params=params,
    )

    return {
        **model_hamiltonian,
        # "basis_psi": basis_psi,
        # "basis": basis,
        "samples": samples,
        "sample_size": sample_size,
        "log_probs": log_probs,
    }


def test_sampling(some_samples):
    """
    Ensures valid tokens are sampled from a probability distribution.
    Enforces samples of shape (sequence, batch, occupation, spin)
    """

    samples = some_samples["samples"]
    sample_size = some_samples["sample_size"]
    log_probs = some_samples["log_probs"]

    # s b o sp
    assert samples.shape == (4, sample_size, 2, 2), "Sample shape mismatch"

    # Log-probs of sampling each of these particular chains
    assert log_probs.shape == (sample_size,)

    assert torch.all(
        samples.sum(dim=-2) == 1
    ), "Occupation axis doesn't meet one-hot constraint"

    # Verify particle numbers are as expected.
    # Particle numbers are one-hot encoded along the occupation axis.
    particle_numbers = samples.argmax(dim=-2)  # s b sp
    particle_numbers = ein.einsum(particle_numbers, "s b o -> b")
    assert torch.all(particle_numbers == 4)


def test_generate_samples(model_hamiltonian):
    """
    Ensures single rounds of token sampling produce samples
    reflecting the distribution provided.
    """

    h_model = model_hamiltonian["h_model"]

    # Single-token probability distribution
    prob_dist = torch.rand(16, 2, 2)  # (batch, occ, spin)
    prob_dist /= prob_dist.sum(
        dim=-2, keepdim=True
    )  # Normalize over the occupation axis, not the spin axis

    all_samples = torch.zeros((3000, 16, 2, 2))  # Preallocate space for samples
    for i in range(3000):
        samples, log_prob = h_model.sampling._generate_samples(
            prob_dist
        )  # (16, 2, 2), (16, 2)
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

    assert (
        kl_div < 0.01
    ), "KL divergence between sampled and original distribution is too high"
