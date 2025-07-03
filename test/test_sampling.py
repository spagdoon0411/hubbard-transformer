from model.hamiltonian import HubbardHamiltonian
from model.model import HubbardWaveFunction
import pytest
import einops as ein
import torch
from utils.logging import chains_to_strings, chain_strings_to_integers
import matplotlib.pyplot as plt

SHOW_PLOTS = False


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
    sample_size = 30000

    basis_psi, basis, _ = h_model.compute_basis_information(
        4,
        params,
    )

    samples, log_probs = h_model.sample(
        num_chains=sample_size,
        chain_length=4,
        params=params,
    )

    return {
        **model_hamiltonian,
        "basis_psi": basis_psi,
        "basis": basis,
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


def test_kl_convergence(some_samples):
    """
    The distance between the sampled distribution and the original
    distribution should be small in the limit of many samples.
    """

    # TODO: check that the basis states are in the same order as basis_psi
    # returned here.

    basis = some_samples["basis"]
    basis_psi = some_samples["basis_psi"]
    samples = some_samples["samples"]
    sample_size = some_samples["sample_size"]

    # Should converge to this distribution, from the model
    basis_dist = basis_psi.abs() ** 2  # (s, b, o, sp)

    basis_strs = chains_to_strings(basis)
    samples_strs = chains_to_strings(samples)
    basis_ints = chain_strings_to_integers(basis_strs)
    samples_ints = chain_strings_to_integers(samples_strs)

    # NOTE: basis ints are in descending order

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

    assert torch.all(
        samples_ints_unique == basis_ints
    ), "Sampled basis states do not match the original basis states after sorting"

    kl_div = ein.einsum(
        torch.nn.functional.kl_div(
            samples_dist.log(),
            basis_dist.flatten().log(),
            reduction="none",
            log_target=True,
        ),
        "b -> ",
    )

    if SHOW_PLOTS:
        # Plot samples_dist and basis_dist as bar charts
        plt.bar(
            range(len(samples_ints_unique)),
            samples_dist.detach().numpy(),
            label="Sampled Distribution",
            alpha=0.5,
        )

        plt.bar(
            range(len(basis_ints)),
            basis_dist.detach().numpy().flatten(),
            label="Basis Distribution",
            alpha=0.5,
        )

        plt.title(f"Sampled vs Basis Distribution, KL Divergence: {kl_div.item():.4f}")

        plt.legend()
        plt.show()

    assert (
        kl_div < 0.3
    ), "KL divergence between sampled and original distribution is too high: {:.4f}".format(
        kl_div.item()
    )
