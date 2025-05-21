import pytest
import torch
import einops as ein
import math
from model.hamiltonian import HubbardHamiltonian
from model.model import HubbardWaveFunction
from utils.dummy_data import create_occupations, create_uniform_params, create_params


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

    data = {
        "ham": ham,
        "h_model": h_model,
    }

    return data


@pytest.fixture
def sample_empty(model_hamiltonian):
    model_hamiltonian["params"] = create_uniform_params(n_params=5, b=16)
    model_hamiltonian["occupations"] = create_occupations(s=0, b=16, sp=2, o=2)
    return model_hamiltonian


@pytest.fixture
def sample_partial(model_hamiltonian):
    model_hamiltonian["params"] = create_uniform_params(n_params=5, b=16)
    model_hamiltonian["occupations"] = create_occupations(s=3, b=16, sp=2, o=2)
    return model_hamiltonian


def test_e_loc_simple(sample_partial):
    h_model = sample_partial["h_model"]
    ham = sample_partial["ham"]
    occupations = sample_partial["occupations"]
    params = create_params(n_params=5)

    e_loc = h_model.e_loc(
        hamiltonian=ham,
        params=params,
        sampled_states=occupations,
    )


def test_partial_psi(sample_partial):
    h_model = sample_partial["h_model"]
    params = sample_partial["params"]
    occupations = sample_partial["occupations"]

    prob, phase = h_model(params, occupations)

    assert prob.shape == (3, 16, 2), "Should output one probability per spin per site"
    assert phase.shape == (3, 16, 2), "Should output one phase per spin per site"
    assert torch.all(prob >= 0), "Probabilities should be geq 0"
    assert torch.all(prob <= 1), "Probabilities should be leq 1"

    assert torch.all(phase >= -torch.pi)
    assert torch.all(phase <= torch.pi)


def test_basis_generation(model_hamiltonian):
    h_model = model_hamiltonian["h_model"]

    num_sites = 4  # So 8 entries in the binary representation
    particle_num = 4
    params = create_params(n_params=5)

    psi, basis = h_model.compute_basis_information(
        num_sites=num_sites,
        params=params,
    )

    canonical_bin = ein.rearrange(
        basis,
        "s b o sp -> (s sp) b o",
    ).argmax(dim=-1)

    particle_counts = ein.einsum(canonical_bin, "s b -> b")

    # Each chain has the right number of particles
    assert torch.all(
        particle_counts == h_model.particle_number
    ), "Particle counts don't match counts of model"

    # We have the right number of chains
    batch = basis.shape[1]
    assert batch == math.comb(
        num_sites * 2, particle_num
    ), f"Batch size should be equal to number of ways to choose {num_sites} from {num_sites * 2}"

    # All of the chains are unique
    for i in range(batch):
        for j in range(i + 1, batch):
            assert not torch.all(
                basis[:, i] == basis[:, j]
            ), "Basis states were not unique"
