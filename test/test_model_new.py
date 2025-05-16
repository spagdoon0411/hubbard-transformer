import pytest
import torch
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
