from model.hamiltonian import HubbardHamiltonian
from utils.logging import display_heatmap, chains_to_strings
from numpy.linalg import eigh
import torch

DISPLAY_HEATMAP = False


def exact_diagonalize(ham: HubbardHamiltonian, basis: torch.Tensor) -> tuple:
    """
    Materializes the Hamiltonian over the s b sp o basis provided and
    diagonalizes it, returning eigenvalues and eigenvectors in ascending algebraic
    order (so that the first eigenvalue is the ground state energy)
    """

    a = basis
    b = basis

    h_a_b = ham.entry(a, b, periodic=False)
    h_a_b_np = h_a_b.detach().cpu().numpy()
    eigenvalues, eigenvectors = eigh(h_a_b_np)

    if DISPLAY_HEATMAP:
        display_heatmap(
            h_a_b,
            title="Hamiltonian Matrix",
            x_labels=chains_to_strings(a),
            y_labels=chains_to_strings(b),
        )

    return eigenvalues, eigenvectors, h_a_b
