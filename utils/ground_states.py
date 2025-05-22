from model.model import HubbardWaveFunction
import torch
from typing import Any
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def display_psi(
    wv: HubbardWaveFunction,
    num_sites: int,
    params: torch.Tensor,
) -> tuple[Any, Any]:
    """
    Plots the real and imaginary parts of the wave function, as well
    as phase information.
    """

    sns.set_theme(style="whitegrid")

    psi, basis, norm = wv.compute_basis_information(
        num_sites=num_sites,
        params=params,
    )

    # psi: (s, b, sp)

    real = psi.real.detach().cpu().numpy()
    imag = psi.imag.detach().cpu().numpy()

    phase = torch.angle(psi).detach().cpu().numpy()

    # Generate x-axis values (e.g., indices of the basis states)
    x = np.arange(len(real))

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot real part
    sns.lineplot(ax=axes[0], x=x, y=real, marker="o", label="Real Part", color="blue")
    axes[0].set_xlabel("Basis Index")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Real Part of Wave Function")
    axes[0].legend()

    # Plot imaginary part
    sns.lineplot(
        ax=axes[1], x=x, y=imag, marker="o", label="Imaginary Part", color="orange"
    )
    axes[1].set_xlabel("Basis Index")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Imaginary Part of Wave Function")
    axes[1].legend()

    # Plot phase
    sns.scatterplot(ax=axes[2], x=x, y=phase, label="Phase", color="green", s=50)
    axes[2].set_xlabel("Basis Index")
    axes[2].set_ylabel("Phase (radians)")
    axes[2].set_title("Phase of Wave Function")
    axes[2].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Return the figure and axes objects
    return fig, axes
