import torch
import einops as ein

from model.hamiltonian import HubbardHamiltonian


def main():
    h = HubbardHamiltonian(t=1.0, U=2.0)

    test_indices = torch.tensor([2, 4, 7])

    print(
        h.anticommutation_mask(
            indices=test_indices,
            target_mask_length=10,
            inclusive=False,
        )
    )


if __name__ == "__main__":
    main()
