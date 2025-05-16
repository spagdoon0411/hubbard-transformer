import torch
import pytest

from model.hamiltonian import HubbardHamiltonian


@pytest.mark.parametrize(
    "l_idx, r_idx, target_mask_length, include_left, include_right, expected",
    [
        (
            torch.tensor([3]),
            torch.tensor([5]),
            8,
            True,
            False,
            torch.tensor([[False, False, False, True, True, False, False, False]]),
        ),
        (
            torch.tensor([3]),
            torch.tensor([5]),
            8,
            False,
            False,
            torch.tensor([[False, False, False, False, True, False, False, False]]),
        ),
        (
            torch.tensor([3]),
            torch.tensor([5]),
            8,
            True,
            True,
            torch.tensor([[False, False, False, True, True, True, False, False]]),
        ),
        (
            torch.tensor([0]),
            torch.tensor([1]),
            2,
            True,
            True,
            torch.tensor([[True, True]]),
        ),
        (
            torch.tensor([0]),
            torch.tensor([1]),
            2,
            True,
            False,
            torch.tensor([[True, False]]),
        ),
        (
            torch.tensor([0]),
            torch.tensor([1]),
            2,
            False,
            False,
            torch.tensor([[False, False]]),
        ),
        (
            torch.tensor([0]),
            torch.tensor([1]),
            1,
            False,
            False,
            torch.tensor([[False]]),
        ),
        (torch.tensor([0]), torch.tensor([1]), 1, True, False, torch.tensor([[True]])),
        (
            torch.tensor([2]),
            torch.tensor([2]),
            4,
            True,
            True,
            torch.tensor([[False, False, True, False]]),
        ),
    ],
)
def test_anticommutation_mask(
    l_idx, r_idx, target_mask_length, include_left, include_right, expected
):
    """
    Tests the anticommutation mask function of the Hubbard Hamiltonian.
    """
    # Create a dummy HubbardHamiltonian instance
    hamiltonian = HubbardHamiltonian(t=1.0, U=2.0)

    # Call the anticommutation_mask method
    mask = hamiltonian.two_index_anticommutation_mask(
        l_idx=l_idx,
        r_idx=r_idx,
        target_mask_length=target_mask_length,
        include_left=include_left,
        include_right=include_right,
    )

    # Assert that the returned mask matches the expected mask
    assert torch.equal(mask, expected), f"Expected {expected}, but got {mask}"
