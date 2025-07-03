import pytest
from model.hamiltonian import HubbardHamiltonian
from itertools import product
import torch
import logging
import einops as ein
from utils.logging import display_heatmap
import torch.nn.functional as F

HEATMAPS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@pytest.fixture
def simple_ham():
    return HubbardHamiltonian(t=1.0, U=2.0)


def test_hermitian(simple_ham):
    basis = spin_occupation_site_basis(3, 2)
    basis = expand_str_chains(basis)
    h_a_b = simple_ham.entry(basis, basis)

    assert torch.all(
        h_a_b.transpose(0, 1).conj().isclose(h_a_b)
    ), "Hamiltonian is not Hermitian."


def expand_str_chain(chain: str):
    """
    Expands a canonical-ordering binary encoding of a
    basis state into a token tensor of shape (s, 1, o, sp).
    """

    if len(chain) % 2 != 0:
        raise ValueError(
            "Chain length must be even for canonical ordering of spin-1/2 particles."
        )

    s = len(chain) // 2
    o = 2
    sp = 2

    # (s sp)
    tens = torch.tensor([int(x) for x in chain])

    # (s sp) o
    tens = F.one_hot(tens, num_classes=o)

    # Unflatten the s and sp dimensions
    tens = ein.rearrange(
        tens,
        "(s sp) o -> s 1 o sp",
        s=s,
        sp=sp,
        o=o,
    )

    assert tens.shape == (s, 1, o, sp)

    return tens


def expand_str_chains(ls: list[str]):
    B_DIM = 1
    tens = [expand_str_chain(s) for s in ls]
    tens = torch.concat(tens, B_DIM)
    return tens  # (s b o sp)


def display_test_tokens(test_str: str):
    tokens = expand_str_chain(test_str)
    print(f"Tokens for {test_str}:")
    print(tokens)


class CreationAnnihilation:
    def __init__(self, max_occs: int):
        d = torch.arange(1, max_occs + 1)
        self.c = torch.diag(d, diagonal=-1)
        self.a = torch.diag(d, diagonal=1)


def circular_shift(lst, shift_by):
    if not lst:
        return lst
    shift_by %= len(lst)  # Handle shifts greater than the list length
    return lst[-shift_by:] + lst[:-shift_by]


def spin_occupation_site_basis(n_sites: int, n_spins: int):
    """
    Generates a spin-occupation-site basis in canonical order.

    Args:
        n_sites (int): Number of sites.
        n_spins (int): Number of spins.

    Returns:
        list[str]: Basis states in canonical order.
    """
    # Total number of states (sites * spins)
    total_states = n_sites * n_spins

    # Generate all possible combinations of 0s and 1s for the total states
    basis = ["".join(config) for config in product("01", repeat=total_states)]

    return basis


def binary_number_basis(n_sites: int):
    """
    Generates a binary number basis for n_sites.
    """

    # Generate all possible combinations of 0s and 1s
    basis = [bin(i)[2:].zfill(n_sites) for i in range(2**n_sites)]
    return basis


def test_batching(simple_ham):
    """
    Some simple cases testing that batching doesn't explode.
    """

    str1 = spin_occupation_site_basis(3, 2)
    str2 = spin_occupation_site_basis(3, 2)

    a = expand_str_chains(str1)
    b = expand_str_chains(str2)  # (s b o sp)

    h_a_b = simple_ham.entry(a, b)  # (b_a, b_b)
    if HEATMAPS:
        display_heatmap(
            h_a_b,
            y_labels=str1,
            x_labels=str2,
            title="Hopping Hamiltonian",
            y_name="Row basis",
            x_name="Column basis",
        )

    assert h_a_b.shape == (len(str1), len(str2)), "Hopping Hamiltonian has wrong shape"


@pytest.mark.parametrize(
    "str1, str2, expected",
    [
        ("0000", "0000", 0.0),
        ("0001", "0000", 0.0),
        ("0001", "0001", 0.0),
        ("0010", "0001", 0.0),
        ("0100", "0001", -1.0),
        ("1000", "0010", -1.0),
        ("0010", "1000", -1.0),
        ("0010", "0010", 0.0),
        ("0011", "0011", 2.0),
        ("0100", "0010", 0.0),
        ("0101", "0101", 0.0),
    ],
)
def test_two_site_entries(simple_ham, str1, str2, expected):
    """
    Parameterized test for two-site Hamiltonian entries.
    """
    # Expand the string chains
    a = expand_str_chains([str1])
    b = expand_str_chains([str2])  # (s b o sp)

    # Compute the Hamiltonian terms
    h_a_b = simple_ham.entry(a, b)  # (b_a, b_b)

    # Validate the expected value
    if expected == "U":
        assert (
            h_a_b[0, 0] == simple_ham.U
        ), f"Expected h_a_b[0, 0] to be {simple_ham.U}."
    else:
        assert h_a_b[0, 0] == expected, f"Expected h_a_b[0, 0] to be {expected}."


@pytest.mark.parametrize(
    "str1, str2, expected",
    [
        ("000000", "000000", 0.0),
        ("000001", "000000", 0.0),
        ("000001", "000001", 0.0),
        ("000001", "000011", 0.0),
        ("010100", "010001", -1.0),
        ("010001", "010100", -1.0),
        ("110010", "101101", 0.0),
        ("110010", "110010", 2.0),
        ("000001", "000000", 0.0),
        ("000001", "000001", 0.0),
        ("111110", "000111", 0.0),
        ("111110", "111011", 1.0),
        ("111011", "111110", 1.0),
        ("011100", "011001", -1.0),
        ("011110", "110110", 1.0),
        ("001100", "001100", 2.0),
        ("111100", "111100", 4.0),
        ("111101", "111101", 4.0),
    ],
)
def test_three_site_entries(simple_ham, str1, str2, expected):
    """
    Parameterized test for simple cases testing that batching doesn't explode.
    """

    # Expand the string chains
    a = expand_str_chains([str1])
    b = expand_str_chains([str2])  # (s b o sp)

    # Compute the Hamiltonian terms
    h_a_b = simple_ham.entry(a, b)  # (b_a, b_b)

    # Validate the expected value
    if expected == "U":
        assert (
            h_a_b[0, 0] == simple_ham.U
        ), f"Expected h_a_b[0, 0] to be {simple_ham.U}."
    else:
        assert h_a_b[0, 0] == expected, f"Expected h_a_b[0, 0] to be {expected}."


def test_batched_entries(simple_ham):
    """
    Tests computation of batched entries.
    """

    # Generate a batch of basis states
    str1 = spin_occupation_site_basis(3, 2)
    str2 = spin_occupation_site_basis(3, 2)

    # Expand the string chains
    a = expand_str_chains(str1)
    b = expand_str_chains(str2)  # (s b o sp)

    # Compute the Hamiltonian terms
    h_a_b = simple_ham.entry(a, b)  # (b_a, b_b)

    # Validate the shape of the result
    assert h_a_b.shape == (len(str1), len(str2)), "Hopping Hamiltonian has wrong shape"


@pytest.mark.parametrize(
    "i_c, i_a, hopped_operators, expected",
    [
        (
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([[0, 1, 1]]).T,
            torch.tensor([0]),
        ),
        (
            torch.tensor([0]),
            torch.tensor([2]),
            torch.tensor([[0, 1, 1]]).T,
            torch.tensor([1]),
        ),
        (
            torch.tensor([0]),
            torch.tensor([2]),
            torch.tensor([[0, 0, 1]]).T,
            torch.tensor([0]),
        ),
        (
            torch.tensor([0]),
            torch.tensor([5]),
            torch.tensor([[0, 0, 1, 1, 0, 1]]).T,
            torch.tensor([2]),
        ),
        (
            torch.tensor([1]),
            torch.tensor([5]),
            torch.tensor([[1, 0, 1, 1, 0, 1]]).T,
            torch.tensor([2]),
        ),
        (
            torch.tensor([5]),
            torch.tensor([1]),
            torch.tensor([[1, 0, 1, 1, 0, 1]]).T,
            torch.tensor([2]),
        ),
    ],
)
def test_hop_counting(i_c, i_a, hopped_operators, expected):
    """
    Counting of creation/annihilation sign flips (not hops).
    """
    ham = HubbardHamiltonian(i_c, i_a)
    assert ham.count_hops(i_c, i_a, hopped_operators) == expected


@pytest.mark.parametrize(
    "str1, str2, expected, U",
    [
        ("000000", "000000", 0.0, 2.0),
        ("110000", "110000", 2.0, 2.0),
        ("111100", "111100", 4.0, 2.0),
        ("001100", "001100", 2.0, 2.0),
        ("000011", "000011", 2.0, 2.0),
        ("000000", "000000", 0.0, 3.0),
        ("110000", "110000", 3.0, 3.0),
        ("111100", "111100", 6.0, 3.0),
        ("001100", "001100", 3.0, 3.0),
        ("000011", "000011", 3.0, 3.0),
    ],
)
def test_diagonal(str1: str, str2: str, expected: float, U: float):
    ham = HubbardHamiltonian(t=1.0, U=U)
    assert (
        ham.entry(
            expand_str_chain(str1),
            expand_str_chain(str2),
        )[0, 0]
        == expected
    ), f"Expected {expected} but got {ham.entry(expand_str_chain(str1), expand_str_chain(str2))[0, 0]}"
