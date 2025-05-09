import pytest
from model.hamiltonian import HubbardHamiltonian
from itertools import product
import torch
import logging
import einops as ein
import matplotlib.pyplot as plt
import functools as ft
import torch.nn.functional as F

HEATMAPS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@pytest.fixture
def simple_ham():
    return HubbardHamiltonian(t=1.0, U=2.0)


def expand_str_chain(chain: str):
    """
    Expands a canonical-ordering binary encoding of a
    basis state into a token tensor of shape (s, 1, o, sp).

    Only accepts single bases for now.
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


# TODO: test two-site hopping explicitly
def test_hopping_1(simple_ham):
    """
    Can we compute hopping terms for simple basis configurations?
    """

    # We can obtain state 2 from state 1 by hopping once.
    str1, str2 = "011110", "110110"
    a, b = expand_str_chain(str1), expand_str_chain(str2)
    h_a_b = simple_ham.term(a, b)

    # <011110|H|110110>
    # <011110|c2u a1u|110110>
    # <011110|c2u a1u c1u c1d c2d c3u|0>
    # <011110|c2u (1 - c1u a1u) c1d c2d c3u|0>
    # <011110|c2u c1d c2d c3u - c2u c1u a1u c1d c2d c3u|0>
    # <011110|c2u c1d c2d c3u|0>
    # - <011110|c1d c2u c2d c3u|0>
    # - <011110|011110>
    # -1

    assert h_a_b.shape == (1, 1), "Hopping Hamiltonian has wrong shape"
    assert h_a_b[0, 0].item() == simple_ham.t, "Hopping Hamiltonian has wrong entry."


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


def exact_hopping(t, n_sites):
    """
    Manual creation of a hopping Hamiltonian as a sum of tensor product'd
    creation and annihilation operators.

    Local number basis: dimension 2

    Occupation basis: dimension 2 * 2 * n_sites
    """

    I_occ_to_spin = torch.eye(2)
    I_spin_to_site = torch.eye(4)

    def reference_list(op: torch.Tensor, I: torch.Tensor, pos: int, num: int):
        """
        Produces a list to tensor-product across to bring an operator from a local
        vector space up to a global vector space.
        """

        l = [I] * num
        l[pos] = op
        return l

    ops = CreationAnnihilation(1)

    @ft.lru_cache(maxsize=None)
    def up_a_space(op: torch.Tensor, I: torch.Tensor, pos: int, num: int):
        r_list = reference_list(op, I, pos, num)
        r_list = circular_shift(r_list, pos)
        res = ft.reduce(torch.kron, r_list)
        return res

    # TODO: be sure cache keys don't involve deep comparisons of tensors

    @ft.lru_cache(maxsize=None)
    def spin_creation(spin_i: int):
        return up_a_space(ops.c, I_occ_to_spin, pos=spin_i, num=2)

    @ft.lru_cache(maxsize=None)
    def spin_annihilation(spin_i: int):
        return up_a_space(ops.a, I_occ_to_spin, pos=spin_i, num=2)

    def hopping_contribution(site_i: int, sp_i: int, site_j: int, sp_j: int):
        """
        The operator describing the hopping contribution between two spins.
        """
        c = spin_creation(sp_i)
        a = spin_annihilation(sp_j)
        c = up_a_space(c, I_spin_to_site, pos=site_i, num=n_sites)
        a = up_a_space(a, I_spin_to_site, pos=site_j, num=n_sites)
        contribution = c @ a
        return contribution

    def site_site_interaction(site_i: int, sp_i: int, site_j: int, sp_j: int):
        i_to_j = hopping_contribution(site_i, sp_i, site_j, sp_j)
        j_to_i = i_to_j.conj().T
        return i_to_j + j_to_i

    def add_interaction_contribution(
        site_i: int,
        sp_i: int,
        site_j: int,
        sp_j: int,
        t: int,
        buf_ref: torch.Tensor,
    ):
        interactions = site_site_interaction(site_i, sp_i, site_j, sp_j)
        buf_ref += -t * interactions

    def interactions():
        for i in range(n_sites):
            for j in range(2):
                print(f"i: {i}, j: {j}")
                yield (i, j, (i + 1) % n_sites, j)

    buf_ref = torch.zeros(
        (2 ** (2 * n_sites), 2 ** (2 * n_sites)),
        dtype=torch.float32,
    )

    # How do we NOT get a power of two?

    for site_i, sp_i, site_j, sp_j in interactions():
        add_interaction_contribution(site_i, sp_i, site_j, sp_j, t, buf_ref)

    return buf_ref


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

    h_a_b = simple_ham.term(a, b)  # (b_a, b_b)
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


def test_three_site_entries(simple_ham):
    """
    Some simple cases testing that batching doesn't explode.
    """

    str1 = spin_occupation_site_basis(3, 2)
    str2 = spin_occupation_site_basis(3, 2)

    a = expand_str_chains(str1)
    b = expand_str_chains(str2)  # (s b o sp)

    h_a_b = simple_ham.term(a, b)  # (b_a, b_b)
    assert str1[1] == "000001" and str2[0] == "000000"
    assert h_a_b[1, 0] == 0.0

    assert str1[1] == "000001" and str2[1] == "000001"
    assert h_a_b[1, 1] == 2.0

    assert str1[1] == "000001" and str2[3] == "000011"
    assert h_a_b[1, 3] == 0.0

    assert str1[20] == "010100" and str2[17] == "010001"
    assert h_a_b[20, 17] == -1.0

    assert str1[50] == "110010" and str2[45] == "101101"
    assert h_a_b[50, 45] == 0.0


def display_heatmap(
    matrix: torch.Tensor,
    y_labels: list[str],
    x_labels: list[str],
    title: str = "Heatmap",
    y_name: str = "Rows",
    x_name: str = "Columns",
):
    """
    Displays a PyTorch matrix as a heatmap grid with custom row and column labels.

    Args:
        matrix (torch.Tensor): The PyTorch matrix to display.
        row_labels (list[str]): Labels for the rows.
        col_labels (list[str]): Labels for the columns.
        title (str): Title of the heatmap.
    """
    if not isinstance(matrix, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    if len(y_labels) != matrix.shape[0]:
        raise ValueError(
            "Number of row labels must match the number of rows in the matrix."
        )
    if len(x_labels) != matrix.shape[1]:
        raise ValueError(
            "Number of column labels must match the number of columns in the matrix."
        )

    # Convert the PyTorch tensor to a NumPy array for plotting
    matrix_np = matrix.numpy()

    # Plot the heatmap
    plt.figure(figsize=(12, 12))
    plt.imshow(matrix_np, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Value")
    plt.title(title)

    # Label rows and columns with custom labels
    plt.xticks(
        ticks=range(matrix_np.shape[1]), labels=x_labels, rotation=45, ha="right"
    )
    plt.yticks(ticks=range(matrix_np.shape[0]), labels=y_labels)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()  # Adjust layout to prevent label overlap
    plt.show()


def main():
    ops = CreationAnnihilation(1)
    print(ops.c)


if __name__ == "__main__":
    main()
