from typing import Literal
import torch
from torch import nn
import numpy as np
import einops as ein
from scipy.sparse.linalg import eigsh


class HubbardHamiltonian(nn.Module):
    def __init__(self, t: float, U: float, **kwargs):
        super(HubbardHamiltonian, self)
        self.t = t
        self.U = U

    def assert_shapes(self, a: torch.Tensor, b: torch.Tensor):
        """
        Defines the shapes that are compatible for hopping-term computations
        """
        a_s, a_b, a_o, a_sp = a.shape
        b_s, b_b, b_o, b_sp = b.shape

        if a_s != b_s:
            raise ValueError(
                f"Basis states must have the same number of sites: {a_s} != {b_s}"
            )
        if a_o != b_o:
            raise ValueError(
                f"Basis states must have the same number of occupations: {a_o} != {b_o}"
            )
        if a_sp != b_sp:
            raise ValueError(
                f"Basis states must have the same number of spins: {a_sp} != {b_sp}"
            )

        return (
            a_s,
            (a_b, b_b),
            a_o,
            a_sp,
        )

    def two_index_anticommutation_mask(
        self,
        l_idx,
        r_idx,
        target_mask_length,
        include_left=True,
        include_right=True,
    ):
        """
        l_idx and r_idx specify left and right bounds for the mask, respectively.
        """

        if l_idx.shape != r_idx.shape:
            raise ValueError(
                f"Left and right indices must have the same shape: {l_idx.shape} != {r_idx.shape}"
            )

        counter = torch.arange(0, target_mask_length)
        counter = ein.rearrange(counter, "i -> 1 i")
        l_idx = ein.rearrange(l_idx, "j -> j 1")
        r_idx = ein.rearrange(r_idx, "j -> j 1")

        left_mask = counter >= l_idx if include_left else counter > l_idx
        right_mask = counter <= r_idx if include_right else counter < r_idx
        mask = left_mask & right_mask
        return mask

    def anticommutation_mask(self, indices, target_mask_length, inclusive=True):
        """
        Pushes a linear index tensor out into a mask over indices greater than or
        equal to the corresponding entry in the input tensor.
        """
        counter = torch.arange(0, target_mask_length)
        counter = ein.rearrange(counter, "i -> 1 i")
        indices = ein.rearrange(indices, "j -> j 1")

        # Inclusive: whether to include the boundary in the masked region
        mask = counter > indices if inclusive else counter >= indices
        return mask

        """
        Find the expected ground state eigenvalue
        """

    def ground_state(
        self,
        basis: torch.Tensor,
        report: bool = False,
    ) -> dict:
        """
        Given a basis, computes the expected ground state energy. Returns the smallest algebraic eigenvalue, with other tangential information.
        """

        a = basis
        b = basis
        entries = self.entry(a, b)
        as_np = entries.cpu().numpy()
        eigvals_SA, eigvecs_SA = eigsh(as_np, k=1, which="SA")
        eigvals_LM, eigvecs_LM = eigsh(as_np, k=1, which="LM")

        if report:
            print(f"Ground state energy (SA): {eigvals_SA}")
            print(f"Ground state energy (LM): {eigvals_LM}")

        return {
            "eigval_SA": eigvals_SA,
            "eigvec_SA": eigvecs_SA,
            "eigval_LM": eigvals_LM,
            "eigvec_LM": eigvecs_LM,
        }

    def count_hops(
        self,
        creation_idx: torch.Tensor,
        annihilation_idx: torch.Tensor,
        creation_operators: torch.Tensor,
    ) -> torch.Tensor:
        """
        creation_idx: (batch, )
            - Index of the particle hole the creation operator is targeting
        annihilation_idx: (batch, )
            - Index of the creation operator the annihilation operator is targeting
        operators: ((s sp), batch)
            - Binary (not bitmap) tensor of site population in canonical order.

        It should not be the case that a creation operator targets a site already
        populated; this function ignores this case. Annihilation is dealt with
        symmetrically.

        Computes the number of sign inversions associated with these hopping target indices,
        using a tensor indicating operators to hop over.
        """

        (b,) = creation_idx.shape

        l_idx = torch.where(
            creation_idx < annihilation_idx,
            creation_idx,
            annihilation_idx,
        )

        r_idx = torch.where(
            creation_idx > annihilation_idx,
            creation_idx,
            annihilation_idx,
        )

        hops_mask = ein.rearrange(
            self.two_index_anticommutation_mask(
                l_idx=l_idx,
                r_idx=r_idx,
                target_mask_length=creation_operators.shape[0],
                include_left=False,
                include_right=False,
            ),
            "b s_sp -> s_sp b",
        )

        hopped_ops = creation_operators.clone()
        hopped_ops[~hops_mask] = 0
        hops = hopped_ops.sum(dim=0)  # (b, )
        return hops

    def flat_tokens_to_string(self, a: torch.Tensor):
        """
        Takes a binary tensor to a string of 0s and 1s
        """
        to_string = lambda x: "".join([str(int(i)) for i in x])
        arr = a.numpy()
        strings = np.apply_along_axis(
            to_string,
            axis=0,
            arr=arr,
        )
        return strings

    def tokens_to_string(self, a: torch.Tensor):
        """
        Takes a chain in s b o sp to a string of 0s and 1s
        (using canonical spin order)
        """
        flattened = ein.rearrange(
            a,
            "s b o sp -> (s sp) b o",
        )
        flattened = flattened.argmax(dim=-1)  # (s sp) b
        return self.flat_tokens_to_string(flattened)

    def entry(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        site_stride: int = 2,
        periodic: bool = False,
    ):
        # Ensure they have the right dimensions
        (seq_dim, (batch_dim, hilbert_dim), occ_dim, spin_dim) = self.assert_shapes(
            a, b
        )

        a = ein.rearrange(a, "s b o sp -> (s sp) b o")
        b = ein.rearrange(b, "s h o sp -> (s sp) h o")

        a_bin = a.argmax(dim=-1)  # (s sp) b
        b_bin = b.argmax(dim=-1)  # (s sp) h

        a_bin = ein.rearrange(a_bin, "s_sp b -> s_sp b 1")
        b_bin = ein.rearrange(b_bin, "s_sp h -> s_sp 1 h")

        # Buffer to populate with Hamiltonian values
        entries = torch.zeros(batch_dim, hilbert_dim)

        diffs = a_bin - b_bin  # (s sp) b h

        # a - b on the right paired with a - b on the left
        diffs_right = torch.roll(
            diffs,
            shifts=-site_stride,
            dims=0,
        )  # (s sp) b h

        # TODO: handle a length-two chain?
        if not periodic and diffs_right.shape[0] >= 2:
            diffs_right[[-2, -1], :, :] = 0

        connections = (diffs == 1) & (diffs_right == -1)  # (s sp) b h
        connections |= (diffs == -1) & (diffs_right == 1)  # (s sp) b h
        one_away = (connections.sum(dim=0) == 1) & (diffs.abs().sum(dim=0) == 2)  # b h

        # Debug individual pairs
        # l = "111010"
        # r = "101011"
        # if (
        #     self.flat_tokens_to_string(a_bin[:, 0]) == l
        #     and self.flat_tokens_to_string(b_bin[:, 0]) == r
        # ):
        #     ipdb.set_trace()

        diagonals = diffs.abs().sum(dim=0) == 0
        entries[diagonals] = self.U

        # Chains that are one away
        diffs_connected = diffs[:, one_away]  # (s sp) num_connected
        creation_idx = diffs_connected.argmax(dim=0)  # num_connected
        annihilation_idx = diffs_connected.argmin(dim=0)  # num_connected

        _, h_idx = torch.nonzero(
            one_away,
            as_tuple=True,
        )

        creation_operators = ein.rearrange(
            b_bin[:, :, h_idx],
            "s_sp 1 h -> s_sp h",
        )

        # TODO: operators is the number of states hopped
        num_hops = self.count_hops(
            creation_idx=creation_idx,
            annihilation_idx=annihilation_idx,
            creation_operators=creation_operators,
        )

        entries[one_away] = torch.where(
            num_hops % 2 == 0,
            -self.t,
            self.t,
        )

        diagonals = diffs.abs().sum(dim=0) == 0  # b h
        _, h_idx = torch.nonzero(
            diagonals,
            as_tuple=True,
        )

        diagonal_kets = ein.rearrange(
            b_bin[:, :, h_idx],
            "s_sp 1 num_diags -> s_sp num_diags",
        )

        even_alignment = torch.arange(0, seq_dim * spin_dim) % 2 == 0  # s sp
        even_alignment = ein.rearrange(
            even_alignment,
            "s_sp -> s_sp 1",
        )

        immediate_right = diagonal_kets.roll(
            shifts=-1,  # Finding paired spins
            dims=0,
        )  # (s sp) num_diags

        if not periodic and immediate_right.shape[0] >= 1:
            immediate_right[[-1], :] = 0

        pairs = (diagonal_kets == 1) & (immediate_right == 1)  # (s sp) num_diags
        pairs &= even_alignment  # (s sp) num_diags
        num_pairs = pairs.sum(dim=0)  # num_diags
        num_pairs = num_pairs.to(torch.float32)

        entries[diagonals] = self.U * num_pairs  # num_diags <- num_diags

        return entries
