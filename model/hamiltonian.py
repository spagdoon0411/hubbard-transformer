import torch
from torch import nn
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

    def anticommutation_mask(self, indices, target_mask_length, inclusive=True):
        """
        Pushes a linear index tensor out into a mask over indices greater than or
        equal to the corresponding entry in the input tensor.
        """
        counter = torch.arange(0, target_mask_length)
        counter = ein.rearrange(counter, "i -> 1 i")
        indices = ein.rearrange(indices, "j -> j 1")
        mask = counter >= indices if inclusive else counter > indices
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
        entries = self.term(a, b)
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

    def term(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        site_stride: int = 2,
    ):
        """
        Takes two basis states, possibly batched, and computes their Hamiltonian entry.
        Returns a batched vector of entries corresponding to the basis states.

        a: (s b o sp) - Arguments to E_loc
        b: (s b o sp) - Sites to use as basis states
        site_stride - When flattened and occupations argmax'd, how many entries correspond to each site?
        w: TODO: estimated weights to use for basis states in E_loc calculations
        """

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

        entries = torch.zeros(batch_dim, hilbert_dim)

        # TODO: can we make this roll less memory inefficient?

        # This is the difference from the basis state to the argument
        diffs = a_bin - b_bin  # (s sp) b h
        diffs_abs = diffs.abs().sum(dim=0)  # b h
        entries[diffs_abs == 0] = self.U

        # Matches the entries in pairing with the entries to their right
        diffs_right = diffs.roll(shifts=-site_stride, dims=0)  # (s sp) b h

        right_hops = (diffs_right == 1) & (diffs == -1)  # (s sp) b h, rightward hopping
        left_hops = (diffs_right == -1) & (diffs == 1)  # (s sp) b h, leftward hopping
        single_right_hops = right_hops.sum(dim=0) == 1  # b h
        single_left_hops = left_hops.sum(dim=0) == 1  # b h
        right_hops &= single_right_hops.unsqueeze(0)
        left_hops &= single_left_hops.unsqueeze(0)  # (s sp) b h

        one_away = diffs_abs == 2  # b h
        right_hops &= one_away.unsqueeze(0)
        left_hops &= one_away.unsqueeze(0)

        # Obtains hopping sequence indices
        s_idx, b_idx, h_idx = torch.nonzero(
            right_hops | left_hops,  # can't have both rightward and leftward hopping
            as_tuple=True,
        )  # (num_linked)

        # Zero out operators not involved in hopping
        hopped_operators_antimask = self.anticommutation_mask(
            indices=s_idx, target_mask_length=seq_dim * spin_dim, inclusive=True
        )  # (num_linked, (s sp))

        hopped_operators_antimask = ein.rearrange(
            hopped_operators_antimask,
            "j s_sp -> s_sp j",
        )

        # This selects the states that participate in hopping
        relevant_states = b_bin[:, :, h_idx]
        relevant_states = ein.rearrange(
            relevant_states,
            "s_sp 1 h -> s_sp h",
        )

        right_hops_new_space = single_right_hops[b_idx, h_idx]
        relevant_states[hopped_operators_antimask] = 0

        # For leftward hops the creation and annihilation operator make the same number of
        # anticommutations. Rightward hops involve one more
        num_hops = 2 * relevant_states.sum(dim=0)  # num_linked

        parity = num_hops % 2  # (num_linked)
        entries[b_idx, h_idx] = torch.where(parity == 0, -self.t, self.t)
        return entries
