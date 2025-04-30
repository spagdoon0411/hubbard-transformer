import torch
from torch import nn
from torchtyping import TensorType
from position_encoding import PositionEncoding
import einops as ein

# TODO: de-embedding should be tied to embedding and should be
# probabilistic.


class GenericParamEncoding(nn.Module):
    """
    Creates a parameter embedding for a TQS model, taking a list of
    physical parameters to do this. Gives the parameters their
    own dimension, then performs a linear combination between
    them using trainable ratios.

    TODO: should there be more degrees of freedom here? Should
    there be fewer?

    TODO: allow a mapping to a parameter embedding space that's
    higher in dimension than the plain number of params.
    """

    def __init__(
        self, n_params: int, target_dim: int, dtype: torch.dtype = torch.complex64
    ):
        super(GenericParamEncoding, self).__init__()
        # TODO: the interaction between parameters probably should not be captured
        # linearly. Maybe use a few small rounds of self-attention?

        self.param_dims = n_params
        self.interaction_weights = nn.Parameter(
            torch.randn(n_params, n_params, dtype=dtype)
        )
        self.n_param_to_target = nn.Parameter(
            torch.randn(target_dim, n_params, dtype=dtype)
        )

    def forward(self, params: TensorType["n_params", "batch"]):
        """
        Returns: TensorType["n_params", "batch", "embed_params"]
        """
        # Arrange parameter list along diag

        # (n_params, batch) -> (n_params, n_params, batch)
        res = torch.diag_embed(params.transpose(0, 1), offset=0, dim1=0, dim2=1)

        # (e p2) @ (p1 p2 b) -> (p1 e b)
        res = torch.matmul(self.n_param_to_target, res)
        res = ein.rearrange(res, "p e b -> p b e")

        return res


class UniformOccupationEmbedding(nn.Module):
    """
    Embeds tokens corresponding to occupations in a spin-occupation model
    to produce logits. Takes a tensor of dimensions

    (seq, batch, occupations, spins)

    that's one-hot across the occupation dimension.
    """

    def __init__(
        self,
        possible_occupations: int,
        possible_spin_states: int,
        token_dims: int,
        max_len: int,
        wavelen_fact: int = 1e6,
        dtype: torch.dtype = torch.complex64,
    ):
        super(UniformOccupationEmbedding, self).__init__()
        self.possible_occupations = possible_occupations
        self.possible_spin_states = possible_spin_states
        self.token_dims = token_dims

        self.pe = PositionEncoding(
            d_model=token_dims,
            max_len=max_len,
            wavelen_fact=wavelen_fact,
            dtype=dtype,
        )

        # TODO: include some weight matrix here before they're treated
        # as logits?
        self.occ_to_token = nn.Parameter(
            torch.randn(
                token_dims,
                possible_occupations * possible_spin_states,
                dtype=dtype,
            )
        ).unsqueeze(0)

        self.occ_to_token.to(dtype=dtype)

    def forward(self, occupations: TensorType["seq", "batch", "occupations", "spins"]):
        # TODO: inefficient?
        # TODO: invariant under switching of s and b in axis order?
        buf = ein.rearrange(occupations, "s b o sp -> s (o sp) b")
        buf = torch.matmul(self.occ_to_token, buf)
        buf = ein.rearrange(buf, "s e b -> s b e")
        return buf


class HubbardEmbedding(nn.Module):
    """
    Translates a sequence of occupations describing a partial Hubbard
    lattice and produces logits to be put through rounds of self-attention.

    embed = token_dims + param_dims

    Produces a buffer of shape (seq, batch, embed)
    """

    def __init__(
        self,
        token_dims: int,
        param_dims: int,
        n_params: int,
        max_len: int,
        possible_occupations: int,
        possible_spin_states: int,
        wavelen_fact=1e6,
        dtype: torch.dtype = torch.complex64,
    ):
        super(HubbardEmbedding, self).__init__()
        self.token_dims = token_dims
        self.param_dims = param_dims
        self.occ_embedding = UniformOccupationEmbedding(
            possible_occupations=possible_occupations,
            possible_spin_states=possible_spin_states,
            token_dims=token_dims,
            max_len=max_len,
            wavelen_fact=wavelen_fact,
            dtype=dtype,
        )

        self.param_embedding = GenericParamEncoding(
            n_params=n_params,
            target_dim=param_dims,
            dtype=dtype,
        )

    def forward(
        self,
        occupations: TensorType["seq", "batch", "occupations", "spins"],
        params: TensorType["n_params", "batch"],
    ):
        """
        Returns: TensorType["seq", "batch", "token_dims + param_dims]"]
        """

        occ = self.occ_embedding(occupations)
        param = self.param_embedding(params)

        n_params = params.size(0)
        n_occs = occupations.size(0)
        batch_occ = occupations.size(1)
        batch_occ = params.size(1)
        if batch_occ != batch_occ:
            raise ValueError("Batch sizes between tokens and params must match")

        seq_len = n_occs + n_params
        embed = self.token_dims + self.param_dims

        buf = torch.zeros(seq_len, batch_occ, embed, dtype=occ.dtype)
        buf[:n_params, :, : self.param_dims] = param
        buf[n_params:, :, self.param_dims :] = occ

        return buf
