import torch
from torch import nn
from torchtyping import TensorType

from position_encoding import PositionEncoding

# End-to-end embedding buffer preparation for a Hubbard-model
# system.


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

    def __init__(self, n_params: int, target_dim: int):
        super(GenericParamEncoding, self).__init__()
        # TODO: the interaction between parameters probably should not be captured
        # linearly. Maybe use a few small rounds of self-attention?

        self.param_dims = n_params
        self.interaction_weights = nn.Parameter(torch.randn(n_params, n_params))
        self.n_param_to_target = nn.Parameter(torch.randn(target_dim, n_params))

    def forward(self, params: TensorType["n_params", "batch"]):
        """
        Returns: TensorType["n_params", "batch", "embed_params"]
        """
        # Arrange parameter list along diag
        # (n_params, batch) -> (n_params, n_params, batch)

        res = torch.diag_embed(
            params.transpose(0, 1), offset=0, dim1=0, dim2=1
        ).transpose(1, 2)

        # (n_params, n_params, 1) * (n_params, n_params, batch) -> (n_params, n_params, batch)
        # res = torch.matmul(self.interaction_weights.unsqueeze(0), res)

        # bring into the same embedding space as the model
        res = torch.einsum("il,jkl->jki", self.n_param_to_target, res)

        return res


class GenericOccupationEncoding(nn.Module):
    def __init__(
        self,
        max_len: int,
        token_dim: int,
        wavelen_fact=1e6,
    ):
        super(GenericOccupationEncoding, self).__init__()
        self.occupation_base = nn.Parameter(torch.randn(token_dim))
        self.pe = PositionEncoding(
            d_model=token_dim,
            max_len=max_len,
            wavelen_fact=wavelen_fact,
        )

    def forward(self, occupations: TensorType["n_sites", "batch"]):
        ob = self.occupation_base.unsqueeze(0).unsqueeze(0)
        o = occupations.unsqueeze(-1)
        return ob * o


class HubbardEmbedding(nn.Module):
    """
    Translates a Hubbard system into a buffer that's pushed through
    rounds of self-attention.
    """

    def __init__(
        self,
        token_dims: int,
        param_dims: int,
        n_params: int,
        max_len: int,
        wavelen_fact=1e6,
    ):
        super(HubbardEmbedding, self).__init__()
        self.token_dims = token_dims
        self.param_dims = param_dims
        self.occ_embedding = GenericOccupationEncoding(
            max_len, token_dims, wavelen_fact
        )
        self.param_embedding = GenericParamEncoding(n_params, target_dim=param_dims)

    def forward(
        self,
        params: TensorType["n_params", "batch", torch.float64],
        occupations: TensorType["n_sites", "batch", torch.int64],
    ):
        n_sites = occupations.size(0)
        n_params = params.size(0)
        seq = n_sites + n_params
        batch = occupations.size(1)
        embed = self.token_dims + self.param_dims

        buf = torch.zeros(seq, batch, embed)

        # TODO: use out parameters for the two embedding submodules
        token_buf = self.occ_embedding(occupations)
        param_buf = self.param_embedding(params)

        buf[:n_sites, :, : self.token_dims] = token_buf
        buf[n_sites:, :, self.token_dims :] = param_buf

        return buf
