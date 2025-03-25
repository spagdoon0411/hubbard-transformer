from complex_model import ComplexAttention
from occupation_embedding import HubbardEmbedding
from torch import nn
import torch


class HubbardTQS(nn.Module):
    def __init__(
        self,
        token_dims: int,
        param_dims: int,
        n_params: int,
        model_dims: int,
        n_heads: int,
        max_len: int,
        wavelen_fact=1e6,
        dtype: torch.dtype = torch.complex64,
    ):
        super(HubbardTQS, self).__init__()

        # TODO: the occupation embedding completely does not care
        # about the possible number of tokens at each site; this coud
        # be a major problem.

        self.embed_dims = token_dims + param_dims
        self.model_dims = model_dims  # TODO: what was this again?
        self.n_heads = n_heads
        self.max_len = max_len

        self.embed = HubbardEmbedding(
            token_dims=token_dims,
            param_dims=param_dims,
            n_params=n_params,
            max_len=max_len,
            wavelen_fact=wavelen_fact,
        )

        self.attention = ComplexAttention(
            embed_dims=self.embed_dims,
            model_dims=self.model_dims,
            n_heads=self.n_heads,
            max_len=self.max_len,
            dtype=dtype,
        )

    def forward(
        self,
        occupations: torch.Tensor,
        params: torch.Tensor,
    ):
        """
        occupations: (n_sites, batch, occupations, spins)
        params: (n_params, batch)
        """

        logits = self.embed(occupations, params)
