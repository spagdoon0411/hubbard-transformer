import torch
from torch import nn
from torchtyping import TensorType
from model.param_embedding import SimpleParamEmbedding
from model.token_embedding import OccupationSpinEmbedding
from model.position_encoding import PositionEncoding


class SiteDegreeEmbedding(nn.Module):
    def __init__(
        self,
        n_params: int,
        embed_dim: int,
        input_token_dims: list[int],
        input_token_rearrange: str,
        param_embedding: type[SimpleParamEmbedding],
        token_embedding: type[OccupationSpinEmbedding],
        position_encoding: type[PositionEncoding],
        max_len: int = 200,
        wavelen_fact: float = 1e6,
        diag: dict = {},
    ):
        super(SiteDegreeEmbedding, self).__init__()

        self.diag = diag

        self.n_params = n_params
        self.embed_dim = embed_dim
        self.input_token_dims = input_token_dims
        self.input_token_rearrange = input_token_rearrange
        self.ParamEmbedding = param_embedding
        self.TokenEmbedding = token_embedding
        self.PositionEncoding = position_encoding

        self.param_embedding = self.ParamEmbedding(
            n_params=n_params,
            target_dim=embed_dim,
        )

        self.token_embedding = self.TokenEmbedding(
            input_token_dims=input_token_dims,
            output_token_dims=embed_dim,
            einops_pattern=input_token_rearrange,
        )

        self.position_encoding = self.PositionEncoding(
            embed_dim=embed_dim,
            max_len=max_len,
            wavelen_fact=wavelen_fact,
        )

    def forward(
        self,
        params: TensorType["n_params", "batch"],
        tokens: TensorType["seq", "batch", "..."],
    ):
        (n_params, param_batch) = params.shape
        n_tokens = tokens.shape[0]
        token_batch = tokens.shape[1]
        if token_batch != param_batch:
            raise ValueError("Token and parameter batch sizes don't match")

        seq = n_tokens + n_params
        batch = token_batch
        buf = torch.zeros(seq, batch, self.embed_dim)

        param_logits = self.param_embedding(params)
        token_logits = self.token_embedding(tokens)

        buf[:n_params, :, :] = param_logits
        buf[n_params:, :, :] = token_logits

        buf = self.position_encoding(buf)

        return buf
