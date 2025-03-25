import torch
from torch import nn
from torchtyping import TensorType


class PositionEncoding(nn.Module):
    """
    Positional encoding module for a transformer model
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 32,
        wavelen_fact=1e6,
        dtype: torch.dtype = torch.complex64,
    ):
        super(PositionEncoding, self).__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.wavelen_fact = wavelen_fact

        if (embed_dim % 2) != 0:
            raise ValueError("d_model must be even")

        # (seq, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Static buffer allocated across model lifetime
        pe = torch.zeros(max_len, embed_dim)  # (max_len, d_model)

        # (embed // 2, )
        # arange represents the advancing index
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-torch.log(torch.tensor(wavelen_fact)) / embed_dim)
        )

        angular_positions = position * div_term
        pe[:, 0::2] = torch.sin(angular_positions)
        pe[:, 1::2] = torch.cos(angular_positions)

        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, embed_dim)

        pe.to(dtype=dtype)

        # Register the positional encoding as a buffer
        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

    def forward(self, x: TensorType["seq", "batch", "embed"]):
        """
        Takes a buffer and mutates it to include the positional encoding
        """
        if x.shape[0] > self.max_len:
            raise ValueError(
                f"Input sequence length {x.shape[0]} is greater than the "
                f"maximum length {self.max_len}"
            )

        seq = x.shape[0]
        return x + self.pe[:seq, :, :]
