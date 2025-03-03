import torch
from torch import nn
from torchtyping import TensorType


class PositionEncoding(nn.Module):
    """
    Positional encoding module for a transformer model
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 32,
        wavelen_fact=1e6,
        dtype: torch.dtype = torch.complex64,
    ):
        super(PositionEncoding, self).__init__()

        if (d_model % 2) != 0:
            raise ValueError("d_model must be even")

        # (seq, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # (d_model, )
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(wavelen_fact)) / d_model)
        )

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)

        # (max_len, 1) * (d_model,) =>  (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        pe.to(dtype=dtype)

        # Register the positional encoding as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x: TensorType["seq", "batch", "embed"]):
        """
        Takes a buffer and mutates it to include the positional encoding
        """
        if x.size(0) > self.pe.size(0):
            raise ValueError(
                f"Input sequence length {x.size(0)} is greater than the "
                f"maximum length {self.pe.size(0)}"
            )

        return x + self.pe[: x.size(0), :].unsqueeze(0)
