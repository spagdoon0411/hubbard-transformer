import torch
from torch import nn
from torchtyping import TensorType


class SimpleParamEmbedding(nn.Module):
    """
    Creates a parameter embedding for a TQS model, taking a list of physical
    parameters to do this. Gives the parameters their own dimension, then
    performs a linear combination between them using trainable ratios.
    """

    def __init__(
        self,
        n_params: int,
        target_dim: int,
    ):
        super(SimpleParamEmbedding, self).__init__()

        self.param_dims = n_params
        self.n_param_to_target = nn.Linear(
            in_features=n_params,
            out_features=target_dim,
        )

    def forward(self, params: TensorType["n_params", "batch"]):
        """
        Returns: TensorType["n_params", "batch", "embed_params"]
        """

        res = torch.diag_embed(
            params.transpose(0, 1), offset=0, dim1=0, dim2=1
        ).transpose(1, 2)

        res = self.n_param_to_target(res)
        return res
