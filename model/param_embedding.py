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
        self.interaction_weights = nn.Parameter(torch.randn(n_params, n_params))
        self.n_param_to_target = nn.Parameter(torch.randn(target_dim, n_params))

    def forward(self, params: TensorType["n_params", "batch"]):
        """
        Returns: TensorType["n_params", "batch", "embed_params"]
        """

        res = torch.diag_embed(
            params.transpose(0, 1), offset=0, dim1=0, dim2=1
        ).transpose(1, 2)

        res = torch.einsum("il,jkl->jki", self.n_param_to_target, res)

        return res
