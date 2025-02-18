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
    """

    def __init__(self, param_dims: int):
        super(GenericParamEncoding, self).__init__()
        self.param_dims = param_dims

        # TODO: try one round of attention to deterine whether
        # allowing the params to interact with each other before
        # interacting with the rest of the system is a reasonable
        # choice.

        # [ M1, M2, M3, M4, ...] -> [comb(M1, M2, ...), comb(M1, M2, ...), ...]
        # [ M1, M2, M3, M4, ...] -> [M * x, M * y, M * z, ...]
        # [ M1, M2, M3, M4, ...] -> M * [x, y, z, ...]
        # [ M1, M2, M3, M4, ...] -> M * W

        # Left-multiplying a weight matrix by the input vectors results in columns
        # that are linear combinations of the inputs. TODO: is this the best way to
        # do it?

        # The matrix below is W.

        self.interaction_weights = nn.Parameter(torch.randn(param_dims, param_dims))

        # NOTE: Any matmul operation is isomorphic to the typical linear combination
        # interpretation.

    def forward(self, params: TensorType["n_params", "batch"]):
        """
        Returns: TensorType["n_params", "batch", "embed_params"]
        """
        # Arrange parameter list along diag
        # (n_params, batch) -> (n_params, n_params, batch)

        res = torch.diag_embed(params, offset=0, dim1=0, dim2=1)
        res = torch.matmul(self.interaction_weights, res)
        return res


class HubbardEmbedding(nn.Module):
    def __init__(self):
        super(HubbardEmbedding, self).__init__()
        self.pe = PositionEncoding(2, 4, 1e6)

    def forward(
        self,
        params: TensorType["n_params", "batch", torch.float64],
        occupations: TensorType["n_sites", "batch", torch.int64],
    ):
        # Inputs should be occupations

        # The whole system relies on us contraining the
        # model so that it can only output valid configurations.

        # A monitoring parameter is probably the delta_num
        # to get to the right number of particles. Add this
        # to the loss function?

        # Generally constraints are tough to enforce in NNs
        # and are usually done creatively

        # Output: a probability distribution over occupation
        # numbers.

        # Remember that the whole sampling
        # process should be autoregressive
        pass
