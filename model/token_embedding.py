import torch
from torch import nn
from torchtyping import TensorType
import einops as ein
import re
import functools as ft


class OccupationSpinEmbedding(nn.Module):
    """
    Embeds tokens corresponding to occupation configurations in a
    spin-occupation model to produce logits. Linearizes the multidimensional
    occupation tokens according to an einops pattern.
    """

    def __init__(
        self,
        input_token_dims: list[int],
        output_token_dims: int,
        einops_pattern: str,
    ):
        """
        The einops pattern describes how to linearize a single occupation token.
        If s and b are provided as dimensions here the behavior of this module is
        undefined.
        """
        super(OccupationSpinEmbedding, self).__init__()
        self.EINOPS_REGEX = re.compile(r"\((.*)\)")

        self.ldims = ft.reduce(lambda x, y: x * y, input_token_dims)
        self.einops_pattern = einops_pattern
        self.occs_to_logits = nn.Parameter(
            torch.randn(
                output_token_dims,
                self.ldims,
            ),
        )

    def forward(self, occupations: TensorType["seq", "batch", "..."]):
        """
        occupations: (seq, batch, d1, ..., dn)
        """

        input, output = self.einops_pattern.split("->")
        input = input.strip()
        output = output.strip()

        assert self.EINOPS_REGEX.match(
            output
        ), "Output einops pattern must be of the form (o1', ..., on')"

        pattern = f"s b {input} -> s b {output}"

        occs = ein.rearrange(occupations, pattern)
        occs = ein.einsum(occs, self.occs_to_logits, "s b l, e l -> s b e")
        return occs
