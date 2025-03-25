from torch import nn
import torch
import einops as ein


class HubbardDeembedding(nn.Module):
    """
    Defines how to take (seq, batch, embed) logits and map them to
    Hubbard probabilities and phases.
    """

    def __init__(
        self,
        token_dims: int,
        param_dims: int,
        possible_occupations: int,
        possible_spins: int,
    ):
        super().__init__()
        self.token_dims = token_dims
        self.param_dims = param_dims
        self.embed_dims = token_dims + param_dims

        # TODO: mention that we can look at the de-embedding for
        # each spin separately, maintaining interpretability

        # TODO: mention possibility of using a single complex head
        # for the output

        # TODO: bring up refactored structure (as opposed to original TQS)

        # For a single logit we produce "spins" probability distributions
        # of shape (seq, batch, occupations). So we need a map from a
        # single logit to (seq, batch, occupations, spins) (then softmax
        # across the occupation dim)
        self.prob_head = nn.Parameter(
            torch.randn(
                self.embed_dims,
                possible_occupations,
                possible_spins,
            )
        )

        self.phase_head = nn.Parameter(
            torch.randn(
                self.embed_dims,
                possible_occupations,
                possible_spins,
            )
        )

        self._softsign = nn.Softsign()
        self.softphase = lambda x: self._softsign(x) * torch.pi
        self.softmax = nn.Softmax(dim=-2)

    # TODO: type this with dimensions
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        During sampling this is run with only one member of the sequence dimension.
        This should only be run with token logits--not parameter logits.

        tokens: (seq, batch, embed)
        """

        # TODO: when we transition to complex, using einops will explode here
        prob = ein.einsum(logits, self.prob_head, "s b e, e o sp -> s b o sp")
        prob = self.softmax(prob)

        phase = ein.einsum(logits, self.phase_head, "s b e, e o sp -> s b o sp")
        phase = self.softphase(phase)

        return prob, phase
