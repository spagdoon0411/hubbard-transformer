import torch
from torch import nn
from torchtyping import TensorType
import functools as ft


class HubbardDeembedding(nn.Module):
    """
    A de-embedding class that focuses on mapping logits to probabilities with dimensions
    (possible_spins, occupations) == (2, occupations) for the Hubbard model.
    """

    def __init__(
        self,
        embed_dim: int,
        target_token_dims: list[int] = [2, 128],
    ):
        super(HubbardDeembedding, self).__init__()
        self.embed_dim = embed_dim
        self.target_token_dims = target_token_dims

        target_flat_dim = ft.reduce(lambda x, y: x * y, target_token_dims)
        self.prob_head = nn.Parameter(torch.randn(embed_dim, target_flat_dim))
        self.phase_head = nn.Parameter(torch.randn(embed_dim, target_flat_dim))

    def compute_psi(
        self,
        probs: torch.Tensor,
        phases: torch.Tensor,
    ):
        """
        The mapping from deembedding outputs (probs, phases) both in the format
        s b sp to values of psi.

        Defines an interpretation for what the model forward pass outputs (e.g., probs,
        root-probs, log probs, etc.).
        """

        # TODO: use log probs for efficiency

        psi = probs * torch.exp(1j * phases)  # s b sp
        return psi

    def forward(
        self,
        logit: TensorType["batch", "embed"] | TensorType["seq", "batch", "embed"],
        calculate_phase: bool = False,
    ):
        """
        Returns: TensorType["seq", "batch", str.(target_token_dims)]
        """

        single = False
        match len(logit.shape):
            case 2:
                single = True
                logit = logit.unsqueeze(0)  # type: ignore
            case 3:
                pass
            case _:
                raise ValueError(
                    "last_logit must have shape (batch, embed) or (seq, batch, embed)"
                )

        (seq, batch, _) = logit.shape
        prob = torch.einsum("ef,sbe->sbf", self.prob_head, logit)
        prob = prob.reshape(seq, batch, *self.target_token_dims)
        prob = torch.softmax(prob, dim=-1)

        if calculate_phase:
            phase = torch.einsum("ef,sbe->sbf", self.phase_head, logit)
            phase = phase.reshape(seq, batch, *self.target_token_dims)
            phase = torch.tanh(phase) * torch.pi

            return (prob.squeeze(0), phase.squeeze(0)) if single else (prob, phase)

        return prob.squeeze(0) if single else prob
