from typing import Callable
from torchtyping import TensorType
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import functools as ft
import einops as ein

from model.hubbard_deembedding import HubbardDeembedding
from model.site_degree_embedding import SiteDegreeEmbedding


class Sampling:
    def __init__(
        self,
        embed_dim: int,
        embedding_function: SiteDegreeEmbedding,
        deembedding_function: HubbardDeembedding,
    ):
        """
        Assumes embedding_function takes the sequence of tokens and produces
        a sequence of logits of the same length, with embedding dimension
        embed_dim. embedding_function is very likely a neural network.
        """

        self.embed_dim = embed_dim
        self.embedding_function = embedding_function
        self.deembedding_function = deembedding_function

    def _generate_samples(
        self,
        prob_dist: TensorType["batch", "..."] | torch.Tensor,
        branching_fact: int,
    ):
        if branching_fact < 1:
            raise ValueError("Branching factor must be at least 1")

        if branching_fact != 1:
            raise NotImplementedError("Branching factors other than 1 not implemented")

        # Bring into simpler reshape space
        # (R^-1) (S) (R)

        # R
        prob_dist = ein.rearrange(
            prob_dist,
            "b o sp -> b sp o",
        )

        # Assuming the token dimensions are tailing the other
        # axes
        flatten_point = 2
        token_shape = list(prob_dist.shape)[flatten_point:]
        batch_shape = list(prob_dist.shape)[:flatten_point]
        token_flat_dim = ft.reduce(
            lambda x, y: x * y,
            token_shape,
        )

        prob_dist = prob_dist.flatten(start_dim=flatten_point)

        # S
        cat = Categorical(probs=prob_dist)
        next = cat.sample()

        # R ^ {-1}

        # Map back to one-hot
        samples = F.one_hot(next, num_classes=token_flat_dim)
        samples.reshape(
            *batch_shape,
            *token_shape,
        )
        prob_dist = prob_dist.reshape(
            *batch_shape,
            *token_shape,
        )

        prob_dist = ein.rearrange(
            prob_dist,
            "b sp o -> b o sp",
        )

        samples = ein.rearrange(
            samples,
            "b sp o -> b o sp",
        )

        return samples

    def sample(
        self,
        params: TensorType["n_params", "batch"],
        tokens: TensorType["n_tokens", "batch", "..."],
        up_to: int,
    ):
        """
        Returns the most probable extension of the passed chain of tokens.
        The caller should be sure to drop the previous tokens reference to prevent
        a memory leak.
        """
        n_params = params.shape[0]
        n_tokens = tokens.shape[0]
        seq = n_params + n_tokens
        tokens_batch = tokens.size(1)
        params_batch = params.size(1)

        if tokens_batch != params_batch:
            raise ValueError("Token and parameter batch sizes don't match")
        else:
            batch = tokens_batch

        if n_tokens > up_to:
            raise ValueError("Number of provided tokens exceeds target length (up_to)")
        if len(tokens.shape) < 3:
            raise ValueError("Tokens must have at least 3 dimensions")

        token_dims = list(tokens.shape)[2:]

        # Preallocate token buffer
        more_tokens = torch.zeros([up_to, batch] + token_dims)  # type: ignore
        more_tokens[:n_tokens, :, :] = tokens

        # Preallocate expected logit buffer
        logits = torch.zeros([n_params + up_to, batch, self.embed_dim])
        logits[:seq, :, :] = self.embedding_function(
            params=params,
            tokens=tokens[:seq, :, :],
        )

        for i in range(n_tokens, up_to):
            logits[: n_params + i, :, :] = self.embedding_function(
                params=params,
                tokens=more_tokens[:i, :, :],
            )

            # Token i is a function of logit i - 1
            # This returns the singleton
            prob_dist = self.deembedding_function(
                logits[i - 1, :, :], calculate_phase=False
            )

            next = self._generate_samples(prob_dist, 1)
            more_tokens[i, :, :] = next

        return more_tokens
