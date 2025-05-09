from typing import Callable
from torch.nn import TransformerEncoder
from torchtyping import TensorType
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import functools as ft
import einops as ein
import pdb

from model.hubbard_deembedding import HubbardDeembedding
from model.site_degree_embedding import SiteDegreeEmbedding

DEBUG_ASSERTIONS = True


class Sampling:
    def __init__(
        self,
        embed_dim: int,
        particle_number: int,
        embedding_function: SiteDegreeEmbedding,
        deembedding_function: HubbardDeembedding,
        transformer_encoder: TransformerEncoder,
        mask: torch.Tensor,
    ):
        """
        Assumes embedding_function takes the sequence of tokens and produces
        a sequence of logits of the same length, with embedding dimension
        embed_dim. embedding_function is very likely a neural network.

        counter_function is a map from (batch1, ..., batchN, token1, ..., tokenN)
        -> (batch1, ..., batchN) that counts the number of particles corresponding
        to each token.
        """

        self.particle_number = particle_number
        self.embed_dim = embed_dim
        self.embedding_function = embedding_function
        self.deembedding_function = deembedding_function
        self.transformer_encoder = transformer_encoder
        self.mask = mask

    def _generate_samples(
        self,
        prob_dist: TensorType["batch", "..."] | torch.Tensor,
        branching_fact: int,
        compute_log_prob: bool = False,
    ):
        if branching_fact < 1:
            raise ValueError("Branching factor must be at least 1")

        if branching_fact != 1:
            raise NotImplementedError("Branching factors other than 1 not implemented")

        # Bring into simpler reshape space

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
        next = cat.sample()  # (b, sp)
        log_prob = cat.log_prob(next)  # (b, sp)

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

        if compute_log_prob:
            # b o sp, b sp
            return samples, log_prob

        return samples

    def _enforce_particle_num(
        self, tokens: TensorType["n_tokens", "batch", "..."]
    ) -> TensorType["n_tokens", "batch", "..."]:
        """
        Ensures the number of particles over the whole chain matches the value prescribed
        on initialization.

        The one-hot encodings either need to be bumped up or down the occupation dimension
        to adjust the number of particles in the chain
        """

        seq, batch, occ, sp = tokens.shape
        max_occ_idx = occ - 1

        site_occs = tokens.argmax(dim=-2)  # (s, b, sp)
        site_occs = ein.rearrange(site_occs, "s b sp -> (s sp) b")

        # Determine how many particles need to be added to each chain to achieve
        # the target particle number
        diffs = site_occs.sum(dim=0)  # (b,)
        diffs = self.particle_number - diffs
        negative_diffs = diffs < 0
        diffs = diffs.abs()

        # Start counting holes instead of particles for the chains that
        # have to have particles removed
        site_occs[:, negative_diffs] = max_occ_idx - site_occs[:, negative_diffs]

        if DEBUG_ASSERTIONS:
            assert diffs.shape == (batch,)
            assert negative_diffs.shape == (batch,)
            assert site_occs.shape == (seq * sp, batch)

        adjustments_needed = diffs > 0  # (b,)
        while adjustments_needed.sum().item():
            available_sites = site_occs < max_occ_idx
            available_count = available_sites.sum(dim=0)  # (b,)
            if available_count.sum().item() == 0:
                raise ValueError(
                    f"No available sites to add particles to. Can your chain support a particle number of {self.particle_number}?"
                )

            site_selection_rand = torch.rand(available_sites.shape)
            site_selection_rand[~available_sites] = -torch.inf
            target_sites = torch.argmax(site_selection_rand, dim=0)
            del site_selection_rand

            remaining_cap = max_occ_idx - site_occs[target_sites, torch.arange(batch)]
            adjustments = torch.rand(adjustments_needed.shape)
            sample_ceil = torch.min(diffs, remaining_cap)
            adjustments = (adjustments * (sample_ceil + 1)).to(torch.int)

            site_occs[target_sites, torch.arange(batch)] += adjustments
            diffs -= adjustments

            adjustments_needed = diffs > 0

        site_occs[:, negative_diffs] = max_occ_idx - site_occs[:, negative_diffs]
        site_occs = ein.rearrange(
            site_occs,
            "(s sp) b -> s b sp",
            s=seq,
            sp=sp,
        )

        new_tokens = F.one_hot(site_occs, num_classes=occ)  # (s, b, sp, o)
        new_tokens = ein.rearrange(new_tokens, "s b sp o -> s b o sp")

        if DEBUG_ASSERTIONS:
            assert new_tokens.shape == (seq, batch, occ, sp)

        return new_tokens  # type: ignore

    def sample(
        self,
        params: TensorType["n_params", "batch"],
        tokens: TensorType["n_tokens", "batch", "..."],
        up_to: int,
        compute_log_prob: bool = False,
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

        # A token tensor to expand
        more_tokens = torch.zeros([n_tokens, batch] + token_dims)

        # A tensor of logits to expand
        logits = self.embedding_function(
            params=params,
            tokens=tokens[:seq, :, :],
        )

        # Log-probs associated with sampling this whole chain
        chain_log_probs = torch.zeros(batch)

        for i in range(n_tokens, up_to):
            # Produce logits up to the i - 1th token, which are used to compute
            # the ith token
            logits = self.embedding_function(
                params=params,
                tokens=tokens,
            )

            logits = self.transformer_encoder(
                logits,
                mask=self.mask[:seq, :seq],
            )

            # Token i is a function of logit i - 1 and the previous tokens. The probability distribution
            # generated by token i - 1 contains information from all prior tokens
            prob_dist = self.deembedding_function(
                logits[i - 1, :, :], calculate_phase=False
            )

            next, log_probs = self._generate_samples(
                prob_dist, 1, compute_log_prob=True
            )

            # If you sum log probs you get a joint probability. Thus we can
            # sum log probs to get the log prob of the whole chain.
            chain_log_probs += ein.einsum(log_probs, "b sp -> b")

            # TODO: Accumulate a log-prob buffer for this run of sampling, to be
            # used in a surrogate loss function later

            more_tokens = torch.cat(
                [more_tokens, next.unsqueeze(0)], dim=0
            )  # Add the new token along the sequence dimension

        # NOTE: so no more in-place operations?

        # type: ignore is because of Tensor -> TensorType error
        more_tokens = self._enforce_particle_num(more_tokens)  # type: ignore

        more_tokens = more_tokens.to(dtype=torch.float32)

        if compute_log_prob:
            # s b o sp, b
            return more_tokens, chain_log_probs

        return more_tokens
