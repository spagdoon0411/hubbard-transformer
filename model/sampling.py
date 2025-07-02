from typing import Callable
from torch.nn import TransformerEncoder
from torchtyping import TensorType
import torch
import pickle
import os
from torch.distributions import Categorical
import torch.nn.functional as F
import functools as ft
import einops as ein
import pdb
import ipdb

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
        diag: dict = {},
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
        self.diag = diag

    def _generate_samples(
        self,
        prob_dist: torch.Tensor,
        branching_fact: int,
    ):
        """
        Generates branching_fact samples from the given probability distribution over
        next tokens.

        prob_dist: (b, o, sp), the distribution to sample next tokens from.
        branching_fact: int, the number of samples to generate from the distribution.
        """

        if branching_fact < 1:
            raise ValueError("Branching factor must be at least 1")

        if branching_fact != 1:
            raise NotImplementedError("Branching factors other than 1 not implemented")

        # Reshape to (b, o, sp) -> (b, sp, o) for Categorical sampling, then reshape back to (b, o, sp)
        # before returning.

        prob_dist = ein.rearrange(
            prob_dist,
            "b o sp -> b sp o",
        )

        cat = Categorical(probs=prob_dist)
        next_token = cat.sample()  # (b, sp)
        log_prob = cat.log_prob(next_token)  # (b, sp)
        samples = F.one_hot(next_token, num_classes=prob_dist.shape[-1])  # (b, sp, o)

        prob_dist = ein.rearrange(
            prob_dist,
            "b sp o -> b o sp",
        )

        samples = ein.rearrange(
            samples,
            "b sp o -> b o sp",
        )

        return samples, log_prob

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

    def _sample_one_more_token(
        self,
        params: torch.Tensor,
        more_tokens: torch.Tensor,
    ):
        """
        Extends the given token sequence by one token, producing log probs
        for sampling that particular token.

        params: (n_params, batch)
        more_tokens: (sequence, batch, occupation, spin)
        """

        seq_len = params.shape[0] + more_tokens.shape[0]

        # FIXME: the embedding function should assume s b o sp
        logits = self.embedding_function(
            params=params,
            tokens=more_tokens,
        )

        logits = self.transformer_encoder(
            logits,
            mask=self.mask[:seq_len, :seq_len],
        )

        # Token i is a function of the last logit sampled so far
        last_logit = logits[seq_len - 1, :, :]
        prob_dist = self.deembedding_function(
            last_logit, calculate_phase=False
        )  # b sp o, softmax over the last dimension

        # FIXME: the de-embedding function should return b o sp
        prob_dist = ein.rearrange(prob_dist, "b sp o -> b o sp")

        # Generate next tokens across the batch dimension, producing one sample
        # per batch element.
        next, log_probs = self._generate_samples(prob_dist, 1)  # b o sp, b sp

        step_log_probs = ein.einsum(log_probs, "b sp -> b")
        return next, step_log_probs

    def sample(
        self,
        params: torch.Tensor,
        tokens: torch.Tensor,
        up_to: int,
        return_log_prob: bool = False,
    ):
        """
        Returns the most probable extension of the passed chain of tokens.
        """

        # A sequence length includes the lengths of both the params and the
        # tokens.
        n_tokens = tokens.shape[0]
        if n_tokens > up_to:
            raise ValueError("Number of provided tokens exceeds target length (up_to)")
        if len(tokens.shape) < 3:
            raise ValueError("Tokens must have at least 3 dimensions")

        tokens_batch = tokens.size(1)
        params_batch = params.size(1)
        if tokens_batch != params_batch:
            raise ValueError("Token and parameter batch sizes don't match")
        else:
            batch = tokens_batch

        # s b o sp
        more_tokens = tokens

        # Log-probs associated with sampling each chain.
        chain_log_probs = torch.zeros(batch)

        for i in range(n_tokens, up_to):
            next, step_log_probs = self._sample_one_more_token(
                params,
                more_tokens,
            )

            chain_log_probs += step_log_probs
            more_tokens = torch.cat(
                [more_tokens, next.unsqueeze(0)], dim=0
            )  # Add the new token along the sequence dimension

        more_tokens = self._enforce_particle_num(more_tokens)  # type: ignore

        more_tokens = more_tokens.to(dtype=torch.float32)

        return more_tokens, chain_log_probs
