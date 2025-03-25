from torch import nn
import torch
from occupation_embedding import HubbardEmbedding
from deembedding import HubbardDeembedding
import einops as ein
from typing import Optional
from torch.distributions import Categorical


# TODO: get familiar with each of these params
class RealHubbardModel(nn.Module):
    def __init__(
        self,
        token_dims: int,
        param_dims: int,
        n_params: int,
        dim_feedforward: int,
        n_heads: int,
        possible_occupations: int,
        possible_spin_states: int,
        max_len: int,
    ):
        super(RealHubbardModel, self).__init__()
        self.token_dims = token_dims
        self.param_dims = param_dims
        self.embed_dims = token_dims + param_dims
        self.n_params = n_params
        self.possible_occupations = possible_occupations
        self.possible_spin_states = possible_spin_states
        self.max_len = max_len

        # Takes occupation tokens with sequence, batch, occupation, and
        # spin dimensions and brings them into logit embedding space.
        # (s b o sp) -> (s b (o sp)) == (s b e)
        self.Embed = HubbardEmbedding(
            token_dims=token_dims,
            param_dims=param_dims,
            n_params=n_params,
            max_len=max_len,
            possible_occupations=possible_occupations,
            possible_spin_states=possible_spin_states,
            dtype=torch.float32,
        )

        # Acts like an operation (domain == codomain). Performs self-attn
        # updates on the logit buffer.
        # (s b e) -> (s b e)
        self.EncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.embed_dims, nhead=n_heads, dim_feedforward=dim_feedforward
        )

        # Produces probability distributions over occupations with a
        # different distribution for each spin.
        # (s b e) -> (s b o sp)
        self.Deembed = HubbardDeembedding(
            token_dims=token_dims,
            param_dims=param_dims,
            possible_occupations=possible_occupations,
            possible_spins=possible_spin_states,
        )

    # TODO: type the "out" parameter
    # TODO: make this more memory-efficient by caching params or setting model
    # params on initialization.
    def sample_next(
        self, params: torch.Tensor, occupations: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """
        Takes a sequence of occupations and spins, populating the next
        occupation token in out.

        sequence: (seq, batch, occupations, spins)
        out: (1, batch, occupations, spins)

        TODO: use out-reference
        TODO: use a return-phase flag
        TODO: does multinomial just treat this as a batched matrix?
        TODO: slices are like "dimensioned references"
        TODO: can/should we sample multiple at a single stage to get
        a better idea of the distribution?
        """

        # TODO: embedding is time-constant and sequence-independent; can we
        # refactor it out to the parent function?
        buf = self.Embed(params=params, occupations=occupations)
        buf = self.EncoderLayer(buf)
        last_logit = buf[:-1, :, :]
        prob, _ = self.Deembed(last_logit)  # (1, batch, occupations, spins)
        prob = prob.squeeze(0)

        # (batch, occupations, spins)
        # TODO: seed all randoms
        # next_token = torch.multinomial(prob, num_samples=1).squeeze(-1)
        probs = ein.rearrange(prob, "b o sp -> b sp o")
        next_token = Categorical(probs).sample()  # (batch, spins)

        out[0, :, :, :] = ein.rearrange(next_token, "b sp o -> 1 b o sp")

        out[0, :, :, :] = next_token

        # TODO: We should only have one occupation number per site
        assert torch.all(next_token.sum(dim=-2) == 1)

        return next_token

    def sample_states(
        self,
        params: torch.Tensor,  # (n_params, batch)
        occupations: Optional[torch.Tensor],  # (seq, batch, occupations, spins)
        target_len: int,
    ):
        """
        Extends the sequence provided to the target length, allocating a new buffer
        to do so.

        TODO: what's the nicest way to explore these sample branches?
        """

        if target_len > self.max_len:
            raise ValueError(
                f"Cannot sample more than {self.max_len} states from the model; see self.max_len."
            )

        if params.shape[0] != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {params.shape[0]}. Was the model initialized with a different param count?"
            )

        batch = params.shape[1]
        buffer = torch.zeros(
            target_len,
            batch,
            self.possible_occupations,
            self.possible_spin_states,
        )

        # Populate buffer with an existing sequence if it's provided
        if occupations is not None:
            (current_len, batch, possible_occs_input, possible_spins_input) = (
                occupations.shape
            )

            if current_len > target_len:
                raise ValueError(
                    f"Sequence length is {current_len} but target length is {target_len}"
                )

            if possible_occs_input != self.possible_occupations:
                raise ValueError(
                    f"Expected {self.possible_occupations} possible occupations, got {possible_occs_input}"
                )

            if possible_spins_input != self.possible_spin_states:
                raise ValueError(
                    f"Expected {self.possible_spin_states} possible spins, got {possible_spins_input}"
                )

            if params.shape[1] != batch:
                raise ValueError(
                    f"Tokens have a batch size of {batch}, but parameters have a batch size of {params.shape[1]}"
                )

            buffer[:current_len, :, :, :] = occupations
        else:
            current_len = 0
            batch = params.shape[1]

        # Autoregressiely populate the empty space in the buffer
        while current_len < target_len:
            # TODO: avoid re-embedding params each time
            # TODO: can we do any elimination here?
            self.sample_next(
                params,
                occupations=buffer[:current_len, :, :, :],
                out=buffer[current_len : current_len + 1, :, :, :],
            )
            current_len += 1

        return params, buffer
