from torch import nn
import torch
from torchtyping import TensorType
from model.site_degree_embedding import SiteDegreeEmbedding
from model.param_embedding import SimpleParamEmbedding
from model.token_embedding import OccupationSpinEmbedding
from model.position_encoding import PositionEncoding
from model.hubbard_deembedding import HubbardDeembedding
from model.sampling import Sampling


class HubbardWaveFunction(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        particle_number: int,
        max_len: int,
        wavelen_fact: float = 1e6,
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        super(HubbardWaveFunction, self).__init__()

        n_params = 5  # t, U, mu, chain length, particle number
        token_dims = [2, 2]  # TODO: occupation, spin?
        input_token_rearrange = "o sp -> (o sp)"

        self.embedding = SiteDegreeEmbedding(
            n_params=n_params,
            embed_dim=embed_dim,
            input_token_dims=token_dims,
            input_token_rearrange=input_token_rearrange,
            param_embedding=SimpleParamEmbedding,
            token_embedding=OccupationSpinEmbedding,
            position_encoding=PositionEncoding,
            max_len=max_len,
            wavelen_fact=wavelen_fact,
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=n_layers,
        )

        self.deembedding = HubbardDeembedding(
            embed_dim=embed_dim,
            target_token_dims=token_dims,
        )

        self.sampling = Sampling(
            embed_dim=embed_dim,
            particle_number=particle_number,
            embedding_function=self.embedding,
            deembedding_function=self.deembedding,
        )

    def sample(
        self,
        num_chains: int,
        chain_length: int,
        params: TensorType["n_params"],
    ):
        """
        Produces num_chains most-probable token chains of length chain_length based
        on the wave function this model represents.
        """

        params = params.unsqueeze(0).expand(num_chains, -1)  # type: ignore
        tokens = torch.zeros(
            0,
            num_chains,
            *self.embedding.input_token_dims,
        )
        chains = self.sampling.sample(
            params=params,
            tokens=tokens,  # type: ignore
            up_to=chain_length,
        )

    def _ham_entry(
        self,
        t,
        U,
        mu,
        lspin: int,
        lpos: int,
        rspin: int,
        rpos: int,
    ):
        """
        Computes the Hamiltonian entry <lspin, lpos|H|rspin, rpos>, for use
        in calculation of E_loc.

        Spins are in {0, 1} and positions are in {0, 1, ..., chain_length}.
        """

        return 1

    def _e_loc(self):
        """
        Computes the expectation value of E_loc using the wave function that this
        model represents.
        """

        # Compute

    def forward(
        self,
        params: TensorType["n_params", "batch"],
        tokens: TensorType["seq", "batch", "occupation", "spin"],
    ):
        """
        Given a sequence of tokens and parameters, produces the probabilities and phases
        associated with the wave function that this model represents.
        """

        assert (
            params.shape[0] == 5
        ), f"HubbardWaveFunction expects 5 parameters, got {params.shape[0]}"

        n_params = params.shape[0]

        logits = self.embedding(params, tokens)
        logits = self.transformer_encoder(logits)
        prob, phase = self.deembedding(logits[n_params:], calculate_phase=True)

        idx = torch.argmax(prob, dim=-2)
        idx = idx.unsqueeze(-2)
        prob = prob.gather(-2, idx).squeeze(-2)
        phase = phase.gather(-2, idx).squeeze(-2)

        return prob, phase
