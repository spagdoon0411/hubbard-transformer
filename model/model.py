from torch import nn
import torch.nn.functional as F
import torch
import einops as ein
from torchtyping import TensorType
from model.site_degree_embedding import SiteDegreeEmbedding
import itertools as it
from functools import lru_cache
from model.param_embedding import SimpleParamEmbedding
from model.token_embedding import OccupationSpinEmbedding
from model.position_encoding import PositionEncoding
from model.hubbard_deembedding import HubbardDeembedding
from model.sampling import Sampling
from model.hamiltonian import HubbardHamiltonian


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

        self.token_dims = token_dims
        self.particle_number = particle_number

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

        # TODO: why do we infinite loop on sampling

        params = ein.repeat(
            params,
            "n_params -> n_params b",
            b=num_chains,
        )
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
        return chains

    @lru_cache(maxsize=None)  # Ideally we only cache once
    def generate_basis(
        self,
        num_sites: int,
    ):
        """
        Produces a basis meeting particle number constraints in the s b o sp format.
        """

        # TODO: this is a leaf operation and should be ignored by grad

        # Enumerate states equivalent under number of particles
        combinations = list(it.combinations(range(num_sites * 2), self.particle_number))
        num_basis_states = len(combinations)

        # (s sp) nCk
        flat_states = torch.zeros(
            num_sites * 2,
            num_basis_states,
            dtype=torch.long,
        )

        # Generate particle distribution indices
        particle_idx = torch.tensor(combinations)  # (nCk p)
        particle_idx = ein.rearrange(particle_idx, "nCk p -> p nCk")
        row_idx = torch.arange(num_basis_states)  # (nCk)
        row_idx = ein.repeat(row_idx, "nCk -> p nCk", p=self.particle_number)

        assert (
            torch.broadcast_shapes(row_idx.shape, particle_idx.shape)
            == particle_idx.shape
        ), "Indices used to distribute particle_number particles are not the same shape"

        flat_states[particle_idx, row_idx] = 1

        # What does the one_hot operation do again?
        # TODO: we can only apply one_hot to LongTensors?
        flat_states = F.one_hot(
            ein.rearrange(flat_states, "s_sp nCk -> nCk s_sp"),
            num_classes=self.token_dims[0],
        )  # nCk (s sp) o

        flat_states = ein.rearrange(
            flat_states,
            "nCk (s sp) o -> s nCk o sp",
            sp=2,
            s=num_sites,
        )

        flat_states = flat_states.to(dtype=torch.float32)

        return flat_states  # s h o sp == s nCk o sp

    @lru_cache(maxsize=None)
    def compute_basis_information(self, num_sites: int, params: torch.Tensor):
        """
        Produces a complete basis tensor for the number of sites given, assuming
        two possible occupations and two possible spin states.

        Passes back the wave function and the basis states.

        num_sites: int
            Number of sites in the system.
        params: torch.Tensor
            Physical system parameters with shape (n_params,).

        Returns:
        psi: torch.Tensor
            The wave function corresponding to the basis states in s sp h format.
        basis: torch.Tensor
            Basis state vectors in s h o sp format. Wave function values are to be aligned
            along the h ("Hilbert") axis.
        """

        # TODO: here, params should not be pre-broadcast

        basis = self.generate_basis(num_sites=num_sites)  # (s, h, o, sp)
        s, h, o, sp = basis.shape
        params = ein.repeat(
            params,
            "n_params -> n_params h",
            h=h,
        )

        probs, phases = self.forward(
            tokens=basis,  # type: ignore
            params=params,  # type: ignore
        )

        psi = self.deembedding.compute_psi(
            probs=probs,  # s b sp
            phases=phases,  # s b sp
        )

        return psi, basis

    def e_loc(
        self,
        hamiltonian: HubbardHamiltonian,
        params: TensorType["n_params"],
        sampled_states: TensorType["seq", "batch", "occupation", "spin"],
    ):
        """
        Computes the expectation value of E_loc using the wave function that this
        model represents.
        """

        # TODO: reconcile parameter information redundancy between Hamiltonian
        # and params tensor

        # NOTE: params are passed pre-broadcasting to this function but are
        # broadcast before the forward pass

        s, b, _, _ = sampled_states.shape

        basis_psi, basis = self.compute_basis_information(
            num_sites=s,
            params=params,  # n_params
        )  # (s b sp, s h o sp)

        params = ein.repeat(
            params,
            "n_params -> n_params b",
            b=b,
        )

        probs, phases = self.forward(
            tokens=sampled_states,  # s b o sp
            params=params,  # n_params b
        )  # (s b sp, s b sp)

        sampled_psi = self.deembedding.compute_psi(
            probs=probs,  # s b sp
            phases=phases,  # s b sp
        )  # (s b sp)

        entries = hamiltonian.term(
            a=sampled_states,
            b=basis,
        )

        # Psi is the wrong shape

        # TODO: these had spins selected in the forward pass
        print(f"Shape of entries: {entries.shape}")
        print(f"Shape of sampled_psi: {sampled_psi.shape}")
        print(f"Shape of basis_psi: {basis_psi.shape}")

        E_loc_terms = ein.einsum(
            entries,
            sampled_psi,
            basis_psi,
            "b h, s b sp, s h sp -> b",
        )

        print(f"E_loc terms after einsumming: {E_loc_terms.shape}")

        # TODO: does the expression above sum over spins?

        expect_E_loc = torch.sum(E_loc_terms) / b
        return expect_E_loc

    def forward(
        self,
        params: TensorType["n_params", "batch"],
        tokens: TensorType["seq", "batch", "occupation", "spin"],
    ):
        """
        Given a sequence of tokens and parameters, produces the probabilities and phases
        associated with the wave function that this model represents.
        """

        # TODO: the caller should broadcast the params tensor to the batch size of the tokens
        # before passing params into this

        assert (
            params.shape[0] == 5
        ), f"HubbardWaveFunction expects 5 parameters, got {params.shape[0]}"

        n_params = params.shape[0]

        logits = self.embedding(params, tokens)  # s b e
        logits = self.transformer_encoder(logits)  # s b e
        prob, phase = self.deembedding(
            logits[n_params:], calculate_phase=True
        )  # s b o sp

        idx = torch.argmax(prob, dim=-2)  # s b sp
        print(f"Shape of idx: {idx.shape}")
        print(f"Shape of prob: {prob.shape}")

        # Gather does prob[i, j, k, l] = prob[i, j, idx[i, j, k, l], l]

        # So this unsqueeze route was wrong
        idx = ein.rearrange(idx, "s b sp -> s b 1 sp")
        prob = prob.gather(-2, idx).squeeze(-2)  # s b sp
        phase = phase.gather(-2, idx).squeeze(-2)  # s b sp

        print(f"Shape of prob after gather: {prob.shape}")

        return prob, phase
