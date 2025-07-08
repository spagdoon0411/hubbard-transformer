from torch import nn
import torch.nn.functional as F
import torch
from utils.logging import get_log_metric, tensor_to_string
import einops as ein
from model.site_degree_embedding import SiteDegreeEmbedding
import itertools as it
from functools import lru_cache
from model.param_embedding import SimpleParamEmbedding
from model.token_embedding import OccupationSpinEmbedding
from model.position_encoding import PositionEncoding
from model.hubbard_deembedding import HubbardDeembedding
from model.sampling import Sampling
from model.hamiltonian import HubbardHamiltonian

DUMP_SAMPLES = False


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
        diag: dict = {},
    ):
        super(HubbardWaveFunction, self).__init__()

        self.diag = diag

        n_params = 5  # t, U, mu, chain length, particle number
        token_dims = [2, 2]
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

        # Making this an attr of self will cause zero grads to be logged. This object is not
        # involved in computation graphs; copies of it (made by TransformerEncoder) are.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        self.logit_norm = nn.LayerNorm(
            embed_dim,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            mask_check=True,
            norm=self.logit_norm,
        )

        # Already applied in the transformer encoder
        # self.post_transform_norm = nn.LayerNorm(
        #     embed_dim,
        # )

        self.deembedding = HubbardDeembedding(
            embed_dim=embed_dim,
            target_token_dims=token_dims,
        )

        sampling_mask = torch.tril(
            torch.ones(
                max_len,
                max_len,
            )
        )

        self.sampling = Sampling(
            embed_dim=embed_dim,
            particle_number=particle_number,
            embedding_function=self.embedding,
            deembedding_function=self.deembedding,
            transformer_encoder=self.transformer_encoder,
            mask=sampling_mask,
        )

    def sample(
        self,
        num_chains: int,
        chain_length: int,
        params: torch.Tensor,  # (n_params, batch)
    ):
        """
        Produces num_chains most-probable token chains of length chain_length based
        on the wave function this model represents.
        """

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

        chains, log_probs = self.sampling.sample(
            params=params,
            tokens=tokens,
            up_to=chain_length,
        )

        return chains, log_probs

    def generate_basis_at_pnum(
        self,
        num_sites: int,
    ) -> torch.Tensor:
        """
        Produces a basis meeting particle number constraints in the s b o sp format.
        """

        # TODO: how does autograd view this function?

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

    @lru_cache(maxsize=None)  # Ideally we only cache once
    def generate_basis(
        self,
        num_sites: int,
    ) -> torch.Tensor:
        """
        Create a full basis of states for the Hubbard model, without particle
        number constraints.

        Args:
            num_sites (int): Number of sites in the system.

        Returns:
            torch.Tensor: Tensor representing all possible binary states.
                          Shape: (num_sites, 4^num_sites, 2, 2)
        """
        # Total number of particle slots (2 per site)
        num_slots = num_sites * 2

        # Generate all possible binary states (2^num_slots combinations)
        flat_states = torch.arange(2**num_slots, dtype=torch.long).unsqueeze(1)

        flat_states = torch.bitwise_and(
            flat_states >> torch.arange(num_slots, dtype=torch.long),
            torch.tensor(1, dtype=torch.long),
        ).T  # Shape: (num_slots, 2^num_slots)

        off_diag_mask = torch.eye(2**num_slots, dtype=torch.bool) == 0
        diffs = (flat_states.unsqueeze(1) - flat_states.unsqueeze(2)).abs()

        assert torch.all(
            ein.einsum(
                diffs[:, off_diag_mask],
                "s_sp n -> n",
            )
            != 0
        ), "All off-diagonal chain pairings should have at least one differing site"

        # These are now unique binary chains with as many slots for particles
        # as there are sites.

        # Reshape to s h o sp format
        flat_states = F.one_hot(
            flat_states, num_classes=2
        )  # Shape: (num_slots, 2^num_slots, 2)
        flat_states = ein.rearrange(
            flat_states,
            "(s sp) h o -> s h o sp",
            sp=2,
            s=num_sites,
        )

        return flat_states.to(dtype=torch.float32)

    def compute_basis_information(
        self,
        num_sites: int,
        params: torch.Tensor,
    ):
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

        basis = self.generate_basis_at_pnum(num_sites)  # (s, h, o, sp)
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

        psi = ein.reduce(
            psi,
            "s b sp -> b",
            reduction="prod",
        )  # b

        # Calculate probability normalization factor from sampling probabilities
        normalization_factor = ein.reduce(
            # The probability of a sample is a product across the chain
            ein.reduce(
                probs,
                "s b sp -> b",
                reduction="prod",
            ),
            # Then sum along all basis states to get the final normalization factor
            "b -> ",
            reduction="sum",
        )

        psi = psi / torch.sqrt(normalization_factor)

        # N = < psi' | psi' >

        # < psi' / sqrt(N) | psi' / sqrt(N) >
        # = (1 / N) ( < psi | psi > )

        return psi, basis, normalization_factor

    def _compute_e_loc(
        self,
        h_entries: torch.Tensor,
        sample_psi: torch.Tensor,  # < a | psi >
        basis_psi: torch.Tensor,  # < b | psi >
    ):

        sample_psi = torch.where(
            sample_psi.abs() < 1e-30,
            torch.ones_like(sample_psi) * 1e-30,
            sample_psi,
        )

        E_loc_values = ein.einsum(
            h_entries,
            sample_psi**-1,
            basis_psi,
            "b h, b, h -> b",  # TODO: does this scale by the b-values?
        )  # The bra-psi are the sampled states

        if E_loc_values.isnan().any():
            raise ValueError(
                "E_loc values contain NaNs but sample_psi and basis psi do not."
            )

        if met := get_log_metric(self.diag, "extra/avg_e_loc_summands"):
            E_loc_abs = ein.einsum(
                h_entries,
                sample_psi**-1,
                basis_psi,
                "b h, b, h -> b h",
            ).abs()

            mean_summands = ein.reduce(
                E_loc_abs,
                "b h -> b",
                reduction="mean",
            )

            met.log(tensor_to_string(mean_summands))

        return E_loc_values  # b

    def e_loc(
        self,
        hamiltonian: HubbardHamiltonian,
        params: torch.Tensor,  # (n_params,)
        sampled_states: torch.Tensor,  # (seq, batch, occupation, spin)
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

        basis_psi, basis, _ = self.compute_basis_information(
            num_sites=s,
            params=params,  # n_params
        )  # (b, s h o sp)

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
        )  # s b sp

        sampled_psi = ein.reduce(
            sampled_psi,
            "s b sp -> b",
            reduction="prod",
        )

        # Calculate norms across the sampled states

        # NOTE: we should have one psi-value per basis entry

        # Invididual entries of < a | H | b > where the sampled states (axis b)
        # are the bras and the basis states (axis h) are the kets
        entries = hamiltonian.entry(
            a=sampled_states,
            b=basis,
        )  # b h

        E_loc_values = self._compute_e_loc(
            h_entries=entries,
            sample_psi=sampled_psi,
            basis_psi=basis_psi,
        )

        return E_loc_values

    def surrogate_loss(self, log_probs, e_loc_values):
        """
        d_theta [ < E_loc(x) > ]
        = < E_loc(x) * d_theta [ log p(x; theta) ] >
        = (1 / N) sum_x ( E_loc(x) * d_theta [ log p(x; theta) ] )
        = d_theta [ (1 / N) sum_x ( E_loc(x) * log p(x; theta) ) ]

        ...which we know to be the gradient of the expectation of E_loc. We
        can apply the regular score function sampling because E_loc is a function
        of the samples themselves, not of model params.
        """

        loss = ein.reduce(log_probs * e_loc_values, "b -> ", reduction="mean")
        return loss

    def forward(
        self,
        params: torch.Tensor,  # (n_params, batch)
        tokens: torch.Tensor,  # (seq, batch, occupation, spin)
    ):
        """
        Given a sequence of tokens and parameters, produces the probabilities and phases
        associated with the wave function that this model represents.
        """

        assert (
            params.shape[0] == 5
        ), f"HubbardWaveFunction expects 5 parameters, got {params.shape[0]}"

        n_params = params.shape[0]

        logits = self.embedding(params, tokens)  # s b e
        logits = self.transformer_encoder(logits)  # s b e

        # Already occurs in transformer_encoder
        # logits = self.post_transform_norm(logits)  # s b e

        prob, phase = self.deembedding(
            logits[n_params:], calculate_phase=True
        )  # s b o sp

        idx = torch.argmax(prob, dim=-2)  # s b sp

        # Gather does prob[i, j, k, l] = prob[i, j, idx[i, j, k, l], l]

        idx = ein.rearrange(idx, "s b sp -> s b 1 sp")
        prob = prob.gather(-2, idx).squeeze(-2)  # s b sp
        phase = phase.gather(-2, idx).squeeze(-2)  # s b sp

        return prob, phase
