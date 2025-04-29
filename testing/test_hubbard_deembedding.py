from model.hubbard_deembedding import HubbardDeembedding
import torch

(seq, batch, embed) = (12, 16, 8)

possible_spins = 32
max_occ = 2

test_logits = torch.randn(seq, batch, embed)

de = HubbardDeembedding(
    embed_dim=embed,
    target_token_dims=[max_occ, possible_spins],
)

# TODO: include a phase head here
prob_dist = de(test_logits)

assert prob_dist.shape == (seq, batch, max_occ, possible_spins)

# Valid prob dist if we integrate along occupations
assert torch.allclose(prob_dist.sum(dim=-1), torch.ones(seq, batch, max_occ))

# If we pass in a single logit, we should get one back out:
prob_dist = de(test_logits[0, :, :])

assert prob_dist.shape == (batch, max_occ, possible_spins)

# Valid prob dist if we integrate along occupations
assert torch.allclose(prob_dist.sum(dim=-1), torch.ones(batch, max_occ))

# As if we're calculating a wave function
prob, phase = de(test_logits[:, :, :], calculate_phase=True)

assert prob.shape == (seq, batch, max_occ, possible_spins)
assert phase.shape == (seq, batch, max_occ, possible_spins)

psi = de.compute_psi(prob, phase)

# Creates complex wave function values that are separated by spin dimension
assert psi.shape == (seq, batch, possible_spins)
