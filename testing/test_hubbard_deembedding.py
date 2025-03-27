from model.hubbard_deembedding import HubbardDeembedding
import torch

(seq, batch, embed) = (12, 16, 8)

max_occ = 32
possible_spins = 2

test_logits = torch.randn(seq, batch, embed)

de = HubbardDeembedding(
    embed_dim=embed,
    target_token_dims=[possible_spins, max_occ],
)

# TODO: include a phase head here
prob_dist = de(test_logits)

assert prob_dist.shape == (seq, batch, possible_spins, max_occ)

# Valid prob dist if we integrate along occupations
assert torch.allclose(prob_dist.sum(dim=-1), torch.ones(seq, batch, possible_spins))

# If we pass in a single logit, we should get one back out:
prob_dist = de(test_logits[0, :, :])

assert prob_dist.shape == (batch, possible_spins, max_occ)

# Valid prob dist if we integrate along occupations
assert torch.allclose(prob_dist.sum(dim=-1), torch.ones(batch, possible_spins))
