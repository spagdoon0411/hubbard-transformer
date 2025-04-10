import torch
import torch.nn.functional as F
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fill_to_particle_num(tens, target_part_num):
    """
    Given a tensor of occupations, fills it to contain expactly particle_num
    particles in total. Consumes a token tensor and provides an updated token tensor.

    TODO: create a nontrivial mapping from occupation index to occupation number.
    Can this reduce overhead?
    """

    seq, batch, occ = tens.shape
    max_occ_idx = occ - 1

    site_occs = tens.argmax(dim=-1)  # (s, b)
    original_site_occs = site_occs.clone()

    # Count the additions necessary to reach the target number of particles.
    # Note the negative adjustments, save a negatives mask, and count particle
    # holes instead of particles for those sequences
    diffs = site_occs.sum(dim=0)  # (b,)
    diffs = target_part_num - diffs  # (b,)
    logger.debug("proposed particle number change: %s", diffs)

    negative_diffs = diffs < 0  # (b,)
    diffs = diffs.abs()  # (b,)
    site_occs[:, negative_diffs] = max_occ_idx - site_occs[:, negative_diffs]

    assert diffs.shape == (batch,)
    assert negative_diffs.shape == (batch,)
    assert site_occs.shape == (seq, batch)

    adjustments_needed = diffs > 0  # (b,)
    while adjustments_needed.sum().item():
        available_sites = site_occs < max_occ_idx

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
    final_site_occs = site_occs.clone()
    new_tokens = F.one_hot(site_occs, num_classes=occ)  # (s, b, o)

    pnum_o = original_site_occs.sum(dim=0)
    pnum_f = final_site_occs.sum(dim=0)

    logger.debug("Original particle numbers: %s", pnum_o)
    logger.debug("Final particle numbers: %s", pnum_f)
    logger.debug("Change in particle numbers: %s", pnum_f - pnum_o)

    assert new_tokens.shape == (seq, batch, occ)
    return new_tokens


TEST_SEQ = 30
TEST_BATCH = 32
TEST_MAX_OCC = 40  # The largest number of particles a site can hold
ADD_PARTICLES = 100

tens = torch.randint(
    0,
    TEST_MAX_OCC + 1,  # [0, TEST_MAX_OCC] inclusive
    (TEST_SEQ, TEST_BATCH),
)

curr_pnum = tens.sum(dim=0)
target_pnum = curr_pnum.max().item() + ADD_PARTICLES
original_tens = tens.clone()

# (s, b, o)
tens = F.one_hot(tens, num_classes=TEST_MAX_OCC + 1)

res = fill_to_particle_num(
    tens,
    target_pnum,
)

res_occupation_numbers = res.argmax(dim=-1).sum(dim=0)

assert torch.all(
    res_occupation_numbers == target_pnum
), "Particle number condition was not met"
