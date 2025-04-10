import torch
import torch.nn.functional as F


# tens: (seq, batch, occ)
def fill_to_particle_num(tens, particle_num):
    _, _, max_occ_idx = tens.shape

    site_totals = torch.argmax(tens, dim=-1)  # (s, b)

    # Determine the number of particles or holes that need to be added per
    # sequence
    part_populated = site_totals.sum(dim=0)  # (b,)
    rem_diffs = particle_num - part_populated
    negatives = rem_diffs < 0  # (b,)

    # seq_target should have dimensions of the batch size

    # For particles that need holes, start counting holes instead of particles
    part_populated = max_occ_idx - part_populated

    # Number of objects (particles, holes) to distribute
    rem_diffs = rem_diffs.abs()  # (b,)

    # While we still have batch members that need attention...
    batch_remaining = rem_diffs > 0  # (b,), new index space along the batch dim
    while (num_remaining := batch_remaining.sum().item()) != 0:
        available_sites = site_totals < max_occ_idx

        # For each batch member, pick a site to populate from those available
        rand_field = torch.rand(available_sites.shape)
        rand_field[~available_sites] = -torch.inf  # Disqualify full sites

        # The site chosen for adjustment for each batch member
        seq_target = torch.argmax(rand_field, dim=0)  # (b,)

        # Should we calculate adjustment room for everything or just for the sequences that
        # we're messing with?
        adjustment_room = (
            max_occ_idx - site_totals[seq_target, torch.arange(site_totals.shape[1])]
        )  # (b,)

        # (b,) - bumps to make to selected sites
        inner_adjust = torch.rand(batch_remaining.shape)
        inner_adjust = (inner_adjust * (adjustment_room + 1)).to(torch.int)

        # NOTE: seq_target should be the exact size of batch_remaining

        # seq_target is a batch index that provides sequence indices
        # what happens when the broadcasting spreads the same index across
        # the other dimension?

        num_batch_remaining = batch_remaining.sum().item()
        tmp_site_totals = site_totals[:, batch_remaining]
        tmp_site_totals[seq_target, torch.arange(num_batch_remaining)] += inner_adjust
        part_populated += inner_adjust.sum(dim=0)

        # Account for the distribution in the remaining diffs to distribute
        rem_diffs[seq_target] -= inner_adjust

    # Un-invert holes into particles now that holes have been distributed
    site_totals[negatives] = max_occ_idx - site_totals[negatives]

    # One-hot encode the new particle numbers
    new_tokens = F.one_hot(
        site_totals,
        num_classes=max_occ_idx + 1,
    )  # (s, b, o)

    return new_tokens


TEST_SEQ = 16
TEST_BATCH = 32
TEST_MAX_OCC = 10
ADD_PARTICLES = 100

tens = torch.randint(
    0,
    TEST_MAX_OCC,  # [0, TEST_MAX_OCC] inclusive
    (TEST_SEQ, TEST_BATCH),
)

curr_pnum = tens.sum(dim=0)
target_pnum = curr_pnum + ADD_PARTICLES

tens = F.one_hot(tens, num_classes=TEST_MAX_OCC)

res = fill_to_particle_num(
    tens,
    target_pnum,
)
