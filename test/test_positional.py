import pytest
import torch
from model.position_encoding import PositionEncoding


@pytest.fixture
def setup_position_encoding():
    embed = 32
    max_len = 100
    wavelen_fact = 1e6
    pe = PositionEncoding(embed, max_len, wavelen_fact)
    return pe, embed


def test_position_encoding_mutation(setup_position_encoding):
    pe, embed = setup_position_encoding
    seq = 10
    batch = 24

    x = torch.randn(seq, batch, embed)
    old_x = x.clone()

    assert x.shape == (seq, batch, embed)

    x = pe(x)

    # Actually did something
    assert not torch.allclose(x, old_x)

    # Just require that we preserve shape; in-place buffer operation
    assert x.shape == (seq, batch, embed)
