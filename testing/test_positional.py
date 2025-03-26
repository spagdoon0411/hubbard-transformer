from model.position_encoding import PositionEncoding
import torch

embed = 32
max_len = 100
wavelen_fact = 1e6
pe = PositionEncoding(embed, max_len, wavelen_fact)

seq = 10
batch = 24

x = torch.randn(seq, batch, embed)
old_x = x.clone()
assert x.shape == (seq, batch, embed)

x = pe(x)

# This operation mutates.
assert not torch.allclose(x, old_x)

# Just require that we preserve shape; in-place buffer operation
assert x.shape == (seq, batch, embed)
