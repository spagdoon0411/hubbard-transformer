# Model Implementation Notes

## Position Encoding

Applied after the map from sequence tokens to embedding logits. Ripped straight
from Attention is All You Need (sinusoids with sines and cosines paired in twos
along the embedding dimension whose wavelengths form a geometric progression).

## Token Embeddings

We perform a map from (seq, batch, d1, ..., dn) tensors to (seq, batch, embed)
tensors.

Deciding whether this is purely one-hot is tough: using plain one-hot vectors
would allow a linear-map weight matrix to learn one embedding per input token
class. Space complexity: O(prod(d1, ..., dn) \* embed). Maybe there's a benefit
to a learned encoding?

Scope limitation: for now, stick to a one-hot encoding with a linear map from
flattened input token space to the embedding space. Complexify (e.g., with a
nonlinear map with lower time/space complexity) later.

## Sampling Routine

## Parameter Embeddings

## TODOs

- Better descriptions of each layer
