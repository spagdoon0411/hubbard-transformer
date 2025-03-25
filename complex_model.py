from torch import nn
import torch
import einops as ein


class ComplexAttention(nn.Module):
    """
    Multiheaded attention with a softmax supporting complex numbers.
    """

    def __init__(
        self,
        embed_dims: int,
        model_dims: int,  # Dimension of the key space
        n_heads: int,
        max_len: int,  # Used to create a mask
        dtype: torch.dtype = torch.complex64,
    ):
        super(ComplexAttention, self).__init__()

        if embed_dims % n_heads != 0:
            raise ValueError("Embedding dimensions must be divisible by n_heads")

        self.value_dims = embed_dims // n_heads
        value_dims = self.value_dims

        print("Using a value space of dimension", self.value_dims)

        self.dtype = dtype

        self.model_dims = torch.tensor(model_dims)

        self.w_k = nn.Parameter(
            torch.randn(n_heads, model_dims, embed_dims, dtype=dtype)
        )
        self.w_q = nn.Parameter(
            torch.randn(n_heads, model_dims, embed_dims, dtype=dtype)
        )
        self.w_v = nn.Parameter(
            torch.randn(n_heads, value_dims, embed_dims, dtype=dtype)
        )
        self.w_o = nn.Parameter(torch.randn(embed_dims, embed_dims, dtype=dtype))

        # (seq, seq)
        self.mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()

        # (1, 1, seq, seq)
        self.mask = self.mask.unsqueeze(0).unsqueeze(0)

        self.register_buffer("attention_mask", self.mask)

    def forward(self, logits: torch.Tensor):
        """
        logits: (seq, batch, embed)
        w_q: (embed, key, head)
        w_k: (embed, key, head)
        w_v: (embed, embed, head)

        # TODO: rotary embedding?

        Returns: (seq, batch, embed) - The updated logits after passing through
        multi-headed attention
        """

        # NOTE: batched matmul focuses on the two last dimensions, but
        # regular matmul is already "batched" where the second dimension of the
        # matrix being operated on is the batch dimension for the vectors. The
        # first dimension of some matrix is the "vector" dimension.

        # Thus for the pre-attention transformations (e.g., to key or value
        # space) the embeddings should be second to last and the last batch
        # axis is completely trivial.

        seq_len = logits.shape[0]

        logits = logits.unsqueeze(1)  # (seq, 1, batch, embed)
        logits = ein.rearrange(logits, "s u b e -> s u e b")  # (seq, 1, embed, batch)

        w_q = self.w_q.unsqueeze(0)  # (1, head, keys, embed)
        w_k = self.w_k.unsqueeze(0)  # (1, head, keys, embed)
        w_v = self.w_v.unsqueeze(0)  # (1, head, values, embed)
        w_o = self.w_o.unsqueeze(0)  # (1, head, embed, values)

        q = torch.matmul(w_q, logits)  # (seq, head, keys, batch)
        k = torch.matmul(w_k, logits)  # (seq, head, keys, batch)
        v = torch.matmul(w_v, logits)  # (seq, head, values, batch)

        # A good inner product between query and key vectors is the
        # standard inner product used in complex vector spaces.

        _q = ein.rearrange(q, "s h k b -> b h s k")
        _kT = ein.rearrange(k, "s h k b -> b h k s")

        # The most important thing to note here is that there are two
        # sequence dimensions: a sequence dimension for "relevances" and a
        # sequence dimension for the output embeddings.

        # QK^T is of dimension (batch, head, s2, s1), where s1 is the
        # relevance dimension and s2 is the output dimension. Later values
        # over s1 should be masked.

        # Applying the attention pattern to the value matrix results in a sum
        # across the sequence dimension of the embedding vectors, weighed by
        # the attention pattern entries. Thus the attention mask should
        # be applied to the axis of summation, as should the softmax.

        # TODO: does masked_fill do a broadcast?
        # (b, h, s2, s1).masked_fill((1, 1, s2, s1)) -> (b, h, s2, s1)
        mask_slice = self.attention_mask[:, :, :seq_len, :seq_len]
        attn_pattern = (
            torch.matmul(_q, _kT)
            .abs()
            .masked_fill(mask_slice, torch.tensor(float("-inf")))
        )
        attn_pattern = attn_pattern / torch.sqrt(self.model_dims)
        attn_pattern = torch.softmax(attn_pattern, dim=-1)
        attn_pattern = attn_pattern.to(dtype=self.dtype)  # (b, h, s2, s1)

        _v = ein.rearrange(v, "s1 h v b -> b h s1 v")

        # (b, h, s2, s1) @ (b, h, s1, v) -> (b, h, s2, v)
        updates = torch.matmul(attn_pattern, _v)
        updates = ein.rearrange(updates, "b h s2 v -> s2 b (h v)")

        # Updates: (s2 b e)

        # TODO: determine whether update masking was right

        # NOTE: w_o: ((h v) (h v)) = (e e)
        updates = torch.matmul(updates, w_o)

        logits = ein.rearrange(logits, "s u e b -> s b e u")
        logits = logits.squeeze(-1)  # (s, b, e)

        return updates + logits


class EmbeddingFF(nn.Module):
    def __init__(self, embed_dims: int, model_dims: int, n_heads: int):
        self.attention = ComplexAttention(embed_dims, model_dims, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dims, model_dims),
            nn.ReLU(),
            nn.Linear(model_dims, embed_dims),
        )

    def forward(self, x: torch.Tensor):
        return self.ff(self.attention(x))
