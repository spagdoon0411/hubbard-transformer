TODO: try one round of attention to determine whether
allowing the params to interact with each other before
interacting with the rest of the system is a reasonable
choice.

```
[ M1, M2, M3, M4, ...] -> [comb(M1, M2, ...), comb(M1, M2, ...), ...]
[ M1, M2, M3, M4, ...] -> [M * x, M * y, M * z, ...]
[ M1, M2, M3, M4, ...] -> M * [x, y, z, ...]
[ M1, M2, M3, M4, ...] -> M * W
```

Left-multiplying a weight matrix by the input vectors results in columns
that are linear combinations of the inputs. TODO: is this the best way to
do it?

# Sanity-Preserving Architecture Choices
- For now use the plain QK^T version; seems like the best for sequence generation
- We definitely have a normalization problem
- For now, append across the spin dimension
- Position encoding is just sinusoidal real for now

# IMMEDIATE
- What dimension is the softmx applied across?

# Optimizations
- Abstractly make some memory optimizations involving sequentializing fundamentally parallel operations
- Use a static parameter buffer
- Use view_as_complex for some real-only static buffers (e.g., position encoding)

# Design/Research Questions 
- The softmax has a preferred axis; the decision of which to conjugate definitely matters. In fact, transposing $Q K ^ T$ matters
    1. Determine what happens when we transpose $Q K^T$
    2. Determine what happens when we choose to tranpose $Q K^T$ or not, and then further choose one to conjugate. How is the attention pattern affected?
- Conjugation is not linear and cannot be learned by W_k?
- Why does the convolution in the research paper's architecture make sense? Where is the convolution and what role does it play?
- What happens when we transpose queries and keys in the context of the attention mask? Does the order really matter?
- TODO: does the performance even change? Shouldn't I be able to think through this?
- Does it matter whether we conjugate the queries or whether we conjugate the keys?
- Should we use the complex inner product or should we use the Attention is All You Need $Q K^T$ when calculating similarities?
- Should position encoding be a plane wave instead of 
real sinusoids?
- Generalize the spin dimension to a more general "feature" dimension (and maybe ask for a pytree map or an einops string over indices to determine how they're interleaved)
- Should de-embedding step take the param context as a parameter?
- Can we just try doing a complex transformation to prob amps instead of doing a prob head and a phase head? The de-embedding head should simply use a matrix of complex type.