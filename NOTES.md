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

