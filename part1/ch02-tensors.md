# Tensors aren't just arrays

*ESM-2's embedding table: vocab_size × hidden_dim. How is this stored?*

**Core insight:** A tensor is not data — it's a *view* into data. It's a pointer + shape + strides. This distinction has direct performance consequences.

```{admonition} TODO
:class: warning

### Memory layout basics
- A tensor = data pointer + shape + strides + dtype + device
- Row-major (C-contiguous) vs column-major (Fortran) — PyTorch is row-major by default
- ESM-2's embedding table: `[vocab_size=33, hidden_dim=1280]` — how is this laid out in memory?

### Strides: the navigation system
- Stride = "how many elements to skip to reach the next index in this dimension"
- `embed.stride()` → `(1280, 1)` — next row = skip 1280, next column = skip 1
- Computing memory offset: `embed[i, j]` lives at `data_ptr + (i * stride[0] + j * stride[1]) * element_size`

### Views vs copies (zero-cost reshaping)
- `.view()`, `.reshape()`, slicing, `.transpose()` — when do they copy?
- Rule: if the new shape is compatible with existing strides → view (free). Otherwise → copy (expensive).
- `.transpose()` is always free — it just swaps strides. But the tensor becomes **non-contiguous**.
- `.contiguous()` — forces a copy to make strides match the "expected" layout again

### Why contiguity matters for performance
- DMA engines (on both GPU and Neuron) load data in contiguous chunks
- Contiguous access = one large transfer. Strided access = many small transfers.
- Benchmark: row access vs column access on a large matrix (show the 2-5x difference)
- On Neuron: tiles loaded from HBM → SBUF must be contiguous for maximum bandwidth

### ESM-2 attention matrices: a real example
- Attention weights shape: `[batch, heads, seq_len, seq_len]`
- Accessing one head: `attn[0, 0]` — is this contiguous? (yes, because heads is dim 1)
- Accessing one position across heads: `attn[0, :, 5, :]` — is this contiguous? (no — strided access)
- This is why attention implementations carefully choose which dimension to parallelize over

### Layout choices that affect the dispatcher
- `channels_last` format for CNNs (relevant for U-Net guest example later)
- Sparse layouts (relevant for PyTorch Geometric later)
- The dispatcher picks different kernels based on layout — same op, different performance

### Connection to Ch 1 and forward to Ch 3
- Ch 1 showed the dispatcher routes ops. Ch 2 shows that *how the data is arranged* affects which kernel is optimal.
- Forward: "A compiler (Ch 3) can see these layout issues and insert `.contiguous()` calls or choose better tiling automatically."
```

*Question raised → "Can something optimize these memory access patterns for me?"*
