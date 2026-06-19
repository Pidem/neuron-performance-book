# Why custom kernels?

*The profiler shows a gap between matmuls in attention. The compiler can't fix it.*

```{admonition} TODO
:class: warning
Draft content.
```

```{admonition} NOTE — scatter_add as the opening motivation
:class: tip
Open with: "The compiler can't fuse what doesn't exist."
- Recall Ch 6: scatter_reduce falls back to CPU. torch.compile can't help — there's no Neuron kernel to compile TO.
- Show the table of missing ops from EquiformerV3: scatter_reduce, index_select, atan2, acos, linalg_cross, sort, repeat_interleave, index_fill_
- The NKI scatter_reduce kernel EXISTS (Ganfu built it) but is 4× slower than XLA because each NKI kernel dispatch reloads a 64KB binary over PCIe (~1.2s/call)
- Key insight: individual NKI op patches split the XLA-fused graph into separate NEFFs. The right approach is either (a) fuse the scatter INTO a larger kernel, or (b) reformulate the math to avoid scatter entirely.
- This motivates learning NKI: not just "write a kernel for one op" but "understand the hardware well enough to reformulate the problem"
```
