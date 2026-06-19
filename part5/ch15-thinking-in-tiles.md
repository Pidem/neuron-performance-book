# Thinking in tiles

*Decompose ESM-2's attention matmul into tile operations.*

```{admonition} TODO
:class: warning
Draft content.
```

```{admonition} NOTE — scatter→matmul reformulation as advanced technique
:class: tip
After teaching basic tiling, show the "reformulate irregular ops as matmul" trick:
- Mathematical equivalence: scatter_add(source, index) == A @ source, where A is a binary selection matrix (A[j,i]=1 if index[i]==j)
- On Neuron: the Tensor Engine can do this matmul at 16,384 MACs/cycle. The Scalar Engine doing sequential scatter gets 1 op/cycle. That's a 16,384× throughput difference.
- Tradeoff: memory O(nodes × edges) for the dense A matrix vs O(edges) for the index. Worth it when graph is small (< ~1000 nodes).
- For larger graphs: block-sparse matmul (if graph has community structure), segment reduction (sort by dest, reduce contiguous segments on Vector Engine), or bucketed scatter (group edges by dest, pad to tile boundary, batched matmul).
- This is the general principle: "if the Tensor Engine can't do your op, can you REFORMULATE it as something the Tensor Engine CAN do?"
- Other examples: softmax via matmul trick (multiply by ones vector for reduction), LayerNorm gamma broadcast via matmul against ones (from Jay Gray's RMSNorm-Quant kernel design).
```
