# Case Study: SchNet on Neuron

*A molecular GNN, its scatter/gather bottleneck, and the matmul trick that gives us 14.7× speedup.*

```{admonition} Run it yourself
:class: tip
Two scripts accompany this chapter (run on a trn2.3xlarge with the Neuron venv activated):

- `scripts/ch7_schnet_example.py` — profiles the original SchNet, shows fallbacks, benchmarks `torch.compile`
- `scripts/ch7_schnet_example_matmulfix.py` — applies the matmul reformulation and benchmarks the speedup

The original SchNetPack library is open source: [github.com/atomistic-machine-learning/schnetpack](https://github.com/atomistic-machine-learning/schnetpack.git) (we use commit `de850eb`).
```

---

## The Model: SchNet

SchNet is a message-passing neural network for predicting molecular properties — energies, forces, potential energy surfaces. It's the workhorse of computational chemistry: drug discovery, materials science, and molecular dynamics all rely on architectures like this.

Each interaction layer does three things:

```python
def forward(self, x, f_ij, idx_i, idx_j, rcut_ij):
    x = self.in2f(x)                              # linear projection
    Wij = self.filter_network(f_ij) * rcut_ij     # edge weights from distances
    
    x_j = x[idx_j]                                # ← GATHER neighbor features
    x_ij = x_j * Wij                              # weight the messages
    x = scatter_add(x_ij, idx_i, dim_size=N)      # ← SCATTER sum into atoms
    
    x = self.f2out(x)                              # linear projection
    return x
```

The linear projections and element-wise ops are fine on any hardware. The **gather** (`x[idx_j]`) and **scatter_add** are the problem — they require indirect memory access that the NeuronCore's tensor engine can't do natively.

---

## Move to Neuron — it just works (slowly)

```python
model = SchNet(n_atom_basis=128, n_interactions=6, ...).eval()
model_neuron = model.to("neuron")
output = model_neuron(inputs)  # ✓ correct output, max diff vs CPU: 0.000012
```

Zero code changes needed. But profiling reveals the cost:

```
Top ops by CPU time:
  aten::index (gather)       ×6    18.3ms   ← falls back to CPU
  aten::index_add (scatter)  ×6     5.5ms   ← falls back to CPU
  aten::to (device transfer) ×55    5.1ms   ← PCIe round-trips

Fallback overhead: 22.7% of forward pass
```

Every interaction layer triggers two CPU fallbacks plus the PCIe data transfers to move tensors back and forth. The tensor engine sits idle while the CPU handles these ops.

`torch.compile` helps — it fuses the linear/activation sequences between fallbacks — but can't eliminate the fallbacks themselves:

| Mode | Time | vs Eager |
|------|------|----------|
| Eager | 28.3ms | — |
| Compiled | 4.3ms | 6.5× |
| CPU baseline | 2.3ms | — |

The compiled model is still **slower than CPU** for small molecules because the fallback overhead exceeds the compute benefit.

---

## Understanding Gather and Scatter

To fix the bottleneck, we need to understand exactly what these operations do. Let's trace through a concrete example.

Consider a tiny molecule with 4 atoms and 6 directed edges:

```{figure} ../assets/gnn_image.png
:width: 400px
:align: center

A 4-atom graph. Each arrow is a directed edge carrying a message from source → destination. idx_i stores destinations, idx_j stores sources.
```

```python
import torch

# 4 atoms, each with a 2D feature vector
x = torch.tensor([[1.0, 2.0],   # atom 0
                  [3.0, 4.0],   # atom 1
                  [5.0, 6.0],   # atom 2
                  [7.0, 8.0]])  # atom 3

# Edge list: idx_i[e] = destination, idx_j[e] = source
idx_i = torch.tensor([0, 0, 1, 1, 2, 3])  # where messages go TO
idx_j = torch.tensor([1, 2, 0, 3, 0, 1])  # where messages come FROM

N = 4  # atoms
E = 6  # edges
```

### Gather: `x[idx_j]`

"For each edge, grab the feature vector of the source atom."

```python
x_j = x[idx_j]
# tensor([[3., 4.],   ← x[1], edge 0 reads from atom 1
#         [5., 6.],   ← x[2], edge 1 reads from atom 2
#         [1., 2.],   ← x[0], edge 2 reads from atom 0
#         [7., 8.],   ← x[3], edge 3 reads from atom 3
#         [1., 2.],   ← x[0], edge 4 reads from atom 0
#         [3., 4.]])  ← x[1], edge 5 reads from atom 1
```

This is **fancy indexing** — non-contiguous memory access. The hardware chases pointers into random locations of x. On Neuron, this falls back to CPU because the DMA engines only handle contiguous blocks.

### Scatter-add: `index_add_(0, idx_i, x_ij)`

"For each edge, ADD its message into the destination atom's accumulator."

```python
x_ij = torch.tensor([[0.3, 0.4],   # edge 0 → atom 0
                     [0.5, 0.6],   # edge 1 → atom 0
                     [0.1, 0.2],   # edge 2 → atom 1
                     [0.7, 0.8],   # edge 3 → atom 1
                     [0.9, 1.0],   # edge 4 → atom 2
                     [0.2, 0.3]])  # edge 5 → atom 3

out = torch.zeros(N, 2)
out.index_add_(0, idx_i, x_ij)
# tensor([[0.8, 1.0],   ← edges 0,1 summed into atom 0
#         [0.8, 1.0],   ← edges 2,3 summed into atom 1
#         [0.9, 1.0],   ← edge 4 into atom 2
#         [0.2, 0.3]])  ← edge 5 into atom 3
```

This is a **reduction with irregular grouping** — multiple edges write to the same atom. The hardware must handle atomic accumulation into arbitrary positions. Also falls back to CPU.

---

## The Matmul Reformulation

Here's the key insight: **any gather or scatter on a 1D index can be expressed as multiplication by a binary matrix.** And matrix multiply is exactly what the tensor engine is built for.

### Gather → G @ x

We construct a **gather matrix** G of shape [E edges × N atoms] where each row has a single 1 in the column of the source atom:

```python
G = torch.zeros(E, N)
G[torch.arange(E), idx_j] = 1.0
```

| | atom 0 | atom 1 | atom 2 | atom 3 |
|---|:---:|:---:|:---:|:---:|
| edge 0 (src=1) | 0 | **1** | 0 | 0 |
| edge 1 (src=2) | 0 | 0 | **1** | 0 |
| edge 2 (src=0) | **1** | 0 | 0 | 0 |
| edge 3 (src=3) | 0 | 0 | 0 | **1** |
| edge 4 (src=0) | **1** | 0 | 0 | 0 |
| edge 5 (src=1) | 0 | **1** | 0 | 0 |

Each row has exactly one 1, so `G[e, :] @ x` picks out row `idx_j[e]` of x — exactly what `x[idx_j[e]]` does.

```python
x_j_matmul = G @ x
assert torch.allclose(x_j, x_j_matmul)  # ✓ identical
```

### Scatter-add → A @ x_ij

We construct a **scatter matrix** A of shape [N atoms × E edges] where row `a` has 1s in every column where `idx_i == a`:

```python
A = torch.zeros(N, E)
A[idx_i, torch.arange(E)] = 1.0
```

| | edge 0 | edge 1 | edge 2 | edge 3 | edge 4 | edge 5 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| atom 0 (receives 0,1) | **1** | **1** | 0 | 0 | 0 | 0 |
| atom 1 (receives 2,3) | 0 | 0 | **1** | **1** | 0 | 0 |
| atom 2 (receives 4) | 0 | 0 | 0 | 0 | **1** | 0 |
| atom 3 (receives 5) | 0 | 0 | 0 | 0 | 0 | **1** |

Row `a` of A sums all edge messages destined for atom a — exactly what scatter_add does.

```python
out_matmul = A @ x_ij
assert torch.allclose(out_scatter, out_matmul)  # ✓ identical
```

### The full reformulated forward pass

```python
# ORIGINAL (irregular memory access → CPU fallback):
x_j = x[idx_j]                    # gather: chase pointers
x_ij = x_j * Wij                  # element-wise (fine)
out = zeros(N, F)
out.index_add_(0, idx_i, x_ij)    # scatter: atomic adds to random locations

# REFORMULATED (pure matmul → tensor engine):
x_j = G @ x                       # gather: [E×N] @ [N×F] → [E×F]
x_ij = x_j * Wij                  # element-wise (unchanged)
out = A @ x_ij                    # scatter: [N×E] @ [E×F] → [N×F]
```

Both produce **identical outputs**. The matmul version just expresses indexing logic as matrix multiplication with binary (0/1) matrices.

```{admonition} Why does this help?
:class: tip

| | Original | Matmul |
|---|---|---|
| Gather | CPU fallback (indirect memory) | Tensor engine GEMM |
| Scatter | CPU fallback (atomic reduction) | Tensor engine GEMM |
| Compiler | Can't fuse across fallbacks | Fuses entire layer into one NEFF |

The tensor engine is 90% of your available FLOPs. Feeding it work — even "wasteful" dense matmuls over sparse binary matrices — beats falling back to CPU.
```

### End-to-end verification

```python
# Full message passing — original vs matmul
x_j_orig = x[idx_j]
msg_orig = x_j_orig * 0.5   # simplified edge weight
out_orig = torch.zeros(N, 2)
out_orig.index_add_(0, idx_i, msg_orig)

x_j_mat = G @ x
msg_mat = x_j_mat * 0.5
out_mat = A @ msg_mat

assert torch.allclose(out_orig, out_mat)
# ✓ Full message passing: original == matmul reformulation
```

---

## Patching SchNet

Applying this to the real model takes 15 lines. We monkey-patch the forward method of each interaction layer:

```python
import types

def matmul_forward(self, x, f_ij, idx_i, idx_j, rcut_ij):
    """Drop-in replacement for SchNetInteraction.forward."""
    x = self.in2f(x)
    Wij = self.filter_network(f_ij) * rcut_ij[:, None]
    x_j = self.G @ x        # gather via matmul
    x_ij = x_j * Wij
    x = self.A @ x_ij       # scatter via matmul
    x = self.f2out(x)
    return x

def patch_schnet_with_matmul(model, idx_i, idx_j, n_atoms, n_pairs, device):
    G = torch.zeros(n_pairs, n_atoms, dtype=torch.float32, device=device)
    G[torch.arange(n_pairs, device=device), idx_j] = 1.0

    A = torch.zeros(n_atoms, n_pairs, dtype=torch.float32, device=device)
    A[idx_i, torch.arange(n_pairs, device=device)] = 1.0

    for interaction in model.interactions:
        interaction.register_buffer("G", G)
        interaction.register_buffer("A", A)
        interaction.forward = types.MethodType(matmul_forward, interaction)
```

Correctness check on CPU (same weights, same inputs, before and after patching):

```
Max diff: 0.000001
✓ Matmul reformulation is numerically exact
```

---

## Benchmark Results

Compiled matmul-reformulated SchNet vs the original, on trn2.3xlarge:

| Config | CPU | Original (Neuron) | Matmul+Compile (Neuron) | vs CPU | vs Original |
|--------|-----|-------------------|------------------------|--------|-------------|
| 50 atoms, 1K edges | 2.32ms | 4.31ms | 1.60ms | **1.4×** | 2.7× |
| 200 atoms, 6K edges | 8.32ms | 15.98ms | 1.45ms | **5.7×** | 11× |
| 500 atoms, 20K edges | 33.09ms | 41.66ms | 2.83ms | **11.7×** | 14.7× |

The original compiled model is *slower than CPU* at every scale because of scatter/gather fallbacks. The matmul reformulation eliminates all fallbacks, letting the compiler fuse the entire interaction layer — and the speedup grows with molecule size because larger matmuls better utilize the systolic array.

```{admonition} Why the scaling is superlinear
:class: note
At 50 atoms the matmul is [1000×50] @ [50×128] — too small to fill the tensor engine's pipeline. At 500 atoms it's [20000×500] @ [500×128] — large enough for efficient systolic array utilization. The compute intensity crosses the roofline threshold.
```

---

## The Memory Tradeoff

G and A are mostly zeros. The memory cost:

| System | Atoms | Edges | G+A Size | Fits in... |
|--------|-------|-------|----------|------------|
| Drug molecule | 50 | 1,000 | 0.4 MB | SBUF (28 MB) ✓ |
| Protein fragment | 200 | 6,000 | 9.2 MB | SBUF ✓ |
| Small protein | 500 | 20,000 | 76 MB | HBM (32 GB) ✓ |
| Full protein | 5,000 | 200,000 | 7.6 GB | Tight in HBM ⚠️ |

For drug discovery (< 1000 atoms) this approach works perfectly. For larger systems, you'd need either:
- A **block-sparse** variant that tiles the adjacency matrices
- A **custom NKI kernel** that operates on the edge list directly (Part V)

```{admonition} When is the graph static enough?
:class: note
The G and A matrices must be rebuilt when the neighbor list changes. In molecular dynamics with a cutoff radius, neighbor lists typically update every 10–50 timesteps. The matrix construction cost (negligible at these sizes) is amortized over many forward passes.
```

---

## The Reformulation Principle

```{admonition} The deepest performance insight for Neuron
:class: important
If the tensor engine can't do your operation directly, ask: **can I reformulate it as something it CAN do?**

The tensor engine is a systolic array. It does one thing: multiply matrices. Everything else — scatter, gather, sort, argmax, custom reductions — either falls back to CPU or runs on the much weaker scalar/vector engines.

Feeding the tensor engine "wasteful" dense matmuls over sparse binary data often beats the alternative of falling back to CPU. The math is the same. The hardware utilization is completely different.
```

This principle extends beyond GNNs:
- **Sparse attention** → dense matmul with masking
- **Embedding lookup** → one-hot @ embedding table
- **Histogram/bincount** → one-hot transpose @ values
- **Top-k selection** → sort via matmul with comparison matrices

Any time you see `tensor[index]` in a hot path on Neuron, ask: "can I express this as a matmul?"

---

## Summary

| Step | What we learned |
|------|----------------|
| `model.to("neuron")` | Works with zero changes, numerically correct |
| Profile | 22.7% of time is fallback overhead (gather + scatter) |
| `torch.compile` | 6.5× over eager, but fallbacks remain → still slower than CPU |
| Matmul reformulation | Numerically exact, eliminates all fallbacks |
| Benchmark | **14.7× faster** than original Neuron, **11.7× faster** than CPU |

The lesson: **correctness is free, performance requires understanding.** The compiler can't fix a hardware mismatch. Understanding that scatter/gather = matmul with binary matrices lets you convert an operation the hardware *can't do* into one it's *optimized for*.

---

*Next: we'll look at how to measure where time actually goes at the hardware level — is it compute-bound or memory-bound? The profiling tools that answer this question.*
