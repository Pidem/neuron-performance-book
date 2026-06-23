# Example 1: Getting SchNet running on Neuron

*You have a molecular GNN. You want it on Neuron. What breaks, what works, and what do you do about it?*

---

## Introducing SchNet

SchNet is a message-passing neural network for predicting molecular properties — energies, forces, potential energy surfaces. It uses continuous-filter convolutions on atom positions with radial basis functions and stacked interaction blocks with residual connections. It's the workhorse of computational chemistry: drug discovery (binding affinity), materials science, and molecular dynamics simulations all rely on architectures like this.

The key characteristic for this chapter: SchNet uses **scatter/gather operations** in every interaction layer. Message passing in GNNs means "sum messages from neighbors into each atom" — and that's an `index_add` (scatter) operation. Let's see what happens when we bring this to Neuron.

---

## Move to Neuron — it just works

```python
import torch
from schnetpack.representation import SchNet
from schnetpack.nn.radial import GaussianRBF
from schnetpack.nn.cutoff import CosineCutoff
import schnetpack.properties as properties

model = SchNet(
    n_atom_basis=128,
    n_interactions=6,
    radial_basis=GaussianRBF(n_rbf=20, cutoff=5.0),
    cutoff_fn=CosineCutoff(cutoff=5.0),
).eval()

# Synthetic inputs: 50-atom molecule with 20 neighbors each
n_atoms, n_neighbors = 50, 20
n_pairs = n_atoms * n_neighbors
inputs = {
    properties.Z: torch.randint(1, 10, (n_atoms,)).to("neuron"),
    properties.Rij: torch.randn(n_pairs, 3).to("neuron"),
    properties.idx_i: torch.randint(0, n_atoms, (n_pairs,)).to("neuron"),
    properties.idx_j: torch.randint(0, n_atoms, (n_pairs,)).to("neuron"),
}

model_neuron = model.to("neuron")
with torch.no_grad():
    output = model_neuron(inputs)

print(f"Output device: {output['scalar_representation'].device}")  # neuron:0
print(f"Output shape: {output['scalar_representation'].shape}")    # [50, 128]
print(f"Matches CPU: {torch.allclose(...)}")                       # True (max diff: 0.000012)
```

**Zero code changes.** The model runs, produces correct output (verified against CPU within 12 µ tolerance), and the first call takes ~15 seconds (JIT compilation of each op into NEFFs). Subsequent calls hit the cache.

---

## The fallback list — what landed on CPU?

Profiling the eager forward pass reveals where time actually goes:

```none
Top ops by CPU time:
  aten::index                                        ×6       18.3ms
  _scatter_add / aten::index_add                     ×6        5.5ms
  aten::to / aten::_to_copy (device transfers)       ×55       5.1ms
  aten::linear                                       ×30       0.4ms
  aten::softplus                                     ×12       0.1ms

Fallback overhead (to/copy ops): 9.75 ms / 42.94 ms total
Fallback fraction: 22.7%
```

Two operations dominate, called 6 times each (once per interaction layer):
1. **`aten::index`** (18.3ms) — the neighbor gather `x[idx_j]`. Fancy indexing with an integer index tensor.
2. **`aten::index_add`** (5.5ms) — the message aggregation `scatter_add(x_ij, idx_i)`. Summing messages into target atoms.

The 55 `aten::to` / `aten::_to_copy` calls are the PCIe round-trips: tensors moving from HBM → CPU → HBM for each fallback. **22.7% of the forward pass is pure data movement overhead** — not compute.

```{admonition} Why these ops fall back
:class: note
`aten::index` with an integer index tensor requires **indirect memory access** — reading from arbitrary, non-contiguous locations determined by the index values at runtime. The NeuronCore's DMA engines operate on contiguous blocks; they can't do random gather natively. Similarly, `index_add` (scatter) requires atomic accumulation into arbitrary positions — a sequential operation that doesn't map to the systolic array or vector engines.
```

---

## Compiling SchNet — what improves, what doesn't

```python
compiled_model = torch.compile(model_neuron, backend="neuron")
```

```none
Compilation time: 1.4s
NEFFs generated: 37
Dynamo graph analysis: Graph Count: 1, Graph Break Count: 0, Op Count: 106
```

Interesting result: **zero graph breaks**. Dynamo successfully traced the entire model — including `index`, `index_add`, and `getitem` — into a single graph of 106 ops. These ops have Dynamo-level support; they simply execute via CPU fallback *within* the compiled graph. The 37 NEFFs represent the segments between fallback boundaries at the HLO compilation level.

**Performance:**
- Eager: 28.3ms
- Compiled: 4.3ms
- **Speedup: 6.5×**

The 6.5× speedup comes from Dynamo eliminating Python dispatch overhead and the Neuron backend fusing the `linear` + `softplus` + `mul` sequences between fallback ops. But the fallbacks themselves remain — the compiler can't eliminate what has no native implementation.

---

## The scaling question — when does Neuron win?

At 50 atoms, Neuron is **slower than CPU** (4.3ms vs 2.5ms). The fallback overhead exceeds the compute benefit. But what happens as molecules get larger?

```none
  Config                     Atoms    Pairs   CPU ms  Neuron ms  Speedup
  ------------------------- ------ -------- -------- ---------- --------
  small molecule                50     1000     2.48       4.29     0.58×
  medium protein fragment      200     6000    10.46      16.06     0.65×
  small protein                500    25000    38.43      51.55     0.75×
  medium protein              1000    50000    86.45      92.57     0.93×
  large protein               2000   100000   311.44     172.55     1.80×
```

```{admonition} The crossover point
:class: important
At ~1500 atoms, Neuron breaks even with CPU. Above 2000 atoms, Neuron is **1.8× faster** — despite the fallbacks still happening every layer. The linear algebra in the filter network (30 `linear` ops per forward pass) finally dominates over the 12 scatter/index fallbacks.
```

The ratio converges toward 1.0 around 1000 atoms, then Neuron pulls ahead. This is the fundamental tradeoff: **fixed fallback cost** (PCIe round-trips don't depend on tensor size) vs **scaling compute benefit** (matmuls scale with atom count). For large enough systems, the matmuls win.

---

## The decision framework

For each fallback op, ask three questions:

| Question | `aten::index` (neighbor gather) | `aten::index_add` (scatter) |
|----------|-------------------------------|---------------------------|
| In the hot path? | Yes — every layer × 6 layers | Yes — every layer × 6 layers |
| How much data moves? | `[n_pairs, 128]` — scales with graph size | `[n_pairs, 128]` — same |
| Neuron-native alternative? | Reformulate as matmul (see below) | Reformulate as matmul |

**Decision matrix:**
- **Small molecules (< 1000 atoms):** keep on CPU, or reformulate scatter→matmul
- **Large molecules (> 1500 atoms):** Neuron wins even WITH fallbacks — deploy as-is
- **Maximum performance at any scale:** NKI kernel for scatter/gather (Part V)

---

## The scatter-as-matmul reformulation

`scatter_add(x_ij, idx_i, dim_size=n_atoms)` is mathematically equivalent to `A @ x_ij` where A is a binary selection matrix:

```python
# Build adjacency matrix: A[j, i] = 1 if edge i targets atom j
A = torch.zeros(n_atoms, n_pairs, dtype=torch.bfloat16)
A[idx_i, torch.arange(n_pairs)] = 1.0

# These produce identical results:
scatter_result = torch.zeros(n_atoms, 128).index_add(0, idx_i, x_ij)
matmul_result = A @ x_ij
# max diff: 0.000004
```

On Neuron, the matmul runs natively on the tensor engine — no fallback:

```none
scatter_add (with CPU fallback): 1.280 ms
matmul (native on Neuron):       0.215 ms
Speedup from reformulation:      6.0×
```

**The tradeoff:** the dense adjacency matrix costs O(atoms × edges) memory.
- 50 atoms × 1000 pairs = 98 KB — trivial
- 500 atoms × 25000 pairs = 24 MB — fits in SBUF
- 2000 atoms × 100000 pairs = 382 MB — **too large**, needs sparse/blocked approach

For small-to-medium graphs, this reformulation eliminates the fallback entirely and keeps everything on the tensor engine. For large graphs, you'd need a block-sparse variant or an NKI kernel.

```{admonition} The reformulation principle
:class: tip
This is the deepest performance engineering insight for Neuron: **if the tensor engine can't do your operation directly, ask whether you can reformulate it as something it CAN do.** The tensor engine is 90% of your available FLOPs. Feeding it work — even "wasteful" dense matmuls over sparse data — often beats the alternative of falling back to CPU or using the weaker scalar engine.
```

---

## When eager is enough

Not every model needs `torch.compile`. The two-persona workflow:

- **Research persona:** eager mode. Fast iteration, `print()` works, `pdb` works, instant feedback. You're scanning hyperparameters or debugging a new architecture — compilation overhead isn't worth it.
- **Production persona:** compiled mode. You've frozen the architecture, you know the input shapes, and you want maximum throughput for deployment.

Rule of thumb: if your forward pass is < 10ms in eager, torch.compile's overhead (guard checking, recompilation risk) may not pay off. For SchNet at 50 atoms, eager is actually fine — the model is too small for Neuron anyway.

---

## Summary

| What we tried | Result |
|---|---|
| `model.to("neuron")` | Works with zero code changes, numerically correct |
| Profile eager mode | 22.7% of time is fallback overhead (index + scatter) |
| `torch.compile` | 6.5× over eager, zero graph breaks, but fallbacks remain |
| Scale to 2000 atoms | Neuron 1.8× faster than CPU despite fallbacks |
| Scatter → matmul | 6× speedup on the isolated op, eliminates fallback entirely |

The story: **correctness is free, performance requires understanding.** SchNet runs on Neuron instantly. Making it run *fast* requires knowing where the bottlenecks are (profiling), understanding why they exist (hardware architecture from Ch 4-5), and choosing the right fix (reformulation, scaling up, or eventually writing a kernel).

---

*The compiler helped with fused ops but couldn't eliminate the scatter fallbacks. Where exactly is the time going at the hardware level? Next we'll learn to measure before we optimize.*
