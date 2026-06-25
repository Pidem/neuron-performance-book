# Thinking in tiles

*ESM-2's weight matrices are 1280×1280. The tensor engine accepts 128×128. Break the problem into pieces.*

```{admonition} Run it yourself
:class: tip
Scripts for this chapter (run on trn2.3xlarge with Neuron venv activated):

- `scripts/ch15_vector_engine.py` — vector/scalar engine ops: element-wise, reductions, activations using `nisa`
- `scripts/ch15_matmul_isa.py` — tensor engine matmul progression (single-tile → tiled → hoisted loads), benchmarked with `wrap_nki` + `synchronize()`

`wrap_nki` + timing measures what your model actually sees through PyTorch's dispatch path — use it to A/B test kernel variants.
```

---

## The 2D memory model

In Chapter 14, we treated SBUF as a simple buffer. Now we need to understand its physical layout.

If you think of a tensor as a matrix, SBUF is a matrix with exactly 128 rows and as many columns as you need. The hardware processes all 128 rows simultaneously on every clock cycle. Each row is a **lane**:

```
Your tensor [128, 512] in SBUF:

Lane 0:   [x₀₀, x₀₁, x₀₂, ... x₀₅₁₁]   ← all 128 lanes fire
Lane 1:   [x₁₀, x₁₁, x₁₂, ... x₁₅₁₁]   ← simultaneously
Lane 2:   [x₂₀, x₂₁, x₂₂, ... x₂₅₁₁]   ← on every cycle
...
Lane 127: [x₁₂₇₀, x₁₂₇₁, ... x₁₂₇₅₁₁] ←
           ─────────────────────────────→
           columns processed sequentially
```

When the vector engine computes a sum across columns, it reduces within each lane independently. All 128 reductions happen at the same time. NKI names these two axes:

- **Partition dimension (P):** which lane. 128 parallel lanes. All 128 fire simultaneously.
- **Free dimension (F):** position within a lane. Data is processed sequentially along this axis.

A **tile** is any tensor whose first dimension maps to the partition dimension. Every NKI instruction operates on tiles. If your data isn't tile-shaped, you must reshape or reindex it first.

```
            Free dimension (F) →
          ┌───────────────────────────┐
      P=0 │ ████████████████████████  │
      P=1 │ ████████████████████████  │
      P=2 │ ████████████████████████  │
Partition  │         ...               │
(128 lanes)│ ████████████████████████  │
    P=127 │ ████████████████████████  │
          └───────────────────────────┘
```

HBM is 1D (flat bytes). When you `nl.load()`, data goes from 1D HBM into this 2D layout. The first logical dimension maps to partition, everything else flattens into the free dimension.

### The #1 mistake: wrong dimension on partition

The partition dimension is fixed at 128. If your tensor's first dimension is larger than 128, you must tile over it — each tile processes 128 rows at a time. If it's smaller than 128, some lanes sit idle.

The critical design decision: **which logical dimension of your data maps to partition?**

```{figure} ../assets/partition_layout.svg
:align: center
:width: 90%

Left: a [128, 512] tensor loads directly — 128 elements fill all partition lanes, 512 goes to the free dimension. Right: a [512, 128] tensor has 512 on partition — it overflows the 128 lanes and must be processed in 4 separate tiles. Same data, 4× more loop iterations.
```

The rule: **put the dimension you want to parallelize across on partition (dim 0), put the dimension you want to iterate over on free (dim 1).** For vector reductions (like computing a mean across features), features go on free and batch/sequence goes on partition — each lane reduces independently in parallel.

When you get this wrong, the kernel still produces correct results. But it runs 4× slower (or worse) because you're serializing work that could be parallel. The profiler won't flag this as an error — it just shows lower throughput. This is the subtlest performance bug in NKI code.

---

## Indexing: how you select tiles from larger tensors

In Chapter 14, our tensors were exactly one tile (128×512). Real tensors are much larger. To process them tile by tile, you need to **index** into the larger tensor and extract each tile.

NKI uses `nl.mgrid` to generate index tensors — the same concept as NumPy's meshgrid:

```python
# Generate a grid of indices for a 128×512 tile
i_p = nl.arange(128)[:, None]  # partition indices [128, 1]
i_f = nl.arange(512)[None, :]  # free indices [1, 512]

# Load a tile starting at row `row_offset`, col `col_offset`
tile = nl.load(hbm_tensor[i_p + row_offset, i_f + col_offset])
```

The pattern: create index arrays once (outside the loop), then add offsets to select different tiles on each iteration. This is a 2-line pattern that appears in virtually every NKI kernel.

---

## Tiled matrix multiplication

ESM-2's FFN weight is 1280×5120 in BF16 (13 MB). SBUF is ~28 MB per core. The matrix fits — but barely, with no room for activations. And the tensor engine only accepts 128×128 (stationary) and 128×512 (moving) tiles. You must tile.

### The algorithm

For `A[M, K] @ B[K, N]`:
1. Partition A into row tiles (M/128 tiles of 128 rows each)
2. Partition B into column tiles (N/512 tiles of 512 columns each)
3. Partition both along the contraction dimension K (K/128 sub-chunks)
4. Each smallest computation produces a **partial sum** → accumulate in PSUM
5. After all K sub-chunks: copy accumulated result from PSUM → SBUF → HBM

```python
@nki.jit
def tiled_matmul(a_t_hbm, b_hbm):
    """Tiled matmul: A_T[K, M] @ B[K, N] → C[M, N]
    
    a_t_hbm: [K, M] transposed left-hand side
    b_hbm:   [K, N] right-hand side
    """
    K, M = a_t_hbm.shape
    _, N = b_hbm.shape
    
    # Tile sizes (maximize hardware utilization)
    TILE_M = 128   # partition dim for stationary
    TILE_K = 128   # contraction dim (partition for both)
    TILE_N = 512   # free dim for moving
    
    # Allocate output in HBM
    out_hbm = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.hbm)
    
    # Index templates (created once, reused with offsets)
    i_p = nl.arange(TILE_K)[:, None]   # partition [128, 1]
    i_m = nl.arange(TILE_M)[None, :]   # free [1, 128]
    i_n = nl.arange(TILE_N)[None, :]   # free [1, 512]
    
    # Loop over output tiles
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            # Accumulator in PSUM (FP32)
            accum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            
            # Loop over contraction dimension
            for k in nl.affine_range(K // TILE_K):
                # Load tiles
                a_tile = nl.load(a_t_hbm[i_p + k * TILE_K, i_m + m * TILE_M])
                b_tile = nl.load(b_hbm[i_p + k * TILE_K, i_n + n * TILE_N])
                
                # Matmul — accumulates into PSUM
                accum += nl.matmul(a_tile, b_tile, transpose_x=True)
            
            # Copy from PSUM to SBUF (cast FP32 → BF16)
            result = nl.copy(accum, dtype=nl.bfloat16)
            
            # Store to HBM
            nl.store(out_hbm[i_p[:TILE_M, :1] + m * TILE_M, i_n + n * TILE_N], value=result)
    
    return out_hbm
```

### Key decisions in this kernel

- **TILE_K = 128:** contraction axis maps to partition dimension (hardware requirement). The tensor engine contracts across the 128 lanes.
- **TILE_M = 128, TILE_N = 512:** max tile sizes for stationary and moving (from Ch 4 engine cost models). Maximizes hardware utilization per instruction.
- **`nl.affine_range`:** tells the compiler iterations are independent — can be reordered, pipelined, parallelized. PSUM accumulation is OK here because addition is associative.
- **Accumulate in PSUM, copy out once:** the inner K-loop accumulates partial products. Only after all K tiles are processed do we copy the final result out. This minimizes PSUM↔SBUF traffic.

---

## Loop types and their meaning

How you write a loop tells the compiler what optimizations are safe:

| Loop type | Meaning | Use case |
|-----------|---------|----------|
| `nl.affine_range` | Iterations are independent (can reorder/parallelize) | Tiling loops, reductions via accumulation |
| `nl.sequential_range` | Iteration N depends on iteration N-1 (strict order) | Scan operations, recurrences |
| `nl.static_range` | Fully unrolls — no compiler optimization | Debugging only (terrible performance) |

Use `affine_range` by default. Switch to `sequential_range` only when you have a true loop-carry dependency.

---

## Engine layout constraints

Each engine has rules about which tensor dimension maps where:

| Engine | Partition dim constraint | Free dim limit |
|--------|------------------------|----------------|
| Tensor engine (stationary) | Contraction axis → partition | 128 |
| Tensor engine (moving) | Contraction axis → partition | 512 |
| Vector engine | Reduction axis → free | — |
| Scalar engine | — | 64K from SBUF, 4K from PSUM |

The tensor engine constraint is the most important: the contraction axis (K in A[M,K]×B[K,N]) **must** map to partition. This is why the LHS needs transposing — so K becomes the first (partition) dimension.

---

## The reformulation principle

Sometimes the right kernel isn't a direct translation of the op — it's a different algorithm that maps better to the tensor engine.

**Example: scatter_add as matmul**

`scatter_add(source, index)` for a GNN with 100 nodes and 500 edges is equivalent to:

```python
# Build binary selection matrix: A[j, i] = 1 if edge i targets node j
A = torch.zeros(100, 500)
for i, j in enumerate(index):
    A[j, i] = 1.0

# scatter_add == matmul
result = A @ source  # [100, features]
```

On Neuron: the tensor engine does this 100×500 matmul at full throughput. A sequential scatter on the scalar engine would be 16,000× slower.

The tradeoff: the dense A matrix costs O(nodes × edges) memory. Worth it when the graph fits in SBUF. For larger graphs: block-sparse patterns or bucketed scatter.

**The general principle:** if the tensor engine can't do your operation directly, ask: *can I reformulate this as something it CAN do?* The tensor engine is 90% of your available FLOPs — feeding it work is always the priority.

---

## Masking: handling non-divisible shapes

Real tensors aren't always multiples of 128 or 512. Without masking, you'd need branching code for edge tiles. NKI provides masks instead:

```python
# Input might be 200×1000 — not divisible by 128×512
M, N = input_hbm.shape

for m in nl.affine_range(nl.cdiv(M, 128)):  # ceiling division
    # Mask: only valid for rows < M
    mask = (i_p + m * 128) < M
    
    tile = nl.load(input_hbm[i_p + m * 128, i_f], mask=mask)
    result = nl.exp(tile, mask=mask)
    nl.store(output_hbm[i_p + m * 128, i_f], value=result, mask=mask)
```

Masked positions generate no instructions — no wasted compute, no branching. Essential for production kernels where 99% of real inputs aren't tile-aligned.

---

## What you learned

| Concept | What it means |
|---------|--------------|
| 2D memory (P × F) | SBUF has 128 parallel lanes × variable depth |
| Tiles | Tensors with leftmost dim mapped to partition |
| `nl.mgrid` / `nl.arange` | Generate index tensors for tile selection |
| Tiled matmul | Triple loop: rows × cols × contraction, accumulate in PSUM |
| `affine_range` | Independent iterations — compiler can optimize aggressively |
| Engine constraints | Contraction axis → partition for tensor engine |
| Masking | Handle non-divisible shapes without branching |
| Reformulation | Recast operations as matmul to use the tensor engine |

This kernel is correct and handles arbitrary sizes. But it's not fast — the triple loop is naive, with no DMA pipelining and no engine overlap. The profiler will show: tensor engine idle while DMA loads, DMA idle while tensor engine computes.

---

*The kernel works but it's slow — the profiler shows engines taking turns instead of working in parallel. How do you pipeline loads, computes, and stores to keep every engine busy?*
