# The memory hierarchy

In Chapter 4, we saw the Neuron Explorer timeline: three matmul waves with gaps between them. The tensor engine was idle during those gaps, waiting for data. The compiler already fused everything into one NEFF. So why is there still idle time?

The answer is memory. Compute is cheap; moving data is expensive. This chapter builds the mental model for *where* data lives and *how* it moves.

---

## Three levels of memory

Every operation on a NeuronCore follows the same pattern: data moves from far away to close by, gets computed on, then moves back. The "distance" is what determines speed:

```{figure} ../assets/memory_speed.png
:alt: Memory hierarchy speeds
:width: 600px
:align: center

The memory hierarchy: closer to compute = faster but smaller.
```

```
┌─────────────────────────────────────────────────────────────────┐
│  PSUM (Partial Sum Buffer)         │  Tiny, instant             │
│  Lives inside the Tensor engine    │  Only holds matmul results │
├────────────────────────────────────┼────────────────────────────┤
│  SBUF (State Buffer)               │  ~28 MB per core           │
│  On-chip SRAM                      │  ~3.5 TB/s bandwidth       │
├────────────────────────────────────┼────────────────────────────┤
│  HBM (High Bandwidth Memory)      │  96 GB total (chip)        │
│  Off-chip, on-package              │  ~2.9 TB/s bandwidth       │
└────────────────────────────────────┴────────────────────────────┘
```

| Level | Capacity | Bandwidth | Latency | What lives here |
|-------|----------|-----------|---------|-----------------|
| **PSUM** | ~8 MB | Internal | ~1 cycle | Matmul accumulation results (named "Partial Sum" because it accumulates partial products during tiled matmuls until the full result is ready) |
| **SBUF** | ~28 MB/core | 3.5 TB/s | ~10 cycles | Active tiles (weights, activations, intermediates) |
| **HBM** | 96 GB | 2.9 TB/s | ~100s cycles | Everything else (model weights, optimizer states, full tensors) |

For our `big_simple` example from Chapter 4: each weight matrix (4096×4096 in bfloat16) is **32 MB**: ($4096\times4096\times2$ bytes=32MB) larger than SBUF. It can't fit on-chip all at once. The activations (1024×4096 = 8 MB) can barely fit. This is why tiling exists. And also a good motivation to use memory efficient number formats. (see also chapter 4)

### Where does ESM-2's attention live?

During a single attention layer, here's what needs to be in memory simultaneously:

```
Q: [1, 20, 32, 64] = 160 KB     ← fits in SBUF
K: [1, 20, 32, 64] = 160 KB     ← fits in SBUF
V: [1, 20, 32, 64] = 160 KB     ← fits in SBUF
Attention scores: [1, 20, 32, 32] = 80 KB  ← fits in SBUF
Q/K/V projection weights: 3 × [1280, 1280] × 2 bytes = 9.8 MB  ← tight fit
Output projection: [1280, 1280] × 2 bytes = 3.3 MB
```

For our small ESM-2 (seq_len=32), the activations fit comfortably. The weights are the bottleneck: they must be streamed through SBUF in tiles. For longer sequences or larger models, even activations spill to HBM.

---

## DMA engines

```{figure} ../assets/memory_movement.png
:alt: Data movement between memory levels
:width: 600px
:align: center

DMA engines move data between HBM and on-chip SRAM. Compute engines operate between SBUF and PSUM.
```

Data doesn't teleport between HBM and SBUF. Each NeuronCore has **16 DMA engines**, each capable of ~277 GB/s throughput. They physically copy bytes between memory levels:

- **Load:** HBM → SBUF (bring data on-chip for compute)
- **Store:** SBUF → HBM (write results back)
- **Internal:** SBUF ↔ PSUM (move between on-chip buffers)

DMA engines have three important properties:

**1. They operate on contiguous addresses only.** A single DMA buffer can only move a contiguous block of memory. This is why tensor contiguity (Chapter 2) matters at the hardware level: a contiguous tensor is one DMA instruction; a strided tensor forces many tiny transfers.

**2. They run independently of compute.** While the tensor engine multiplies tiles, DMA engines can simultaneously load the *next* set of tiles. This overlap is the key to performance. The goal is to keep the TensorEngine busy !

**3. They parallelize across the partition dimension.** DMA reads a full vertical stripe across all 128 partition lanes simultaneously, then pipelines across the free dimension. This means data organized with the "interesting" dimension on partition gets loaded in one parallel burst.

When DMA and compute overlap perfectly, the tensor engine never waits. When they don't overlap (load takes longer than compute), you see idle gaps. 

### Transfer size matters

DMA throughput depends heavily on transfer size:

| Transfer size | Effective bandwidth | Notes |
|--------------|--------------------|----|
| < 32 KB | Poor | Startup overhead dominates |
| 32–512 KB | Good | Amortizes overhead |
| ≥ 512 KB | Near peak (3.5 TB/s) | Ideal — one tile of 128 partitions × 1024 elements |

This explains the 128×128 tile size the compiler uses: 128×128 × 2 bytes (bfloat16) = **32 KB** — right at the threshold for efficient DMA. Larger tiles would be better for DMA but might not fit enough copies in SBUF for double-buffering.

---

## Tiling and double-buffering

### Tiling: breaking the problem into SBUF-sized pieces

A 4096×4096 matmul can't execute in one shot — the weight matrix alone is 32 MB and SBUF is 28 MB. The compiler decomposes it into tiles:

```
4096×4096 weight matrix
┌───┬───┬───┬───┬─── ─── ───┐
│ T │ T │ T │ T │   ...     │  
├───┼───┼───┼───┤           │
│ T │ T │ T │ T │           │
├───┼───┼───┼───┤           │ 
│   │   │   │   │           │
│       ...                 │
└───┴───┴───┴───┴─── ─── ───┘

If each tile is of size 128, you have 4096/128=32 tiles accross and down.
Each tile T = 128×128 × 2 bytes = 32 KB
Total tiles = 32×32 = 1,024 tiles
```

The tensor engine processes tiles in two internal steps:

1. **Load stationary** — load a weight tile (128×128) into the systolic array. This is pure data movement, no computation. It's fast — **4× faster** than the next step for equal tile sizes.
2. **Multiply moving** — stream the activation tile (128×512) through the loaded weights, accumulating into PSUM. This is where computation happens.

The design choice: put the matrix with more tiles as stationary (cheaper to reload), and the one with fewer tiles as moving. The optimal tile sizes are 128×128 for stationary and 128×512 for moving.

For our 4096×4096 matmul: 1,024 weight tiles get loaded as stationary, and activation tiles stream through as moving. The compiler tiles this automatically with `torch.compile`.

### Double-buffering: hiding load latency

If you load a tile and *then* compute it, the tensor engine waits during every load. Double-buffering eliminates this: The idea is literally to have "two buffers." You allocate twice the SBUF space per tile (buffer A and buffer B), and alternate which one is being loaded vs computed. The cost is 2× memory for that tile, but the benefit is zero idle time on the tensor engine. It's the simplest form of pipelining — you trade memory for latency hiding.

The compute and load alternate between buffers, fully overlapping. The tensor engine always has fresh data waiting. This is what the "dual weight cache" in Chapter 4 implements — foreground weights (active compute) and background weights (loading next).

Double-buffering works when **compute time ≥ load time** for each tile. If loading is slower. This might happen for strided/fragemented loads or if tiles are small for example. 
This is exactly the roofline concept: if arithmetic intensity is below the critical threshold (230 ops/byte on Trainium2), you're memory-bound and no amount of compute optimization helps — you need to reduce data movement.

### When SBUF overflows: silent spilling

What happens if your kernel allocates more data in SBUF than physically fits? The compiler handles this automatically — it inserts spill instructions that write excess data to HBM and refill instructions that read it back when needed. Your code still produces correct results. There is no error, no warning, no crash. The only signal is degraded performance: unexpected DMA activity during compute, visible only in the profiler as extra load/store operations you didn't write. Part of becoming a performance engineer is learning to spot these spills in a profiler trace and restructuring your data flow to eliminate them.

---

## The four optimization principles

Everything in Neuron performance engineering reduces to four rules about data movement:

### 1. Pipeline operations

Load the next tile while computing the current tile while storing the previous result. Three stages running in parallel:

```
DMA load:    [tile 2][tile 3][tile 4][tile 5]
Compute:     [tile 1][tile 2][tile 3][tile 4]
DMA store:   [tile 0][tile 1][tile 2][tile 3]
```

This is what the compiler does automatically with `torch.compile`. In Part V, you'll orchestrate this manually with NKI for cases the compiler misses.

### 2. Minimize data movement

The fastest data transfer is the one that never happens. If an intermediate result is consumed immediately by the next operation, keep it in SBUF — don't write it to HBM and read it back.

This is *exactly* what op fusion achieves. In Chapter 3, the 8× speedup from `torch.compile` came primarily from eliminating intermediate HBM round-trips:

```
Eager (unfused):                    Compiled (fused):
matmul → write result to HBM       matmul → result stays in SBUF
layer_norm → read from HBM         layer_norm → reads from SBUF
           → write result to HBM                → result stays in SBUF
gelu → read from HBM               gelu → reads from SBUF
     → write result to HBM               → result stays in SBUF
matmul → read from HBM             matmul → reads from SBUF
```

The fused version avoids 6 HBM transfers. At 2.9 TB/s bandwidth and 8 MB per transfer, that's ~16 μs saved — per layer, per iteration.

### 3. Maximize transfer size

Large contiguous transfers amortize DMA startup costs. This connects directly to Chapter 2: a contiguous tensor enables one large DMA instruction; a non-contiguous tensor forces many small ones.

Practical implications:
- Prefer batch sizes that make tiles fill SBUF efficiently
- Avoid operations that fragment memory layout (non-contiguous transposes before matmul)
- The compiler's tile size choices (128×128, 128×512) are optimized for DMA efficiency

### 4. Overlap communication with compute

On multi-chip configurations, the 16 CC-Cores handle collective operations (all-reduce, all-gather) independently of the compute engines. While the tensor engine computes layer N, the collective cores can simultaneously communicate gradients from layer N-1:

```
Tensor engine: [compute layer 1][compute layer 2][compute layer 3]
CC-Cores:      [allreduce grad 0][allreduce grad 1][allreduce grad 2]
```

Communication is effectively free as long as compute takes longer — which it usually does for large models. This is why distributed training on Neuron scales efficiently without communication bottlenecks (Chapter 17).

---

## The endgame: keep everything on-chip

The ultimate optimization is to never touch HBM during the forward pass at all. If your entire working set fits in SBUF (activations + current weight tiles), every op reads from and writes to fast on-chip memory.

For large models this isn't possible for the full model — but for individual operations it is. A fused attention kernel that keeps Q, K, V, and the attention scores entirely in SBUF (no intermediate HBM writes) is dramatically faster than one that spills intermediates. This is what FlashAttention achieves on GPU, and what NKI attention kernels achieve on Neuron.

An internal Amazon team achieved **80% MFU** by designing their training pipeline to keep activations in SRAM the entire time, using SRAM-to-SRAM collectives between chips that skip HBM round-trips entirely. That's the ceiling — and it's achievable when you control the full stack (more on this in Chapter 17).

---

*The memory hierarchy exists. The engines exist. But how does your PyTorch code actually reach them? What connects `model.to("neuron")` to DMA engines and systolic arrays?*
