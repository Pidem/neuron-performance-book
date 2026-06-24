# Advanced NKI

*From 24μs to 8.5μs: pipeline the attention kernel until the tensor engine never stops.*

```{admonition} Run it yourself
:class: tip
Scripts for this chapter (run on trn2.3xlarge with Neuron venv activated):

- `scripts/ch16_rmsnorm.py` — RMSNorm kernel combining reductions, scalar ops, and broadcasting (a real LLM building block)
- `scripts/ch16_fused_attention.py` — fused self-attention at ISA level, profiled with `NeuronProfiler` for device timeline

`NeuronProfiler` captures a device-level trace showing which engines are active or idle — wall clock time tells you *how slow*, the profiler tells you *why*.
```

---

## Language vs ISA

In Chapter 14-15 we used `nki.language` (nl) — high-level, NumPy-like. Each `nl.*` call may map to multiple hardware instructions, with the compiler choosing the lowering. This is fine for correctness but unpredictable for performance.

`nki.isa` (nisa) gives you one-to-one control: one call ≈ one hardware instruction. The compilation is deterministic — what you write is what executes.

```python
# Language: compiler decides how to lower
result = nl.matmul(a_tile, b_tile, transpose_x=True)

# ISA: you specify exactly what happens
nisa.nc_matmul(moving=b_tile, stationary=a_tile, is_transposed=True)
```

Same line count. But ISA gives you:
- Deterministic instruction mapping (no unintended overhead)
- Access to engine-specific parameters (DMA modes, scheduling hints)
- Ability to mix operations from different engines explicitly

**Rule of thumb:** use language for your first working version, then move to ISA for the performance-critical path.

---

## The optimization workflow

The NKI bootcamp teaches a systematic optimization flow. Applied to self-attention (`Q×K^T → softmax → ×V`):

### Phase 1: Get it correct (Ch 14-15 territory)

| Step | What | Result |
|------|------|--------|
| 1 | NumPy implementation | Ground truth for testing |
| 2 | NKI language on smallest shapes | Correct kernel, single tile |
| 3 | Move to NKI ISA | Same logic, deterministic instructions |
| 4 | Add tiling loops | Works on real shapes |
| 5 | Loop fusion (merge separate loops) | Fewer HBM round-trips |

After step 5: **24μs** steady-state. Profile shows vector engine oversubscribed — blocking everything else.

### Phase 2: Algorithm-specific optimizations

| Step | What | Speedup | Insight |
|------|------|---------|---------|
| 6 | Delay division to end | 39% (→15μs) | Divide smaller output tensor instead of large intermediate |
| 7 | Combine scalar instructions | 1μs saved | Pack 3 scalar ops into one `activation` instruction |
| 8 | Downcast scores to BF16 before transpose | 10% (→11μs) | Half the data to transpose, numerically safe after reduction |

Key principle: all of steps 6-8 work on a **single tile** (`num_tiles=1`). Profiling one tile is simpler than profiling a multi-tile kernel — you can map every instruction to your code.

### Phase 3: Pipelining (the big win)

| Step | What | Result |
|------|------|--------|
| 9 | Cache max-reduce on vector engine | Opens PSUM capacity for multi-tile |
| 10 | Software refactor for pipeline structure | Code reorganization (no perf change) |
| 11 | Triple-tile software pipelining | **8.5μs** — 3× improvement |
| 12 | Double buffering / PSUM eviction | Final polish |

---

## Pipelining: the core technique

The profiler after step 5 shows: tensor engine idle during softmax, vector/scalar idle during matmul. The engines take turns instead of overlapping.

**Triple-tile pipelining** works on 3 tiles simultaneously:

```
Time →
Tile I+1:  [QK matmul on Tensor Engine]
Tile I:    [softmax on Scalar/Vector]
Tile I-1:  [score×V matmul on Tensor Engine]
```

The tensor engine processes tiles I+1 and I-1 (two different matmuls) while scalar/vector handle the softmax for tile I. No engine is idle.

Implementation requires:
- **`nl.sequential_range`** — forces tile ordering (tile I depends on tile I-1's output)
- **`compiler.no_reorder`** — prevents the compiler from rearranging your carefully pipelined instruction sequence
- **Manual allocation** — prevents the compiler from overwriting live SBUF variables when 3 tiles coexist

---

## Manual allocation

Most of the time, the compiler allocates SBUF/PSUM locations automatically. But with 3 tiles in flight simultaneously, it can accidentally overwrite a live variable (it doesn't know you're still using it for tile I-1).

Manual allocation lets you pin variables to specific SBUF locations:

```python
# Define allocator with explicit byte offsets
sbuf_alloc = nki.allocator(
    buffer=nl.sbuf,
    assignments={
        'q_tile': (0, TILE_SIZE_BYTES),
        'k_tile': (TILE_SIZE_BYTES, 2 * TILE_SIZE_BYTES),
        'scores': (2 * TILE_SIZE_BYTES, 3 * TILE_SIZE_BYTES),
    }
)

# Load into the pinned location
q_tile = nl.load(q_hbm[...], buffer=sbuf_alloc['q_tile'])
```

Key details:
- Allocation is in **bytes**, not elements — changes with data type (BF16 = 2 bytes, FP32 = 4 bytes)
- You're responsible for ensuring no overlap between simultaneously live variables
- The compiler still handles scheduling within each tile — you're only constraining placement

---

## DMA transfer size: the hidden bottleneck

From the profiler: DMA engines show 95% active time, but throughput is low. The issue: transfer sizes are too small.

Target: **≥32 KB per DMA transfer** for optimal bandwidth on Trn2.

The math for optimal free dimension size:
- 16 DMA engines, 128 partitions → 8 partitions per engine
- Transfer size = partitions_per_engine × free_dim × element_size
- For BF16 (2 bytes): 8 × F × 2 ≥ 32,768 → **F ≥ 2048**
- For FP8 (1 byte): 8 × F × 1 ≥ 32,768 → **F ≥ 4096**

Kernel design changes based on data type. Smaller data types need larger free dimensions to hit good DMA utilization.

---

## The three optimization categories

After the 12-step bootcamp, optimizations fall into three buckets:

### 1. Improve arithmetic intensity (reduce unnecessary data movement)

- **Loop hoisting:** move invariant loads to outer loops (don't reload the same tile every inner iteration)
- **Loop fusion:** merge loops over the same data (avoid intermediate HBM spills)
- **Delayed operations:** postpone expensive ops to where the tensor is smaller (step 6)

### 2. Improve compute efficiency (reduce engine idle time)

- **Engine pipelining:** slice work so multiple engines overlap (the triple-tile technique)
- **Instruction combination:** pack multiple scalar ops into one `activation` instruction
- **Partition vectorization:** if using only 64/128 lanes, map two independent results across all 128

### 3. Improve data movement efficiency (reduce DMA waste)

- **Maximize transfer size:** ensure F ≥ 2048 for BF16 (see above)
- **Choose transpose method wisely:** DMA transpose (crossbar on Trn2) vs tensor engine transpose — pick based on which resource is less constrained
- **Avoid strided access:** contiguous HBM reads are far faster than gathered reads

---

## Composable kernels vs mega-kernels

A natural question: should you write one giant kernel for the entire forward pass?

**Mega-kernels** (one kernel = entire attention + MLP) exist as tactical solutions but aren't scalable:
- Impossible to shard across multiple chips
- Hard to support different model dimensions
- Maintenance nightmare

The better pattern: **small composable kernels** orchestrated by the framework. Flash attention is one kernel. RMSNorm-Quant is another. The framework chains them together, and the compiler handles the boundaries.

The NKI Library follows this pattern — each kernel is a focused, reusable building block:
- `attention_tkg` — fused attention for token generation
- `attention_cte` — fused attention for context encoding
- `rmsnorm_quant` — fused normalization + quantization
- `fused_adam` — optimizer step

---

## The full optimization cycle

```
Write correct kernel (Ch 14)
    │
    ▼
Add tiling (Ch 15)
    │
    ▼
Profile → identify bottleneck (Tensor idle? DMA tiny? Spilling?)
    │
    ├─ Tensor idle during other engines → pipeline (triple-tile)
    ├─ DMA transfers too small → increase free dimension
    ├─ Unexpected spills → reduce SBUF pressure or add manual allocation
    ├─ HFU >> MFU → too many transposes, reformulate
    └─ Arithmetic intensity low → fuse more ops, hoist loads
    │
    ▼
Apply fix → reprofile → repeat
```

This is iterative. The 12-step attention optimization took the Annapurna team months to develop. But the pattern is learnable, and each step compounds.

```{admonition} New in 2025: Scheduling and Allocation API
:class: note
NKI now exposes fine-grained control over instruction scheduling (which engine fires when) and tensor allocation (which SBUF bank holds which data). This is what enables the structured pipelines shown in this chapter — without it, you'd rely on the compiler's heuristics for scheduling, which may not discover the optimal overlap pattern for your kernel.

The NKI compiler itself is being open-sourced — giving full transparency into how your kernel code compiles to hardware instructions. No more black boxes between your Python and the silicon.
```

---

## Reference: the 98% HFU matmul

The NKI documentation contains a fully optimized tiled matmul tutorial that achieves **98% HFU** on a zero-shot profile. It demonstrates every technique in this chapter: optimal tile sizes, DMA pipelining, double-buffering, and instruction-level scheduling.

Start there after this chapter: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials/matrix_multiplication.html

---

## Case study: Anthropic's flash attention kernel

Jay Gray (Anthropic's Trainium inference lead) optimized a fused flash attention kernel — QKV projection → self-attention → output projection — following exactly the cycle above. Here's what the profiler revealed and how they fixed it:

**The unexpected bottleneck.** After basic optimization, the tensor engine showed dense matmul activity in the QKV projection... but gaps in the attention section. The profiler trace showed bursts of matrix multiplications interspersed with clusters of small vector operations. The tensor engine was idle during those vector bursts.

**Root cause.** Reading the ISA view in Neuron Explorer, the bottleneck was *not* compute. It was vector ops shuffling intermediate results between SBUF memory banks — an inefficiently large number of small transfers, each paying instruction launch overhead.

**The fix.** Rewrite the tiling so data moves between banks using fewer, larger vector operations — amortizing the per-instruction overhead. Same math, different scheduling. **Result: 13% speedup on attention from this single change.**

**The progression on hardware generations.** The same kernel achieves ~60% tensor engine utilization on Trn2. On Trn3 — with 4× FP8 matmuls, faster vector/scalar engines, and faster comms — the same kernel reaches **over 90% tensor utilization** without algorithmic changes. Better hardware turns a good kernel into a great one.

```{admonition} The lesson
:class: tip
The bottleneck wasn't where you'd guess. Not matmul speed, not softmax, not memory bandwidth — but instruction launch overhead on small vector shuffles. You can only find this with an instruction-level profiler. This is why Neuron Explorer exists.
```

Ron (Chief Architect): "Your goal is to write a kernel that hits 100% MFU. I have customers who do this. If my customers can do this, you can do this."

*Part V complete. You can now write, tile, and optimize NKI kernels. Next: scaling beyond a single chip.*

*Next: [Chapter 17](../part6/ch17-distributed) — Multi-core, multi-chip (Part VI).*
