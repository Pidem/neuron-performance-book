# The memory hierarchy

*ESM-2's attention layer: Q, K, V matrices are large. Where do they live? How do they move?*

```{admonition} TODO
:class: warning
This chapter answers: "The compiler fused everything — so why is there still idle time in the profiler?"
This is the bridge between "software optimization" (Part II) and "measuring" (Part III).
```

---

## Act 1: Three levels of memory

```{admonition} TODO
:class: warning
- HBM (32GB on trn2.3xlarge): large, slow (~900 GB/s bandwidth)
- SBUF/SRAM (on-chip, per NeuronCore): small (~24MB), fast (TB/s bandwidth)
- Registers (inside engines): tiny, instant
- Diagram: show the hierarchy with sizes and bandwidths for Trainium2
- ESM-2 context: where do Q, K, V live during attention? (HBM until loaded into SBUF)
- The fundamental tension: compute is fast, memory is slow → most time is spent waiting for data
```

---

## Act 2: DMA engines — the trucks

```{admonition} TODO
:class: warning
- DMA = Direct Memory Access — hardware engines that move data without involving compute
- HBM → SBUF (load), SBUF → HBM (store)
- Contiguous transfers are fast (ch02's contiguity benchmark was showing this!)
- Strided/scattered transfers are slow or impossible (explains why layout matters)
- DMA scheduling: the compiler decides WHEN to issue loads/stores
- The key insight: DMA and compute can overlap — load next tile while computing current tile
```

---

## Act 3: Tiling and double-buffering

```{admonition} TODO
:class: warning
- Tiling: break large tensors into SBUF-sized chunks
- ESM-2 attention: Q is [1, 32, 1280] — too big for SBUF all at once → tile it
- The compiler's tiling decisions (from ch07): tile sizes affect both compute and memory efficiency
- Double-buffering: allocate two SBUF buffers — load into buffer B while computing from buffer A
- Pipeline: Load(tile_n+1) | Compute(tile_n) | Store(tile_n-1) — all three overlap
- When it works perfectly: compute time ≥ load time → compute-bound (good!)
- When it breaks: load time > compute time → memory-bound (the profiler shows idle tensor engine)
```

---

## Act 4: The optimization principles

```{admonition} TODO
:class: warning
Four rules for Neuron performance:
1. **Pipeline operations:** load next while computing current while saving previous
2. **Minimize data movement:** keep activations in SRAM across ops (this is what fusion achieves!)
3. **Maximize transfer size:** large contiguous chunks > many small transfers
4. **Overlap communication:** collective units run independently of compute (free for multi-chip)

- Connect each principle back to what we've already seen:
  - Principle 2 explains WHY fusion gives 8x (ch03)
  - Principle 3 explains WHY contiguity matters (ch02)
  - Principle 1 is what NKI kernels do manually (Part V preview)
- The endgame: DART achieved 80% MFU by keeping activations in SRAM the entire time
- SRAM-to-SRAM collectives between chips (skip HBM round-trips entirely)
```

*Question raised → "I understand the theory. How do I SEE what's actually happening on my model?"*

*Next: [Chapter 9](../part3/ch09-profiler) — The profiler.*
