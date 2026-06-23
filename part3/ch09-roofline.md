# Roofline thinking

*You've seen the hardware, the memory hierarchy, and two real models on Neuron. Now: how do you know whether your workload is fast or slow — and what's limiting it?*

---

## Where does the time go?

Every operation on a NeuronCore is bounded by three things:

1. **Compute** — how fast the engines can do math (FLOPs/second)
2. **Bandwidth** — how fast data moves between HBM and SBUF (bytes/second)
3. **Total memory** — how much data can be stored (bytes)

These constraints let us bound the runtime of any operation *before profiling it*. This is the roofline model — paper-and-pencil performance estimation.

---

## The two times

For any operation, we can estimate two quantities:

$$T_\text{math} = \frac{\text{FLOPs}}{\text{Accelerator FLOPs/s}}$$

$$T_\text{comms} = \frac{\text{Bytes moved}}{\text{Bandwidth (bytes/s)}}$$

The actual runtime is bounded:

- **Lower bound** (perfect overlap): $\max(T_\text{math}, T_\text{comms})$
- **Upper bound** (no overlap): $T_\text{math} + T_\text{comms}$

With good pipelining (Ch 5), we approach the lower bound. The upper bound is never more than 2× the lower bound.

---

## Arithmetic intensity

The ratio that determines which bound dominates:

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes moved}}$$

This measures "how much work per byte loaded." When intensity is high, compute dominates ($T_\text{math} > T_\text{comms}$) — we're **compute-bound**. When low, bandwidth dominates — we're **memory-bound**.

The crossover point is the hardware's **critical intensity**:

$$\text{Critical Intensity} = \frac{\text{Peak FLOPs/s}}{\text{HBM Bandwidth}}$$

For a single NeuronCore-v3 (BF16):

$$\frac{667 \text{ TFLOPS} / 8 \text{ cores}}{2.9 \text{ TB/s} / 8 \text{ cores}} = \frac{83.4 \text{ TFLOPS}}{362 \text{ GB/s}} \approx 230 \text{ FLOPs/byte}$$

Any operation below 230 FLOPs/byte is memory-bound. Above, it's compute-bound.

```{figure} ../assets/roofline.png
:alt: Roofline model plot
:width: 600px
:align: center

The roofline model: peak achievable throughput vs arithmetic intensity. Operations left of the ridge point are memory-bound; right of it, compute-bound. (Figure from [How To Scale Your Model](https://jax-ml.github.io/scaling-book/roofline/), Austin et al., Google DeepMind, 2025.)
```

---

## Worked examples on Neuron

### Dot product (always memory-bound)

Dot product of two BF16 vectors of length N:
- Load: $2N + 2N = 4N$ bytes (two inputs)
- FLOPs: $2N - 1 \approx 2N$ (N multiplies + N-1 adds)
- Write: 2 bytes (scalar output)

$$\text{Intensity} = \frac{2N}{4N + 2} \rightarrow \frac{1}{2} \text{ as } N \rightarrow \infty$$

Intensity = 0.5 FLOPs/byte. This is 460× below the critical intensity. The dot product is hopelessly memory-bound on any hardware — the engines will always be waiting for data.

### BF16 matrix multiplication

$X[B, D] \times Y[D, F] \rightarrow Z[B, F]$ in BF16:
- Load: $2BD + 2DF$ bytes (inputs)
- FLOPs: $2BDF$ (B×F output elements, each requiring D multiply-adds)
- Write: $2BF$ bytes (output)

$$\text{Intensity} = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF}$$

When $B \ll D$ and $B \ll F$ (typical for transformers where D, F > 4096 but local batch B is small):

$$\text{Intensity} \approx \frac{BDF}{DF} = B$$

The arithmetic intensity of a matmul is approximately equal to the batch size! We become compute-bound when:

$$B > 230$$

For a single NeuronCore-v3 in BF16. This is remarkably similar to TPUs (~240) and H100s (~295).

### What this means in practice

- **Token generation** (B=1): intensity ≈ 1. Hopelessly memory-bound. The tensor engine sits idle most of the time, waiting for weights to load. Speed is determined entirely by HBM bandwidth.
- **Small batch inference** (B=8): intensity ≈ 8. Still memory-bound, but 8× better than B=1.
- **Context encoding / prefill** (B=512): intensity ≈ 512. Solidly compute-bound. The tensor engine is fully utilized.
- **Training** (B=1024+): intensity >> 230. Compute-bound. This is the ideal regime.

### INT8/FP8 weights with BF16 compute

If weights are quantized to FP8 (1 byte) but activations stay in BF16 (2 bytes) and compute runs at BF16 speed:
- Load: $BD×2 + DF×1 = 2BD + DF$ bytes (activations in BF16, weights in FP8)
- FLOPs: still $2BDF$

When B is small relative to D, F:

$$\text{Intensity} \approx \frac{2BDF}{DF} = 2B$$

Critical batch size drops to $B > 115$. Quantizing weights halves the critical batch size — you escape the memory-bound regime sooner. This is why quantization helps inference latency even when it doesn't change the number of FLOPs.

---

## ESM-2 attention: compute-bound or memory-bound?

Let's apply this to our running example. In ESM-2 with seq_len=32, heads=20, head_dim=64:

**QK^T matmul** (per head): $[32, 64] \times [64, 32]$
- FLOPs: $2 \times 32 \times 64 \times 32 = 131,072$
- Bytes: $2(32×64) + 2(64×32) + 2(32×32) = 8192 + 8192 + 2048 = 18,432$
- Intensity: $131,072 / 18,432 = 7.1$ → **memory-bound**

**QK^T matmul** (seq_len=512): $[512, 64] \times [64, 512]$
- FLOPs: $2 \times 512 \times 64 \times 512 = 33,554,432$
- Bytes: $2(512×64) + 2(64×512) + 2(512×512) = 65,536 + 65,536 + 524,288 = 655,360$
- Intensity: $33,554,432 / 655,360 = 51.2$ → **still memory-bound** (but much better)

**FFN matmul**: $[512, 1280] \times [1280, 5120]$
- FLOPs: $2 \times 512 \times 1280 \times 5120 = 6.7$ billion
- Bytes: $2(512×1280) + 2(1280×5120) + 2(512×5120) = 1.3M + 13.1M + 5.2M = 19.7M$
- Intensity: $6.7B / 19.7M = 340$ → **compute-bound** ✓

Key insight: the FFN layers (large weight matrices) are compute-bound and efficient. The attention QK^T computation is memory-bound at typical sequence lengths — this is why flash attention and fused attention kernels matter.

---

## Network communication rooflines (multi-chip)

When you shard a model across multiple NeuronCores (Ch 17), a new roofline emerges. After computing partial results, chips must exchange data via NeuronLink.

Example: a matmul sharded along the contraction dimension across 2 chips. Each chip computes half, then they all-reduce the partial sums.

- $T_\text{math}$: halved (each chip does half the FLOPs)
- $T_\text{comms}$: $\frac{2BF}{\text{NeuronLink bandwidth}}$

The critical threshold now depends on **D** (model dimension), not B:

$$D > \frac{2 \times \text{FLOPs/s per chip}}{\text{NeuronLink bandwidth}} = \frac{2 \times 83.4 \text{ TFLOPS}}{1.28 \text{ TB/s}} \approx 130,000$$

Since D is typically 4096–16384, sharded matmuls are often **communication-bound**. This is why tensor parallelism has diminishing returns beyond a few chips — and why NeuronLink bandwidth matters so much for distributed training.

---

## From roofline to action

The roofline doesn't tell you *how* to optimize — it tells you *what* to optimize:

| Regime | Symptom | Action |
|--------|---------|--------|
| Memory-bound | Tensor engine idle, DMA active | Reduce data movement: fusion, keep data on-chip, increase batch size |
| Compute-bound | DMA idle, Tensor engine saturated | Already at peak — reduce total FLOPs (pruning, sparsity, smaller model) |
| Communication-bound | Compute idle between collectives | Overlap compute/communication, reduce sharding degree |

Most NKI kernel optimization (Part V) operates in the memory-bound regime — the goal is to move UP toward the roofline by eliminating unnecessary data movement (spills, reloads, unfused intermediates).

---

## The roofline is a ceiling, not a floor

The roofline represents 100% hardware utilization — you never actually reach it. Real performance falls below due to:

- Tiling overhead (not all tiles are perfectly sized)
- Pipeline bubbles (first/last tile don't overlap)
- Instruction overhead (~100 cycles per instruction regardless of payload)
- Memory bank conflicts
- Compiler scheduling inefficiency

The profiler (next chapter) shows you *where* on the roofline you actually sit and the gap between actual and theoretical tells you how much optimization opportunity remains.
