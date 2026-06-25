# Number formats for humans

You profiled a kernel in Neuron Explorer and spotted something odd: the Tensor Engine is idle 60% of the time, waiting on DMA. The operation is memory-bound, not because your matrix is small, but because each element is 4 bytes wide and HBM bandwidth is finite. What if each element were 2 bytes? Or 1?

*ESM-2 attention scores: do they need 32 bits?*

Smaller numbers mean fewer bytes crossing the bus, which means higher effective arithmetic intensity for the same math. But how small can you go before the model stops working? And what exactly do you lose when you shrink a floating-point number from 32 bits to 8?

---

## Why precision matters for performance

From Chapter 9: the arithmetic intensity of a matmul `[B, D] × [D, D]` is approximately B. A training matmul with B=1024 has intensity well above Trn2's critical threshold of 230: the tensor engine is the bottleneck, and DMA finishes loading the next weight tile before compute needs it. But token generation runs at B=1. You load the entire weight matrix (D×D×2 bytes) from HBM to produce a single output vector (D elements). Intensity ≈ 1. The tensor engine sits idle most of the time, waiting for the next weight tile to arrive.

This is where precision connects to performance:

- **Memory-bound ops (token generation, B=1):** switching from BF16 (2 bytes) to FP8 (1 byte) halves the data that crosses the bus. Same math, half the wait. Nearly 2× faster.
- **Compute-bound ops (training, B=1024):** the bottleneck is already the tensor engine, not DMA. Reduced precision helps only if the hardware computes faster in that format (Trn2: 667 TFLOPS BF16, 1299 TFLOPS FP8).
- **The tradeoff:** smaller numbers can't represent all values accurately. Potential accuracy loss.

In short: precision reduction helps memory-bound workloads by shrinking the data, and helps compute-bound workloads by unlocking faster hardware paths. Token generation benefits from both.

### Concrete example: ESM-2's FFN weight

```python
import torch

# ESM-2 FFN weight: [1280, 5120]
shape = (1280, 5120)
elements = shape[0] * shape[1]

formats = {
    "FP32":  (torch.float32, 4),
    "BF16":  (torch.bfloat16, 2),
    "FP16":  (torch.float16, 2),
    "FP8":   (torch.float8_e4m3fn, 1),
    "INT8":  (torch.int8, 1),
}

print(f"Tensor shape: {shape} ({elements:,} elements)\n")
print(f"{'Format':<8} {'Bytes/elem':<12} {'Total size':<12} {'vs FP32'}")
print("-" * 48)

fp32_size = elements * 4
for name, (dtype, nbytes) in formats.items():
    total = elements * nbytes
    ratio = total / fp32_size
    print(f"{name:<8} {nbytes:<12} {total/1024/1024:.1f} MB      {ratio:.2f}×")
```

```none
Tensor shape: (1280, 5120) (6,553,600 elements)

Format   Bytes/elem   Total size   vs FP32
------------------------------------------------
FP32     4            25.0 MB      1.00×
BF16     2            12.5 MB      0.50×
FP16     2            12.5 MB      0.50×
FP8      1            6.2 MB       0.25×
INT8     1            6.2 MB       0.25×
```

One FFN layer in FP8 fits in 6.2 MB. In FP32 it's 25 MB. ESM-2 has 33 layers with two FFN projections each, so the full model shrinks from ~1.6 GB (FP32) to ~400 MB (FP8). That's the difference between fitting in SBUF-resident tiling vs. spilling to HBM on every layer.

---

## The number format zoo

### What's in a floating-point number?

Every format has three fields: **sign** (1 bit), **exponent** (range), **mantissa** (precision):
- More exponent bits → wider range (can represent very large and very small numbers)
- More mantissa bits → finer precision (more distinct values between two powers of 2)
- Total bits = sign + exponent + mantissa = storage cost per element

Consider the number $0.000012346789$. In scientific notation: $1.2346789 \times 10^{-5}$. The **exponent** ($-5$) tells you where the decimal point sits. The **mantissa** ($1.2346789$) carries the significant digits. Now limit the mantissa:

| Mantissa digits | Stored as | Lost detail |
|---|---|---|
| 8 (full) | $1.2346789 \times 10^{-5}$ | none |
| 4 | $1.235 \times 10^{-5}$ | last 4 digits rounded |
| 2 | $1.2 \times 10^{-5}$ | only 2 significant figures survive |
| 1 | $1 \times 10^{-5}$ | order of magnitude only |

Binary floating point works the same way. FP32 gives you 23 mantissa bits (~7 decimal digits). BF16 gives you 7 bits (~2 digits). FP8 E4M3 gives you 3 bits (~1 digit). The number stays in the right ballpark (the exponent handles that), but fewer mantissa bits mean coarser rounding at each step.

### The formats you'll encounter

| Format | Bits | Exponent | Mantissa | Range | Precision | Use case |
|--------|------|----------|----------|-------|-----------|----------|
| FP32 | 32 | 8 | 23 | ±3.4×10³⁸ | ~7 decimal digits | Optimizer states, loss computation |
| TF32 | 19 | 8 | 10 | Same as FP32 | ~3 decimal digits | NVIDIA-specific; same range, less precision |
| BF16 | 16 | 8 | 7 | Same as FP32 | ~2 decimal digits | Default training/inference on Neuron |
| FP16 | 16 | 5 | 10 | ±65,504 | ~3 decimal digits | More precise than BF16 but narrower range |
| FP8 E5M2 | 8 | 5 | 2 | ±57,344 | ~0.6 decimal digits | Backward pass (gradients need range) |
| FP8 E4M3 | 8 | 4 | 3 | ±448 | ~1 decimal digit | Forward pass (activations need precision) |
| INT8 | 8 | — | — | -128 to 127 | Exact integers | Weight quantization |

### The key insight: BF16 vs FP16

```{figure} ../assets/data_types_neuron.png
:alt: Floating-point data type bit layouts
:width: 600px
:align: center

Bit layouts of common floating-point formats. BF16 keeps FP32's 8-bit exponent (same dynamic range), while FP16 trades range for precision. FP8 variants split their 8 bits differently: E5M2 favors range, E4M3 favors precision.
```

- BF16 keeps the same 8-bit exponent as FP32 → same dynamic range, just less precision
- FP16 has only 5-bit exponent → can overflow (values > 65,504 become infinity)
- This is why BF16 "just works" as a drop-in for FP32 training  (no loss scaling needed)
- FP16 training requires loss scaling to prevent gradient overflow (added complexity)
- Neuron natively supports both; BF16 is the default and recommended choice

---

## FP8: the new frontier

FP8 halves memory vs BF16 and can double throughput on hardware with FP8 tensor cores:

### Two flavors for different needs

- **E5M2** (5-bit exponent, 2-bit mantissa): wide range, low precision. Best for gradients (backward pass) where values span many orders of magnitude
- **E4M3** (4-bit exponent, 3-bit mantissa): narrower range, better precision. Best for activations/weights (forward pass) where values cluster in a tighter range

### The outlier problem

- Neural network tensors often have a few extreme values ("outliers") while most values are small
- Naive FP8 casting: the representable range covers the outliers → all small values collapse to the same few FP8 values → catastrophic accuracy loss
- Example: if max value is 1000 but 99% of values are between -1 and 1, FP8's 448 (E4M3) or 57,344 (E5M2) range wastes most of its precision on the empty space

### Per-tensor vs per-group scaling

- **Per-tensor scaling (absmax):** find the max absolute value in the tensor, scale everything so max maps to FP8 max. Simple but one outlier ruins precision for all other elements
- **Per-group scaling (block-wise):** divide tensor into groups of 32-128 elements, scale each independently. Outliers only affect their own group. Better accuracy, slightly more metadata
- **Microscaling (MXFP8/MXFP4):** hardware-native per-group quantization. Each group of elements shares a small scale factor stored alongside. Trn3 has dedicated silicon circuits for this — zero compute overhead for the scaling step

---

## When precision improves: bottlenecks move

Reducing precision doesn't always give you the speedup you expect. Here's a real example from self-attention that illustrates why:

**Self-attention pipeline:** QK^T matmul → softmax → V matmul

**Step 1: BF16 baseline.** In BF16, the timeline is roughly balanced with matmuls and softmax taking comparable time. The tensor engine is well-utilized.

**Step 2: Switch matmuls to FP8.** The tensor engine runs at 2× throughput (1299 vs 667 TFLOPS). The QK^T and V matmuls finish in half the time. But total latency doesn't halve — because **softmax is now the bottleneck**. It runs on the scalar/vector engines at the same speed as before, and the tensor engine sits idle waiting for it.

**Step 3: Hardware-accelerated softmax (Trn3).** Trainium 3 adds dedicated softmax circuits that run 4× faster while maintaining full precision. Now the pipeline is balanced again, and the end-to-end self-attention achieves the full 2× speedup.

```{admonition} The lesson
:class: important
Optimizing one stage of a pipeline exposes the next bottleneck. Casting matmuls to FP8 without addressing the surrounding ops (softmax, layer norm, activations) gives you less than the theoretical 2× improvement. Always profile the full pipeline — not just individual ops — after a precision change.
```

This is why Neuron's approach combines lower-precision compute with specialized hardware acceleration for the operations that become bottlenecks. It's not enough to make numbers smaller — you need the entire pipeline to speed up together.

---

## What Neuron supports

### NeuronCore-v2 (Trn1, Inf2)

| Engine | Input types | Accumulation | Output |
|--------|------------|--------------|--------|
| Tensor engine | BF16, FP16 | FP32 | FP32 (in PSUM) |
| Vector engine | FP32, BF16, FP16 | FP32 | FP32 or BF16 |
| Scalar engine | FP32, BF16, FP16 | — | Same as input |

- No FP8 on Trn1
- PSUM always accumulates in FP32 (zero-overhead — hardware design)
- Copy PSUM → SBUF can cast (FP32 → BF16) at zero cost

### NeuronCore-v3 (Trn2)

| Engine | Input types | Accumulation | Output |
|--------|------------|--------------|--------|
| Tensor engine | BF16, FP16, **FP8 (E5M2, E4M3)** | FP32 | FP32 (in PSUM) |
| Vector engine | FP32, BF16, FP16 | FP32 | FP32 or BF16 |
| Scalar engine | FP32, BF16, FP16 | — | Same as input |

- FP8 inputs to tensor engine → doubled throughput (1299 TFLOPS vs 667 for BF16)
- Accumulation still in FP32 — no precision loss during the matmul itself
- The precision loss happens at the *casting boundary* (when you convert your BF16/FP32 tensor to FP8 before feeding the engine)

### NeuronCore-v4 (Trn3 — announced)

- Adds hardware microscaling circuits (MXFP8, MXFP4)
- Per-group scaling computed in silicon at line-rate — no extra instructions needed
- Accelerated softmax in hardware (stays high precision but runs 4× faster)
- FP4 support for weight storage (4 bits per parameter)

---

## Stochastic rounding

When casting from higher to lower precision, the default is "round to nearest even" (RNE). But this introduces a bias: values always round toward the nearest representable number, which can accumulate errors over millions of gradient updates.

**Stochastic rounding:** randomly round up or down with probability proportional to the distance to each representable value. Over many iterations, the expected value of the rounded number equals the original — unbiased. This can improve model convergence in low-precision training.

Neuron supports stochastic rounding via environment variable:
```bash
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
```

---

## Choosing a format: the decision framework

Three questions decide your number format:

**1. Is this operation accumulating many values?** (optimizer states, loss computation, running statistics) → **FP32.** Precision compounds over millions of additions.

**2. Is this a matmul weight or activation in forward/backward?** → **BF16 by default.** Switch to FP8 when you've validated accuracy AND the operation is memory-bound (intensity < 230). FP8 halves the bytes and doubles effective bandwidth — but only helps if bandwidth was the bottleneck.

**3. Is this inference-only with a latency target?** → **FP8 or INT8 for weights** (halves memory, fits larger models on-chip). Keep activations in BF16 unless you've measured acceptable accuracy loss.

The rule underneath: **never downcast for compute-bound ops** (you gain nothing — the tensor engine is already saturated). **Always consider downcasting for memory-bound ops** (you shift the bottleneck from bandwidth to compute, which is where you want to be).

---

*Question raised → "How do I actually apply this on Neuron — which layers to cast, which to keep?"*

*Next: [Chapter 12](ch12-casting-neuron) — Casting on Neuron.*
