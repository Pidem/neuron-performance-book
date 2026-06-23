# Number formats for humans

*ESM-2 attention scores: do they really need 32 bits?*

In Chapter 9, we learned that arithmetic intensity determines whether an operation is compute-bound or memory-bound. Smaller numbers = fewer bytes to move = higher effective arithmetic intensity. This chapter explains what you lose when you shrink numbers — and what you gain.

---

## Why precision matters for performance

- Moving data is the bottleneck (Ch 5). If you halve the bytes per element, you halve the DMA time
- For a memory-bound operation (like token generation, B=1): switching from BF16 (2 bytes) to FP8 (1 byte) is nearly 2× faster — same FLOPs, half the data movement
- For a compute-bound operation (like large matmul in training): reduced precision may also enable higher FLOPs/s if the hardware supports it (Trn2: 667 TFLOPS BF16, 1299 TFLOPS FP8)
- The tradeoff: smaller numbers can't represent all values accurately → potential accuracy loss

---

## The number format zoo

### What's in a floating-point number?

Every format has three fields: **sign** (1 bit), **exponent** (range), **mantissa** (precision):
- More exponent bits → wider range (can represent very large and very small numbers)
- More mantissa bits → finer precision (more distinct values between two powers of 2)
- Total bits = sign + exponent + mantissa = storage cost per element

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
- This is why BF16 "just works" as a drop-in for FP32 training — no loss scaling needed
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

## Rules of thumb

- **Training forward pass:** BF16 (safe default) or FP8 E4M3 (if you validate accuracy)
- **Training backward pass:** BF16 (safe) or FP8 E5M2 (wider range for gradients)
- **Optimizer states:** always FP32 (Adam's running averages need precision)
- **Inference weights:** FP8 or INT8 (half the memory, fits larger models on-chip)
- **Inference activations:** BF16 (safe) or FP8 (if latency-critical and accuracy validated)
- **Softmax, LayerNorm internals:** always FP32 (exponentiation and reduction need precision)

*Question raised → "How do I actually apply this on Neuron — which layers to cast, which to keep?"*

*Next: [Chapter 12](ch12-casting-neuron) — Casting on Neuron.*
