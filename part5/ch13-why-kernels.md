# Why custom kernels?

*The compiler can't fuse what doesn't exist.*

---

## The gap the compiler can't close

In Chapter 7, we saw SchNet's `scatter_reduce` fall back to CPU — the dispatcher found no Neuron implementation, so it round-tripped through PCIe. In Chapter 10, we profiled OpenCLIP and saw dense tensor engine activity with small gaps between matmuls. Those gaps are vector and scalar operations the compiler couldn't overlap perfectly.

`torch.compile` can only fuse operations that have Neuron implementations. It can't:
- Create a kernel for an unsupported op
- Fuse operations across the boundary of a CPU fallback
- Optimize a tiling pattern that its heuristics don't handle well
- Pipeline DMA and compute in a way the scheduler doesn't discover

When you hit these limits, you have two options: file a ticket and wait, or write it yourself. NKI (Neuron Kernel Interface) is the second option.

---

## Three reasons to write a kernel

### 1. Unsupported operators

The model uses an operation that has no Neuron implementation. The compiler can't help — there's nothing to compile TO.

Real examples:
- **Autodesk (trilinear interpolation):** the model's upsampling layer used an op the compiler couldn't lower. It didn't just fall back to CPU — it crashed the compiler entirely. An NKI kernel unlocked the model.
- **SchNet (scatter_reduce):** message passing in GNNs requires scatter operations. Every message-passing layer round-tripped through CPU. An NKI kernel kept it on-chip.
- **EquiformerV3:** 5 unsupported ops (`atan2`, `linalg_cross`, `acos`, `scatter_reduce`, `uniform_`) caused 87× slowdown vs the model's theoretical performance.

### 2. Performance hotspots the compiler misses

The operation IS supported, but the compiler's tiling/scheduling is suboptimal for your specific shapes or access patterns.

Real examples:
- **Mobilai (Max Pooling):** the compiler's lowering for max pool was correct but catastrophically slow. An NKI kernel was 100× faster.
- **Attention gaps:** between the two attention matmuls (QK^T and scores×V), the compiler inserts softmax as a sequence of scalar/vector ops with unnecessary HBM round-trips. A fused attention kernel keeps the intermediate scores in SBUF.

### 3. Algorithm-level reformulation

Sometimes the best kernel isn't a direct implementation of the original op — it's a different algorithm that achieves the same mathematical result but maps better to the hardware.

The classic example: **scatter as matmul.** `scatter_add(source, index)` is mathematically equivalent to `A @ source` where A is a binary selection matrix. On Neuron: the tensor engine does this matmul at 16,384 MACs/cycle. The scalar engine doing sequential scatter gets 1 op/cycle. That's a 16,384× throughput difference — if the graph is small enough for the dense matrix to fit in SBUF.

This is the deepest reason to learn NKI: not just "implement the missing op" but "understand the hardware well enough to reformulate the problem into something the hardware is great at."

---

## When NOT to write a kernel

- **Don't optimize what already works well.** Profile first. If the compiler gives you 80% MFU, a kernel won't help much.
- **Don't write a kernel just because a GPU version exists.** The hardware is completely different. GPU bottlenecks may not exist on Neuron, and vice versa.
- **Don't start at the keyboard.** Start at the whiteboard: What am I trying to do? What shapes? What's the theoretical best time? (roofline from Ch 9). Only then write code.

```{admonition} The one rule of kernel engineering
:class: important
"The goal of a kernel and the goal of a kernel engineer is to make sure the tensor engine is always doing matrix multiplications. Everything else is essentially auxiliary data movement to ensure that when the tensor engine is done with one matmul, the data needed for the next one is ready to go."

— Jay Gray, Trainium Inference Lead, Anthropic
```

The Neuron team's advice: "It's not because there is a kernel on GPU that you need one on Neuron — the hardware is totally different, bottlenecks may be different."

---

## What NKI is

NKI is a Python domain-specific language for writing kernels that run on NeuronCore. The API is heavily inspired by NumPy and Triton:

- If you know NumPy indexing, the syntax is familiar
- If you've written Triton kernels for GPU, the programming model (tiles, explicit loads/stores) maps directly
- It integrates into PyTorch and JAX via a `@nki.jit` decorator — use it like any other function

NKI has two levels:
- **`nki.language` (nl):** High-level, NumPy-like. One call may map to multiple hardware instructions. Good for fast prototyping.
- **`nki.isa` (nisa):** One call ≈ one hardware instruction. Maximum control. Good for performance engineering.

You can mix both in the same kernel. Start with language for correctness, move to ISA for performance.

---

## The NKI development cycle

From the NKI bootcamp (taught by the Annapurna Labs team):

1. **Write unit tests** — NumPy or PyTorch reference as ground truth. "I strongly encourage you to start by writing these unit tests. It's not time wasted."
2. **Build a working kernel on smallest inputs** — 128×512 (fits tile constraints perfectly). Get correct results before anything else.
3. **Scale your inputs** — this usually requires changing your tiling strategy or algorithm.
4. **Profile → optimize → repeat** — enter the performance optimization loop (Ch 14-16).

The Autodesk trilinear interpolation kernel went through 5 implementation attempts over 4-6 weeks before converging on a working, performant solution. Kernel development is iterative. The first version is never the final version.

---

## The kernel ecosystem

You don't always need to write from scratch:

**NKI Library** (production-ready kernels): https://github.com/aws-neuron/nki-library
- Pre-optimized kernels you can drop into models: flash attention, fused softmax, RMSNorm-Quant, MoE, etc.
- Designed to be generalizable across shapes, sizes, and architectures
- Good for studying how experts tile, pipeline, and schedule

**NKI Samples** (learning examples): https://github.com/aws-neuron/nki-samples
- Tutorial-style kernels with step-by-step explanations
- Start here for your first kernel
- Community contributions welcome — even non-optimized kernels help others

The NKI Library kernels have "prefill" and "decode" flavors — different kernels optimized for different arithmetic intensity regimes (compute-bound context encoding vs memory-bound token generation).

---

## What's ahead in Part V

| Chapter | What you'll build | Concept introduced |
|---------|------------------|--------------------|
| Ch 14 | Vector add, simple matmul | Load/store, tiles, the basic workflow |
| Ch 15 | Tiled ESM-2 attention matmul | Tile sizing, engine constraints, advanced indexing |
| Ch 16 | Pipelined attention (3× faster) | Double-buffering, allocation, the 12-step optimization |

By the end, you'll have a kernel that takes ESM-2 attention from 24μs to 8.5μs — a 3× improvement over the compiler's output. And you'll understand *why* each optimization step works, because you'll have the hardware mental model from Parts II-IV.

*Next: [Chapter 14](ch14-first-nki-kernel) — Your first NKI kernel.*
