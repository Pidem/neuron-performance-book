# Migrating from GPU to Neuron

*Setting the right expectations: this is performance engineering, not a drop-in replacement.*

Neuron's PyTorch Native integration means your model code runs with a one-word change (`cuda` → `neuron`). But running ≠ running well. The API compatibility gets you to *functional* quickly. Getting to *performant* requires understanding the hardware — which is what this entire book teaches.

If you're coming from GPU expecting identical behavior with better price-performance, recalibrate. Neuron is a different architecture with different strengths, different bottlenecks, and a different optimization path. The engineers who succeed on Neuron are those who approach it as a performance engineering challenge, not a migration checkbox.

---

## When Neuron is the right fit

Neuron's architecture excels in specific scenarios. Understanding where it shines helps you set the right expectations and invest your engineering effort where it compounds.

**Strong fit:**

- **Transformer-based models (LLMs, diffusion, vision transformers)** — attention and MLP layers are dominated by large matrix multiplications, which map directly to the tensor engine's 128×128 systolic array. This is the workload Neuron was purpose-built for.
- **Teams with a performance engineering culture** — if you already profile your GPU workloads, track MFU, and iterate on kernel efficiency, Neuron rewards that discipline. The profiler (Neuron Explorer) gives you deeper visibility than most GPU profiling tools, and NKI gives you more direct hardware control than CUDA in many respects.
- **Capacity, cost, or compute diversity goals** — Neuron offers a path to large-scale training and inference without competing for GPU allocation. Trainium instances are available in regions and at price points that complement GPU fleets rather than replacing them.
- **Long-term hardware investment** — the Neuron SDK, NKI, and PyTorch Native integration improve with each generation (Trn2 → Trn3 → Trn4). Teams that invest in understanding the architecture carry that knowledge forward as the hardware scales, with each generation delivering 4–6× improvements on the same code.
- **Standard PyTorch/JAX workflows without hard CUDA coupling** — if your training loop uses `torch.compile`, HuggingFace Transformers, FSDP, or vLLM, the migration path is straightforward. These frameworks already support Neuron as a backend.

**Weaker fit (today):**

- **Models with extensive custom CUDA kernels** — if your forward pass depends on hand-written `.cu` files or libraries that wrap CUDA-specific implementations (custom cuDNN layers, NCCL-specific communication patterns), those will need NKI equivalents. The NKI Kernel Library covers common operations (flash attention, fused softmax, RMSNorm), but niche custom kernels require porting effort.
- **Expectation of zero engineering effort** — Neuron is not a transparent GPU emulator. The one-line device change gets you *functional*, but competitive performance requires profiling and optimization work — the same work this book teaches. Teams that approach it as "just swap the device and ship" will be disappointed.
- **Heavy scatter/gather workloads without workarounds** — GNNs, recommendation models with large sparse lookups, or architectures that rely heavily on irregular memory access patterns will hit more CPU fallbacks. These are solvable (reformulate as matmul, write NKI kernels) but require upfront investment.

```{admonition} The honest framing
:class: tip
Neuron is not "GPUs but cheaper." It's a different architecture with a different optimization surface. The teams that succeed treat it as a performance engineering opportunity — and the ones that get the most value are those building on top of transformers at scale, where Neuron's architectural choices (large SRAM, dedicated tensor engine, independent collectives) create genuine advantages over GPU for their workload.
```

---

## The migration path

```
Day 1:     model.to("neuron") — it runs (possibly with fallbacks)
Week 1:    torch.compile — 2-8× speedup, identify fallback ops
Month 1:   Profile and optimize — tile sizes, precision, batch strategy
Month 2+:  NKI kernels for remaining hotspots (if needed)
```

Most models reach competitive performance at the `torch.compile` stage without NKI. The models that need NKI are those with:
- Custom attention patterns (sliding window, sparse, cross-attention variants)
- Non-standard activations or normalization layers
- GNN/scatter-heavy architectures
- Latency-sensitive decode paths where every microsecond matters

---

## What's improving on the roadmap

The Neuron SDK evolves rapidly. Areas of active development (as of May 2026):
- **Operator coverage** — the gap between supported and unsupported ops shrinks every release. Check the [supported operators list](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/pytorch-neuron-supported-operators.html) for current status.
- **vLLM native integration** — inference serving with paged attention, chunked prefill, speculative decoding
- **NKI compiler** — a dedicated compiler for NKI kernels (separate from the model compiler) that will enable better kernel-to-kernel fusion
- **Collectives in NKI** — writing distributed kernels that communicate directly between chips
- **PyTorch ecosystem compatibility** — TorchTitan, HuggingFace Transformers v5, FSDP2 all being validated

---

## The 5-minute functional migration

```python
# Before (CUDA)
device = "cuda"
model = Model().to(device)
output = model(x.to(device))

# After (Neuron)
device = "neuron"
model = Model().to(device)
output = model(x.to(device))
```

Then immediately:

1. **Check fallbacks:** profile for `copy_` chains indicating CPU round-trips
2. **Compile:** `torch.compile(model, backend="neuron")` — measure speedup
3. **Bucket:** if variable shapes, implement bucketing (Ch 3)
4. **Profile:** open Neuron Explorer, check MFU, identify the bottleneck

From here, the rest of this book applies.
