# Migrating from GPU to Neuron

*Setting the right expectations: this is performance engineering, not a drop-in replacement.*

Neuron's PyTorch Native integration means your model code runs with a one-word change (`cuda` → `neuron`). But running ≠ running well. The API compatibility gets you to *functional* quickly. Getting to *performant* requires understanding the hardware — which is what this entire book teaches.

If you're coming from GPU expecting identical behavior with better price-performance, recalibrate. Neuron is a different architecture with different strengths, different bottlenecks, and a different optimization path. The engineers who succeed on Neuron are those who approach it as a performance engineering challenge, not a migration checkbox.

---

## When Neuron is the right fit

**Good candidates:**
- Transformer models (LLMs and diffusion) — the architecture maps naturally to the tensor engine
- Teams open to optimization work — willing to profile, iterate, and go low-level when needed
- Workloads where you need capacity, price-performance, or compute diversity beyond GPU availability
- Strategic mindset — investing time to understand the hardware pays compounding returns
- No hard CUDA dependencies (custom CUDA kernels, NCCL-specific communication patterns)

**Poor candidates:**
- Expecting a drop-in GPU replacement with zero engineering effort
- Models heavily dependent on CUDA-specific libraries (cuDNN custom layers, custom CUDA kernels with no NKI equivalent yet)
- Just looking for credits or trying Neuron only because GPUs are unavailable

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
