# PyTorch Native on Neuron

In Chapter 4, we saw what the NeuronCore hardware looks like — engines, SBUF, DMA. But how does your PyTorch code actually *reach* those engines? This chapter covers the integration layer: how Neuron registers itself as a PyTorch backend, what happens when you call `.to("neuron")`, and why a one-line change is enough to run any model.

---

## The PrivateUse1 backend

PyTorch has a built-in extension mechanism for hardware accelerators. Every device type — CPU, CUDA, MPS — is a registered backend that responds to dispatched ops. Neuron uses the same mechanism via `PrivateUse1`, a dispatch key reserved for out-of-tree backends:

```python
import torch

x = torch.randn(4, 4, device="neuron")
print(x.device)  # neuron:0
```

The registration happens once at import time. After that, any ATen op dispatched on a neuron tensor routes to the Neuron backend automatically.

```
torch.matmul(A, B)      A.device == "neuron"
       │
       ▼
┌─────────────────┐
│   Dispatcher     │    Checks device → finds PrivateUse1 registration
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Neuron backend  │    Compiles op → NEFF → executes on NeuronCore
└─────────────────┘
```

This is the same pattern CUDA uses — there's nothing special about Neuron's integration. The `torch.accelerator` API auto-detects the available backend:

```python
print(torch.accelerator.current_accelerator())  # "neuron"
print(torch.accelerator.device_count())         # number of NeuronCores (4 cores for trn2.3xlarge)
```

---

## A brief history of PyTorch Native support for Neuron (optional read)

Before 2024, Neuron's PyTorch integration went through **XLA** (the same framework TPUs use). XLA works by capturing an entire computation graph lazily and compiling it all at once:

```python
# The OLD way (XLA) — don't do this anymore
import torch_xla.core.xla_model as xm

device = xm.xla_device()
model = model.to(device)
output = model(x)
xm.mark_step()  # ← required: triggers compilation and execution of everything above
```

XLA had real advantages — whole-graph compilation enabled aggressive optimizations. But it also had painful problems:

- **Debugging was impossible.** Errors surfaced as compiler failures deep inside XLA, not at the Python line that caused them.
- **Dynamic control flow broke.** Any `if/else` depending on tensor values caused graph breaks that XLA couldn't recover from gracefully.
- **Ecosystem incompatibility.** Libraries that assumed eager semantics (HuggingFace, most training frameworks) needed XLA-specific forks.
- **Falling behind.** PyTorch evolved faster than the XLA team could follow — new features like `torch.compile`, scaled_dot_product_attention, and FSDP2 all required XLA reimplementations.

The PyTorch Native approach eliminates all of this.

---

## The two execution paths

When you run PyTorch on Neuron, your code takes one of two paths depending on whether you use `torch.compile`. These aren't just "slow vs fast" — they serve different stages of the development lifecycle:

- **Eager mode** is for iteration and debugging. You get immediate feedback, can print any tensor, set breakpoints anywhere. It's what researchers use when experimenting — fast time to first result, even if each result is slower.
- **torch.compile** is for production delivery. You pay compilation cost upfront, then get maximum hardware utilization on every subsequent call. It's what performance engineers hand off for deployment.

### Eager mode (op-by-op)

```python
model = model.to("neuron")
output = model(x)  # Each op compiled and executed individually
```

For each ATen op the dispatcher sends to the Neuron backend:
1. **Cache check** — has this exact op + shape been compiled before?
2. **If hit** — load the cached NEFF, execute immediately
3. **If miss** — compile via neuronx-cc, cache the resulting NEFF, then execute

```
aten::matmul [1024, 4096] × [4096, 4096] bfloat16
       │
       ▼
  Cache lookup: /tmp/neff_cache/<hash>.neff
       │
  ┌────┴────┐
  │ HIT     │ MISS
  │         │
  ▼         ▼
Execute   neuronx-cc compile (seconds)
            │
            ▼
       Save to cache
            │
            ▼
         Execute
```

The first iteration is slow (compilation). Subsequent iterations with the same shapes are fast (cache hits). This is what we observed in Chapter 1: first call ~8ms, second call ~49µs.

### torch.compile mode (graph)

```python
compiled_model = torch.compile(model, backend="neuron")
output = compiled_model(x)  # Entire graph compiled as one NEFF
```

The compilation pipeline:
1. **Dynamo** captures the Python execution into an FX graph of ATen ops
2. **torch-MLIR** lowers the FX graph to Stable HLO (an intermediate representation)
3. **neuronx-cc** compiles the HLO graph into a single optimized NEFF
4. The NEFF is cached and reused on subsequent calls

```A
Python code
    │
    ▼ (Dynamo traces)
FX Graph (DAG of ATen ops with shapes)
    │
    ▼ (torch-MLIR lowers)
Stable HLO
    │
    ▼ (neuronx-cc compiles)
NEFF (one binary for the entire graph)
    │
    ▼
Execute on NeuronCore
```

Both paths use the same compiler backend (neuronx-cc). The difference is *granularity*: eager compiles one op at a time; torch.compile compiles the full graph. This is why torch.compile gives the 8× speedup we saw in Chapter 3 — the compiler can fuse ops, eliminate intermediate memory writes, and pipeline DMA across the entire graph.

---

## CPU fallback — the safety net

Not every PyTorch operation has a Neuron implementation. When the dispatcher encounters an unsupported op on a neuron tensor, it falls back to CPU automatically:

1. Transfer input tensors from NeuronCore HBM → CPU memory
2. Execute the op on CPU (using MKL, etc.)
3. Transfer the result back to NeuronCore HBM

```python
x = torch.randn(8, 128, device="neuron")
indices = torch.zeros(8, 1, dtype=torch.long, device="neuron")
src = torch.ones(8, 1, device="neuron")

# scatter has no Neuron implementation → silent CPU fallback
result = torch.scatter(x, 1, indices, src)
print(result.device)  # neuron:0 — looks normal!
```

The output tensor reports `device=neuron:0` because the runtime moves the result back automatically. **The fallback is functionally invisible** — your model produces correct results. But the PCIe round-trip (device → host → compute → host → device) adds latency.

The contract is: **correctness first, performance second.** Any PyTorch model should *run* on Neuron. Not all operations will run *fast*. The fallback mechanism ensures you can always get a model working, then optimize the hot path.

```{admonition} Detecting fallbacks
:class: tip
Profile with `torch.profiler` and look for `aten::to` / `aten::copy_` chains surrounding an op — that's the transfer signature. We covered this in Chapter 3.
```

---

## What changes in your code — and what doesn't

Here's the minimal diff to move a training loop from CUDA to Neuron:

```python
# Before (CUDA)                        # After (Neuron)
device = "cuda"                         device = "neuron"
model = Model().to(device)              model = Model().to(device)
optimizer = Adam(model.parameters())    optimizer = Adam(model.parameters())

for batch in dataloader:                for batch in dataloader:
    x = batch.to(device)                    x = batch.to(device)
    loss = model(x).sum()                   loss = model(x).sum()
    loss.backward()                         loss.backward()
    optimizer.step()                        optimizer.step()
    optimizer.zero_grad()                   optimizer.zero_grad()
```

One word changed. Everything else — the optimizer, the data loading, `loss.backward()`, gradient accumulation — works identically because it all goes through the same PyTorch APIs.

```{admonition} The performance ladder
:class: note
The API is identical to CUDA — same dispatcher, same device change. But the path to peak performance is different. On GPU, vendor libraries (cuBLAS, cuDNN) provide strong defaults — you start fast and optimize from there. On Neuron, the ladder is: eager mode (correctness) → `torch.compile` (graph-level optimization) → NKI kernels (hardware-level control). Each step unlocks more performance by giving the compiler — or you — more information about the workload. The upside: NKI gives you direct hardware access that no GPU vendor exposes. The book walks you up this ladder.
```

What you *do* need to be aware of:

**Async execution.** Neuron executes asynchronously (Chapter 1). For accurate timing, add `torch.neuron.synchronize()` before measuring:

```python
torch.neuron.synchronize()
start = time.time()
output = model(x)
torch.neuron.synchronize()  # Wait for hardware to actually finish
elapsed = time.time() - start
```

**Compilation cost.** The first forward pass (eager) or first call to a compiled model triggers NEFF compilation. This is seconds to minutes depending on model size. Subsequent calls use the cache.

**Fixed shapes.** NEFFs are compiled for specific tensor shapes. Dynamic sequence lengths trigger recompilation. Use bucketing (Chapter 3) for variable-length inputs.

---

## NKI integration — custom kernels in native PyTorch

When the compiler can't handle an operation (or you need more performance than it provides), NKI kernels plug into the native PyTorch stack the same way Triton kernels do for CUDA:

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def my_kernel(input_tensor, output_tensor):
    # Direct control over NeuronCore engines
    ...

# Register for eager mode
torch.ops.my_namespace.my_op = my_kernel

# Or use inside torch.compile — the compiler treats it as a custom op
```

The pattern mirrors Triton: `nki.jit` ↔ `triton.jit`. Custom kernels participate in autograd (you can define backward passes), work in both eager and compiled modes, and can be reused across models. We'll write our first NKI kernel in Part V.

---

## The full picture

```
Your PyTorch code
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Dispatcher                                     │
│  device="neuron" → route to Neuron backend      │
└─────────────┬────────────────────┬──────────────┘
              │                    │
    ┌─────────┴──-───┐   ┌──-──────┴────────┐
    │  Eager mode    │   │  torch.compile   │
    │  op-by-op      │   │  full graph      │
    └───────┬────────┘   └────────┬─-───────┘
            │                     │
            ▼                     ▼
    ┌───────────────┐   ┌────────────────────┐
    │ NEFF per op   │   │ One NEFF per graph │
    │ (cached)      │   │ (Dynamo → HLO →    │
    │               │   │  neuronx-cc)       │
    └───────┬───────┘   └────────┬───────────┘
            │                    │
            └──────────┬─────────┘
                       ▼
              NeuronCore execution
              (engines from Ch4)
```

The key insight: **the PyTorch Native integration means Neuron is a first-class PyTorch backend.** There's no separate framework to learn, no XLA quirks to work around, no special APIs. Your existing PyTorch knowledge transfers directly. The performance optimization story (torch.compile, profiling, NKI) layers on top of working code, you never need to change your model to make it *run*, only to make it *fast*.

