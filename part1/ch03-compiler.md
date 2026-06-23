# The Compiler

## Recap 

In chapter 1 and 2, we saw that in eager mode, each operation is independent: PyTorch dispatches each operation without taking into consideration the "big picture". No operation knows what comes next, layout choices are made locally, fusion are made manually. This the problem that the compiler solves for: The compiler looks at the entire computational graph and optimizes globally. 

---

## Benchmarking inference with and without compilation

In Chapter 1 we ran ESM-2 in eager mode — each of the ~330 ops compiled and executed independently. What if the compiler could see the *entire* forward pass as one graph?

```python
import torch, time
from transformers import EsmModel, EsmTokenizer

device = "neuron"
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained(
    "facebook/esm2_t33_650M_UR50D", attn_implementation="eager"
).to(device)
model.eval()

inputs = {k: v.to(device) for k, v in tokenizer(
    "FVNQHLCGSHLVEALYLVCGERGFFYTPKT", return_tensors="pt"
).items()}

# --- Eager baseline ---
with torch.no_grad():
    for _ in range(3):
        _ = model(**inputs)
torch.neuron.synchronize()

torch.neuron.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = model(**inputs)
torch.neuron.synchronize()
t_eager = (time.time() - start) / 10

# --- Compiled ---
compiled_model = torch.compile(model, backend="neuron")

with torch.no_grad():
    for _ in range(3):
        _ = compiled_model(**inputs)
torch.neuron.synchronize()

torch.neuron.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = compiled_model(**inputs)
torch.neuron.synchronize()
t_compiled = (time.time() - start) / 10

print(f"Eager:    {t_eager*1000:.1f}ms")
print(f"Compiled: {t_compiled*1000:.1f}ms")
print(f"Speedup:  {t_eager/t_compiled:.2f}x")
```

```none
Eager:    124.5ms
Compiled:  15.6ms
Speedup:  7.99x
```

**8x faster.** We executed the same model, with the same input, and only added this line: `torch.compile(model, backend="neuron")`.
But *why*? The hardware hasn't gotten faster. What changed?

---

## Looking under the hood with the neuron profiler

Let's profile both paths and compare what the hardware actually does and look at the computational graph being dispatched:

```python
import os, torch
from torch.profiler import profile, ProfilerActivity
from torch_neuronx.profiling import NeuronConfig, ProfileMode, NeuronProfiler

# Profile eager
os.system("rm -rf /workshop/profile_eager")
config = NeuronConfig(
    modes=[ProfileMode.RUNTIME],
    profile_output_dir="/workshop/profile_eager",
    capture_enabled_for_nc="0", 
)
exporter = NeuronProfiler(config)

with torch.no_grad():
    for _ in range(3):
        _ = model(**inputs)
torch.neuron.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    experimental_config=config,
    on_trace_ready=exporter.export_trace,
) as prof:
    with torch.no_grad():
        _ = model(**inputs)
    torch.neuron.synchronize()
```

```bash
$ neuron-explorer view -d /workshop/profile_eager --display-name eager --output-format summary-text
$ neuron-explorer view -d /workshop/profile_compiled --display-name compiled --output-format summary-text
```

| Metric | Eager | Compiled | Reduction |
|--------|------:|--------:|-----------:|
| NEFF submissions (`nrt_model_submit`) | 1,050 | 1 | 1050× |
| Hardware executions (`nc_exec_running`) | 2,100 | 2 | 1050× |
| DMA transfers (`dmem_buf_copyin`) | 13,785 | 14 | 985× |
| Memory allocations (`nrt_dma_mem_alloc`) | 5,196 | 6 | 866× |
| NEFF swaps (`nc_model_switch`) | 1,968 | 0 | eliminated |
| Total trace events | 58,707 | 531 | 110× |

The numbers tell a clear story:

- **Eager**: 1,050 separate NEFFs submitted one by one. Each needs its own DMA setup, memory allocation, and model switch. The NeuronCore spends most of its time *managing* work — loading NEFFs, shuffling data, switching contexts.

- **Compiled**: the entire ESM-2 forward pass (33 transformer layers, 650M parameters) fuses into **1 NEFF**, submitted once, executed in 2 pipelined hardware passes with 14 DMA transfers total.

```{admonition} The 8x speedup isn't faster math
:class: important
The tensor engine runs at the same speed in both cases. The speedup comes entirely from eliminating overhead: fewer NEFF loads, fewer DMA transfers, fewer memory allocations, zero context switches. The compiler made the *plumbing* disappear.
```

---

## What `torch.compile` actually does

Three stages turn your Python code into a single optimized NEFF:

```
torch.compile(model, backend="neuron")
              │
              ▼
┌──────────────────────────────────┐
│ 1. GRAPH CAPTURE (Dynamo)        │  Traces Python → FX graph of aten ops
└──────────────┬───────────────────┘
               │ FX Graph (DAG of aten ops with shapes)
               ▼
┌──────────────────────────────────┐
│ 2. OPTIMIZATION (neuronx-cc)     │  Fuses ops, plans memory, tiles for hardware
└──────────────┬───────────────────┘
               │ HLO (optimized intermediate representation)
               ▼
┌──────────────────────────────────┐
│ 3. CODE GENERATION               │  Generates NEFF binary for NeuronCore
└──────────────────────────────────┘
               │
               ▼
         Cached in /tmp/neff_cache
         Reused on subsequent calls
```

**Stage 1: Graph capture (Dynamo)** — Dynamo is a Python bytecode analyzer. It watches what your code *does* and records a graph of `aten` operations. Output: an FX graph — a DAG of operations with concrete shapes.

**Stage 2: Optimization (neuronx-cc)** — The Neuron compiler takes the graph and:
- Fuses adjacent ops into single kernels (LayerNorm + Linear + GELU → 1 kernel)
- Plans memory layout so intermediates stay in on-chip SBUF
- Tiles operations for the tensor engine's fixed-size compute units

**Stage 3: Code generation** — Produces a NEFF binary: a sequence of hardware instructions for DMA engines, tensor engine, and vector engine. This is what gets loaded onto the NeuronCore.

The key insight: **Dynamo is hardware-agnostic.** The same graph capture works for any backend. The Neuron-specific magic happens in stages 2 and 3.

---
## What breaks compilation

Dynamo traces by *running* your code and recording what happens. This works great for static models. But certain patterns force it to split the graph — and on Neuron, each split means a separate NEFF.

### Graph breaks: one `print()` doubles your NEFFs

```python
import torch, glob, os

class CleanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(64, 128))
    def forward(self, x):
        x = torch.relu(x)
        x = torch.nn.functional.linear(x, self.w)
        return x

class BrokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(64, 128))
    def forward(self, x):
        x = torch.relu(x)
        print(f"shape: {x.shape}")  # ← graph break
        x = torch.nn.functional.linear(x, self.w)
        return x

device = "neuron"
x = torch.randn(8, 128, device=device)

# Clean model
os.system("rm -rf /tmp/neff_cache")
torch._dynamo.reset()
compiled_clean = torch.compile(CleanModel().to(device), backend="neuron")
with torch.no_grad():
    _ = compiled_clean(x)
torch.neuron.synchronize()
clean_neffs = len(glob.glob("/tmp/neff_cache/**/*.neff", recursive=True))
print(f"Clean model NEFFs:  {clean_neffs}")

# Broken model
os.system("rm -rf /tmp/neff_cache")
torch._dynamo.reset()
compiled_broken = torch.compile(BrokenModel().to(device), backend="neuron")
with torch.no_grad():
    _ = compiled_broken(x)
torch.neuron.synchronize()
broken_neffs = len(glob.glob("/tmp/neff_cache/**/*.neff", recursive=True))
print(f"Broken model NEFFs: {broken_neffs}")
```

```none
Clean model NEFFs:  1
shape: torch.Size([8, 128])
Broken model NEFFs: 2
```

One `print()` statement forced Dynamo to split `relu → linear` into two separate graphs — and therefore two separate NEFFs. The clean model fuses both ops into a single NEFF; the broken model compiles each in isolation.

We can confirm this with `torch._dynamo.explain`:

```python
print(torch._dynamo.explain(BrokenModel().to(device))(x))
```

```none
Graph Count: 2
Graph Break Count: 1
Break Reasons:
  Reason: Failed to trace builtin operator
  Explanation: Dynamo does not know how to trace builtin operator `print`
```

Common graph break causes:
- **`print()`, logging, any Python side effect** — Dynamo traces tensor operations, not arbitrary Python. When it encounters something that requires the Python interpreter (I/O, side effects), it must stop tracing, hand control back to Python, then start a new graph after.
- **Data-dependent control flow** (`if x.sum() > 0: ...`) — the compiler needs to know the graph structure at trace time. If a branch depends on a tensor *value* (not shape), Dynamo can't know which path to take without actually running the computation — so it splits.
- **Unsupported ops** (no Neuron implementation) — the op falls back to CPU, which forces a device transfer boundary. The graph before the op becomes one NEFF, the graph after becomes another.

In short: anything that forces execution back to the Python interpreter or CPU creates a graph break. The compiler can only optimize what it can see as a continuous graph.

```{admonition} Graph breaks on Neuron are expensive
:class: warning
Each graph break means a separate NEFF. Data must be written to HBM at the boundary, the new NEFF loaded, and data read back. In our ESM-2 example, the compiler collapsed 1,050 NEFFs into 1. A single graph break in a transformer layer would split that back into 2 — losing half the fusion benefit. **Aim for zero breaks.**
```

### Dynamic shapes: every new length recompiles

Proteins have variable lengths (4 to 2000+ residues). On Neuron, NEFFs are compiled for *specific* shapes — the tensor engine needs fixed tile dimensions. What happens when you pass a new sequence length?

```python
import torch, time
from transformers import EsmModel, EsmTokenizer

device = "neuron"
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D", attn_implementation="eager").to(device)

torch._dynamo.reset()
compiled_model = torch.compile(model, backend="neuron")

inputs = {k: v.to(device) for k, v in tokenizer("ACGT", return_tensors="pt").items()}
start = time.time()
with torch.no_grad():
    _ = compiled_model(**inputs)
torch.neuron.synchronize()
print(f"seq_len=4 → {time.time()-start:.1f}s (first compilation)")
```

```none
seq_len=4 → 17.7s (first compilation)
```

17.7 seconds — for a tiny 8M-parameter model with a 4-token input. That's the cost of NEFF compilation. Every new sequence length would trigger this again.

When Dynamo captures a graph, it records **guards**: assumptions about tensor shapes, dtypes, devices, and even model structure. On subsequent calls, PyTorch checks all guards before reusing the compiled graph. If any guard fails (different sequence length, different batch size, model parameter changed), the graph is invalid and must be recompiled from scratch. This is why a single `torch.compile` call doesn't generalize across shapes — the NEFF was specialized to one set of assumptions.

The solution on Neuron: **bucketing**. Pre-compile for a fixed set of sizes and pad inputs to the nearest bucket:

```python
buckets = [8, 16, 32, 64, 128, 256, 512]

def pad_to_bucket(inputs, buckets):
    seq_len = inputs["input_ids"].shape[1]
    target = next(b for b in buckets if b >= seq_len)
    pad_len = target - seq_len
    if pad_len > 0:
        inputs["input_ids"] = torch.nn.functional.pad(inputs["input_ids"], (0, pad_len))
        inputs["attention_mask"] = torch.nn.functional.pad(inputs["attention_mask"], (0, pad_len))
    return inputs
```

Pre-compile each bucket during startup (pay the cost once), then at inference time every input maps to a cached NEFF — no recompilation, no matter the sequence length.

```{admonition} Why not dynamic=True?
:class: note
PyTorch's `dynamic=True` uses symbolic integers (SymInts) to generalize across shapes. The Neuron backend does not support SymInts — NEFFs require concrete dimensions. Bucketing is the Neuron-native solution: finite shapes, pre-compiled, zero runtime compilation cost.
```

```{admonition} Why does Neuron need fixed shapes?
:class: tip
The NeuronCore tensor engine operates on **fixed-size tiles** (e.g., 128×128 for matmuls). When the compiler generates a NEFF, it must decide:
- How many tiles to split the input into
- How to schedule DMA transfers for each tile
- How to pipeline compute and data movement across tiles
- How much SBUF (on-chip memory) to allocate per tile

All of these decisions depend on the *exact* input dimensions. A 32-token sequence tiles differently than a 512-token sequence — different number of tiles, different DMA patterns, different pipeline schedules. There's no single instruction sequence that works for both. One shape = one NEFF.
```

### Silent CPU fallbacks: the output lies

Not all ops run on the NeuronCore — some silently fall back to CPU. The tricky part: the output tensor still reports `device=neuron:0`, because the runtime moves the result back automatically.

```python
import torch

device = "neuron"
x = torch.randn(8, 128, device=device)
indices = torch.zeros(8, 1, dtype=torch.long, device=device)
src = torch.ones(8, 1, device=device)

result = torch.scatter(x, 1, indices, src)
print(f"Result device: {result.device}")  
```

```none
Result device: neuron:0
```

Looks normal, but **CAREFUL**, if you look at the profile of the operation:

```python
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:
    _ = torch.scatter(x, 1, indices, src)
    torch.neuron.synchronize()
print(p.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

```none
                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
          aten::scatter        92.41%     368.869us       100.00%     399.182us     399.182us             1  
            aten::empty         4.90%      19.548us         4.90%      19.548us       6.516us             3  
               aten::to         0.50%       1.991us         2.70%      10.765us      10.765us             1  
         aten::_to_copy         1.00%       4.005us         2.20%       8.774us       8.774us             1  
            aten::copy_         0.86%       3.434us         0.86%       3.434us       3.434us             1  
```

The `aten::to → _to_copy → copy_` chain shows the issue: the data is being moved off-device and back.

```{admonition} Detecting fallbacks
:class: warning
The output device is not proof of native execution. To detect fallbacks:
1. **Profile**: look for `aten::to` / `aten::copy_` surrounding the op
2. **Time it**: native ops on small tensors run in ~50µs; fallbacks take 500µs+ (PCIe transfer dominates)
3. **Neuron Explorer**: `dmem_buf_copyout` events in the system trace indicate data leaving the device mid-execution
```

---

## Conclusion: What we've learned — and where we're going

In three chapters, we've gone from "call `model(x)` and hope for the best" to understanding why one line of code (`torch.compile`) gives us an 8x speedup: We now understand the impact of graph breaks, of fused operations, and are begining to form an intuition on why profiling is key. 

These are software-level decisions that directly translate into hardware-level outcomes. You don't need to change the math. You need to understand how the hardware wants to receive it.

But we've also hit the compiler's limits. It can't fix algorithmic complexity. It can't optimize across compilation boundaries. It can't beat a hand-written kernel that exploits hardware-specific knowledge. When you need that level of control — *how* tiles move through the memory hierarchy, *when* engines overlap, *which* data stays on-chip — you need to understand the hardware itself.

That's the art of performance engineering: designing your software layer with the hardware's strengths and constraints in mind. The first time you do it without a hardware background, there's a learning curve. But it's deeply satisfying work — and in today's era of AI-assisted coding, the barrier to entry is lower than ever.

The rest of this book helps you on this journey.

```{figure} ../assets/meme_Firsttime.png
:alt: First time?
:width: 400px
:align: center

Performance engineering on custom silicon, it gets easier.
```

---

*The compiler turns your Python into optimized hardware instructions. But what hardware is it targeting? What are the engines, the memory, the silicon that actually executes those instructions?*


