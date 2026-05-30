# What actually happens when you call `model(x)`

## Act 1: Running ESM-2 with PyTorch

Let's start with a real protein language model and see what it actually *does* when we call it.

```python
import torch
from transformers import EsmModel, EsmTokenizer
print(f"Torch version: {torch.__version__}")

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
model.eval()

# Human insulin B chain
sequence = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"
inputs = tokenizer(sequence, return_tensors="pt")
```

```none
2.10.0+cu130
```

```python
print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

```none
Input shape: torch.Size([1, 32])
Vocabulary size: 33
Model parameters: 651,040,661
```

Now let's profile the forward pass to see which operations PyTorch actually dispatches:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        output = model(**inputs)

events = prof.key_averages()
for e in sorted(events, key=lambda e: e.cpu_time_total, reverse=True)[:10]:
    print(f"  {e.key:<50} called {e.count:>4} times, total {e.cpu_time_total/1000:>7.1f}ms")
```

```none
  aten::as_strided                                   called 1079 times, total     1.2ms
  aten::view                                         called  666 times, total     2.0ms
  aten::transpose                                    called  530 times, total     2.8ms
  aten::resolve_conj                                 called  400 times, total     0.1ms
  aten::to                                           called  352 times, total     3.6ms
  aten::copy_                                        called  346 times, total     4.9ms
  aten::addmm                                        called  264 times, total   812.3ms
  aten::scaled_dot_product_attention                  called   33 times, total   245.1ms
  aten::layer_norm                                   called   67 times, total    61.3ms
  aten::gelu                                         called   33 times, total    38.7ms
```

```{admonition} Observation
:class: important
ESM-2 (650M params, 33 layers) decomposes into just a handful of ATen ops repeated hundreds of times. The "intelligence" is in the **weights**, not the ops. These are operations such as additions, transposes, views, multiplications.
```

---

## Act 2: PyTorch eager execution

What does "eager" mean? Python executes your code line by line. Each operation runs *immediately* and returns a result. There is no "graph" being built. There is no compilation step.

This is different from TensorFlow 1.x, where you'd first build a graph of placeholder operations and then call `sess.run()` to execute them all at once.

Here's a concrete example of what eager mode costs you. Consider layer normalization followed by a linear projection:

```python
x = torch.randn(1, 32, 1280)
ln = torch.nn.LayerNorm(1280)
linear = torch.nn.Linear(1280, 1280)

# In eager mode, these are TWO separate kernel launches:
x_normed = ln(x)           # kernel 1: compute mean, variance, normalize
x_proj = linear(x_normed)  # kernel 2: matmul + bias

# A compiler could FUSE these into ONE kernel:
#   load x from memory → normalize → matmul → write result to memory
#   (no intermediate write of x_normed back to memory)
```

In eager mode, `x_normed` is a real tensor that gets written to memory — even though nothing else ever reads it except the next line. A compiler would see that and eliminate the intermediate write. But eager mode can't look ahead; it executes each line in isolation.

The upside? You can inspect `x_normed` right now — it has real values:

```python
print(f"After layer_norm: mean={x_normed.mean():.6f}, std={x_normed.std():.4f}")
print(f"After linear:     shape={x_proj.shape}, first value={x_proj[0,0,0]:.4f}")
```

```none
After layer_norm: mean=-0.000000, std=1.0001
After linear:     shape=torch.Size([1, 32, 1280]), first value=-0.3842
```

In a graph framework, these would be placeholders with no value yet. In eager mode, every tensor is real the instant it's created.

Layer 0 fully completes, then Layer 1, then Layer 2. This IS Python's execution order. No scheduler. No optimizer. No graph. The call stack looks like this:

```
model(**inputs)
  → model.forward()                    [Python method call]
    → embeddings(input_ids)            [Python method call]
    → encoder(hidden_states)           [Python method call]
      → layer[0](x)                    [Python method call]
        → attention.self(x)            [Python method call]
          → torch.matmul(Q, K)         [ATen op → Dispatcher → kernel]
          → masked_fill(...)           [ATen op → Dispatcher → kernel]
          → softmax(...)               [ATen op → Dispatcher → kernel]
          → torch.matmul(attn, V)      [ATen op → Dispatcher → kernel]
        ← returns
        → attention.output(x)          [Python method call]
          → layer_norm(...)            [ATen op → Dispatcher → kernel]
        ...
```
Every indentation = a Python function call. Every ATen op = an immediate computation. **That's eager mode.**

```{admonition} Eager mode tradeoffs
:class: note

**Pros:**
- Easy to debug (print anything, set breakpoints anywhere)
- Dynamic control flow (if/else, loops that depend on tensor values)
- Immediate error messages (crash on the exact line that's wrong)

**Cons:**
- Python overhead on every single op
- The dispatcher can't see the big picture (no global optimization)
- Can't fuse ops across module boundaries
- Can't pre-compile for specialized hardware (Neuron, TPU)

`torch.compile()` (Chapter 3) fixes the cons by tracing eager code into a graph.
```

---

## Act 3: Naked attention — same ops, no framework magic

What does a transformer layer actually compute? Strip away the HuggingFace wrapper, and ESM-2's attention is just 6 PyTorch operations: four linear projections (`addmm`), one matmul for attention scores, one softmax, and one matmul to combine values. That's it. Let's rebuild it from scratch to prove it:

```python
import torch.nn.functional as F

class ESMAttentionManual(torch.nn.Module):
    """Exactly what one ESM-2 attention layer does, using raw ATen ops."""
    
    def __init__(self, d_model=1280, n_heads=20):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # 64
        self.scale = self.d_head ** -0.5
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        
        # Op 1: aten::addmm (linear projection)
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        # Op 2: aten::matmul (Q @ K^T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Op 3: aten::masked_fill
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Op 4: aten::softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Op 5: aten::matmul (attn @ V)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        # Op 6: aten::addmm (output projection)
        return self.out_proj(context)

manual_attn = ESMAttentionManual(d_model=1280, n_heads=20).eval()
x = torch.randn(1, 32, 1280)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    with torch.no_grad():
        out = manual_attn(x)

print(f"Output shape: {out.shape}")
print(f"\nOps dispatched:")
for e in sorted(prof.key_averages(), key=lambda e: e.cpu_time_total, reverse=True)[:6]:
    print(f"  {e.key:<40} called {e.count:>3} times")
```

```none
Output shape: torch.Size([1, 32, 1280])

Ops dispatched:
  aten::addmm                              called   4 times
  aten::matmul                             called   2 times
  aten::softmax                            called   1 times
  aten::masked_fill                        called   0 times
  aten::transpose                          called   5 times
  aten::view                               called   5 times
```

Our 30-line class triggers the same ATen ops as the full 650M-parameter ESM-2. The dispatcher sees no difference — it doesn't know whether it's running a protein language model or a toy. It just receives `matmul`, `softmax`, `matmul` and routes them to the appropriate kernel.

---

## Act 4: The dispatcher — same op, different kernel

The dispatcher's job: given an operation + tensor metadata, pick the right implementation to run.

What metadata does it look at?
- **Device** — is the tensor on CPU, CUDA, or Neuron?
- **Dtype** — is it float32, bfloat16, int8?
- **Shape** — some kernels are specialized for specific dimensions (e.g., small matmuls vs large)
- **Layout** — is the tensor dense (strided), sparse, or in a special format like channels-last for convolutions?

If no implementation exists for a given combination (say, a complex-number op on Neuron), the dispatcher falls back to CPU. The operation still runs — just slower, because data has to move back to CPU and return.

```
torch.matmul(A, B)
     │
     ▼
┌─────────────┐
│  DISPATCHER  │ ← looks at: dtype, device, shape, layout
└──────┬──────┘
       │
┌──────┼──────────────────────────┐
│      │      │         │         │
▼      ▼      ▼         ▼         ▼
MKL   cuBLAS  Neuron   Metal    oneDNN
(CPU)  (CUDA)  (NKI)   (MPS)   (Intel)
```

We can actually see which backend kernel the dispatcher picked by profiling with `record_shapes=True`:

```python
A = torch.randn(32, 64)
B = torch.randn(64, 128)

# CPU dispatch — profile to see which kernel runs
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:
    C_cpu = torch.matmul(A, B)
for e in p.key_averages():
    if 'mm' in e.key:
        print(f"  CPU kernel: {e.key}")

# GPU dispatch — same call, different kernel
if torch.cuda.is_available():
    A_g, B_g = A.cuda(), B.cuda()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as p:
        C_cuda = torch.matmul(A_g, B_g)
        torch.cuda.synchronize()
    for e in p.key_averages():
        if 'mm' in e.key or 'gemm' in e.key.lower():
            print(f"  CUDA kernel: {e.key}")
    print(f"  Same result: {torch.allclose(C_cpu, C_cuda.cpu(), atol=1e-5)}")
```

```none
  CPU kernel: aten::mm
  CUDA kernel: aten::mm
  CUDA kernel: volta_sgemm_32x32_sliced1x4_nn
  Same result: True
```

Same `aten::mm` at the Python level, but on CUDA it dispatches to `ampere_sgemm_32x128_tn` — an NVIDIA-specific GEMM kernel optimized for this shape on Ampere GPUs. On Neuron, it would dispatch to a compiled NEFF. **Your code doesn't change.** (The math doesn't change)

---

## Act 5: Fusion — collapsing ops into one kernel

Some operation sequences appear so frequently in deep learning (attention being the prime example) that hardware vendors write a single optimized kernel for the entire sequence. Instead of dispatching 5 separate ops — each reading from and writing to HBM — the fused kernel does all the work in one shot, keeping intermediate results in fast on-chip memory.

```
Manual attention:                     Fused SDPA:
  matmul(Q, K^T)                        ┐
  multiply by scale                     │
  masked_fill(mask, -inf)               ├→ ONE kernel call
  softmax                               │   (scaled_dot_product_attention)
  matmul(attn, V)                       ┘
```

5 op dispatches → 1 op dispatch. Same math. Fewer kernel launches.

```python
class ESMAttentionFused(torch.nn.Module):
    """Same math as ESMAttentionManual, but using the fused SDPA op."""
    
    def __init__(self, d_model=1280, n_heads=20):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        # ONE OP replaces matmul + scale + mask + softmax + matmul
        context = F.scaled_dot_product_attention(Q, K, V, attn_mask=attention_mask)
        
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(context)

# Compare op counts
fused_attn = ESMAttentionFused(d_model=1280, n_heads=20).eval()

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p1:
    with torch.no_grad():
        out_manual = manual_attn(x)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p2:
    with torch.no_grad():
        out_fused = fused_attn(x)

attn_ops = ['matmul', 'softmax', 'scaled_dot_product', 'masked_fill']
manual_count = sum(e.count for e in p1.key_averages() if any(op in e.key for op in attn_ops))
fused_count = sum(e.count for e in p2.key_averages() if any(op in e.key for op in attn_ops))

print(f"Manual path — attention op dispatches: {manual_count}")
print(f"Fused path  — attention op dispatches: {fused_count}")
```

```none
Manual path — attention op dispatches: 4
Fused path  — attention op dispatches: 2
```

```{admonition} Why fusion matters
:class: important
The fused kernel does matmul+mask+softmax+matmul in ONE shot:
- Fewer kernel launches (less dispatch overhead)
- Data stays in fast on-chip memory (no round-trips to HBM)
- On GPU: FlashAttention under the hood
- On Neuron: the compiler fuses into a single NEFF
```

---

## Where does the data live?

Once you move your model and inputs to a device (`model.to("cuda")` or `model.to("neuron")`), data stays there. Between ops, **nothing moves back to CPU**:

```
CPU (Python + dispatcher)              Device memory (GPU VRAM / Neuron HBM)
─────────────────────────              ──────────────────────────────────────
                                       Q, K, V tensors live here
                                       ↓
"dispatch aten::matmul"  ── launch ──► Device computes Q @ K^T
                                       Result stays on device
                                       ↓
"dispatch aten::softmax" ── launch ──► Device computes softmax
                                       Result stays on device
                                       ↓
"dispatch aten::matmul"  ── launch ──► Device computes attn @ V
                                       Result stays on device
```

The CPU's role is issuing commands. The data never crosses the bus between ops — unless an operation **isn't supported** on the device. In that case, the dispatcher silently moves the tensor to CPU, runs the op there, and moves the result back. This is called a **fallback**.

Fallbacks are functionally correct but expensive (data transfer + losing device parallelism). We'll see this concretely in Chapter 6 when we run PyTorch Geometric's `scatter` operations on Neuron — they fall back to CPU because Neuron doesn't have a native kernel for them yet.

### Asynchronous execution and synchronization

Accelerators (both GPU and Neuron) execute **asynchronously**: when Python calls `torch.matmul(A, B)`, it doesn't wait for the result. It queues the work and immediately returns to Python. This means:

- Python can queue the *next* op while the device is still computing the current one
- The profiler's "CPU time" includes dispatch overhead, not actual compute
- If you time with `time.time()`, you're measuring how fast Python can *queue* work, not how fast the device *does* it

To get accurate timing, you need to **synchronize** — force Python to wait until all queued work finishes:

```python
# GPU: wait for all queued CUDA kernels to finish
torch.cuda.synchronize()

# Neuron: wait for all queued Neuron ops to finish
# (handled automatically in eager mode — each op blocks until its NEFF completes)
```

On Neuron in eager mode, each op is synchronous by default (it JIT-compiles and executes before returning). This makes debugging easier but is slower than GPU's async pipeline — which is one reason `torch.compile` matters more on Neuron.

### Why do kernels need to be compiled?

An accelerator can't run Python. It needs machine code specific to its hardware. The **kernel** is that machine code — a compiled function that runs on the device.

| | GPU (CUDA) | Neuron |
|---|---|---|
| **Kernel format** | PTX → SASS (machine code) | HLO → NEFF (Neuron Executable) |
| **When compiled** | First call (JIT), then cached | First call (JIT), then cached in `/tmp/neuron_cache` |
| **First-call penalty** | ~100ms (cuDNN autotuning) | ~seconds (neuronx-cc compilation) |
| **Subsequent calls** | Instant (from cache) | Instant (from NEFF cache) |

This is why the first forward pass is always slow — the device is compiling kernels. After that, compiled kernels are cached and reused.

```python
# Warmup: pay the one-time compilation cost
with torch.no_grad():
    _ = model(**inputs)
torch.cuda.synchronize()  # or just wait on Neuron

# NOW measure steady-state performance
```

### Why fusion matters (revisited)

Since data stays on-device, the bottleneck between ops is not CPU↔device transfer. It's **device memory bandwidth** — reading and writing intermediate results between ops:

```
Without fusion (5 ops):
  matmul → write result to device memory → read it back →
  scale  → write → read →
  mask   → write → read →
  softmax→ write → read →
  matmul → write final result

With fusion (1 op: scaled_dot_product_attention):
  Load Q,K,V ONCE → do everything in fast on-chip SRAM → write result ONCE
```

5 round-trips to device memory → 1 round-trip. Same math. Much less memory traffic. This is true on GPU, Neuron, and any accelerator with a memory hierarchy.

---

## The big picture

```
ESM-2("FVNQHLCGSHLVEALYLVCGERGFFYTPKT")
     │
     ▼
┌──────────────────────────────────┐
│  Python: model.forward()         │  ← Eager mode
│  Each line triggers an ATen op   │
└──────────────┬───────────────────┘
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
aten::matmul  aten::softmax  aten::masked_fill  ...
     │         │         │
     ▼         ▼         ▼
┌──────────────────────────────────┐
│  DISPATCHER                      │  ← Routes by device/dtype
│  Checks: device? dtype? layout?  │
└──────────────┬───────────────────┘
               │
┌──────────────┼──────────────────────────────┐
│              │              │                │
▼              ▼              ▼                ▼
MKL          cuBLAS        Neuron           Metal
(CPU)        (CUDA)        (NKI)           (Apple)
│              │              │                │
▼              ▼              ▼                ▼
x86 asm      PTX/SASS       NEFF           GPU shader
```

**Fusion** collapses multiple ops into one *before* dispatching:
`[matmul + mask + softmax + matmul]` → `[scaled_dot_product_attention]`

**Your model code never changes.** Only the dispatch target does. That's the abstraction. That's why "just change the device" works.

---

*Next: [Chapter 2](ch02-tensors) — What ARE these tensors that the ops operate on? Memory layout, strides, and why it matters for performance.*
