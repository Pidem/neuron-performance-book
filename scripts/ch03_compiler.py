"""
Chapter 3: The compiler enters
===============================
What torch.compile does: graph capture via Dynamo, optimization, code generation.
Why graph mode matters for hardware acceleration.
"""

import torch
from transformers import EsmModel, EsmTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).eval()

sequence = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"
inputs = tokenizer(sequence, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# --- 1. Eager mode baseline ---
print("=== Eager Mode (no compilation) ===")
import time

# Warmup
with torch.no_grad():
    for _ in range(3):
        _ = model(**inputs)
if device.type == "cuda":
    torch.cuda.synchronize()

# Time it
start = time.perf_counter()
with torch.no_grad():
    for _ in range(10):
        _ = model(**inputs)
if device.type == "cuda":
    torch.cuda.synchronize()
eager_time = (time.perf_counter() - start) / 10
print(f"Eager forward pass: {eager_time*1000:.2f} ms")

# --- 2. torch.compile: what happens ---
print("\n=== torch.compile ===")
print("Compiling ESM-2 with torch.compile...")

# torch.compile captures the computation graph via Dynamo
compiled_model = torch.compile(model)

# First call triggers compilation (slow)
start = time.perf_counter()
with torch.no_grad():
    _ = compiled_model(**inputs)
if device.type == "cuda":
    torch.cuda.synchronize()
compile_time = time.perf_counter() - start
print(f"First call (includes compilation): {compile_time*1000:.1f} ms")

# Subsequent calls use the compiled graph (fast)
with torch.no_grad():
    for _ in range(3):
        _ = compiled_model(**inputs)
if device.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()
with torch.no_grad():
    for _ in range(10):
        _ = compiled_model(**inputs)
if device.type == "cuda":
    torch.cuda.synchronize()
compiled_time = (time.perf_counter() - start) / 10
print(f"Compiled forward pass: {compiled_time*1000:.2f} ms")
print(f"Speedup: {eager_time/compiled_time:.2f}x")

# --- 3. What Dynamo captures: the graph ---
print("\n=== What Dynamo Sees ===")

# We can inspect what Dynamo captures using torch._dynamo.explain
from torch._dynamo import explain

explanation = explain(model, **inputs)
print(f"Number of graph breaks: {explanation.graph_break_count}")
print(f"Number of graphs captured: {len(explanation.graphs)}")
if explanation.break_reasons:
    print(f"Break reasons: {explanation.break_reasons[:3]}")

# --- 4. The compiled graph: fused operations ---
print("\n=== Operator Fusion ===")
print("""
What the compiler does with the graph:
1. Captures the full computation as a graph (Dynamo)
2. Identifies fusion opportunities:
   - LayerNorm = mean + subtract + variance + divide + scale + bias → ONE kernel
   - Linear + GELU = matmul + activation → ONE kernel
3. Generates optimized code for the target backend (Inductor for GPU, Neuron compiler for Neuron)

In eager mode: each op launches a separate kernel (overhead per op)
In compiled mode: fused ops = fewer kernel launches = less overhead
""")

# --- 5. The tracing problem: what breaks compilation ---
print("=== What Breaks Compilation ===")

# Dynamic shapes: ESM-2 processes variable-length proteins
seq_short = tokenizer("ACGT", return_tensors="pt")
seq_long = tokenizer("ACGT" * 100, return_tensors="pt")
seq_short = {k: v.to(device) for k, v in seq_short.items()}
seq_long = {k: v.to(device) for k, v in seq_long.items()}

# With dynamic=False (default), different shapes trigger recompilation
print("Different sequence lengths trigger recompilation:")
start = time.perf_counter()
with torch.no_grad():
    _ = compiled_model(**seq_short)
if device.type == "cuda":
    torch.cuda.synchronize()
print(f"  Short seq (4 tokens):  {(time.perf_counter()-start)*1000:.1f} ms (may recompile)")

start = time.perf_counter()
with torch.no_grad():
    _ = compiled_model(**seq_long)
if device.type == "cuda":
    torch.cuda.synchronize()
print(f"  Long seq (400 tokens): {(time.perf_counter()-start)*1000:.1f} ms (may recompile)")

# --- 6. Backend selection ---
print("\n=== Backend Selection ===")
print(f"Current backend: inductor (default for {device})")
print("""
Available backends:
  - 'inductor'  → generates Triton kernels for GPU, C++ for CPU
  - 'neuron'    → generates NEFFs for Neuron hardware
  - 'eager'     → no optimization (useful for debugging)

On Neuron, you'd write:
  compiled_model = torch.compile(model, backend="neuron")

The compiler then:
  1. Captures the graph (same Dynamo step)
  2. Passes it to the Neuron compiler (neuronx-cc)
  3. Generates NEFFs (Neuron Executable File Format)
  4. Caches them for subsequent runs
""")

print(f"""
=== Key Takeaway ===
torch.compile transforms your model from "execute ops one by one" to 
"execute an optimized graph." This matters because:

1. Fewer kernel launches (fused ops)
2. Better memory access patterns (compiler can reorder)
3. Backend-specific optimizations (Triton for GPU, Neuron compiler for Neuron)

Eager mode: {eager_time*1000:.2f} ms
Compiled:   {compiled_time*1000:.2f} ms  ({eager_time/compiled_time:.2f}x faster)

But the compiler needs a GRAPH to optimize. That's why it matters
what hardware you're targeting — the graph is the interface between
your PyTorch code and the hardware underneath.

Next chapter: What IS that hardware? GPU vs Neuron — different engines,
different constraints, different mental models.
""")
