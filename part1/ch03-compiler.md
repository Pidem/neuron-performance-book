# The compiler enters

*`torch.compile(esm_model, backend="neuron")` — what just happened?*

```{admonition} TODO
:class: warning

## The problem (recap from Ch 1 + Ch 2)

- In eager mode, each op is independent: dispatch → kernel → write to device memory → next op
- No op knows what comes next. Layout choices are local. Fusion is manual (SDPA).
- Ch 2 showed: if kernel A writes in layout X and kernel B needs layout Y, someone pays for rearrangement
- **The compiler's promise**: see the whole computation, optimize globally

## Act 1: What `torch.compile` actually does — the three stages

1. **Graph capture (Dynamo)** — traces your Python code into a graph of `aten` ops
   - Dynamo is a Python bytecode analyzer — it watches what your code *does*, not what it *says*
   - Output: FX graph (a DAG of aten operations with concrete shapes)
   - Show: `torch._dynamo.explain(model, **inputs)` → number of graphs, break reasons
   - Show: `torch.compile(model, backend="aot_eager")` to see the captured graph without optimization

2. **Optimization (backend-specific)** — the backend transforms the graph
   - On GPU: Inductor decomposes ops, fuses pointwise chains, generates Triton kernels
   - On Neuron: neuronx-cc lowers to HLO, tiles for the tensor engine, generates NEFFs
   - Key optimizations both do: operator fusion, dead code elimination, layout propagation
   - Key difference: Inductor generates code at Python level (Triton); Neuron compiler generates hardware instructions

3. **Code generation** — produces executable code for the target
   - GPU: Triton kernels (compiled to PTX → SASS at runtime)
   - Neuron: NEFFs (cached in `/tmp/neuron_cache`, loaded by Neuron Runtime)
   - Both: cached after first compilation, reused on subsequent calls

## Act 2: Why graph mode unlocks performance

Show the same ESM-2 forward pass, eager vs compiled:

```python
model_eager = model
model_compiled = torch.compile(model)

# Benchmark both (with proper warmup)
```

What the compiler does that eager can't:
- **Fusion**: LayerNorm = 5 ops → 1 kernel. Linear+GELU = 2 ops → 1 kernel.
  - Fewer kernel launches (less dispatch overhead)
  - Fewer HBM round-trips (intermediates stay in SRAM)
- **Layout propagation**: compiler knows kernel B follows kernel A, can choose a layout that works for both
- **Memory planning**: compiler knows tensor lifetimes, can reuse buffers (less peak memory)
- **Constant folding**: if something is known at compile time, compute it once

ASCII diagram:

```
Eager (33 layers × ~10 ops each = ~330 kernel launches):
  op → HBM → op → HBM → op → HBM → ...

Compiled (33 layers × ~3 fused kernels = ~99 launches):
  [fused: norm+linear+gelu] → HBM → [fused: attention] → HBM → [fused: ffn] → ...
```

## Act 3: The tracing problem — what breaks compilation

Dynamo traces by *running* your code and recording what happens. This works great for static models. It breaks when:

1. **Data-dependent control flow**
   ```python
   if x.sum() > 0:  # can't trace — depends on runtime values
       ...
   ```
   → Graph break: Dynamo splits into two subgraphs with Python in between

2. **Dynamic shapes** — ESM-2's key challenge
   - Proteins have variable lengths (4 to 2000+ residues)
   - Default: each new shape triggers recompilation (expensive!)
   - Fix: `torch.compile(model, dynamic=True)` — compiler generates shape-generic code
   - Tradeoff: dynamic shapes disable some optimizations (can't hardcode tile sizes)
   - On Neuron: `dynamic=True` uses bucketing — compile for a few representative sizes, pad inputs to nearest bucket

3. **Unsupported ops**
   - If an op has no backend implementation → graph break → fallback to eager
   - On Neuron: some ops (scatter, complex indexing) cause graph breaks → CPU fallback
   - Show: `torch._dynamo.explain()` to diagnose breaks

4. **Python side effects**
   - `print()`, logging, global state mutation → graph break
   - Rule: compiled code should be pure computation

## Act 4: ESM-2 and dynamic shapes — a concrete example

```python
# Compile with dynamic shapes for variable-length proteins
compiled_model = torch.compile(model, dynamic=True)

# First call: compiles a shape-generic graph
short = tokenizer("ACGT", return_tensors="pt")
_ = compiled_model(**short)  # slow (compilation)

# Second call with different length: NO recompilation
long = tokenizer("FVNQHLCGSHLVEALYLVCGERGFFYTPKT", return_tensors="pt")
_ = compiled_model(**long)  # fast (reuses compiled graph)
```

On Neuron, the story is different:
- NEFFs are compiled for *specific* shapes (the tensor engine needs fixed tile dimensions)
- Solution: bucket sizes (e.g., compile for seq_len = 32, 64, 128, 256, 512)
- Input is padded to the nearest bucket, attention mask hides the padding
- Tradeoff: wasted compute on padding vs. compilation cost for every unique length

Show the bucketing pattern:
```python
# Neuron-style bucketing
compiled_model = torch.compile(model, backend="neuron")
for bucket_size in [32, 64, 128, 256, 512]:
    dummy = torch.zeros(1, bucket_size, dtype=torch.long)
    _ = compiled_model(input_ids=dummy)  # pre-compile each bucket
```

## Act 5: Backend selection — what "backend" means

```
torch.compile(model, backend=???)
                          │
                          ▼
              ┌─────────────────────┐
              │  Dynamo (graph      │  ← same for all backends
              │  capture)           │
              └──────────┬──────────┘
                         │ FX Graph
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │Inductor│ │ Neuron │ │  eager  │
         │(GPU)   │ │compiler│ │(debug)  │
         └───┬────┘ └───┬────┘ └───┬────┘
             ▼          ▼          ▼
          Triton      NEFFs     No-op
          kernels   (cached)   (just runs)
```

- `backend="inductor"` (default on GPU): generates Triton → PTX → SASS
- `backend="neuron"`: passes graph to neuronx-cc → HLO → NEFF
- `backend="aot_eager"`: captures graph but runs eagerly (for debugging)
- `backend="eager"`: no compilation at all (baseline)

The key insight: **Dynamo is hardware-agnostic**. The same graph capture works regardless of target. The backend is where hardware-specific decisions happen.

## Act 6: What the compiler CANNOT do

- It can't fix algorithmic complexity (O(n²) attention is still O(n²))
- It can't move data between devices (if your tensor is on CPU, it stays there)
- It can't optimize across `torch.compile` boundaries (each compiled region is independent)
- It can't handle truly dynamic computation (tree-structured networks, variable-depth recursion)
- On Neuron: it can't fuse ops that span multiple NeuronCores (that's collective communication — Ch 14+)

## The big picture

```
Chapter 1: model(x) → dispatcher → kernel (one op at a time)
Chapter 2: tensor = pointer + strides (layout affects kernel efficiency)
Chapter 3: compiler sees the WHOLE graph → fuses ops, optimizes layouts, generates hardware code
                         │
                         ▼
         "generates hardware code" — but for WHAT hardware?
         What does the Neuron chip actually look like?
         What are its constraints? Its strengths?
```

*Question raised → "The compiler generates code for 'the hardware.' But what IS that hardware?"*

→ Chapter 4: The hardware landscape (GPU vs Neuron architecture)
```

*Question raised → "A graph of what operations, running on what hardware?"*
