# Hardware deep dive

In Chapter 3, we saw the compiler collapse 1,050 NEFFs into one. But what hardware does that NEFF actually run on? In this chapter, we dive into the NeuronCore architecture.

```{admonition} Production silicon, not research prototype
:class: note
AWS is one of TSMC's top 5 customers for advanced process nodes. Unlike typical chip development (A0, B0, C0 iterations), 14 out of the last 16 AWS chips went into multi-million unit production from the very first silicon revision. Trainium is battle-tested at scale — over a million chips deployed running customer workloads today.
```

---

## The NeuronCore architecture

If you're coming from the CUDA world, here's what you're used to a world of many small Stream Multiprocessors (SMs) following the SIMT model: Thousands of indepedendent threads execute the same instruction. Each SM has only a small shared memory (~100-200 KB) and the main philosophy is that the massive parallelism compensates. The GPU bet is basically to throw enough threads at the problem and latency will disappear: If thousands of other threads keep the hardware busy while some stall, you are utilizing your compute well. 

Neuron makes the opposite architectural design choice: *dedicate die area to massive compute engines and fast on-chip memory, not thread management.*

```{figure} ../assets/trainium2_architecture.png
:alt: Trainium2 chip architecture
:width: 600px
:align: center

The Trainium2 chip: 8 NeuronCore-v3 engines, 96 GiB HBM, 224 MiB SBUF.
```

Each Trainium2 chip contains **8 NeuronCore-v3** cores. Each NeuronCore has specialized engines:

```{admonition} Why you see 4 cores on trn2.3xlarge
:class: note
The diagram shows the full physical chip (8 cores). A `trn2.3xlarge` is a single-chip instance, but exposes only **4 NeuronCores** — half of the chip's 8. DMA bandwidth is proportionally reduced as well. The full 8 cores per chip are accessible on `trn2.48xlarge` (16 chips × 8 cores = 128 NeuronCores).
```

| Engine | Role | Think of it as... |
|--------|------|-------------------|
| **Tensor engine** | Matrix multiplication (systolic array) | One input → one output, via massive multiply-accumulate |
| **Vector engine** | One output depends on *multiple* inputs (reductions, layer norm) | The reducer — 128 parallel lanes |
| **Scalar engine** | One output depends on *one* input (exp, reciprocal, activations) | The activator — 128 parallel polynomial evaluators |
| **GPSIMD engine** | Arbitrary C code for custom ops | The escape hatch — "invent operators we never thought of" |
| **DMA engines** | Shuttle data between SRAM ↔ HBM | The trucks — keep the engines fed |

```{figure} ../assets/neuroncore-v3.png
:alt: NeuronCore-v3 architecture
:width: 600px
:align: center

Inside a single NeuronCore-v3: specialized engines connected by on-chip SRAM (SBUF).
```
### Engine cost models

Each engine has a distinct cost profile shaped by how it parallelizes work:

**Tensor engine (128×128 systolic array).** The dominant engine — 90% of a model's TFLOPs live here. It has two phases: *load stationary* (filling the array with a weight tile) is pure data movement and runs **4× faster** than *multiply moving* (streaming activations through and accumulating). Design principle: map the larger tensor to stationary, the smaller to moving.

The tensor engine has a **dual weight cache** — while the current matmul runs using the "foreground" weights, DMA loads the next weight tile into the "background" slot. When the current tile finishes, foreground and background swap instantly — no stall. This is why the profiler shows "Tensor Engine" (load stationary) and "TensorMatrix Engine" (multiply moving) as separate rows: they overlap.

**Vector engine (128 parallel lanes).** Each lane processes one element simultaneously, so work that spans all 128 lanes costs the same as work that spans 2 — the parallelism is free. But within each lane, cost is linear: reducing 1000 elements takes twice as long as reducing 500.

**Scalar engine (128 parallel lanes).** Same parallel model as vector. Each lane evaluates one element through polynomial lookup hardware (for transcendentals like exp, rsqrt).

**GPSIMD (8 cores, not 128 lanes).** The weakest engine by throughput — only 8 parallel units instead of 128. Handles edge cases: indirect memory access, triangular masks, random number generation, and custom C++ operators.

```{figure} ../assets/GmSimd_engine.png
:alt: GpSimd engine
:width: 500px
:align: center

The GpSimd engine: 8 general-purpose SIMD cores for operations that don't fit the specialized engines.
```

**DMA engines (16 per core, ~277 GB/s each).** A single DMA transfer writes one contiguous block. Large contiguous transfers (≥32 KB) saturate bandwidth; many small transfers (4-byte strided reads) are catastrophic.

```{admonition} The rule of thumb
:class: tip
If an operation doesn't map naturally to one of these engines (if it requires random-access writes, irregular gather patterns, or complex control flow,...) it probably won't perform well on Neuron. Large embedding table lookups are a classic example.
```

### Trainium2 specs

| Component | Spec |
|-----------|------|
| Compute (BF16) | 667 TFLOPS |
| Compute (FP8) | 1,299 TFLOPS |
| HBM capacity | 96 GiB |
| HBM bandwidth | 2.9 TB/s |
| SBUF (on-chip SRAM) | 224 MiB | 
| DMA bandwidth | 3.5 TB/s |
| NeuronLink (chip-to-chip) | 1.28 TB/s |
| CC-Cores (collectives) | 16 |

### Peak vs. sustained: the sprinter and the marathoner

Those spec numbers are *peak* performance — what the chip can sustain for a single instruction under ideal conditions. Real workloads are marathons, not sprints.

```{admonition} The sprinter vs. marathoner analogy
:class: tip
A sprinter runs faster than a marathoner over 100m. But a marathoner wins the race that matters. AI training is a marathon — what matters is *sustained* throughput over hours, not peak FLOPs for one instruction.
```

Why sustained < peak:
- **Softmax and other non-matmul ops** block the tensor engine between matmuls
- **Tiling inefficiencies** — not all tile shapes perfectly fill the 128×128 systolic array
- **Memory bank shuffles** — data in the wrong SBUF partition must be reorganized
- **Thermal and power delivery** — real racks under full load for hours

What "good" looks like: a customer doing real-time video generation recently exceeded **80% MFU** (Model FLOPs Utilization) sustained on Trainium 3. Getting from 30% MFU (naive eager) to 60% (torch.compile) to 80%+ (NKI-optimized) is the journey this book teaches.

---

## From Python to hardware: a first-principles walkthrough

Let's trace a concrete function through the chip. Consider the following fictional layer consisting of three matmuls, one layer normalization, and one activation function:

```python
def big_simple(x, w1, w2, w3):
    x = torch.matmul(x, w1)              
    x = torch.layer_norm(x, [4096])      
    x = torch.matmul(x, w2)              
    x = torch.nn.functional.gelu(x)      
    x = torch.matmul(x, w3)              
    return x

x = torch.randn(1024, 4096, device="neuron", dtype=torch.bfloat16)
w1 = torch.randn(4096, 4096, device="neuron", dtype=torch.bfloat16)
w2 = torch.randn(4096, 4096, device="neuron", dtype=torch.bfloat16)
w3 = torch.randn(4096, 4096, device="neuron", dtype=torch.bfloat16)

compiled = torch.compile(big_simple, backend="neuron")
```

### Step 1: What does this function actually compute?

| Line | Math | What kind of operation? |
|------|------|------------------------|
| `matmul(x, w1)` | X × W₁ | Multiply-accumulate: for each output element, dot 4096 pairs and sum |
| `layer_norm(x)` | (x - μ) / √(σ² + ε) | Reduce (mean), subtract, square, reduce (variance), rsqrt, scale |
| `matmul(x, w2)` | X × W₂ | Same as first matmul |
| `gelu(x)` | x · 0.5 · (1 + erf(x/√2)) | Polynomial approximation, then element-wise multiply |
| `matmul(x, w3)` | X × W₃ | Same as first matmul |

### Step 2: Map each primitive to an engine

Each of these primitives needs to run on specific hardware. Let's think through the mapping:

**Matrix multiplication** is a large number of multiply-accumulate operations. This is the most common operation in modern AI, so NeuronCore has a dedicated 128×128 systolic array purpose-built for it — the Tensor engine. The full 4096×4096 matrix doesn't fit in the array at once, so the compiler breaks it into 128×128 tiles and processes them sequentially.

**Reductions and element-wise math** (the subtract, multiply, and sum operations inside layer_norm) operate on vectors of values. The Vector engine handles these with 128-wide SIMD — it processes 128 elements per cycle in parallel.

```{figure} ../assets/vector_engine.png
:alt: Vector engine parallel lanes
:width: 500px
:align: center

The Vector engine: 128 parallel lanes reducing across the free dimension.
```

**Transcendental functions** like rsqrt and erf can't be computed with simple arithmetic. The Scalar engine has dedicated piecewise polynomial hardware — essentially lookup tables with interpolation — that approximate these functions efficiently.

**Data movement** between HBM (where your tensors live) and SBUF (where engines read from) is handled by 16 parallel DMA engines. They operate on contiguous chunks and can run concurrently with compute.

Putting it together:

| Primitive operation | Engine | Why |
|---|---|---|
| Dot product (multiply-accumulate) | **Tensor** | 128×128 systolic array |
| Reduce (sum across a dimension) | **Vector** | 128-wide SIMD partial sums |
| Subtract, multiply (element-wise) | **Vector** | 128-wide SIMD |
| rsqrt, erf (transcendentals) | **Scalar** | Polynomial lookup tables |
| Move tiles HBM ↔ SBUF | **DMA** | 16 parallel engines |

So our 5-line function decomposes like this:

```
matmul(x, w1)     →  DMA: load W₁ tiles from HBM → SBUF
                      Tensor: LDWEIGHTS (fill systolic array with W₁ tile)
                      Tensor: MATMUL (stream X tiles through, accumulate in PSUM)
                      ... repeated for each 128×128 tile pair

layer_norm(x)     →  Vector: REDUCE (sum all elements → compute mean)
                      Vector: TENSOR_TENSOR SUBTRACT (x - mean)
                      Scalar: ACTIVATE SQUARE (compute (x - mean)²)
                      Vector: REDUCE (sum squares → compute variance)
                      Scalar: TENSOR_SCALAR RSQRT (1 / √variance)
                      Vector: MULTIPLY (normalize)

matmul(x, w2)     →  same pattern as first matmul

gelu(x)           →  Scalar: ACTIVATE (polynomial approximation of erf)
                      Vector: MULTIPLY (x × Φ(x))

matmul(x, w3)     →  same pattern as first matmul
```

### Step 3: The problems the compiler must solve

The compiler can't just execute these sequentially — it must orchestrate parallelism:

**Tiling.** A 4096×4096 weight matrix in bfloat16 is 32 MB. SBUF is ~28 MB per core. The entire matrix doesn't fit at once. The compiler breaks it into 128×128 tiles (32 KB each) and processes them in sequence, loading the next tile while computing the current one.

**Pipelining.** While the tensor engine multiplies tile N, DMA loads tile N+1 into a different SBUF bank. This hides memory latency — the tensor engine never waits for data (ideally).

**Multi-core split.** The compiler splits your 1024 rows across 2 NeuronCores: nc0 handles rows 0–511, nc1 handles rows 512–1023. Both cores execute the same instruction sequence in parallel. This is why Neuron Explorer shows two of everything (Tensor nc0, Tensor nc1, etc.).

**Synchronization.** The engines run concurrently but have data dependencies — you can't normalize before the matmul finishes. The compiler inserts **semaphores** (hardware counters) that gate instructions: "don't start this SUBTRACT until the REDUCE has completed 55 iterations." You'll see these as `S[4] (Scalar)>=55` in the profiler tooltips.

**Weight loading.** The tensor engine has a **weight-stationary** design with a dual weight cache. While the current matmul runs using the "foreground" weights, the DMA loads the next weight tile into the "background" slot. When the current tile finishes, foreground and background swap instantly — no stall.

### Step 4: Seeing it in Neuron Explorer

Let's compile and profile this function, then open the result in Neuron Explorer:

```python
import os, torch, shutil, glob
from torch.profiler import profile, ProfilerActivity
from torch_neuronx.profiling import NeuronConfig, ProfileMode, NeuronProfiler

os.environ["NEURON_RT_NUM_CORES"] = "1"
device = "neuron"
profile_dir = "/workshop/profile_simple"
os.system(f"rm -rf {profile_dir}")

def big_simple(x, w1, w2, w3):
    x = torch.matmul(x, w1)
    x = torch.layer_norm(x, [4096])
    x = torch.matmul(x, w2)
    x = torch.nn.functional.gelu(x)
    x = torch.matmul(x, w3)
    return x

x = torch.randn(1024, 4096, device=device, dtype=torch.bfloat16)
w1 = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
w2 = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
w3 = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

compiled = torch.compile(big_simple, backend="neuron")

# Warmup (triggers compilation)
with torch.no_grad():
    for _ in range(3):
        _ = compiled(x, w1, w2, w3)
torch.neuron.synchronize()

# Profile
config = NeuronConfig(
    modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
    profile_output_dir=profile_dir,
    capture_enabled_for_nc="0",
)
exporter = NeuronProfiler(config)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    experimental_config=config,
    on_trace_ready=exporter.export_trace,
) as prof:
    with torch.no_grad():
        _ = compiled(x, w1, w2, w3)
    torch.neuron.synchronize()

# Copy NEFFs so Neuron Explorer can decode traces
neff_cache = os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR", "/tmp/neff_cache")
ntff_files = glob.glob(f"{profile_dir}/**/*.ntff", recursive=True)
if ntff_files:
    session_dir = os.path.dirname(ntff_files[0])
    for neff in glob.glob(f"{neff_cache}/**/*.neff", recursive=True):
        shutil.copy2(neff, session_dir)
```

Now open the output in the **Neuron Explorer**: In VSCode open `>Neuron Explorer: open profile manager`, give your profile a unique name, click on `Upload Profile`, select a directoy upload and point the `profile_dir` and select your python script as the source code which will allow neuron explorer to map the profile with specific pytorch instructions. 

alternatively, you can pen the profile through the CLI and port-forwarding. 
```bash
neuron-explorer view -d /workshop/profile_simple --port 3001 --display-name simple
```

We'll use Neuron Explorer extensively in Part III (Chapters 9-10) to build roofline models and diagnose bottlenecks.

#### The System Timeline

```{figure} ../assets/ne-ex1-systemtimeline.png
:alt: System Timeline showing the full execution
:width: 800px
:align: center

The System Timeline: PyTorch compiles the graph (row 2), then a single NEFF executes on two NeuronCores in parallel (pink blocks, rows 5-6).
```

The System Timeline reads top to bottom:
1. **framework/TID:0** — the main Python thread, active for the entire call
2. **framework/TID** (second row) — the `torch.compile` dispatch: Dynamo captures the graph, hands it to the Neuron backend
3. **framework/Stream:0** — the compiled NEFF executing as one continuous block
4. **neuron_rt/NC:0** — runtime setup (memory allocation, NEFF submission) then waiting for hardware
5. **neuron_hw/NC0:4, NC0:5** — two physical NeuronCores executing in parallel

Double-click on one of the pink `nc_exec_running` blocks to open the Device Timeline.

#### The Device Timeline

The Device Timeline shows what happens *inside* the NeuronCore during execution. Click and drag to select a region of interest — the view will zoom into that time window.

```{figure} ../assets/ne-ex1-devicetimeline.png
:alt: Device Timeline showing engine-level activity
:width: 800px
:align: center

The Device Timeline: each row is a hardware engine. The execution splits into three visible "waves" corresponding to our three matmuls.
```

Every engine is duplicated (nc0, nc1) because the compiler split work across 2 NeuronCores. Reading the timeline left to right, you can see three distinct phases:

**Wave 1 (0–300,000 ns): First matmul.** The Tensor rows show dense activity — DMA engines are loading weight tiles while the systolic array computes `matmul(x, w1)`. The TensorMatrix rows show individual matmul instructions tightly packed.

**Wave 2 (~300,000–320,000 ns): Layer norm.** The Tensor engine goes quiet. Vector and Scalar engines take over — computing mean, variance, rsqrt, and normalization. This is the gap between the first and second matmul bursts.

**Wave 3 (~320,000 ns–end): Second matmul, GELU, and third matmul.** The compiler is smart here: it interleaves the GELU activation with the second matmul's output tiles and the third matmul's weight loading. Rather than waiting for all of `matmul(x, w2)` to complete before starting GELU, the compiler pipelines them — you'll see Vector/Scalar activity overlapping with Tensor activity in this region.

#### Hovering: tracing instructions back to Python

Hover over any block in the Device Timeline to see exactly what hardware instruction it represents and which line of Python generated it:

```{figure} ../assets/ne-ex1-matmul.png
:alt: Hovering over a MATMUL instruction
:width: 800px
:align: center

A MATMUL instruction on the TensorMatrix engine — traced back to `test.py:13` (our first `torch.matmul` call). Shows the tile dimensions, duration, and source location.
```

```{figure} ../assets/ne-ex1-vector-tensor.png
:alt: Hovering over a TENSOR_TENSOR SUBTRACT instruction
:width: 800px
:align: center

A TENSOR_TENSOR SUBTRACT on the Vector engine — part of layer_norm's `x - mean` computation, traced to `test.py:14`.
```

```{figure} ../assets/ne-ex1-scalar-activate.png
:alt: Hovering over an ACTIVATE instruction
:width: 800px
:align: center

An ACTIVATE instruction on the Scalar engine — the polynomial approximation inside GELU, traced to `test.py:16`.
```

#### Memory usage

```{figure} ../assets/ne-ex1-memusage.png
:alt: State Buffer and PSUM usage over time
:width: 800px
:align: center

State Buffer (SBUF) and PSUM usage. The sawtooth pattern in PSUM shows accumulation during matmuls followed by drain when results write back to SBUF.
```

The memory rows at the bottom confirm the tiling strategy:
- **State Buffer Usage** oscillates as weight tiles load in and get consumed
- **PSUM Usage** spikes during matmul accumulation (partial products building up), then drops when results are written back to SBUF for the next operation

```{admonition} The key insight
:class: important
Five lines of Python became a choreography across 2 NeuronCores × (32 DMA engines + Tensor + Vector + Scalar + GpSimd), all running concurrently with semaphore synchronization. The compiler did this automatically — from `torch.compile`, we went from Python to hardware instructions with full source-level traceability.
```

---

## Neuron architectural choices and tradeoff 

Neuron makes a deliberate tradeoff: It bets that **95% of ML compute is matmul-shaped**: Most of the die area is allocated to larger systolic arrays, so when you have layer operations that have more exotic memory access patterns, it can be more challenging to support them well (think scatter_reduce, etc.)

---
## Engineering Principles

> **Keep the tensor engine always doing matmuls. Everything else is auxiliary data movement.**

This single principle explains most of Neuron performance engineering:

- **Why `torch.compile` helps** (Chapter 3): it fuses all the small vector/scalar ops between matmuls so the tensor engine never stalls waiting for data
- **Why contiguity matters** (Chapter 2): DMA engines transfer contiguous blocks fastest — strided access slows the data movement that feed the tensor engine
- **Why memory hierarchy matters** (Chapter 8): the DMA engines' job is to keep SBUF fed so the tensor engine never idles
- **Why collectives are "free"** (Chapter 17): the 16 CC-Cores run independently of compute — communication overlaps computation at zero cost

The critical arithmetic intensity: **667 TFLOPS ÷ 2.9 TB/s ≈ 230 ops/byte**. If your operation does fewer than 230 FLOPs per byte loaded from HBM, it's memory-bound — the tensor engine is waiting for data. If more, it's compute-bound — the ideal state.

A 1024×1024 BF16 matmul has arithmetic intensity ~512 ops/byte — solidly compute-bound. A vector sum has intensity ~1 op/byte — hopelessly memory-bound. 

```{seealso}
For detailed NeuronCore-v3 engine specifications and the full Trn2 instance architecture, see the [Neuron SDK documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html).
```
---

## When the compiler isn't enough

The Neuron compiler handles the common cases well — matmuls, layer norms, attention patterns. But some operations have no native implementation, and some algorithms need tighter control over how data moves through SBUF and PSUM than the compiler can provide.

When you hit that wall, an unsupported op falling back to CPU, a novel recurrence like DeltaNet or Mamba2, or a fusion pattern the compiler misses, you write directly to the hardware using **NKI (Neuron Kernel Interface)**. NKI gives you explicit control over DMA scheduling, engine assignment, and tile layout. You become the compiler.

That's Part V of this book. For now, know that the engines and memory hierarchy we just explored are exactly what you'll be programming against.

---

*The chip has engines, SRAM, and HBM. But how does data actually flow between these levels? What decides what lives where, and when it moves?*


