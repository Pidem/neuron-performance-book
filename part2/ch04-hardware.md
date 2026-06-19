# Hardware deep dive

In Chapter 3, we ended with: "The compiler generates machine code. In this chapter, we will dive deep into the chip architecture choices made by Neuron. 

---

## The GPU mental model (brief)

If you're coming from CUDA, here's what you're used to:

- **Many small SMs** (Streaming Multiprocessors), each running warps of 32 threads
- **SIMT model:** thousands of independent threads execute the same instruction
- **Tiny shared memory per SM** (~100-200 KB), massive parallelism compensates
- **Hardware atomicAdd:** enables scatter — each thread writes to a random destination, atomics serialize conflicts. Works because thousands of other threads keep the hardware busy while some stall.

The GPU bet: *throw enough threads at the problem and latency disappears.*

---

## The NeuronCore architecture

Neuron makes the opposite architectural design choice: *dedicate die area to massive compute engines and fast on-chip memory, not thread management.*

```{figure} ../assets/trainium2_architecture.png
:alt: Trainium2 chip architecture
:width: 600px
:align: center

The Trainium2 chip: 8 NeuronCore-v3 engines, 96 GiB HBM, 224 MiB SBUF.
```

Each Trainium2 chip contains **8 NeuronCore-v3** cores. Each NeuronCore has specialized engines:

| Engine | Role | Think of it as... |
|--------|------|-------------------|
| **Tensor engine** | Matrix multiplication (systolic array) | This is where FLOPs happen |
| **Vector engine** | Reductions, accumulations, SIMD ops across 128 elements | The reducer — sum, mean, normalize |
| **Scalar engine** | Element-wise ops (exp, reciprocal, comparisons) | The activator — one element at a time |
| **GPSIMD engine** | Arbitrary C code for custom ops | The escape hatch |
| **DMA engines** | Shuttle data between SRAM ↔ HBM | The trucks — keep the engines fed |

```{figure} ../assets/neuroncore-v3.png
:alt: NeuronCore-v3 architecture
:width: 600px
:align: center

Inside a single NeuronCore-v3: specialized engines connected by on-chip SRAM (SBUF).
```

**Key architectural insight:** NO independent threads. NO random-access write hardware. DMA only does contiguous bulk transfers. Die area went to a bigger systolic array instead of atomic units.

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

---

## Seeing the engines in action

Theory is nice. Let's prove it. Let's profile four isolated ops on our trn2.3xlarge and measure which engine the hardware actually uses:

```python
import torch, os, glob, shutil
from torch.profiler import profile, ProfilerActivity
from torch_neuronx.profiling import NeuronConfig, ProfileMode, NeuronProfiler

os.environ["NEURON_RT_NUM_CORES"] = "1"

device = "neuron"
profile_dir = "./experiment_engine_activation"
neff_cache = f"{profile_dir}/neff_cache"

os.system(f"rm -rf {profile_dir}")
os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = neff_cache

def profile_op(name, op_fn, warmup_fn):
    """Profile a single op and copy NEFFs into the session directory."""
    out_dir = f"{profile_dir}/{name}"

    # Warmup populates the NEFF cache
    warmup_fn()
    torch.neuron.synchronize()

    config = NeuronConfig(
        modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
        profile_output_dir=out_dir,
        capture_enabled_for_nc="0",
    )
    exporter = NeuronProfiler(config)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        experimental_config=config,
        on_trace_ready=exporter.export_trace,
    ) as prof:
        op_fn()
        torch.neuron.synchronize()

    # neuron-explorer needs .neff files alongside .ntff files to decode traces
    ntff_files = glob.glob(f"{out_dir}/**/*.ntff", recursive=True)
    if ntff_files:
        session_dir = os.path.dirname(ntff_files[0])
        for neff in glob.glob(f"{neff_cache}/**/*.neff", recursive=True):
            shutil.copy2(neff, session_dir)

# Setup tensors
a = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
b = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
z = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

profile_op("tensor_matmul",
    op_fn=lambda: [torch.matmul(a, b) for _ in range(5)],
    warmup_fn=lambda: [torch.matmul(a, b) for _ in range(3)])

profile_op("vector_sum",
    op_fn=lambda: [torch.sum(a, dim=-1) for _ in range(5)],
    warmup_fn=lambda: [torch.sum(a, dim=-1) for _ in range(3)])

profile_op("scalar_gelu",
    op_fn=lambda: [torch.nn.functional.gelu(a) for _ in range(5)],
    warmup_fn=lambda: [torch.nn.functional.gelu(a) for _ in range(3)])

profile_op("dma_clone",
    op_fn=lambda: [z.clone() for _ in range(5)],
    warmup_fn=lambda: [z.clone() for _ in range(3)])
```

View the results with `neuron-explorer`:

```bash
neuron-explorer view -d ./experiment_engine_activation/tensor_matmul --output-format summary-text 2>&1 | grep -E "engine_active_time_percent|dma_active_time_percent|matmul_instruction"
neuron-explorer view -d ./experiment_engine_activation/vector_sum --output-format summary-text 2>&1 | grep -E "engine_active_time_percent|dma_active_time_percent|matmul_instruction"
neuron-explorer view -d ./experiment_engine_activation/scalar_gelu --output-format summary-text 2>&1 | grep -E "engine_active_time_percent|dma_active_time_percent|matmul_instruction"
neuron-explorer view -d ./experiment_engine_activation/dma_clone --output-format summary-text 2>&1 | grep -E "engine_active_time_percent|dma_active_time_percent|matmul_instruction"
```

Here's what the hardware reported:

| Engine | `matmul` | `sum` | `gelu` | `clone` |
|--------|----------|-------|--------|---------|
| **Tensor** | **17.1%** | 3.1% | 1.2% | 1.5% |
| **Vector** | 2.7% | **5.4%** | **25.7%** | 1.2% |
| **Scalar** | 11.6% | 2.9% | **20.3%** | 1.1% |
| **DMA** | **13.0%** | 5.8% | 2.6% | **29.2%** |
| matmul instructions | **960** | 10 | 0 | 0 |

The story:

- **matmul** → tensor engine dominant, DMA feeding it data, 960 matmul instructions executed
- **sum** → vector engine highest — reducing 1024 elements per row is a SIMD reduction
- **gelu** → vector (25.9%) + scalar (20.4%) — *not* what you'd naively expect
- **clone** → DMA dominant (29.2%), compute engines nearly idle — pure data movement

```{admonition} GELU isn't "scalar" — the compiler is smarter than you think
:class: tip
We expected GELU (an element-wise activation) to be scalar-dominated. Instead, the compiler vectorized the polynomial approximation (`x * 0.5 * (1 + tanh(...))`) into 128-wide SIMD operations on the vector engine. This is why you **measure, not assume**. The engine assignment isn't always intuitive.
```

```{admonition} Why is total utilization low?
:class: note
The individual op profiles show low total utilization (16-29%) because we're profiling single ops in eager mode — each op compiles its own tiny NEFF, dispatches, and returns. The overhead between ops dwarfs the compute. This is exactly why `torch.compile` gives 8× speedup (Chapter 3): it eliminates the between-op overhead and keeps the engines busy continuously.
```

---

## Why scatter falls back — the architecture explains it

In Chapter 3 we discovered that `torch.scatter` silently falls back to CPU (8.9× slower than a native matmul). Now we can explain *why*:

- **GPU:** hardware atomicAdd in the memory controller makes scatter work natively. Each thread writes to a random destination; atomics serialize conflicts. Thousands of other threads keep the hardware busy while some stall.
- **Neuron:** no atomic units. No random-access write hardware. DMA only does contiguous bulk transfers. There is physically no circuit on the chip that can do `output[random_index] += value`.

The tradeoff is deliberate: Neuron bets that **95% of ML compute is matmul-shaped**. The die area that would have gone to atomic units instead went to a larger systolic array (667 TFLOPS at BF16). For the 5% of ops that need scatter/gather semantics, the runtime falls back to CPU.

```{admonition} The EquiformerV3 story
:class: note
The EquiformerV3 model (equivariant GNN for molecular simulations) has 5 ops that fall back to CPU: `scatter_reduce`, `atan2`, `linalg_cross`, `acos`, and `uniform_`. The model still *runs correctly* — but the PCIe round-trips make it 87× slower than running entirely on CPU. In Part V, we'll write NKI kernels to solve some of these on-chip.
```

---

## The one rule

> **Keep the tensor engine always doing matmuls. Everything else is auxiliary data movement.**

This single principle explains most of Neuron performance engineering:

- **Why `torch.compile` helps** (Chapter 3): it fuses all the small vector/scalar ops between matmuls so the tensor engine never stalls waiting for data
- **Why contiguity matters** (Chapter 2): DMA engines transfer contiguous blocks fastest — strided access slows the trucks that feed the tensor engine
- **Why memory hierarchy matters** (Chapter 8): the DMA engines' job is to keep SBUF fed so the tensor engine never idles
- **Why collectives are "free"** (Chapter 17): the 16 CC-Cores run independently of compute — communication overlaps computation at zero cost

The critical arithmetic intensity: **667 TFLOPS ÷ 2.9 TB/s ≈ 230 ops/byte**. If your operation does fewer than 230 FLOPs per byte loaded from HBM, it's memory-bound — the tensor engine is waiting for data. If more, it's compute-bound — the ideal state.

A 1024×1024 BF16 matmul has arithmetic intensity ~512 ops/byte — solidly compute-bound. A vector sum has intensity ~1 op/byte — hopelessly memory-bound. This is why the profiler showed matmul at 16% tensor engine utilization but sum at only 5% vector — the sum is bottlenecked on data movement, not compute.

```{seealso}
For detailed NeuronCore-v3 engine specifications and the full Trn2 instance architecture, see the [Neuron SDK documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html).
```

---

*Question raised → "OK, but how does my PyTorch code actually get to these engines?"*

*Next: [Chapter 5](ch05-pytorch-native) — PyTorch Native on Neuron.*
