# The profiler

*You can't optimize what you can't measure. Neuron Explorer is your microscope.*

---

## What is Neuron Explorer?

Neuron Explorer is an instruction-level, cycle-accurate profiler with near-zero overhead. It shows you exactly what every engine is doing at every nanosecond — which instructions fired, what data moved, which engines were idle and why.

```{admonition} No Heisenberg effect
:class: tip
Most profilers perturb what they measure — adding overhead that shifts bottlenecks. Neuron Explorer has dedicated hardware circuits that emit trace notifications for every instruction at silicon speed. The act of measurement doesn't move the bottleneck. In practice, you can run with profiling **always on** — the overhead is negligible enough that Neuron's runtime essentially keeps a profiler armed at all times.
```

> "Every flop, every nanosecond, every byte of memory in every operation of every kernel can be traced to this level of detail. This is a level of visibility into the performance of your kernels that you really just don't get anywhere else."
>
> — Jay Gray, Trainium Inference Lead, Anthropic

It exposes two levels:
- **Device profiling** — what's happening inside the NeuronCore: instructions on each engine, DMA transfers, memory utilization
- **System profiling** — the holistic view: host CPU, runtime overhead, multi-core synchronization, NEFF loading

---

## Setting up a profile capture

The Native PyTorch API integrates directly with `torch.profiler`:

```python
import torch
from torch.profiler import profile, ProfilerActivity
from torch_neuronx.profiling import NeuronConfig, ProfileMode, NeuronProfiler

# Configure what to capture
config = NeuronConfig(
    modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
    profile_output_dir="/workshop/my_profile",
)
exporter = NeuronProfiler(config)

# Warmup (triggers compilation — don't profile this)
with torch.no_grad():
    _ = model(x)
torch.neuron.synchronize()

# Profile a real execution
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    experimental_config=config,
    on_trace_ready=exporter.export_trace,
) as prof:
    with torch.no_grad():
        _ = model(x)
    torch.neuron.synchronize()
    prof.step()
```

Two environment variables enable richer profiling (set before compilation):

```bash
export NEURON_FRAMEWORK_DEBUG=1      # Source code → instruction mapping
export XLA_IR_DEBUG=1                # Descriptive layer names
```

### Profile modes

| Mode | What it captures | When to use |
|------|-----------------|-------------|
| `DEVICE` | Hardware instructions per engine | Kernel optimization, tiling analysis |
| `RUNTIME` | System-level trace (runtime → hardware) | End-to-end latency debugging |
| `CPU_UTIL` | Host CPU utilization | Detecting CPU bottlenecks |
| `HOST_MEMORY` | Host memory usage | OOM debugging |

90% of the time you want `DEVICE` + `RUNTIME`. Note: `DEVICE` mode reserves ~5 GB of HBM for trace storage on Trn2.

### Viewing the profile

```bash
neuron-explorer view -d /workshop/my_profile --port 3001 --display-name "my model"
```

Then SSH port-forward and open `localhost:3001`. Or use the VS Code Neuron Explorer extension: `>Neuron Explorer: Open Profile Manager` → Upload Profile → point to the directory.

---

## The views that matter

Neuron Explorer has ~9 views. Three matter most:

### Summary View

Your first stop. Shows holistic metrics in one screen:
- **MFU** (Model FLOPs Utilization) — only counts actual model computation (matmuls). The single most important number.
- **HFU** (Hardware FLOPs Utilization) — counts ALL tensor engine ops including transposes and broadcasts
- **Arithmetic intensity** — achieved FLOPs / bytes moved
- **Per-engine activity %** — which engines are busy, which are idle
- **DMA statistics** — read/write throughput, spill/reload ratio

If HFU >> MFU, you're spending too much tensor engine time on transposes rather than useful compute.

Ron (Chief Architect, Annapurna): "If you see MFU in the teens or low single digits, that's very bad. 30-40% means you're doing pretty good. Goal is to push toward 100%."

### System Trace View

Shows the timeline across the full stack — Python thread, torch.compile dispatch, runtime, and hardware. Useful for spotting:
- Framework overhead (time between Python and hardware execution)
- Multi-core synchronization gaps
- Runtime setup costs (NEFF loading, memory allocation)

Double-click on a hardware execution block to drill into the Device Trace.

### Device Trace View

The most important view for performance engineering. Each row is a hardware engine:

```
DMA engines (16)     ████░░████░░████░░████
Tensor Engine        ██████████░░██████████     (load stationary)
TensorMatrix Engine  █████████░░░█████████░     (multiply moving)
Vector Engine        ░░░░░███░░░░░░░███░░░░
Scalar Engine        ░░░░░░░██░░░░░░░░██░░░
GPSIMD               ░░░░░░░░░░░░░░░░░░░░░░
```

Note: "Tensor Engine" and "TensorMatrix Engine" are the same physical engine. The profiler separates load stationary (pure data movement) from multiply moving (actual computation) to make the distinction visible.

Red blocks in the trace = semaphore waits (one engine waiting for another to complete). These are scheduling artifacts, not bugs.

---

## Reading a profile: what to look for

### Anti-pattern 1: Idle tensor engine

If the tensor engine row has large gaps, something else is blocking it. The tensor engine is 90% of your available FLOPs — every idle nanosecond is wasted.

Causes: DMA still loading (memory-bound), vector/scalar ops not pipelined (serialized compute), spill/refill activity.

### Anti-pattern 2: No engine pipelining

If engines fire sequentially (tensor finishes, then vector starts, then scalar starts), you're leaving parallelism on the table. Good profiles show stacked rows — multiple engines active simultaneously.

### Anti-pattern 3: Tiny DMA transfers

Look at individual DMA blocks. If they show 4-byte or 32-byte transfers instead of 32+ KB, bandwidth utilization is terrible even if DMA engines show high % active time. The transfer size is visible when you hover over a DMA event.

Target: ≥32 KB per DMA transfer for good bandwidth utilization.

### Anti-pattern 4: Unexpected spills

If you see DMA store/load activity during compute that you didn't program (in an NKI kernel) or don't expect (in a compiled model), that's the compiler spilling SBUF to HBM. The spill/reload ratio in the Summary view quantifies this.

---

## Source code mapping

With `NEURON_FRAMEWORK_DEBUG=1` set, Neuron Explorer maps every hardware instruction back to the Python source line that generated it.

- In the Device Trace, hover over any instruction block → tooltip shows source file and line number
- In the Source Code View, click a line → highlights all corresponding instructions in the timeline
- For NKI kernels: your kernel code lines map 1:1 to instruction groups

This is what makes Neuron Explorer uniquely powerful for NKI development: write a kernel, profile it, see exactly which line generates which instructions, identify the gap, fix it, reprofile.

---

## Profiling NKI kernels specifically

For NKI kernels, `nki.benchmark` gives quick P50/P99 latency without the full profiler UI:

```python
import neuronxcc.nki as nki

bench = nki.benchmark(my_kernel, input_a, input_b)
print(f"P50: {bench.p50_latency_ms:.3f} ms")
print(f"P99: {bench.p99_latency_ms:.3f} ms")
```

Use this for rapid iteration ("932μs... changed something... 873μs, making progress"), then switch to Neuron Explorer when you need to understand *why*.

---

## Practical tips

- **Profile a single operation first.** A full model profile is unreadable — billions of data points. Isolate the attention layer, or just the QK matmul, and profile that. You can always zoom out later.
- **Skip compilation.** In compile mode, warm up the model first (one call), then profile the second call. Otherwise you're profiling compilation, not execution.
- **Drag to select.** In the Device Trace, horizontal drag selects a time region and updates all statistics for just that region.
- **Watch for the NEF gotcha.** In eager mode, NEFs sometimes write to `/tmp/neff_cache/` instead of your profile directory. If Explorer says "no NEF found," check the cache.
- **Use JSON output for automation.** Explorer can output profiling statistics as JSON — useful for CI pipelines or agentic tools that track performance regressions.

---

## What good looks like

From the NKI bootcamp demo — a 512×512 NKI tiled matmul:
- MFU: 3.6% (tiny matmul, expected to be low — not enough work to pipeline)
- MFU max achievable: 77%
- Timeline shows: DMA loads → load_weights → multiply_moving → cast copy (FP32 PSUM → BF16 SBUF) → memset PSUM → repeat

A well-optimized kernel (from NKI docs): 98% HFU on a fully tiled matmul. The gap between 3.6% and 98% is the optimization journey of Part V.

*Question raised → "I can see the bottleneck in the profiler. The tensor engine is idle. Now how do I fix it?"*

*Next: Part IV — Precision & Numerics (make the numbers smaller to move less data).*
