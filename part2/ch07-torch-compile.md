# torch.compile on Neuron

*The compiler optimizes the graph. Longer compilation, better runtime.*

```{admonition} TODO
:class: warning
This chapter answers: "I've seen the 8x speedup in ch03. Now HOW does the compiler achieve it?"
Goes DEEPER than ch03 — ch03 showed the benchmark and gotchas, this shows the compilation pipeline internals.
```

---

## Act 1: The three-stage pipeline (detailed)

```{admonition} TODO
:class: warning
- Stage 1: TorchDynamo → captures FX graph (Python-level tracing)
- Stage 2: neuronx-cc → hardware-aware optimizations (op fusion, tiling decisions, DMA scheduling)
- Stage 3: NEFF generation → binary that runs on NeuronCore
- Show the intermediate representations at each stage (if accessible)
- Compilation flags and options: `torch.compile(model, backend="neuron", options=...)`
- What optimizations happen at each stage: dead code elimination, constant folding, op fusion, layout transformation
```

---

## Act 2: What the compiler fuses — and why it matters

```{admonition} TODO
:class: warning
- Revisit the 1050 → 1 NEFF reduction from ch03, but now explain WHAT was fused
- Concrete fusion examples in ESM-2:
  - LayerNorm = mean + variance + normalize + scale + bias → one fused kernel
  - GELU = x * 0.5 * (1 + tanh(...)) → one fused kernel
  - Attention: QKV projection + reshape + transpose → fused
- Each fusion eliminates HBM round-trips (preview of ch08 memory hierarchy)
- Show in Neuron Explorer: fused vs unfused instruction count
```

---

## Act 3: NEFF cache management

```{admonition} TODO
:class: warning
- Where NEFFs live: `/tmp/neff_cache/` (eager) vs compilation cache (compile)
- Cache invalidation: when does recompilation trigger? (code change, shape change, config change)
- `NEURON_CC_FLAGS` for cache control
- Production workflow: compile once, deploy cached NEFFs
- Compilation time budget: ESM-2 650M takes ~X minutes first compile (measure this)
- Parallel compilation: neuronx-cc can compile multiple subgraphs in parallel
```

---

## Act 4: The two personas — research vs production

```{admonition} TODO
:class: warning
- Research persona: eager mode, fast iteration, Python debugging
- Production persona: compiled mode, max throughput, cached NEFFs
- The transition workflow: develop in eager → validate → compile → profile → deploy
- When NOT to compile: dynamic control flow, heavy Python interop, debugging
- Bucketing revisited: the production solution for variable-length inputs (from ch03)
- Warm-up strategies for inference servers
```

*Question raised → "Compile helped, but there are still gaps in the profiler. Where is the time going?"*

*Next: [Chapter 8](ch08-memory-hierarchy) — The memory hierarchy.*
