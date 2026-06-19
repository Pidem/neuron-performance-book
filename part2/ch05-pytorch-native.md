# PyTorch Native on Neuron

*Change one word: `model.to("cuda")` → `model.to("neuron")`. What happens underneath?*

```{admonition} TODO
:class: warning
This chapter answers: "How does my PyTorch code actually get to the NeuronCore engines?"
```

---

## Act 1: Neuron as a private backend plugin

```{admonition} TODO
:class: warning
- Same registration pattern as CUDA in PyTorch (PrivateUse1 dispatch key)
- The dispatcher routes ops to the Neuron backend automatically
- `torch.device("neuron")` → torch sees it like any other accelerator
- `torch.accelerator` auto-detection when torch-neuronx is installed
- Show the registration code from torch_neuronx source (we have it at `/workshop/workspace/torch_neuron_eager/`)
- Diagram: PyTorch dispatcher → PrivateUse1 → Neuron runtime → NeuronCore
```

---

## Act 2: What "native" means — the XLA history

```{admonition} TODO
:class: warning
- **The old world:** XLA, LazyTensor, `xm.mark_step()`, whole-graph compilation
- Why it was abandoned: poor debugging, eager semantics broken, graph breaks catastrophic
- **The new world:** PyTorch Native — op-by-op eager works, torch.compile is optional optimization
- Same code runs on CPU, CUDA, Neuron — no framework-specific APIs
- Timeline: Beta 1 → Beta 2 → Beta 3 (what we're using)
- Reference: [[NDS Weekly - PyTorch Native and Eager on Neuron]] (Jin Ying broadcast)
```

---

## Act 3: CPU fallback — the safety net

```{admonition} TODO
:class: warning
- Unsupported ops run on CPU automatically (functionality preserved, perf hit)
- The runtime handles device transfers transparently (that copy_ chain from ch03)
- `torch_neuronx.get_fallback_ops()` — your first debugging tool (does this exist in Beta 3? validate)
- The contract: "all models should work" — correctness first, performance second
- When to accept CPU fallback vs when to rewrite (foreshadow NKI in Part V)
```

---

## Act 4: What changes in your code — and what doesn't

```{admonition} TODO
:class: warning
- Minimal code diff: `device = "neuron"` + `torch.neuron.synchronize()` for timing
- Same training loop, same optimizer, same data loading
- What DOES change: async execution (ch01 covered this), NEFF compilation (ch01/03 covered this)
- The async trap revisited: why `torch.neuron.synchronize()` is needed for accurate benchmarks
- Show ESM-2 running: same code from ch01 but now we understand every layer underneath
```

*Question raised → "So I just change the device and it works? What about ops Neuron doesn't support?"*

*Next: [Chapter 6](ch06-eager-mode) — Eager mode: getting your model running.*
