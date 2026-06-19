# Eager mode: getting your model running

*Run ESM-2 in eager mode on Neuron. Some ops fall back to CPU. Now what?*

```{admonition} TODO
:class: warning
This chapter answers: "I changed the device. What's actually happening under the hood in eager mode?"
Goes DEEPER than ch01 — ch01 showed the "what", this shows the "how and why" of eager execution on Neuron.
```

---

## Act 1: Op-by-op NEFF compilation

```{admonition} TODO
:class: warning
- Each op compiles to a small NEFF individually (contrast with torch.compile's single NEFF)
- First-iteration penalty: compilation time per op (revisit the 8506µs first-call from ch01)
- NEFF cache: `/tmp/neff_cache/` — show what's in there, how it grows
- Subsequent calls hit cache (49µs from ch01)
- Many small NEFFs vs one monolithic NEFF (the old XLA approach compiled everything up front)
- Analogy: eager = JIT interpreting line by line; compile = AOT compiling the whole program
```

---

## Act 2: The fallback debugging workflow

```{admonition} TODO
:class: warning
- `torch_neuronx.get_fallback_ops()` — list what's falling back (validate this API exists in Beta 3)
- Reading the fallback list: categorize ops as "accept on CPU" vs "needs fixing"
- Decision framework:
  - Is it in the critical path? (attention inner loop vs one-time preprocessing)
  - How much data moves? (small tensor fallback = cheap; large tensor = expensive)
  - Is there a Neuron-native alternative? (e.g., replace scatter with a matmul-based approach)
- Profiling fallbacks: the copy_ chain pattern from ch03, but now in a real model context
```

---

## Act 3: Guest example — SchNet (molecular GNN)

```{admonition} TODO
:class: warning
- SchNet: message-passing GNN for molecular property prediction (HCLS-relevant)
- scatter/gather ops produce many fallbacks — natural illustration of the debugging workflow
- Profile it: show which ops fall back, measure the PCIe round-trip cost
- The 87× slower finding from EquiformerV3 (5 CPU fallbacks: atan2, linalg_cross, acos, scatter_reduce, uniform_)
- Alternative: could use ESM-2 with a custom head that uses scatter? (simpler setup)
- Key insight: even with fallbacks, the model RUNS correctly — performance is the only issue
- Foreshadow: "We'll fix scatter_reduce in Part V by writing our own NKI kernel"
```

---

## Act 4: When eager is enough

```{admonition} TODO
:class: warning
- Research/debugging: eager gives you Python-level stack traces, print() works, pdb works
- Small models / prototyping: compilation overhead > runtime savings
- The two-persona workflow: eager for iteration, compile for production
- Rule of thumb: if your forward pass is < 10ms eager, compilation may not help
- When to move to torch.compile: next chapter
```

*Question raised → "Eager works but it's slow. How do I make it faster?"*

*Next: [Chapter 7](ch07-torch-compile) — torch.compile on Neuron.*
